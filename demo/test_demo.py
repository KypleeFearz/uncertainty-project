"""Tests for the demo adapter and app. Run from the project root:  pytest -q demo

No network: live model calls are replaced by a fake client. Tests that need the real project data
(stored responses, fitted assessors) are skipped when those files are absent.
"""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import numpy as np

import adapter as A
from uncertainty_project.features import feature_frame
from app import CURATED, make_handler

ROOT = A.ROOT
HAS_DATA = (A.experiment_directory(ROOT) / "frozen_protocol.json").exists() and (ROOT / "data/processed/test_inputs.jsonl").exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="needs the real project outputs")


class FakeOpenAI:
    """Mimics client.responses.parse (OpenAI) and client.messages.create (Anthropic)."""
    def __init__(self):
        self.calls = 0
        self.responses = SimpleNamespace(parse=self._parse)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, model, system, messages, output_config, **kw):
        self.calls += 1
        schema_keys = set(output_config["format"]["schema"]["properties"])
        if "entailment" in schema_keys:
            payload = {"entailment": 0.2, "neutral": 0.7, "contradiction": 0.1, "lexical_ambiguity": 0.8,
                       "unclear_reference": 0.1, "probabilistic_inference": 0.3, "brief_reason": "Fake output."}
        else:
            payload = {"answer_text": "Fake framing.", "reading_1": "", "reading_2": "",
                       "divergence": "", "question": "Which x do you mean?"}
        return SimpleNamespace(stop_reason="end_turn", model=model, id="fake", usage=None,
                               content=[SimpleNamespace(type="text", text=json.dumps(payload))])

    def _parse(self, model, input, text_format, **kw):
        self.calls += 1
        if text_format is A.NLIResult:
            payload = A.NLIResult(entailment=0.2, neutral=0.7, contradiction=0.1, lexical_ambiguity=0.8,
                                  unclear_reference=0.1, probabilistic_inference=0.3, brief_reason="Fake output.")
        else:
            payload = A.Wording(answer_text="Fake framing.", reading_1="One reading.", reading_2="Another reading.",
                                divergence="The word 'x'.", question="Which x do you mean?")
        return SimpleNamespace(status="completed", output_parsed=payload, usage=None, model=model, id="fake")


@pytest.fixture(scope="module")
def adapter_fake(tmp_path_factory):
    fake = FakeOpenAI()
    ad = A.Adapter(cache_dir=tmp_path_factory.mktemp("democache"), client_factory=lambda settings: fake)
    return ad, fake


@needs_data
def test_stored_decisions_equal_frozen_evaluation(adapter_fake):
    """Every stored test/SNLI decision from the adapter must equal the batch decision of the frozen pipeline."""
    ad, _ = adapter_fake
    for split in ("test", "shift"):
        labels = A.load_split(ROOT, split, include_test_labels=True)
        for key in ad.keys:
            batch, _ = A.run_policies(ad.bundles[key], feature_frame(ROOT, labels, key))["decomposed"]
            results = [ad.assess(key, uid=u) for u in labels.uid]
            single = [r["decision"]["action"] for r in results]
            assert list(batch) == single, (split, key)
            features = feature_frame(ROOT, labels, key)
            predictions = A.predict_assessors(ad.bundles[key], features)
            assert [r["decision"]["model_label"] for r in results] == [A.LABELS[int(i)] for i in features.base_label]
            np.testing.assert_allclose([r["decision"]["scores"]["error_risk"] for r in results],
                                       predictions["risks"]["epistemic_tree_full"], atol=1e-12)
            np.testing.assert_allclose([r["decision"]["scores"]["disagreement"] for r in results],
                                       A.entropy(predictions["distributions"]["tree_full"]), atol=1e-12)


@needs_data
def test_baseline_rules_equal_frozen_policies(adapter_fake):
    """The 'simpler rules' shown in the demo must be exactly the frozen comparison policies."""
    ad, _ = adapter_fake
    labels = A.load_split(ROOT, "test", include_test_labels=True)
    for key in ad.keys:
        batch = A.run_policies(ad.bundles[key], feature_frame(ROOT, labels, key))
        for i, uid in enumerate(labels.uid[:120]):
            shown = {b["key"]: b["action"] for b in ad.assess(key, uid=uid)["decision"]["baselines"]}
            assert shown == {k: str(batch[k][0][i]) for k in A.BASELINES}, (key, uid)


@needs_data
def test_compare_returns_every_model(adapter_fake):
    ad, fake = adapter_fake
    res = ad.compare(uid="95883e")
    assert set(res) == set(ad.keys)
    assert res["gpt"]["decision"]["action"] == "answer" and res["claude"]["decision"]["action"] == "alternatives"
    assert res["gpt"]["human"]["matches_reference"] is False and res["claude"]["human"]["matches_reference"] is True
    before = fake.calls
    live = ad.compare(premise="A dog runs in the park.", hypothesis="An animal is outside.")
    assert set(live) == set(ad.keys) and fake.calls - before == 10


@needs_data
def test_curated_examples_exist_and_show_what_they_claim(adapter_fake):
    ad, _ = adapter_fake
    expected = {"65353n": "answer", "106411n": "hedge", "66260c": "alternatives", "13531e": "clarify",
                "95883e": "answer", "snli_3552435734.jpg#0r1e": "answer"}
    for case in CURATED:
        r = ad.assess(case["model"], uid=case["uid"])
        assert r["decision"]["action"] == expected[case["uid"]], case["uid"]
        assert r["human"] is not None and r["mode"] == "stored"


@needs_data
def test_live_input_uses_demo_cache_only(adapter_fake):
    ad, fake = adapter_fake
    research = sorted(p.name for p in (ROOT / "outputs/runs").rglob("*.json"))
    before = fake.calls
    r = ad.assess("gpt", premise="The bank was steep.", hypothesis="The river bank was steep.")
    assert r["mode"] == "live" and fake.calls - before == 5 and r["human"] is None
    c = ad.assess("claude", premise="The bank was steep.", hypothesis="The river bank was steep.")
    assert fake.calls - before == 10 and c["model"] == "claude"
    before = fake.calls
    assert r["decision"]["action"] in ("answer", "hedge", "alternatives", "clarify")
    again = ad.assess("gpt", premise="The bank was steep.", hypothesis="The river bank was steep.")
    assert fake.calls == before, "second request must come from the demo cache"
    assert again["decision"] == r["decision"]
    assert sorted(p.name for p in (ROOT / "outputs/runs").rglob("*.json")) == research, "research cache must not change"


@needs_data
def test_llm_wording_keeps_the_frozen_decision(adapter_fake):
    ad, _ = adapter_fake
    plain = ad.assess("claude", uid="104805e")
    worded = ad.assess("claude", uid="104805e", wording="llm")
    assert worded["decision"] == plain["decision"] and worded["wording_source"] == "llm"
    assert worded["text"]["question"].endswith("?")


@needs_data
def test_input_validation(adapter_fake):
    ad, _ = adapter_fake
    for kwargs in ({"premise": "", "hypothesis": "x"}, {"premise": "x" * 700, "hypothesis": "y"}, {"uid": "nope"}):
        with pytest.raises(ValueError):
            ad.assess("gpt", **kwargs)
    with pytest.raises(ValueError):
        ad.assess("llama", uid="65353n")


@needs_data
def test_template_wording_for_every_action(adapter_fake):
    ad, _ = adapter_fake
    for case in CURATED:
        text = ad.assess(case["model"], uid=case["uid"])["text"]
        assert text["answer_text"]


@needs_data
def test_http_endpoints(adapter_fake):
    ad, _ = adapter_fake
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ad, CURATED))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"Uncertain or Genuinely Debatable?" in urllib.request.urlopen(base + "/").read()
        meta = json.loads(urllib.request.urlopen(base + "/api/meta").read())
        assert meta["protocol_id"] and len(meta["curated"]) == 6
        req = urllib.request.Request(base + "/api/assess", data=json.dumps({"uid": "66260c", "model": "gpt"}).encode(),
                                     headers={"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req).read())["decision"]["action"] == "alternatives"
        cmp = urllib.request.Request(base + "/api/compare", data=json.dumps({"uid": "66260c"}).encode(),
                                     headers={"Content-Type": "application/json"})
        assert set(json.loads(urllib.request.urlopen(cmp).read())) == set(ad.keys)
        bad = urllib.request.Request(base + "/api/assess", data=b'{"model":"gpt","premise":""}',
                                     headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(bad)
        assert err.value.code == 400
    finally:
        server.shutdown()


@needs_data
def test_freeze_seal_still_valid_with_demo():
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from verify_freeze import check
    _, changed, missing, added = check(ROOT)
    assert not (changed or missing or added)


@needs_data
def test_coarse_and_fine_reference_are_distinct(adapter_fake):
    ad, _ = adapter_fake
    r = ad.assess("gpt", uid="121910c")
    assert r["decision"]["action"] == "alternatives"
    assert r["human"]["reference_coarse"] == "debatable"
    assert r["human"]["reference_fine"] == "clarify"
    assert r["human"]["matches_reference"] is True
    assert r["human"]["matches_reference_fine"] is False


@needs_data
def test_comparison_references_are_model_specific(adapter_fake):
    ad, _ = adapter_fake
    pair = ad.compare(uid="103354n")
    assert pair["gpt"]["human"]["reference_coarse"] == "hedge"
    assert pair["claude"]["human"]["reference_coarse"] == "answer"


@needs_data
def test_corrupt_live_cache_is_rejected(adapter_fake):
    ad, fake = adapter_fake
    row = {"premise": "Cache validation fixture.", "hypothesis": "Fixture."}
    p = ad._demo_cache_path("gpt", A.build_messages(row, 0))
    A.write_json(p, {"parsed": {"entailment": 4.0}})
    with pytest.raises(A.ValidationError):
        ad.assess("gpt", **row)


@needs_data
def test_invalid_wording_falls_back_without_losing_decision(adapter_fake, monkeypatch):
    ad, _ = adapter_fake
    plain = ad.assess("gpt", uid="65353n")
    def broken(*args):
        raise ValueError("invalid provider output")
    monkeypatch.setattr(ad, "llm_wording", broken)
    r = ad.assess("gpt", uid="65353n", wording="llm")
    assert r["decision"] == plain["decision"]
    assert r["wording_source"] == "template" and r["wording_warning"]
    with pytest.raises(ValueError):
        ad.validate_wording({"answer_text": "Answer.", "reading_1": "Unexpected", "reading_2": "", "divergence": "", "question": ""}, "answer")


@needs_data
def test_concurrent_identical_inputs_make_only_five_calls(adapter_fake):
    from concurrent.futures import ThreadPoolExecutor
    ad, fake = adapter_fake
    before = fake.calls
    def assess(_):
        return ad.assess("gpt", premise="Concurrency regression fixture.", hypothesis="A test fixture.")
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(assess, range(2)))
    assert fake.calls - before == 5
    assert rows[0]["decision"] == rows[1]["decision"]


@needs_data
def test_local_http_rejects_cross_origin_and_non_json(adapter_fake):
    ad, fake = adapter_fake
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ad, CURATED))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    before = fake.calls
    try:
        for headers, status in [({"Content-Type": "application/json", "Origin": "https://example.org"}, 403),
                                ({"Content-Type": "text/plain"}, 415)]:
            req = urllib.request.Request(base + "/api/assess", data=b'{}', headers=headers)
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(req)
            assert err.value.code == status
        assert fake.calls == before
    finally:
        server.shutdown(); server.server_close()


@needs_data
def test_no_research_file_bytes_change_during_demo_calls(adapter_fake):
    ad, _ = adapter_fake
    files = [p for directory in ("src", "config", "data/processed", "outputs")
             for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    before = {p: A.digest(p.read_bytes()) for p in files}
    ad.compare(uid="95883e")
    ad.assess("gpt", premise="Read-only research fixture.", hypothesis="A test.")
    assert before == {p: A.digest(p.read_bytes()) for p in files}


@needs_data
def test_startup_refuses_an_invalid_freeze(monkeypatch, tmp_path):
    import verify_freeze
    monkeypatch.setattr(verify_freeze, "check", lambda root: ({}, ["config/project.json"], [], []))
    with pytest.raises(RuntimeError, match="changed since freezing"):
        A.Adapter(cache_dir=tmp_path)


# ---------------------------------------------------------------- post-hoc Apertus column (added 2026-11-07)
HAS_APERTUS = (ROOT / "extension_apertus/outputs_v2/mnli_test").exists()
needs_apertus = pytest.mark.skipif(not (HAS_DATA and HAS_APERTUS), reason="needs the Apertus extension outputs")


@needs_apertus
def test_apertus_column_equals_batch_transfer_for_curated_cases(adapter_fake):
    """The demo's Apertus decisions must equal the batch transfer evaluation in extension_apertus."""
    import sys
    sys.path.insert(0, str(ROOT / "extension_apertus"))
    import evaluate_apertus as ev
    from uncertainty_project.models import run_policies
    ad, fake = adapter_fake
    before = fake.calls
    examples = ad.examples.set_index("uid")
    for case in CURATED:
        res = ad.apertus(case["uid"])
        assert res["posthoc"] is True and set(res["assessors"]) == set(ad.keys)
        split = examples.loc[case["uid"], "split"]
        inputs = A.load_split(ROOT, split)
        feats, _ = ev.apertus_features(inputs, ev.load_records(ev.SPLITS[split]))
        feats = feats[feats.uid == case["uid"]].reset_index(drop=True)
        for k in ad.keys:
            actions, _ = run_policies(ad.bundles[k], feats)["decomposed"]
            assert res["assessors"][k]["action"] == actions[0], (case["uid"], k)
    assert fake.calls == before  # stored answers only, never a live call


@needs_apertus
def test_apertus_excluded_and_unknown_cases_are_refused(adapter_fake):
    ad, _ = adapter_fake
    with pytest.raises(LookupError):
        ad.apertus("1073c")  # all five Apertus answers invalid -> excluded from the evaluation
    with pytest.raises(ValueError):
        ad.apertus("not-a-case")


@needs_apertus
def test_apertus_http_endpoint(adapter_fake):
    ad, _ = adapter_fake
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ad, CURATED))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert json.loads(urllib.request.urlopen(base + "/api/meta").read())["apertus"] is True
        req = urllib.request.Request(base + "/api/apertus", data=json.dumps({"uid": "104805e"}).encode(),
                                     headers={"Content-Type": "application/json"})
        body = json.loads(urllib.request.urlopen(req).read())
        assert body["model"] == "apertus" and body["posthoc"] is True
        bad = urllib.request.Request(base + "/api/apertus", data=json.dumps({"uid": "1073c"}).encode(),
                                     headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(bad)
        assert err.value.code == 400
    finally:
        server.shutdown(); server.server_close()

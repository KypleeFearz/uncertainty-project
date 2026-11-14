"""Technical checks use artificial examples. They are never research observations."""
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from anthropic import Anthropic
from openai import OpenAI

from uncertainty_project.api import (NLIResult, build_messages, cache_path, collect_predictions, experiment_directory,
                                     model_keys, run_spec)
from uncertainty_project.common import digest, read_json, write_json, write_jsonl
from uncertainty_project.data import make_splits, normalized_text
from uncertainty_project.experiment import final_evaluation, freeze_experiment, train_development, verify_frozen
from uncertainty_project.features import (FULL_FEATURES, credal_signals, entropy, majority_correct,
                                          summarize_predictions)
from uncertainty_project.models import label_error_rates
from uncertainty_project.metrics import dissociation_table, group_bootstrap_metric, js_divergence, risk_coverage
from uncertainty_project.policy import choose_actions, choose_actions_single, policy_metrics, reference_actions

REPO = Path(__file__).resolve().parents[1]


def prediction(p=(0.6, 0.3, 0.1), cue=0.4):
    return dict(entailment=p[0], neutral=p[1], contradiction=p[2], lexical_ambiguity=cue,
                unclear_reference=cue / 2, probabilistic_inference=cue, brief_reason="Artificial unit-test output.")


def test_shared_premises_cannot_cross_splits():
    frame = pd.DataFrame({"uid": [str(i) for i in range(120)], "group_id": [str(i // 3) for i in range(120)]})
    a = make_splits(frame, seed=17)
    b = make_splits(frame.sample(frac=1, random_state=7), seed=17)
    pd.testing.assert_frame_equal(a, b)
    assert a.groupby("group_id").split.nunique().max() == 1
    assert set(a.split) == {"train", "dev", "test"}
    assert normalized_text(" HELLO\n  café ") == normalized_text("hello café")


def test_gold_labels_and_ids_are_not_sent_to_model():
    row = {"premise": "A person reads.", "hypothesis": "A person is asleep.",
           "uid": "NEVER_SEND_THIS_ID", "human_distribution": [0.13, 0.14, 0.73],
           "taxonomy_categories": ["PRIVATE_GOLD_MARKER"]}
    messages = build_messages(row, 0)
    text = json.dumps(messages)
    assert "NEVER_SEND_THIS_ID" not in text and "PRIVATE_GOLD_MARKER" not in text
    payload = json.loads(messages[1]["content"].split("\n", 1)[1])
    assert set(payload) == {"premise", "hypothesis"}


def test_metrics_known_extremes_and_tied_majorities():
    assert js_divergence([1, 0, 0], [1, 0, 0]) == pytest.approx(0)
    assert js_divergence([1, 0, 0], [0, 1, 0]) == pytest.approx(1)
    assert entropy([1 / 3] * 3) == pytest.approx(1)
    assert entropy([1, 0, 0]) == pytest.approx(0)
    np.testing.assert_array_equal(majority_correct([0, 1, 2], [[.5, .5, 0]] * 3), [1, 1, 0])
    curve = risk_coverage([.2, .2, .2, .2], [0, 0, 1, 1])
    np.testing.assert_allclose(curve.risk, 0.5)


def test_decomposition_separates_aleatoric_from_epistemic():
    # Five heads that agree on a flat distribution: all aleatoric, no epistemic part.
    flat = summarize_predictions([prediction((1 / 3, 1 / 3, 1 / 3)) for _ in range(5)])
    assert flat["epistemic_mi"] == pytest.approx(0)
    assert flat["expected_entropy"] == pytest.approx(1)
    assert flat["credal_width"] == pytest.approx(0)
    # Five confident heads that disagree: low expected entropy, high epistemic part, wide intervals.
    sharp = [(.98, .01, .01), (.01, .98, .01), (.01, .01, .98), (.98, .01, .01), (.01, .98, .01)]
    split = summarize_predictions([prediction(p) for p in sharp])
    assert split["expected_entropy"] < 0.15
    assert split["epistemic_mi"] > 0.7
    assert split["credal_width_max"] == pytest.approx(0.97)
    assert split["credal_determinate"] == 0
    assert credal_signals([(.8, .1, .1), (.7, .2, .1)])["credal_determinate"] == 1
    assert flat["mean_entropy"] == pytest.approx(flat["expected_entropy"] + flat["epistemic_mi"])
    assert not any("human" in f or "taxonomy" in f for f in FULL_FEATURES)
    with pytest.raises(ValueError): summarize_predictions([prediction()])
    with pytest.raises(ValueError): NLIResult.model_validate(prediction((.8, .8, .8)))


def test_four_actions_and_reference_actions():
    p = [[.9, .05, .05], [.9, .05, .05], [.4, .35, .25], [.4, .35, .25]]
    actions, _ = choose_actions(p, [.05, .9, .1, .1], [.1, .1, .1, .9],
                                aleatoric_threshold=.8, epistemic_threshold=.5, cue_threshold=.5)
    assert actions.tolist() == ["answer", "hedge", "alternatives", "clarify"]
    human = [[.95, .03, .02], [.95, .03, .02], [.4, .35, .25], [.4, .35, .25]]
    ref, fine = reference_actions(human, [0, 1, 0, 0], .8, [None, None, ["Probabilistic"], ["Coreference"]])
    assert ref.tolist() == ["answer", "hedge", "debatable", "debatable"]
    assert fine.tolist() == ["answer", "hedge", "alternatives", "clarify"]
    m = policy_metrics(actions, ref, fine)
    assert m["balanced_accuracy"] == 1 and m["fine_balanced_accuracy_annotated"] == 1


def test_missed_clarifications_are_not_hidden_by_conditional_metric():
    # 100 cases that call for a clarification; the system asks once and answers directly 99 times.
    ref = np.array(["debatable"] * 100, dtype=object)
    fine = np.array(["clarify"] * 100, dtype=object)
    actions = np.array(["clarify"] + ["answer"] * 99, dtype=object)
    m = policy_metrics(actions, ref, fine)
    assert m["conditional_clarify_vs_alternatives_accuracy"] == 1 and m["conditional_n"] == 1
    assert m["recall_clarify_annotated"] == pytest.approx(0.01)
    assert m["recall_debatable"] == pytest.approx(0.01)


def test_label_only_baseline_uses_train_error_rate_per_answer_label():
    labels = np.array([0, 0, 1, 1, 1, 1, 2])
    errors = np.array([0, 0, 1, 1, 0, 0, 0], float)
    clear = np.array([1, 1, 1, 1, 0, 0, 0], bool)
    rates = label_error_rates(labels, errors, clear)
    assert rates["clear"] == [0.0, 1.0, 0.5]  # label 2 never clear -> overall clear rate
    assert rates["all"] == [0.0, 0.5, 0.0]


def test_single_score_policy_and_metric_bootstrap():
    actions = choose_actions_single([.1, .5, .9, .9], [0, 0, 0, .8], hedge_threshold=.4, debatable_threshold=.8, cue_threshold=.5)
    assert actions.tolist() == ["answer", "hedge", "alternatives", "clarify"]
    values = np.arange(40, dtype=float)
    ci = group_bootstrap_metric(lambda i: float(values[i].mean()), np.repeat(np.arange(20), 2), 200, 1)
    assert ci["ci_95_low"] < ci["estimate"] < ci["ci_95_high"]


def test_dissociation_detects_the_intended_pattern():
    rng = np.random.default_rng(1)
    n = 400
    q = rng.dirichlet([1, 1, 1], n)
    h = entropy(q)
    errors = (rng.random(n) < 0.3).astype(float)
    aleatoric_signal = h + rng.normal(0, .05, n)
    epistemic_signal = errors + rng.normal(0, .3, n)
    clear = h <= np.quantile(h, .33)
    t = dissociation_table({"a": aleatoric_signal, "e": epistemic_signal}, q, errors, clear, np.quantile(h, .5)).set_index("signal")
    assert t.loc["a", "auroc_debatable"] > .9 and abs(t.loc["a", "auroc_error_on_clear_cases"] - .5) < .15
    assert t.loc["e", "auroc_error_on_clear_cases"] > .8 and abs(t.loc["e", "auroc_debatable"] - .5) < .15


def _chaos_row(uid, counts, premise):
    codes = ("e", "n", "c")
    return {"uid": uid, "label_counter": {c: n for c, n in zip(codes, counts) if n}, "majority_label": codes[int(np.argmax(counts))],
            "label_dist": [n / 100 for n in counts], "label_count": list(counts),
            "example": {"uid": uid, "premise": premise, "hypothesis": "Artificial hypothesis.", "source": "artificial"}}


def test_unfamiliar_input_sample_is_fixed_prefixed_and_hides_labels(tmp_path):
    from uncertainty_project.data import load_split, prepare_shift
    root = tmp_path / "shift_fixture"
    raw = [_chaos_row(f"x{i}", (50, 30, 20) if i % 2 else (90, 5, 5), f"Artificial premise {i % 7}") for i in range(30)]
    write_jsonl(root / "data/raw/artificial_snli.jsonl", raw)
    data = (root / "data/raw/artificial_snli.jsonl").read_bytes()
    write_json(root / "config/data_sources.json", {"chaos_snli": {"filename": "artificial_snli.jsonl", "url": "offline",
               "sha256": digest(data), "expected_rows": 30}})
    write_json(root / "config/project.json", {"shift": {"source": "chaos_snli", "size": 12, "seed": 3}})
    # premise 0 also occurs in the (artificial) MNLI data -> those pairs must be excluded before sampling
    write_json(root / "data/processed/splits.json", {"assignment": [{"uid": "m1", "split": "train",
               "group_id": digest(normalized_text("Artificial premise 0"))}]})
    first = prepare_shift(root)
    assert first["sample"] == 12 and first["source_rows"] == 30 and first["excluded_premise_overlap_with_mnli"] == 5
    assert "Artificial premise 0" not in set(load_split(root, "shift").premise)
    inputs = load_split(root, "shift")
    assert len(inputs) == 12 and inputs.uid.str.startswith("snli_").all()
    assert "human_distribution" not in inputs.columns
    assert "human_distribution" in load_split(root, "shift", include_test_labels=True).columns
    before = (root / "data/processed/shift_inputs.jsonl").read_bytes()
    prepare_shift(root)  # deterministic: same sample again
    assert (root / "data/processed/shift_inputs.jsonl").read_bytes() == before
    write_json(root / "config/project.json", {"shift": {"source": "chaos_snli", "size": 12, "seed": 4}})
    with pytest.raises(ValueError, match="differs"):
        prepare_shift(root)


def make_artificial_project(tmp_path):
    root = tmp_path / "artificial_fixture"
    root.mkdir()
    cfg = read_json(REPO / "config/project.json")
    cfg["trees"] = {"max_depth": [2], "min_samples_leaf": [10]}
    cfg["bootstrap_repetitions"] = 25
    cfg["api"]["parallel_requests"] = 2
    write_json(root / "config/project.json", cfg)
    write_json(root / "config/data_sources.json", read_json(REPO / "config/data_sources.json"))
    (root / "src/uncertainty_project").mkdir(parents=True)
    (root / "src/uncertainty_project/fixture.py").write_text("# Artificial test fixture\n")
    assignment, all_rows = [], []
    rng = np.random.default_rng(13)
    for split, n in (("train", 160), ("dev", 80), ("test", 60), ("shift", 40)):
        rows = []
        for i in range(n):
            q = rng.dirichlet([1.2, 1.2, 1.2]) if i % 3 else rng.dirichlet([20, 1, 1])
            uid = f"fixture_{split}_{i}"
            row = {"uid": uid, "group_id": uid, "premise": f"Artificial premise {uid}",
                   "hypothesis": "Artificial hypothesis.", "human_distribution": q.tolist(),
                   "human_counts": None, "taxonomy_categories": ["Lexical"] if i % 2 else ["Probabilistic"],
                   "taxonomy_round": 1}
            rows.append(row); all_rows.append(row)
            if split != "shift":
                assignment.append({"uid": uid, "group_id": uid, "split": split})
        if split in ("test", "shift"):
            write_jsonl(root / f"data/processed/{split}_inputs.jsonl", [{k: r[k] for k in ("uid", "group_id", "premise", "hypothesis")} for r in rows])
            write_jsonl(root / f"data/processed/{split}_labels.jsonl", [{k: v for k, v in r.items() if k not in ("group_id", "premise", "hypothesis")} for r in rows])
        else:
            write_jsonl(root / f"data/processed/{split}.jsonl", rows)
    write_json(root / "data/processed/splits.json", {"assignment": assignment})
    return root, all_rows


def seed_artificial_cache(root, rows):
    # Technical fixtures only in pytest's temporary directory, never the real project.
    for key, noise in (("gpt", .15), ("claude", .25)):
        rng = np.random.default_rng(53 if key == "gpt" else 54)
        for row in rows:
            for v in range(5):
                q = np.array(row["human_distribution"])
                noisy = np.clip(q + rng.normal(0, noise, 3), .005, 1); noisy /= noisy.sum()
                payload = prediction(noisy, float(rng.random()))
                write_json(cache_path(root, row, v, key), {"status": "ok", "uid": row["uid"], "variant": v,
                           "run_digest": digest(run_spec(root, key)), "messages_digest": digest(build_messages(row, v)),
                           "parsed": payload, "usage": {}, "fixture_notice": "Artificial integration-test data; not research observations."})


def test_openai_sdk_format_and_resume_without_network(tmp_path, monkeypatch):
    root, rows = make_artificial_project(tmp_path)
    calls = []
    def fake_http(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["store"] is False
        assert body["text"]["format"]["type"] == "json_schema"
        assert body["reasoning"]["effort"] == "low"
        return httpx.Response(200, json={"id": "resp_offline_fixture", "object": "response", "created_at": 0,
             "status": "completed", "model": "gpt-6.1-sol",
             "output": [{"id": "msg_fixture", "type": "message", "role": "assistant", "status": "completed",
                         "content": [{"type": "output_text", "text": json.dumps(prediction()), "annotations": []}]}],
             "usage": {"input_tokens": 100, "output_tokens": 40, "total_tokens": 140}})
    client = OpenAI(api_key="offline-test-placeholder", http_client=httpx.Client(transport=httpx.MockTransport(fake_http)))
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: client)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-placeholder")
    frame = pd.DataFrame(rows[:1])
    assert collect_predictions(root, frame, "gpt")["mode"] == "dry_run"
    assert not calls
    collect_predictions(root, frame, "gpt", allow_api=True, max_new_requests=5)
    assert len(calls) == 5
    assert collect_predictions(root, frame, "gpt", allow_api=True, max_new_requests=5)["mode"] == "cache_only"
    assert len(calls) == 5
    with pytest.raises(RuntimeError, match="freeze"):
        collect_predictions(root, pd.DataFrame(rows[-1:]), "gpt", allow_api=True, max_new_requests=5)


def test_anthropic_structured_output_retry_and_failure_records(tmp_path, monkeypatch):
    import httpx2  # the anthropic SDK (v1+) ships its own httpx fork
    root, rows = make_artificial_project(tmp_path)
    calls = []
    def fake_http(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["model"] == "claude-sonnet-5-5"
        assert body["output_config"]["format"]["type"] == "json_schema"
        assert body["output_config"]["effort"] == "low"
        assert "NEVER" not in json.dumps(body) and body["system"].startswith("You perform")
        # First answer violates the sum-to-one rule and must be retried, never repaired.
        text = json.dumps(prediction((.8, .8, .8)) if len(calls) == 1 else prediction())
        return httpx2.Response(200, json={"id": f"msg_{len(calls)}", "type": "message", "role": "assistant",
             "model": "claude-sonnet-5-5", "stop_reason": "end_turn", "stop_sequence": None,
             "content": [{"type": "text", "text": text}],
             "usage": {"input_tokens": 120, "output_tokens": 60}})
    client = Anthropic(api_key="offline-test-placeholder", http_client=httpx2.Client(transport=httpx2.MockTransport(fake_http)))
    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-test-placeholder")
    cfg = read_json(root / "config/project.json"); cfg["api"]["parallel_requests"] = 1
    write_json(root / "config/project.json", cfg)
    result = collect_predictions(root, pd.DataFrame(rows[:1]), "claude", allow_api=True, max_new_requests=5)
    assert result["cached"] == 5 and len(calls) == 6
    first = read_json(cache_path(root, rows[0], 0, "claude"))
    assert first["attempts"] == 2 and first["failed_attempts"][0]["status"] == "schema_validation_failed"


def test_end_to_end_two_models_freeze_transfer_and_evaluation(tmp_path):
    root, rows = make_artificial_project(tmp_path)
    assert model_keys(root) == ["gpt", "claude"]
    seed_artificial_cache(root, rows)
    bundles, dev_report = train_development(root)
    assert set(bundles) == {"gpt", "claude"}
    assert set(dev_report["transfer"]) == {"gpt_to_claude", "claude_to_gpt"}
    diss = dev_report["per_model"]["gpt"]["dissociation"]
    assert {"learned_epistemic_risk", "joint_epistemic_mi"} <= set(diss.signal)
    policies = dev_report["per_model"]["gpt"]["policies"]
    assert {"decomposed", "risk_trained_all_cases", "single_score_risk", "single_score_confidence",
            "raw", "always_answer"} <= set(policies.method)
    intervals = dev_report["per_model"]["gpt"]["policy_intervals"]
    assert len(intervals) == 4 and intervals.comparison.str.startswith("balanced_accuracy_decomposed_minus_").all()
    assert "auroc_debatable_ci_low" in diss.columns
    assert {"learned_entropy_direct", "baseline_label_confidence", "baseline_answer_label"} <= set(diss.signal)
    spec = dev_report["per_model"]["gpt"]["specialisation"]
    assert (spec["rank"] == "primary").sum() == 3 and {"auroc_difference", "ci_95_low"} <= set(spec.columns)
    risk = dev_report["per_model"]["gpt"]["error_risk"]
    assert {"label_confidence_tree", "epistemic_label_confidence_tree"} <= set(risk.method)
    assert (experiment_directory(root) / "development" / "dissociation.png").exists()
    from uncertainty_project.data import load_split
    with pytest.raises(RuntimeError, match="freeze"):
        collect_predictions(root, load_split(root, "shift"), "gpt", allow_api=True, max_new_requests=0)
    frozen = freeze_experiment(root)
    assert verify_frozen(root)["protocol_id"] == frozen["protocol_id"]
    report = final_evaluation(root)
    assert len(report["transfer_summary"]) == 2
    assert report["shift"]["meta"]["split"] == "shift" and report["shift"]["meta"]["n"] == 40
    assert (experiment_directory(root) / "shift" / "gpt_dissociation.csv").exists()
    assert read_json(experiment_directory(root) / "evaluation_record.json")["unfamiliar_inputs"]["n"] == 40
    assert report["per_model"]["claude"]["distributions"].js_divergence.between(0, 1).all()
    with pytest.raises(RuntimeError, match="Test"):
        train_development(root)
    (root / "src/uncertainty_project/fixture.py").write_text("# changed after test\n")
    with pytest.raises(RuntimeError, match="changed"):
        verify_frozen(root)

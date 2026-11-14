"""Tests for the post-hoc Apertus extension.  Run:  python -m pytest extension_apertus -q"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))

import evaluate_apertus as ev  # noqa: E402

HAS_OUTPUTS = (ev.OUTPUTS / "mnli_test").exists()


def _record(uid, variant, e=0.6, n=0.3, c=0.1):
    return {"status": "ok", "uid": uid, "variant": variant, "model": ev.MODEL, "temperature": 0.0,
            "prompt_fingerprint": ev.prompt_fingerprint(),
            "parsed": {"entailment": e, "neutral": n, "contradiction": c, "lexical_ambiguity": 0.1,
                       "unclear_reference": 0.1, "probabilistic_inference": 0.2, "brief_reason": "x"}}


def test_prompt_fingerprint_matches_the_collection_run():
    assert ev.prompt_fingerprint() == "37d7e209e7fc9178"


def test_incomplete_cases_are_excluded_not_filled(tmp_path):
    folder = tmp_path / "mnli_test"
    folder.mkdir()
    for v in range(5):
        (folder / f"a__v{v}.json").write_text(json.dumps(_record("a", v)))
    for v in range(4):  # case b misses one prompt
        (folder / f"b__v{v}.json").write_text(json.dumps(_record("b", v)))
    (tmp_path / "mnli_test_failed.jsonl").write_text(
        json.dumps({"uid": "b", "variant": 4, "reason": "Summe 0.00"}) + "\n" +
        json.dumps({"uid": "a", "variant": 2, "reason": "Summe 0.00"}) + "\n")  # a/v2 later succeeded
    records = ev.load_records("mnli_test", tmp_path)
    inputs = pd.DataFrame({"uid": ["a", "b"], "group_id": ["g1", "g2"], "premise": "p", "hypothesis": "h"})
    feats, excluded = ev.apertus_features(inputs, records)
    assert list(feats.uid) == ["a"] and excluded == ["b"]
    invalid = ev.invalid_answers("mnli_test", records, tmp_path)
    assert list(zip(invalid.uid, invalid.variant)) == [("b", 4)]
    assert bool(invalid.all_labels_zero.iloc[0])


def test_records_with_other_prompts_are_rejected(tmp_path):
    folder = tmp_path / "mnli_test"
    folder.mkdir()
    bad = _record("a", 0)
    bad["prompt_fingerprint"] = "0000000000000000"
    (folder / "a__v0.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        ev.load_records("mnli_test", tmp_path)


@pytest.mark.skipif(not HAS_OUTPUTS, reason="Apertus outputs not present")
def test_real_outputs_counts_and_exclusions():
    for split, expected in (("test", 310), ("shift", 294)):
        from uncertainty_project.data import load_split
        inputs = load_split(ev.ROOT, split)
        records = ev.load_records(ev.SPLITS[split])
        feats, excluded = ev.apertus_features(inputs, records)
        invalid = ev.invalid_answers(ev.SPLITS[split], records)
        assert len(feats) == expected
        assert set(excluded) == set(invalid.uid)
        assert len(feats) + len(excluded) == len(inputs)


@pytest.mark.skipif(not HAS_OUTPUTS, reason="project data not present")
def test_notebook03_is_reproduced_and_seal_is_intact():
    from uncertainty_project.data import load_split
    from uncertainty_project.experiment import _labels_for, load_bundle
    from uncertainty_project.features import feature_frame, joint_feature_frame
    import verify_freeze
    bundles = {k: load_bundle(ev.ROOT, k) for k in ev.FROZEN_MODELS}
    inputs, labels = load_split(ev.ROOT, "test"), _labels_for(ev.ROOT, "test")
    full = {k: feature_frame(ev.ROOT, inputs, k) for k in ev.FROZEN_MODELS}
    joint = joint_feature_frame(ev.ROOT, inputs, list(ev.FROZEN_MODELS))
    check = ev.reproduce_notebook03(ev.ROOT, "test", full, joint, labels, bundles)
    assert check.ok.all(), check[~check.ok]
    _, changed, missing, added = verify_freeze.check(ev.ROOT)
    assert not (changed or missing or added)

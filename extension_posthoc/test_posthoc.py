"""Checks for the post-hoc robustness extension. Run: python -m pytest -q extension_posthoc/test_posthoc.py

The tests read results/ produced by evaluate_posthoc.py and compare the rows that use the FROZEN definition of
'clear' with the sealed exports of notebook 03. If these agree, the extension measures the same quantities as
the frozen evaluation and the stricter definition is the only thing that changes.
"""
from pathlib import Path
import sys

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from uncertainty_project.experiment import experiment_directory  # noqa: E402
from verify_freeze import check  # noqa: E402

RESULTS = HERE / "results"
FROZEN = "frozen entropy definition"


def _sens():
    return pd.read_csv(RESULTS / "clear_sensitivity.csv")


def test_seal_is_intact():
    _, changed, missing, added = check(ROOT)
    assert not (changed or missing or added)


@pytest.mark.parametrize("split", ["test", "shift"])
@pytest.mark.parametrize("model", ["gpt", "claude"])
def test_frozen_definition_reproduces_sealed_exports(split, model):
    sealed = pd.read_csv(experiment_directory(ROOT) / split / f"{model}_error_risk.csv")
    sealed = sealed[sealed.subset == "clear_cases"].set_index("method")
    mine = _sens()
    mine = mine[(mine.split == split) & (mine.model == model) & (mine.definition == FROZEN)].set_index("quantity")
    pairs = {"auroc_clear_case_risk_tree": "epistemic_tree_full",
             "auroc_label_and_confidence_tree": "epistemic_label_confidence_tree", "auroc_raw_confidence": "raw_confidence"}
    for quantity, method in pairs.items():
        assert mine.loc[quantity, "estimate"] == pytest.approx(sealed.loc[method, "error_auroc"], abs=1e-9)
        assert int(mine.loc[quantity, "clear_cases"]) == int(sealed.loc[method, "n"])
        assert int(mine.loc[quantity, "deviations"]) == int(sealed.loc[method, "errors"])


def test_strict_cases_are_a_subset_with_fewer_deviations():
    s = _sens()
    for (split, model), g in s.groupby(["split", "model"]):
        frozen = g[g.definition == FROZEN].iloc[0]
        strict = g[g.definition != FROZEN].iloc[0]
        assert strict.deviations <= frozen.deviations


def test_composition_adds_up():
    comp = pd.read_csv(RESULTS / "clear_composition.csv")
    summ = pd.read_csv(RESULTS / "clear_summary.csv").set_index(["split", "model"])
    for key, g in comp.groupby(["split", "model"]):
        assert g.clear_cases.sum() == summ.loc[key, "clear_cases"]
        assert g.deviations.sum() == summ.loc[key, "deviations"]


def test_source_separation_table_is_complete():
    src = pd.read_csv(RESULTS / "source_separation.csv")
    assert set(src.model) == {"gpt", "claude"}
    assert src.auroc_snli.between(0, 1).all()
    assert (src[src.stratum == "all cases"].n_mnli == 327).all() and (src[src.stratum == "all cases"].n_snli == 300).all()

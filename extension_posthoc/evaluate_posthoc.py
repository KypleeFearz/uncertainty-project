"""Post-hoc robustness checks added after the freeze (10 November 2026).

Only READS the frozen project (src/, config/, data/processed/, fitted assessors, stored responses) and the
Apertus features exported by extension_apertus. Nothing is refitted or retuned. New files are written to
extension_posthoc/results/ only. Everything here is exploratory and not part of the sealed analysis plan.

Two questions:

A. Sensitivity to the definition of a 'clear' case. The frozen definition uses three-class entropy, which admits
   two-way splits such as 59/41/0. Do the clear-case findings hold when 'clear' requires a human majority of at
   least 80 percent? The frozen assessors are applied unchanged; only the evaluation subset changes.

B. Source separation. Do the frozen signals tell MNLI test inputs from SNLI inputs? This is a first, simple
   look at out-of-distribution detection. It is reported for all cases and within human-agreement strata,
   because the two sources differ strongly in how often people disagree.

    python extension_posthoc/evaluate_posthoc.py            # 500 bootstrap repetitions
    python extension_posthoc/evaluate_posthoc.py --quick    # 50 repetitions
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from uncertainty_project.common import config, now, write_json  # noqa: E402
from uncertainty_project.data import load_split  # noqa: E402
from uncertainty_project.experiment import _labels_for, experiment_directory, load_bundle  # noqa: E402
from uncertainty_project.features import entropy, feature_frame, majority_correct  # noqa: E402
from uncertainty_project.metrics import group_bootstrap_metric, safe_auroc  # noqa: E402
from uncertainty_project.models import aligned_human, predict_assessors  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from verify_freeze import check as _freeze_check  # noqa: E402  (path-neutral, works on Windows and Linux)

RESULTS = HERE / "results"


def verify_seal():
    record, changed, missing, added = _freeze_check(ROOT)
    if changed or missing or added:
        raise RuntimeError("Sealed files differ from the frozen state. Run python tools/verify_freeze.py for details.")
    return record


MODELS = ("gpt", "claude")
SPLITS = {"test": "MNLI test", "shift": "SNLI sample"}
STRICT_MAJORITY = 0.80
LABELS = "ENC"
BANDS = ((0.0, 0.7, "below 0.70"), (0.7, 0.8, "0.70 to 0.79"), (0.8, 0.9, "0.80 to 0.89"), (0.9, 1.01, "0.90 or more"))
RISK_SIGNALS = {"clear_case_risk_tree": "epistemic_tree_full",
                "label_and_confidence_tree": "epistemic_label_confidence_tree",
                "raw_confidence": "raw_confidence"}


def _auroc(y, score):
    value = safe_auroc(np.asarray(y), np.asarray(score))
    return None if value is None or not np.isfinite(value) else float(value)


def _interval(metric, groups, repetitions, seed):
    out = group_bootstrap_metric(metric, groups, repetitions, seed)
    return out["estimate"], out["ci_95_low"], out["ci_95_high"]


def _prepare(split, model):
    inputs = load_split(ROOT, split)
    labels = _labels_for(ROOT, split)
    bundle = load_bundle(ROOT, model)
    features = feature_frame(ROOT, inputs, model)
    return bundle, features, labels


def _case_table(bundle, features, labels):
    """One row per case: human statistics, model answer and the frozen signals."""
    q = aligned_human(features, labels)
    pred = predict_assessors(bundle, features)
    answer = features.base_label.to_numpy(int)
    frame = pd.DataFrame({
        "uid": features.uid.to_numpy(), "group_id": features.group_id.to_numpy(),
        "human_entropy": entropy(q), "majority_share": q.max(axis=1), "majority_label": q.argmax(axis=1),
        "answer": answer, "deviation": 1 - majority_correct(answer, q),
        "support_for_answer": q[np.arange(len(q)), answer],
        "disagreement_signal": entropy(pred["distributions"]["tree_full"]),
        "model_total_entropy": features.mean_entropy.to_numpy(float),
        "model_expected_entropy": features.expected_entropy.to_numpy(float),
        "model_prompt_mi": features.epistemic_mi.to_numpy(float),
        "model_credal_width": features.credal_width.to_numpy(float),
        "model_one_minus_confidence": 1 - features.mean_confidence.to_numpy(float),
    })
    for name, key in RISK_SIGNALS.items():
        frame[name] = pred["risks"][key]
    th = bundle["human_thresholds"]
    frame["clear_frozen"] = frame.human_entropy <= th["clear_max_entropy"]
    frame["debatable_frozen"] = frame.human_entropy >= th["debatable_min_entropy"]
    frame["clear_strict"] = frame.majority_share >= STRICT_MAJORITY
    return frame


# ---------------------------------------------------------------------------------- A. definition of 'clear'

def composition(cases, split, model):
    rows = []
    clear = cases[cases.clear_frozen]
    for lo, hi, name in BANDS:
        band = clear[(clear.majority_share >= lo) & (clear.majority_share < hi)]
        rows.append({"split": split, "model": model, "majority_share": name, "clear_cases": len(band),
                     "deviations": int(band.deviation.sum())})
    dev = clear[clear.deviation == 1]
    summary = {"split": split, "model": model, "clear_cases": len(clear), "deviations": len(dev),
               "clear_cases_majority_below_0.80": int((clear.majority_share < STRICT_MAJORITY).sum()),
               "deviations_majority_below_0.80": int((dev.majority_share < STRICT_MAJORITY).sum()),
               "median_human_support_for_deviating_answer": float(dev.support_for_answer.median()) if len(dev) else None,
               "deviations_with_support_at_least_0.25": int((dev.support_for_answer >= 0.25).sum())}
    return rows, summary


def error_profile(cases, split, model):
    rows = []
    for name, mask in (("frozen entropy definition", cases.clear_frozen), ("majority at least 0.80", cases.clear_strict)):
        dev = cases[mask & (cases.deviation == 1)]
        counts = dev.groupby(["answer", "majority_label"]).size()
        for (a, m), n in counts.items():
            rows.append({"split": split, "model": model, "definition": name, "clear_cases": int(mask.sum()),
                         "deviations": len(dev), "model_answer": LABELS[a], "human_majority": LABELS[m], "n": int(n)})
    return rows


def sensitivity(cases, split, model, repetitions, seed):
    rows = []
    groups = cases.group_id.to_numpy()
    dev = cases.deviation.to_numpy()
    signals = {**{k: cases[k].to_numpy() for k in RISK_SIGNALS}, "disagreement_signal": cases.disagreement_signal.to_numpy()}
    for name, column in (("frozen entropy definition", "clear_frozen"), ("majority at least 0.80", "clear_strict")):
        mask = cases[column].to_numpy()
        for signal, values in signals.items():
            est, lo, hi = _interval(lambda i: _auroc(dev[i][mask[i]], values[i][mask[i]]), groups, repetitions, seed)
            rows.append({"split": split, "model": model, "definition": name, "clear_cases": int(mask.sum()),
                         "deviations": int(dev[mask].sum()), "quantity": f"auroc_{signal}",
                         "estimate": est, "ci_95_low": lo, "ci_95_high": hi})
        pairs = (("risk_tree_minus_label_and_confidence", "clear_case_risk_tree", "label_and_confidence_tree"),
                 ("risk_tree_minus_disagreement_signal", "clear_case_risk_tree", "disagreement_signal"))
        for label, a, b in pairs:
            def diff(i, a=a, b=b):
                x, y = _auroc(dev[i][mask[i]], signals[a][i][mask[i]]), _auroc(dev[i][mask[i]], signals[b][i][mask[i]])
                return None if x is None or y is None else x - y
            est, lo, hi = _interval(diff, groups, repetitions, seed)
            rows.append({"split": split, "model": model, "definition": name, "clear_cases": int(mask.sum()),
                         "deviations": int(dev[mask].sum()), "quantity": label,
                         "estimate": est, "ci_95_low": lo, "ci_95_high": hi})
    return rows


def apertus_sensitivity(repetitions, seed):
    """Frozen GPT and Claude risk assessors applied to Apertus, under both definitions of 'clear'."""
    rows = []
    for split in SPLITS:
        path = ROOT / "extension_apertus" / "results" / split / "apertus_features.csv"
        if not path.exists():
            continue
        features = pd.read_csv(path, dtype={"uid": str, "group_id": str})
        labels = _labels_for(ROOT, split)
        for source in MODELS:
            cases = _case_table(load_bundle(ROOT, source), features, labels)
            groups, dev = cases.group_id.to_numpy(), cases.deviation.to_numpy()
            for name, column in (("frozen entropy definition", "clear_frozen"), ("majority at least 0.80", "clear_strict")):
                mask = cases[column].to_numpy()
                for signal in ("clear_case_risk_tree", "raw_confidence"):
                    values = cases[signal].to_numpy()
                    est, lo, hi = _interval(lambda i: _auroc(dev[i][mask[i]], values[i][mask[i]]), groups, repetitions, seed)
                    rows.append({"split": split, "assessor": source, "target": "apertus", "definition": name,
                                 "clear_cases": int(mask.sum()), "deviations": int(dev[mask].sum()),
                                 "quantity": f"auroc_{signal}", "estimate": est, "ci_95_low": lo, "ci_95_high": hi})
            if source == MODELS[0]:
                for name, column in (("frozen entropy definition", "clear_frozen"), ("majority at least 0.80", "clear_strict")):
                    d = cases[cases[column] & (cases.deviation == 1)]
                    for (a, m), n in d.groupby(["answer", "majority_label"]).size().items():
                        rows.append({"split": split, "assessor": "", "target": "apertus", "definition": name,
                                     "clear_cases": int(cases[column].sum()), "deviations": len(d),
                                     "quantity": f"profile_model_{LABELS[a]}_humans_{LABELS[m]}", "estimate": int(n),
                                     "ci_95_low": None, "ci_95_high": None})
    return rows


# ---------------------------------------------------------------------------------- B. source separation

SOURCE_SIGNALS = ("model_total_entropy", "model_expected_entropy", "model_prompt_mi", "model_credal_width",
                  "model_one_minus_confidence", "disagreement_signal", "clear_case_risk_tree")


def source_separation(tables, model, repetitions, seed):
    """AUROC for 'this input comes from SNLI'. Values below 0.5 mean the signal is lower on SNLI."""
    both = pd.concat([tables["test"].assign(is_snli=0), tables["shift"].assign(is_snli=1)], ignore_index=True)
    groups = (both.is_snli.astype(str) + "_" + both.group_id.astype(str)).to_numpy()
    y = both.is_snli.to_numpy()
    strata = {"all cases": np.ones(len(both), bool), "clear cases only": both.clear_frozen.to_numpy(),
              "debatable cases only": both.debatable_frozen.to_numpy()}
    rows = []
    for stratum, mask in strata.items():
        for signal in SOURCE_SIGNALS:
            values = both[signal].to_numpy()
            est, lo, hi = _interval(lambda i: _auroc(y[i][mask[i]], values[i][mask[i]]), groups, repetitions, seed)
            rows.append({"model": model, "stratum": stratum, "n_mnli": int(((y == 0) & mask).sum()),
                         "n_snli": int(((y == 1) & mask).sum()), "signal": signal,
                         "auroc_snli": est, "ci_95_low": lo, "ci_95_high": hi,
                         "separation": None if est is None else max(est, 1 - est)})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="50 bootstrap repetitions")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    cfg = config(ROOT)
    repetitions = 50 if args.quick else cfg["bootstrap_repetitions"]
    seed = cfg["seed"]
    verify_seal()

    comp, summ, prof, sens, src = [], [], [], [], []
    for model in MODELS:
        tables = {}
        for split in SPLITS:
            bundle, features, labels = _prepare(split, model)
            cases = _case_table(bundle, features, labels)
            tables[split] = cases
            rows, summary = composition(cases, split, model)
            comp += rows
            summ.append(summary)
            prof += error_profile(cases, split, model)
            sens += sensitivity(cases, split, model, repetitions, seed)
        src += source_separation(tables, model, repetitions, seed)

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(comp).to_csv(RESULTS / "clear_composition.csv", index=False)
    pd.DataFrame(summ).to_csv(RESULTS / "clear_summary.csv", index=False)
    pd.DataFrame(prof).to_csv(RESULTS / "clear_error_profile.csv", index=False)
    pd.DataFrame(sens).to_csv(RESULTS / "clear_sensitivity.csv", index=False)
    pd.DataFrame(apertus_sensitivity(repetitions, seed)).to_csv(RESULTS / "clear_sensitivity_apertus.csv", index=False)
    pd.DataFrame(src).to_csv(RESULTS / "source_separation.csv", index=False)
    protocol = experiment_directory(ROOT)
    write_json(RESULTS / "run_info.json", {
        "at": now(), "bootstrap_repetitions": repetitions, "seed": seed, "strict_majority": STRICT_MAJORITY,
        "experiment_folder": protocol.name,
        "status": "post-hoc, exploratory, frozen assessors applied unchanged, evaluated on held-out data"})
    verify_seal()
    print(f"Done. Tables written to {RESULTS}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .features import CUE_FIELDS, TAXONOMY_MAP, entropy, majority_correct, normalize


def js_divergence(p, q):
    """Jensen-Shannon divergence, base 2, not its square-root distance."""
    p, q = normalize(p), normalize(q)
    m = (p + q) / 2
    def kl(a, b):
        return (a * (np.log2(np.clip(a, 1e-15, 1)) - np.log2(np.clip(b, 1e-15, 1)))).sum(axis=-1)
    return np.maximum(0, (kl(p, m) + kl(q, m)) / 2)


def calibration_table(confidence, outcome, bins=10):
    confidence, outcome = np.asarray(confidence, float), np.asarray(outcome, float)
    groups = np.minimum((np.clip(confidence, 0, 1) * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        mask = groups == b
        if mask.any():
            rows.append({"bin": b, "count": int(mask.sum()), "confidence": float(confidence[mask].mean()),
                         "observed": float(outcome[mask].mean())})
    return pd.DataFrame(rows, columns=["bin", "count", "confidence", "observed"])


def ece(confidence, outcome):
    table = calibration_table(confidence, outcome)
    return float(np.average(abs(table.confidence - table.observed), weights=table["count"]))


def safe_spearman(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def distribution_metrics(predicted, human):
    p, q = normalize(predicted), normalize(human)
    labels = p.argmax(axis=1)
    support = q[np.arange(len(q)), labels]
    return {"js_divergence": float(js_divergence(p, q).mean()),
            "expected_brier": float((1 + (p ** 2).sum(axis=1) - 2 * (p * q).sum(axis=1)).mean()),
            "majority_agreement": float(majority_correct(labels, q).mean()),
            "human_support_of_answer": float(support.mean()),
            "human_support_ece": ece(p.max(axis=1), support),
            "entropy_spearman": safe_spearman(entropy(p), entropy(q))}


def risk_coverage(error_risk, errors):
    """Expected ordering within tied risk scores avoids arbitrary ID-based tie advantages."""
    risks, errors = np.asarray(error_risk, float), np.asarray(errors, float)
    order = np.argsort(risks, kind="stable")
    risks, errors = risks[order], errors[order]
    out, previous_count, previous_errors = [], 0, 0.0
    for value in np.unique(risks):
        block = errors[risks == value]
        for j in range(1, len(block) + 1):
            count = previous_count + j
            out.append({"coverage": count / len(errors),
                        "risk": (previous_errors + j * block.mean()) / count})
        previous_count += len(block)
        previous_errors += block.sum()
    return pd.DataFrame(out)


def error_metrics(predicted_risk, errors):
    r, y = np.clip(predicted_risk, 0, 1), np.asarray(errors, float)
    if len(y) == 0:
        return {"error_brier": None, "correctness_ece": None, "error_auroc": None, "aurc": None}
    curve = risk_coverage(r, y)
    return {"error_brier": float(np.mean((r - y) ** 2)),
            "correctness_ece": ece(1 - r, 1 - y),
            "error_auroc": float(roc_auc_score(y, r)) if len(np.unique(y)) > 1 else None,
            "aurc": float(curve.risk.mean())}


def safe_auroc(y, score):
    y, score = np.asarray(y, float), np.asarray(score, float)
    return float(roc_auc_score(y, score)) if len(y) > 1 and len(np.unique(y)) > 1 else None


def group_bootstrap_metric(metric, group_ids, repetitions=500, seed=2026):
    """Percentile CI for any metric(indices) -> float, resampling whole premise groups.
    Used for metrics that are not per-example means (balanced accuracy, AUROC)."""
    group_ids = np.asarray(group_ids)
    groups = np.unique(group_ids)
    indices = [np.flatnonzero(group_ids == g) for g in groups]
    rng = np.random.default_rng(seed)
    point = metric(np.arange(len(group_ids)))
    estimates = []
    for _ in range(repetitions):
        sample = np.concatenate([indices[i] for i in rng.integers(0, len(groups), size=len(groups))])
        value = metric(sample)
        if value is not None and np.isfinite(value):
            estimates.append(float(value))
    if point is None or len(estimates) < repetitions * 0.8:
        return {"estimate": point, "ci_95_low": None, "ci_95_high": None, "valid_resamples": len(estimates)}
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return {"estimate": float(point), "ci_95_low": float(lo), "ci_95_high": float(hi), "valid_resamples": len(estimates)}


def dissociation_table(signals, human, errors, clear_mask, debatable_threshold, group_ids=None, repetitions=0, seed=2026):
    """Core test of the project: does a signal track HUMAN disagreement (aleatoric, 'debatable')
    or MODEL error on clear cases (epistemic, 'uncertain')? AUROC 0.5 means no information.
    Expected pattern for a successful decomposition: aleatoric signals high in the first column
    and near 0.5 in the second; epistemic signals the reverse."""
    q = normalize(human)
    h = entropy(q)
    debatable = (h >= debatable_threshold).astype(float)
    errors, clear_mask = np.asarray(errors, float), np.asarray(clear_mask, bool)
    rows = []
    for name, values in signals.items():
        values = np.asarray(values, float)
        row = {"signal": name,
               "auroc_debatable": safe_auroc(debatable, values),
               "auroc_error_on_clear_cases": safe_auroc(errors[clear_mask], values[clear_mask]),
               "spearman_human_entropy": safe_spearman(values, h),
               "auroc_error_all_cases": safe_auroc(errors, values),
               "n": len(values), "n_clear": int(clear_mask.sum()), "errors_on_clear": int(errors[clear_mask].sum())}
        if group_ids is not None and repetitions:
            ci_d = group_bootstrap_metric(lambda i: safe_auroc(debatable[i], values[i]), group_ids, repetitions, seed)
            ci_e = group_bootstrap_metric(lambda i: safe_auroc(errors[i][clear_mask[i]], values[i][clear_mask[i]]), group_ids, repetitions, seed)
            row.update({"auroc_debatable_ci_low": ci_d["ci_95_low"], "auroc_debatable_ci_high": ci_d["ci_95_high"],
                        "auroc_error_on_clear_ci_low": ci_e["ci_95_low"], "auroc_error_on_clear_ci_high": ci_e["ci_95_high"]})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_group_bootstrap(delta, group_ids, repetitions=500, seed=2026, direction="negative favours the first method"):
    """CI for an average paired loss difference; resample whole premise groups."""
    delta = np.asarray(delta, float)
    group_ids = np.asarray(group_ids)
    groups = np.unique(group_ids)
    indices = [np.flatnonzero(group_ids == g) for g in groups]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        chosen = rng.integers(0, len(groups), size=len(groups))
        sample = np.concatenate([indices[i] for i in chosen])
        estimates.append(float(delta[sample].mean()))
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return {"mean_difference": float(delta.mean()), "ci_95_low": float(lo), "ci_95_high": float(hi),
            "groups": len(groups), "bootstrap_repetitions": repetitions,
            "direction": direction}


def concept_metrics(features, labels):
    joined = features.merge(labels[["uid", "taxonomy_categories", "taxonomy_round"]], on="uid", validate="one_to_one")
    available = joined.taxonomy_categories.map(lambda x: isinstance(x, list) and "N/A" not in x)
    selected = joined.loc[available]
    rows = []
    for cue in CUE_FIELDS:
        if len(selected) == 0:
            continue
        y = selected.taxonomy_categories.map(lambda x: int(TAXONOMY_MAP[cue] in x)).to_numpy()
        score = selected[f"{cue}_mean"].to_numpy()
        binary = score >= 0.5  # Fixed before test; not adjusted from test results.
        rows.append({"cue": cue, "n": len(y), "positive_annotations": int(y.sum()),
                     "f1_at_0_5": float(f1_score(y, binary, zero_division=0)),
                     "average_precision": float(average_precision_score(y, score)) if y.sum() > 0 else None,
                     "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else None,
                     "round1_n": int((selected.taxonomy_round == 1).sum()),
                     "round2_n": int((selected.taxonomy_round == 2).sum())})
    return pd.DataFrame(rows)

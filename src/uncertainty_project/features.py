"""Signals computed from stored model outputs.

Each prompt variant is treated as one "head" of a small ensemble. With 5 prompts per model
(or 10 heads when GPT and Claude are pooled) the standard ensemble decomposition applies:

    total uncertainty      H(mean_p)             -> mean_entropy
    = expected entropy     mean_k H(p_k)         -> expected_entropy   (the model's own aleatoric estimate)
    + mutual information   H(mean_p) - mean H    -> epistemic_mi       (disagreement between heads)

Credal-style signals use the per-class interval [min_k p_k, max_k p_k] over heads. Interval width is
an epistemic signal; interval dominance says whether one label is determinate across all heads.
Prompt variation is an empirical proxy for epistemic uncertainty, not a proven identification of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .api import PROMPT_VARIANTS, read_cached
from .common import LABELS

CUE_FIELDS = ("lexical_ambiguity", "unclear_reference", "probabilistic_inference")
TAXONOMY_MAP = dict(zip(CUE_FIELDS, ("Lexical", "Coreference", "Probabilistic")))
# Disagreement sources that a clarification question could plausibly resolve.
CLARIFIABLE_CUES = ("lexical_ambiguity", "unclear_reference")
CLARIFIABLE_TAXONOMY = ("Lexical", "Coreference")

SCORE_FEATURES = ["single_e", "single_n", "single_c", "single_confidence", "single_entropy"]
SIGNAL_FEATURES = SCORE_FEATURES + ["mean_e", "mean_n", "mean_c", "mean_confidence", "mean_entropy",
                                    "expected_entropy", "epistemic_mi", "answer_disagreement",
                                    "credal_width", "credal_width_max", "credal_determinate", "margin"]
FULL_FEATURES = SIGNAL_FEATURES + [f"{c}_{stat}" for c in CUE_FIELDS for stat in ("mean", "range")]
FEATURE_SETS = {"score_only": SCORE_FEATURES, "no_concepts": SIGNAL_FEATURES, "full": FULL_FEATURES}
# Simple risk baseline: which label the model answered (from the prompt mean) and how confident it was.
# No MI, no credal width, no cues: tests whether those add anything beyond label + confidence.
LABEL_CONFIDENCE_FEATURES = ["answer_is_e", "answer_is_n", "answer_is_c", "mean_confidence"]


def normalize(p):
    p = np.asarray(p, dtype=float)
    if p.shape[-1] != 3 or not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p.sum(axis=-1) <= 0):
        raise ValueError("Invalid three-class distribution.")
    return p / p.sum(axis=-1, keepdims=True)


def entropy(p):
    """Normalised Shannon entropy in [0, 1] (divided by log 3)."""
    p = normalize(p)
    return -(p * np.log(np.clip(p, 1e-15, 1))).sum(axis=-1) / np.log(3)


def credal_signals(heads):
    """Interval [min, max] per label across heads, its width, and interval dominance."""
    p = normalize(heads)
    lower, upper = p.min(axis=0), p.max(axis=0)
    width = upper - lower
    top = int(p.mean(axis=0).argmax())
    others = [k for k in range(3) if k != top]
    determinate = float(all(lower[top] > upper[k] for k in others))
    return {"credal_width": float(width.mean()), "credal_width_max": float(width.max()),
            "credal_determinate": determinate, "credal_lower_top": float(lower[top]),
            "credal_upper_top": float(upper[top])}


def summarize_predictions(predictions, expected_heads=None):
    expected_heads = len(PROMPT_VARIANTS) if expected_heads is None else expected_heads
    if len(predictions) != expected_heads:
        raise ValueError(f"All {expected_heads} heads are required; incomplete cases cannot silently enter evaluation.")
    p = normalize([[r[label] for label in LABELS] for r in predictions])
    first, mean = p[0], p.mean(axis=0)
    counts = np.bincount(p.argmax(axis=1), minlength=3)
    total = float(entropy(mean))
    expected = float(entropy(p).mean())
    result = dict(zip(["single_e", "single_n", "single_c"], first))
    result.update(dict(zip(["mean_e", "mean_n", "mean_c"], mean)))
    result.update(single_confidence=float(first.max()), single_entropy=float(entropy(first)),
                  mean_confidence=float(mean.max()), mean_entropy=total,
                  expected_entropy=expected, epistemic_mi=float(max(0.0, total - expected)),
                  answer_disagreement=float(1 - counts.max() / len(p)),
                  margin=float(np.sort(mean)[-1] - np.sort(mean)[-2]),
                  base_label=int(mean.argmax()), single_label=int(first.argmax()))
    result.update({f"answer_is_{c}": float(int(mean.argmax()) == i) for i, c in enumerate("enc")})
    result.update(credal_signals(p))
    for cue in CUE_FIELDS:
        vals = np.array([r[cue] for r in predictions], dtype=float)
        result[f"{cue}_mean"] = float(vals.mean())
        result[f"{cue}_range"] = float(np.ptp(vals))
    result["clarifiable_cue"] = float(max(result[f"{c}_mean"] for c in CLARIFIABLE_CUES))
    return result


def _cached_heads(root, row, model_key):
    return [read_cached(root, row, v, model_key) for v in range(len(PROMPT_VARIANTS))]


def feature_frame(root, inputs, model_key):
    """Single-provider features. Column names are identical for every model, which makes transfer possible."""
    rows, missing = [], []
    for row in inputs.to_dict("records"):
        cached = _cached_heads(root, row, model_key)
        if any(r is None for r in cached):
            missing.append(row["uid"])
            continue
        vals = summarize_predictions([r["parsed"] for r in cached])
        rows.append({"uid": row["uid"], "group_id": row["group_id"], **vals})
    if missing:
        raise RuntimeError(f"[{model_key}] Missing complete predictions for {len(missing)} examples. "
                           f"Run notebook 01 (or 03 for test). First missing IDs: {missing[:3]}")
    return pd.DataFrame(rows)


def joint_feature_frame(root, inputs, model_keys):
    """Pooled heads across providers (e.g. 5 GPT + 5 Claude = 10 heads). Joint-model setting only:
    these signals need BOTH providers and are kept separate from single-provider transfer results."""
    rows = []
    for row in inputs.to_dict("records"):
        per_model = {}
        for key in model_keys:
            cached = _cached_heads(root, row, key)
            if any(r is None for r in cached):
                raise RuntimeError(f"[{key}] Missing predictions for {row['uid']}; joint signals need all models.")
            per_model[key] = [r["parsed"] for r in cached]
        pooled = [p for key in model_keys for p in per_model[key]]
        vals = summarize_predictions(pooled, expected_heads=len(pooled))
        means = [normalize([[r[label] for label in LABELS] for r in per_model[k]]).mean(axis=0) for k in model_keys]
        cross = {"cross_model_label_disagreement": float(len({int(m.argmax()) for m in means}) > 1),
                 "cross_model_entropy_gap": float(entropy(np.mean(means, axis=0)) - np.mean([entropy(m) for m in means]))}
        rows.append({"uid": row["uid"], "group_id": row["group_id"],
                     **{f"joint_{k}": v for k, v in vals.items()}, **cross})
    return pd.DataFrame(rows)


def probabilities(frame, which="mean"):
    return normalize(frame[[f"{which}_{x}" for x in "enc"]].to_numpy())


def majority_correct(labels, human_p):
    """Any tied majority label counts as agreement. This is not absolute semantic truth."""
    human_p = normalize(human_p)
    labels = np.asarray(labels, dtype=int)
    return np.isclose(human_p[np.arange(len(labels)), labels], human_p.max(axis=1)).astype(float)

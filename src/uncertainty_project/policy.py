"""Response policy for the 2x2 framing: uncertain vs genuinely debatable.

                       humans agree (clear)        humans disagree (debatable)
    model reliable     answer                      alternatives / clarify
    model unreliable   hedge (say uncertain)       alternatives / clarify

Debatable cases take precedence: if humans disagree, showing alternatives already communicates
uncertainty. Within debatable cases, "clarify" is chosen when the predicted source of disagreement
is one a question could resolve (word meaning, unclear reference); otherwise "alternatives".

Evaluation does not invent utility costs. Each example gets a REFERENCE action derived from the
human votes and the model's answer, and policies are scored by how well they recover it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import CLARIFIABLE_TAXONOMY, entropy, majority_correct, normalize

ACTIONS = ("answer", "hedge", "alternatives", "clarify")
COARSE = ("answer", "hedge", "debatable")


def choose_actions(human_prediction, epistemic_risk, clarifiable_cue, *,
                   aleatoric_threshold, epistemic_threshold, cue_threshold):
    p = normalize(human_prediction)
    actions = np.full(len(p), "answer", dtype=object)
    actions[np.asarray(epistemic_risk, float) >= epistemic_threshold] = "hedge"
    debatable = entropy(p) >= aleatoric_threshold
    actions[debatable] = "alternatives"
    actions[debatable & (np.asarray(clarifiable_cue, float) >= cue_threshold)] = "clarify"
    top2 = np.argsort(-p, axis=1, kind="stable")[:, :2]
    return actions, top2


def choose_actions_single(score, clarifiable_cue, *, hedge_threshold, debatable_threshold, cue_threshold):
    """ONE score drives everything: low -> answer, middle -> hedge, high -> alternatives/clarify.
    Baseline for the question whether two separate signals are needed at all."""
    s = np.asarray(score, float)
    actions = np.full(len(s), "answer", dtype=object)
    actions[s >= hedge_threshold] = "hedge"
    debatable = s >= debatable_threshold
    actions[debatable] = "alternatives"
    actions[debatable & (np.asarray(clarifiable_cue, float) >= cue_threshold)] = "clarify"
    return actions


def coarse(actions):
    actions = np.asarray(actions, dtype=object)
    return np.where(np.isin(actions, ("alternatives", "clarify")), "debatable", actions)


def reference_actions(human, base_labels, debatable_threshold, taxonomy=None):
    """Coarse reference: debatable if human vote entropy >= threshold; otherwise answer if the model's
    answer matches the majority, else hedge. Fine reference splits debatable cases into clarify /
    alternatives where a disagreement-taxonomy annotation exists (else stays 'debatable')."""
    q = normalize(human)
    debatable = entropy(q) >= debatable_threshold
    correct = majority_correct(base_labels, q) == 1
    ref = np.where(debatable, "debatable", np.where(correct, "answer", "hedge")).astype(object)
    fine = ref.copy()
    if taxonomy is not None:
        for i, cats in enumerate(taxonomy):
            if ref[i] == "debatable" and isinstance(cats, (list, tuple)) and "N/A" not in cats:
                fine[i] = "clarify" if any(c in CLARIFIABLE_TAXONOMY for c in cats) else "alternatives"
    return ref, fine


def balanced_accuracy(predicted, reference, classes=COARSE):
    recalls = [float(np.mean(predicted[reference == c] == c)) for c in classes if np.any(reference == c)]
    return float(np.mean(recalls)) if recalls else None


def policy_metrics(actions, ref_coarse, ref_fine=None):
    pred = coarse(actions)
    out = {"balanced_accuracy": balanced_accuracy(pred, ref_coarse),
           "accuracy": float(np.mean(pred == ref_coarse)),
           **{f"recall_{c}": (float(np.mean(pred[ref_coarse == c] == c)) if np.any(ref_coarse == c) else None) for c in COARSE},
           **{f"{a}_fraction": float(np.mean(np.asarray(actions) == a)) for a in ACTIONS},
           # How often the system answers directly although humans disagree or the model is wrong on a clear case.
           "unwarranted_answer_rate": (float(np.mean(ref_coarse[np.asarray(actions) == "answer"] != "answer"))
                                       if np.any(np.asarray(actions) == "answer") else None)}
    if ref_fine is not None:
        acts = np.asarray(actions, dtype=object)
        # Unconditional: on ALL annotated debatable cases, was exactly the right fine action chosen?
        # Missed cases (answer/hedge) count as wrong here.
        recalls = {}
        for target in ("clarify", "alternatives"):
            mask = ref_fine == target
            out[f"n_reference_{target}"] = int(mask.sum())
            recalls[target] = float(np.mean(acts[mask] == target)) if mask.any() else None
            out[f"recall_{target}_annotated"] = recalls[target]
        valid = [v for v in recalls.values() if v is not None]
        out["fine_balanced_accuracy_annotated"] = float(np.mean(valid)) if valid else None
        # Conditional: only cases where the system already chose clarify or alternatives.
        both = np.isin(ref_fine, ("clarify", "alternatives")) & np.isin(acts, ("clarify", "alternatives"))
        out["conditional_n"] = int(both.sum())
        out["conditional_clarify_vs_alternatives_accuracy"] = float(np.mean(acts[both] == ref_fine[both])) if both.any() else None
    return out


def confusion(actions, ref_fine):
    table = pd.crosstab(pd.Series(ref_fine, name="reference"), pd.Series(actions, name="chosen_action"))
    return table.reindex(columns=list(ACTIONS), fill_value=0)


def tune_policy(human_prediction, epistemic_risk, clarifiable_cue, ref_coarse, settings, cue_threshold):
    """Grid search on DEVELOPMENT data only. Objective: balanced accuracy over answer/hedge/debatable.
    Ties prefer answering more often (less friction for the user)."""
    rows = []
    for a in settings["aleatoric_thresholds"]:
        for e in settings["epistemic_thresholds"]:
            actions, _ = choose_actions(human_prediction, epistemic_risk, clarifiable_cue,
                                        aleatoric_threshold=a, epistemic_threshold=e, cue_threshold=cue_threshold)
            m = policy_metrics(actions, ref_coarse)
            rows.append({"aleatoric_threshold": a, "epistemic_threshold": e,
                         "dev_balanced_accuracy": m["balanced_accuracy"], "dev_answer_fraction": m["answer_fraction"]})
    table = pd.DataFrame(rows).sort_values(["dev_balanced_accuracy", "dev_answer_fraction", "aleatoric_threshold", "epistemic_threshold"],
                                           ascending=[False, False, False, False])
    best = table.iloc[0]
    return {"kind": "two_signal", "aleatoric_threshold": float(best.aleatoric_threshold),
            "epistemic_threshold": float(best.epistemic_threshold), "cue_threshold": float(cue_threshold)}, table


def tune_single_score_policy(score, clarifiable_cue, ref_coarse, cue_threshold, quantiles=(.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01)):
    """Thresholds on quantiles of the score on DEVELOPMENT data (scores have different scales)."""
    s = np.asarray(score, float)
    cuts = sorted({float(np.quantile(s, q)) if q <= 1 else float(s.max() + 1) for q in quantiles})
    rows = []
    for h in cuts:
        for d in cuts:
            if d < h:
                continue
            actions = choose_actions_single(s, clarifiable_cue, hedge_threshold=h, debatable_threshold=d, cue_threshold=cue_threshold)
            m = policy_metrics(actions, ref_coarse)
            rows.append({"hedge_threshold": h, "debatable_threshold": d,
                         "dev_balanced_accuracy": m["balanced_accuracy"], "dev_answer_fraction": m["answer_fraction"]})
    table = pd.DataFrame(rows).sort_values(["dev_balanced_accuracy", "dev_answer_fraction"], ascending=[False, False])
    best = table.iloc[0]
    return {"kind": "single", "hedge_threshold": float(best.hedge_threshold),
            "debatable_threshold": float(best.debatable_threshold), "cue_threshold": float(cue_threshold)}, table


def apply_policy(policy, inputs, clarifiable_cue):
    """inputs: (distribution, risk) for two-signal policies, or a 1-D score for single-score policies."""
    p = dict(policy)
    if p.pop("kind", "two_signal") == "single":
        return choose_actions_single(inputs, clarifiable_cue, **p), None
    dp, rp = inputs
    return choose_actions(dp, rp, clarifiable_cue, **p)

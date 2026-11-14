from __future__ import annotations

import importlib.metadata
from itertools import permutations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .api import PROMPT_VARIANTS, cache_path, experiment_directory, model_keys, run_spec
from .common import config, digest, now, read_json, write_json
from .data import load_split
from .features import entropy, feature_frame, joint_feature_frame, majority_correct
from .metrics import (concept_metrics, dissociation_table, distribution_metrics, group_bootstrap_metric,
                      js_divergence, paired_group_bootstrap, risk_coverage, safe_auroc)
from .models import (aligned_human, aligned_taxonomy, evaluation_tables, fit_assessors, run_policies,
                     tree_rules)
from .policy import balanced_accuracy, coarse, confusion, policy_metrics, reference_actions

# Which component of the decomposition each raw or learned signal is meant to capture.
SIGNAL_KINDS = {
    "model_total_entropy": "mixed",
    "model_expected_entropy": "aleatoric",
    "model_epistemic_mi": "epistemic",
    "model_credal_width": "epistemic",
    "model_label_flips": "epistemic",
    "model_one_minus_confidence": "mixed",
    "learned_human_entropy": "aleatoric",
    "learned_epistemic_risk": "epistemic",
    "learned_risk_all_cases": "mixed",
    "joint_expected_entropy": "aleatoric",
    "joint_epistemic_mi": "epistemic",
    "joint_credal_width": "epistemic",
    "cross_model_label_disagreement": "epistemic",
    "baseline_answer_label": "baseline",
    "learned_entropy_direct": "aleatoric",
    "baseline_label_confidence": "baseline",
}


def model_path(root, model_key):
    return experiment_directory(root) / model_key / "assessor.joblib"


def load_bundle(root, model_key):
    path = model_path(root, model_key)
    if not path.exists():
        raise FileNotFoundError(f"Train the {model_key} assessor in notebook 02 first.")
    return joblib.load(path)


def _test_started(root):
    return (experiment_directory(root) / "test_started.json").exists()


def train_development(root):
    """Fit one assessor per model on train, select on dev, and write development reports."""
    root = Path(root)
    if _test_started(root):
        raise RuntimeError("Test collection/evaluation has started. Do not refit or tune this experiment from its test results.")
    cfg = config(root)
    train, dev = load_split(root, "train"), load_split(root, "dev")
    bundles = {}
    for key in model_keys(root):
        folder = experiment_directory(root) / key
        ft, fd = feature_frame(root, train, key), feature_frame(root, dev, key)
        bundle, search, policy_search = fit_assessors(ft, train, fd, dev, cfg, key)
        bundle["run_spec_digest"] = digest(run_spec(root, key))
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_path(root, key))
        search.to_csv(folder / "development_tree_search.csv", index=False)
        policy_search.to_csv(folder / "development_policy_search.csv", index=False)
        for target, rules in tree_rules(bundle).items():
            (folder / f"full_{target}_tree.txt").write_text(rules, encoding="utf-8")
        write_json(folder / "development_choices.json", {"selected_parameters": bundle["selected_parameters"],
                   "temperature": bundle["temperature"], "policies": bundle["policies"],
                   "human_thresholds": bundle["human_thresholds"],
                   "train_n": len(train), "dev_n": len(dev), "created_at": now()})
        bundles[key] = bundle
    report = evaluate_split(root, "dev", bundles)
    _write_report(experiment_directory(root) / "development", report)
    return bundles, report


# ---------------------------------------------------------------- shared evaluation

def _signals(features, pred, joint=None):
    s = {"model_total_entropy": features.mean_entropy, "model_expected_entropy": features.expected_entropy,
         "model_epistemic_mi": features.epistemic_mi, "model_credal_width": features.credal_width,
         "model_label_flips": features.answer_disagreement, "model_one_minus_confidence": 1 - features.mean_confidence,
         "learned_human_entropy": entropy(pred["distributions"]["tree_full"]),
         "learned_epistemic_risk": pred["risks"]["epistemic_tree_full"],
         "learned_risk_all_cases": pred["risks"]["tree_full"],
         # Baseline: error rate of the chosen answer label on clear train cases (no uncertainty at all).
         "baseline_answer_label": pred["risks"]["epistemic_answer_label_only"],
         # Robustness check for the aleatoric signal: tree trained on human entropy directly.
         "learned_entropy_direct": pred["entropy_direct"],
         # Simple risk baseline: answer label + confidence, trained on clear train cases.
         "baseline_label_confidence": pred["risks"]["epistemic_label_confidence_tree"]}
    if joint is not None:
        j = joint.set_index("uid").loc[features.uid]
        s.update({"joint_expected_entropy": j.joint_expected_entropy.to_numpy(),
                  "joint_epistemic_mi": j.joint_epistemic_mi.to_numpy(),
                  "joint_credal_width": j.joint_credal_width.to_numpy(),
                  "cross_model_label_disagreement": j.cross_model_label_disagreement.to_numpy()})
    return s


POLICY_COMPARISONS = ("risk_trained_all_cases", "single_score_risk", "single_score_confidence", "calibrated")


def policy_intervals(actions_by_policy, ref, group_ids, repetitions, seed):
    """Paired CIs for the MAIN policy metric (balanced accuracy over answer/hedge/debatable):
    decomposed minus each comparison, premise groups resampled together."""
    out = {}
    for other in POLICY_COMPARISONS:
        a, b = coarse(actions_by_policy["decomposed"]), coarse(actions_by_policy[other])
        def diff(i):
            x, y = balanced_accuracy(a[i], ref[i]), balanced_accuracy(b[i], ref[i])
            return None if x is None or y is None else x - y
        out[f"balanced_accuracy_decomposed_minus_{other}"] = {
            **group_bootstrap_metric(diff, group_ids, repetitions, seed),
            "direction": f"positive favours decomposed over {other}"}
    return out


# Fixed BEFORE the test (see config "analysis_plan"): paired AUROC differences, premise groups resampled.
SPECIALISATION_COMPARISONS = (
    ("primary", "debatable", "learned_human_entropy", "learned_epistemic_risk",
     "Does the disagreement signal detect debatable cases better than the risk signal?"),
    ("primary", "error_on_clear", "learned_epistemic_risk", "learned_human_entropy",
     "Does the risk signal detect majority deviations on clear cases better than the disagreement signal?"),
    ("primary", "error_on_clear", "learned_epistemic_risk", "baseline_label_confidence",
     "Do ensemble signals (MI, credal width, cues) add anything beyond answer label + confidence?"),
    ("secondary", "debatable", "learned_entropy_direct", "learned_human_entropy",
     "Robustness: does predicting human entropy directly change disagreement detection?"),
)


def specialisation_table(signals, human, errors, clear, debatable_threshold, group_ids, repetitions, seed):
    debatable = (entropy(human) >= debatable_threshold).astype(float)
    errors, clear = np.asarray(errors, float), np.asarray(clear, bool)
    rows = []
    for rank, target, a, b, question in SPECIALISATION_COMPARISONS:
        if a not in signals or b not in signals:
            continue
        x, y = np.asarray(signals[a], float), np.asarray(signals[b], float)
        if target == "debatable":
            def diff(i, x=x, y=y):
                u, v = safe_auroc(debatable[i], x[i]), safe_auroc(debatable[i], y[i])
                return None if u is None or v is None else u - v
        else:
            def diff(i, x=x, y=y):
                j = i[clear[i]]
                u, v = safe_auroc(errors[j], x[j]), safe_auroc(errors[j], y[j])
                return None if u is None or v is None else u - v
        res = group_bootstrap_metric(diff, group_ids, repetitions, seed)
        rows.append({"rank": rank, "target": target, "signal": a, "minus": b, "auroc_difference": res["estimate"],
                     "ci_95_low": res["ci_95_low"], "ci_95_high": res["ci_95_high"],
                     "valid_resamples": res["valid_resamples"], "question": question})
    return pd.DataFrame(rows)


def evaluate_bundle(bundle, features, labels, joint=None, repetitions=0, seed=2026):
    """All tables for one assessor applied to one model's features (same model or transfer)."""
    th = bundle["human_thresholds"]
    q = aligned_human(features, labels)
    errors = 1 - majority_correct(features.base_label, q)
    clear = entropy(q) <= th["clear_max_entropy"]
    dist, risk, pred = evaluation_tables(bundle, features, labels)
    ref, fine = reference_actions(q, features.base_label, th["debatable_min_entropy"], aligned_taxonomy(features, labels))
    policies, actions_by_policy = [], {}
    for name, (actions, top2) in run_policies(bundle, features, pred).items():
        policies.append({"method": name, **policy_metrics(actions, ref, fine)})
        actions_by_policy[name] = actions
    for action in ("answer", "hedge", "alternatives"):
        trivial = np.full(len(q), action, dtype=object)
        policies.append({"method": f"always_{action}", **policy_metrics(trivial, ref, fine)})
    signals = _signals(features, pred, joint)
    dissociation = dissociation_table(signals, q, errors, clear, th["debatable_min_entropy"],
                                      group_ids=features.group_id.to_numpy(), repetitions=repetitions, seed=seed)
    dissociation.insert(1, "intended_component", dissociation.signal.map(SIGNAL_KINDS))
    specialisation = (specialisation_table(signals, q, errors, clear, th["debatable_min_entropy"],
                                           features.group_id.to_numpy(), repetitions, seed) if repetitions else pd.DataFrame())
    reference_counts = pd.Series(fine).value_counts().rename_axis("reference_action").reset_index(name="n")
    intervals = policy_intervals(actions_by_policy, ref, features.group_id.to_numpy(), repetitions, seed) if repetitions else {}
    return {"distributions": dist, "error_risk": risk, "policies": pd.DataFrame(policies),
            "policy_intervals": pd.DataFrame(intervals).T.reset_index(names="comparison") if intervals else pd.DataFrame(),
            "confusion_decomposed": confusion(actions_by_policy["decomposed"], fine),
            "dissociation": dissociation, "specialisation": specialisation, "reference_counts": reference_counts,
            "_pred": pred, "_errors": errors, "_q": q, "_actions": actions_by_policy, "_ref": fine}


def _labels_for(root, split):
    return load_split(root, split, include_test_labels=True) if split in ("test", "shift") else load_split(root, split)


def evaluate_split(root, split, bundles=None):
    """Same-model evaluation for every model, model comparison (RQ1) and transfer (RQ2)."""
    keys = model_keys(root)
    bundles = bundles or {k: load_bundle(root, k) for k in keys}
    inputs = load_split(root, split)
    features = {k: feature_frame(root, inputs, k) for k in keys}
    joint = joint_feature_frame(root, inputs, keys) if len(keys) > 1 else None
    labels = _labels_for(root, split)
    cfg = config(root)
    report = {"per_model": {}, "transfer": {}}
    reps = cfg["bootstrap_repetitions"]
    for k in keys:
        report["per_model"][k] = evaluate_bundle(bundles[k], features[k], labels, joint, reps, cfg["seed"])
        report["per_model"][k]["concepts"] = concept_metrics(features[k], labels)
    for src, tgt in permutations(keys, 2):
        report["transfer"][f"{src}_to_{tgt}"] = evaluate_bundle(bundles[src], features[tgt], labels)
    report["model_comparison"] = model_comparison(features, labels, cfg)
    report["transfer_summary"] = transfer_summary(report)
    report["meta"] = {"split": split, "n": len(inputs), "premise_groups": int(inputs.group_id.nunique()), "models": keys,
                      "human_thresholds": bundles[keys[0]]["human_thresholds"]}
    return report


def model_comparison(features, labels, cfg):
    """RQ1: the raw models on identical examples, before any assessor."""
    keys = list(features)
    rows = []
    for k in keys:
        f = features[k]
        q = aligned_human(f, labels)
        p = f[["mean_e", "mean_n", "mean_c"]].to_numpy()
        rows.append({"model": k, **distribution_metrics(p, q),
                     "mean_epistemic_mi": float(f.epistemic_mi.mean()),
                     "label_flip_rate": float((f.answer_disagreement > 0).mean()),
                     "mean_credal_width": float(f.credal_width.mean())})
    table = pd.DataFrame(rows)
    if len(keys) == 2:
        a, b = (features[k].set_index("uid") for k in keys)
        b = b.loc[a.index]
        q = aligned_human(a.reset_index(), labels)
        agree_a = majority_correct(a.base_label, q)
        agree_b = majority_correct(b.base_label, q)
        boot = paired_group_bootstrap(agree_a - agree_b, a.group_id.to_numpy(), cfg["bootstrap_repetitions"], cfg["seed"],
                                      direction=f"positive: {keys[0]} agrees with the majority more often than {keys[1]}")
        table.attrs["paired_majority_agreement_difference"] = boot
        table.attrs["same_answer_rate"] = float((a.base_label.to_numpy() == b.base_label.to_numpy()).mean())
    return table


def transfer_summary(report):
    rows = []
    for name, res in report["transfer"].items():
        src, tgt = name.split("_to_")
        own = report["per_model"][tgt]
        def pick(r, method, col, table="policies"):
            t = r[table]
            return float(t.loc[t.method == method, col].iloc[0])
        def risk_auc(r, method, subset):
            t = r["error_risk"]
            v = t.loc[(t.method == method) & (t.subset == subset), "error_auroc"].iloc[0]
            return None if pd.isna(v) else float(v)
        rows.append({"assessor_from": src, "applied_to": tgt,
                     "policy_bal_acc_transfer": pick(res, "decomposed", "balanced_accuracy"),
                     "policy_bal_acc_own": pick(own, "decomposed", "balanced_accuracy"),
                     "human_js_transfer": pick(res, "tree_full", "js_divergence", "distributions"),
                     "human_js_own": pick(own, "tree_full", "js_divergence", "distributions"),
                     "clear_error_auroc_transfer": risk_auc(res, "epistemic_tree_full", "clear_cases"),
                     "clear_error_auroc_own": risk_auc(own, "epistemic_tree_full", "clear_cases")})
    return pd.DataFrame(rows)


def _write_report(folder, report):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for k, res in report["per_model"].items():
        for name, table in res.items():
            if not name.startswith("_") and isinstance(table, pd.DataFrame):
                table.to_csv(folder / f"{k}_{name}.csv", index=name.startswith("confusion"))
    for k, res in report["transfer"].items():
        for name in ("policies", "distributions", "error_risk", "dissociation"):
            res[name].to_csv(folder / f"transfer_{k}_{name}.csv", index=False)
    report["transfer_summary"].to_csv(folder / "transfer_summary.csv", index=False)
    report["model_comparison"].to_csv(folder / "model_comparison.csv", index=False)
    write_json(folder / "model_comparison_extra.json", report["model_comparison"].attrs)
    write_json(folder / "meta.json", {**report["meta"], "written_at": now()})
    _save_plots(folder, report)


# ---------------------------------------------------------------- freeze and final test

def _fingerprint(root):
    root = Path(root)
    keys = model_keys(root)
    tracked = list((root / "src/uncertainty_project").glob("*.py"))
    tracked += list((root / "config").glob("*.json"))
    tracked += list((root / "data/processed").glob("*.json*"))
    for key in keys:
        if not model_path(root, key).exists():
            raise FileNotFoundError("Train the assessors in notebook 02 first.")
        tracked += [model_path(root, key), experiment_directory(root) / key / "development_choices.json"]
        for split in ("train", "dev"):
            for row in load_split(root, split).to_dict("records"):
                for v in range(len(PROMPT_VARIANTS)):
                    tracked.append(cache_path(root, row, v, key))
    return {str(p.relative_to(root)): digest(p.read_bytes()) for p in sorted(set(tracked))}


def freeze_experiment(root):
    """Record chosen method BEFORE the held-out test; no claims of formal preregistration."""
    folder = experiment_directory(root)
    path = folder / "frozen_protocol.json"
    if config(root).get("shift") and not (Path(root) / "data/processed/shift_inputs.jsonl").exists():
        raise RuntimeError("Prepare the unfamiliar-input sample (prepare_shift in notebook 02) before freezing, "
                           "so it is sealed together with the method.")
    files = _fingerprint(root)
    if path.exists():
        old = read_json(path)
        if old["files"] == files:
            return old
    if _test_started(root):
        raise RuntimeError("Cannot change the frozen experiment after accessing test predictions/results.")
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "openai", "anthropic", "pydantic"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    frozen = {"protocol_id": digest(files), "frozen_at": now(), "files": files, "config": config(root),
              "run_spec_digests": {k: digest(run_spec(root, k)) for k in model_keys(root)}, "package_versions": versions,
              "holdout_claim": "Held out from this project's fitting and tuning, not a guarantee of absence from LLM pretraining.",
              "claims": ["Separate signals for human disagreement (aleatoric) and model error on clear cases (epistemic)",
                         "Dissociation test of these signals", "Four-action response policy scored against reference actions",
                         "Comparison of two providers and transfer of assessors between them"],
              "not_claimed": ["Proof that prompt variation identifies epistemic uncertainty", "CREDENCE reproduction",
                              "Individual annotator reliability", "Validated user utility"]}
    write_json(path, frozen)
    return frozen


def verify_frozen(root):
    path = experiment_directory(root) / "frozen_protocol.json"
    if not path.exists():
        raise RuntimeError("Finish notebook 02 and freeze the method before collecting test predictions.")
    frozen = read_json(path)
    if frozen["files"] != _fingerprint(root):
        raise RuntimeError("Code, configuration, data or fitted models changed after freezing. Resolve before test evaluation.")
    return frozen


def final_evaluation(root):
    root = Path(root)
    frozen = verify_frozen(root)
    folder = experiment_directory(root)
    write_json(folder / "test_started.json", {"protocol_id": frozen["protocol_id"], "at": now()})
    report = evaluate_split(root, "test")
    cfg = config(root)
    intervals = {}
    for k, res in report["per_model"].items():
        features = feature_frame(root, load_split(root, "test"), k)
        q, pred = res["_q"], res["_pred"]
        intervals[f"{k}_human_js_tree_minus_prompt_mean"] = paired_group_bootstrap(
            js_divergence(pred["distributions"]["tree_full"], q) - js_divergence(pred["distributions"]["prompt_mean"], q),
            features.group_id, cfg["bootstrap_repetitions"], cfg["seed"], direction="negative favours the tree")
        for _, row in res["policy_intervals"].iterrows():
            intervals[f"{k}_{row.comparison}"] = {c: row[c] for c in res["policy_intervals"].columns if c != "comparison"}
    report["paired_intervals"] = intervals
    _write_report(folder / "test", report)
    write_json(folder / "test" / "paired_intervals.json", intervals)
    record = {"protocol_id": frozen["protocol_id"], "at": now(), **report["meta"],
              "interpretation": "Empirical results for this split and these models. Prompt variation is a proxy, not a proven identification of epistemic uncertainty."}
    if config(root).get("shift"):
        # Unfamiliar inputs: same frozen assessors, thresholds and policies, applied once to a different source.
        report["shift"] = evaluate_split(root, "shift")
        _write_report(folder / "shift", report["shift"])
        record["unfamiliar_inputs"] = {**report["shift"]["meta"], "source": config(root)["shift"]["source"],
                                       "note": "Same frozen method; no refitting or threshold changes on this source."}
    write_json(folder / "evaluation_record.json", record)
    return report


def _save_plots(folder, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"aleatoric": "#2a7ab9", "epistemic": "#d95f02", "mixed": "#7f7f7f", "baseline": "#000000"}
    keys = list(report["per_model"])
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.6), squeeze=False)
    for ax, k in zip(axes[0], keys):
        d = report["per_model"][k]["dissociation"].dropna(subset=["auroc_debatable", "auroc_error_on_clear_cases"])
        for _, r in d.iterrows():
            ax.scatter(r.auroc_debatable, r.auroc_error_on_clear_cases, color=colors.get(r.intended_component, "black"), s=40)
            ax.annotate(r.signal.replace("model_", "").replace("learned_", "L:"), (r.auroc_debatable, r.auroc_error_on_clear_cases),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0.5, color="gray", lw=0.8, ls="--"); ax.axvline(0.5, color="gray", lw=0.8, ls="--")
        ax.set(xlim=(0.3, 1), ylim=(0.3, 1), xlabel="AUROC: humans disagree (debatable)",
               ylabel="AUROC: model wrong on clear cases (uncertain)", title=f"{k}: which signal tracks what?")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=n) for n, c in colors.items()]
    axes[0][-1].legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(folder / "dissociation.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4), squeeze=False)
    for ax, k in zip(axes[0], keys):
        res = report["per_model"][k]
        for name in ("raw_confidence", "calibrated_confidence", "tree_full", "epistemic_tree_full"):
            curve = risk_coverage(res["_pred"]["risks"][name], res["_errors"])
            ax.plot(curve.coverage, curve.risk, label=name)
        ax.set(xlabel="Fraction answered", ylabel="Majority disagreement among answered", title=f"{k}: risk vs coverage",
               xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(folder / "risk_coverage.png", dpi=170); plt.close(fig)

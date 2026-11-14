"""Post-hoc extension: Apertus as a third model (requested by the supervisor after the freeze).

Only READS the frozen project (src/, config/, data/processed/, fitted assessors, GPT/Claude responses).
Every metric comes from the frozen functions in src/uncertainty_project; nothing is refitted or retuned.
New files are written to extension_apertus/results/ only.

Not part of the pre-registered analysis plan: everything here is exploratory.

    python extension_apertus/evaluate_apertus.py            # full run (500 bootstrap repetitions)
    python extension_apertus/evaluate_apertus.py --quick    # 50 repetitions, for a fast check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from uncertainty_project import api as project_api  # noqa: E402
from uncertainty_project.common import LABELS, config, now, read_json, write_json  # noqa: E402
from uncertainty_project.data import load_split  # noqa: E402
from uncertainty_project.experiment import (_labels_for, evaluate_bundle, experiment_directory,  # noqa: E402
                                            load_bundle)
from uncertainty_project.features import (entropy, feature_frame, joint_feature_frame,  # noqa: E402
                                          majority_correct, normalize, summarize_predictions)
from uncertainty_project.metrics import (dissociation_table, distribution_metrics, error_metrics,  # noqa: E402
                                         group_bootstrap_metric, js_divergence, paired_group_bootstrap,
                                         safe_auroc)
from uncertainty_project.models import aligned_human  # noqa: E402

MODEL = "swiss-ai/apertus-v1.5-70b"
FROZEN_MODELS = ("gpt", "claude")
ALL_MODELS = ("gpt", "claude", "apertus")
SPLITS = {"test": "mnli_test", "shift": "snli"}          # project split name -> Apertus output folder
SPLIT_NAMES = {"test": "MNLI-Test", "shift": "SNLI (unbekannte Eingaben)"}
OUTPUTS = HERE / "outputs_v2"
RESULTS = HERE / "results"
RAW_SIGNALS = ("model_total_entropy", "model_expected_entropy", "model_epistemic_mi", "model_credal_width",
               "model_label_flips", "model_one_minus_confidence")
# Tolerances for the reproduction check of notebook 03. Learned assessors, policies and error metrics must match
# exactly. Spearman correlations and AUROCs of RAW model signals can differ in the 3rd-4th decimal between
# computers, because nearly tied entropies / MI values are ordered differently (floating point).
TOL_EXACT, TOL_SPEARMAN = 1e-9, 0.005


# ------------------------------------------------------------------------------------------ Apertus responses

def prompt_fingerprint():
    """Same fingerprint as run_apertus.py: proves the stored answers used the frozen prompts."""
    return hashlib.sha256(json.dumps({"system": project_api.SYSTEM_PROMPT, "variants": list(project_api.PROMPT_VARIANTS)},
                                     sort_keys=True).encode()).hexdigest()[:16]


def load_records(folder_name, outputs=OUTPUTS):
    """All valid Apertus answers of one part, as {uid: {variant: record}}. Checks model, prompts and settings."""
    fp, records = prompt_fingerprint(), {}
    for path in sorted((Path(outputs) / folder_name).glob("*.json")):
        r = read_json(path)
        if r.get("status") != "ok" or r.get("model") != MODEL or r.get("prompt_fingerprint") != fp or r.get("temperature") != 0.0:
            raise ValueError(f"Unexpected Apertus record {path.name}: model, prompts or settings differ.")
        records.setdefault(str(r["uid"]), {})[int(r["variant"])] = r
    return records


def invalid_answers(folder_name, records, outputs=OUTPUTS):
    """Final status of every failed request: the LAST failure per (uid, variant) that never became valid."""
    path = Path(outputs) / f"{folder_name}_failed.jsonl"
    last = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                f = json.loads(line)
                last[(str(f["uid"]), int(f["variant"]))] = f
    rows = [{"uid": uid, "variant": v, "reason": f.get("reason"), "finish_reason": f.get("finish_reason"),
             "all_labels_zero": str(f.get("reason", "")).startswith("Summe 0.00"), "raw_start": f.get("raw_start", "")[:200]}
            for (uid, v), f in sorted(last.items()) if v not in records.get(uid, {})]
    return pd.DataFrame(rows, columns=["uid", "variant", "reason", "finish_reason", "all_labels_zero", "raw_start"])


def apertus_features(inputs, records, heads=None):
    """Features with the FROZEN summarize_predictions. Frozen rule: a case needs all five prompts,
    incomplete cases are excluded (never filled in). Returns (features, excluded uids)."""
    heads = len(project_api.PROMPT_VARIANTS) if heads is None else heads
    rows, excluded = [], []
    for row in inputs.to_dict("records"):
        got = records.get(str(row["uid"]), {})
        if len(got) < heads or any(v not in got for v in range(heads)):
            excluded.append(str(row["uid"]))
            continue
        vals = summarize_predictions([got[v]["parsed"] for v in range(heads)])
        rows.append({"uid": row["uid"], "group_id": row["group_id"], **vals})
    unknown = set(records) - set(inputs.uid.astype(str))
    if unknown:
        raise ValueError(f"Apertus answers for uids that are not in this split: {sorted(unknown)[:3]}")
    return pd.DataFrame(rows), excluded


# ------------------------------------------------------------------------------------------ reproduction check

def reproduce_notebook03(root, split, features, joint, labels, bundles):
    """Recompute the stored notebook-03 tables for GPT and Claude (no bootstrap) and compare.
    If this fails, the environment differs from the frozen one and nothing else should be trusted."""
    folder = experiment_directory(root) / split
    rows = []
    for k in FROZEN_MODELS:
        res = evaluate_bundle(bundles[k], features[k], labels, joint, 0, config(root)["seed"])
        for name in ("policies", "distributions", "error_risk", "dissociation"):
            old = pd.read_csv(folder / f"{k}_{name}.csv")
            new = res[name]
            cols = [c for c in old.select_dtypes("number").columns if c in new.columns and not c.endswith(("_ci_low", "_ci_high"))]
            # Raw model signals (entropy, MI, ...) contain near-ties whose order can flip between computers.
            raw = (old.signal.str.startswith(("model_", "joint_", "cross_")).to_numpy() if "signal" in old
                   else np.zeros(len(old), bool))
            for c in cols:
                diff = np.abs(old[c].to_numpy(float) - new[c].to_numpy(float))
                for part, mask in (("exact", ~raw), ("raw_signals", raw)):
                    d = diff[mask]
                    if not len(d):
                        continue
                    d = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
                    tol = TOL_SPEARMAN if ("spearman" in c or part == "raw_signals") else TOL_EXACT
                    rows.append({"split": split, "model": k, "table": name, "rows": part, "column": c,
                                 "max_abs_diff": d, "tolerance": tol, "ok": d <= tol})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ analyses

def raw_comparison(subset, labels, cfg):
    """A: the three raw models on identical cases, before any assessor (same metrics as notebook 03)."""
    rows, q = [], None
    for k, f in subset.items():
        q = aligned_human(f, labels)
        p = f[["mean_e", "mean_n", "mean_c"]].to_numpy()
        errors = 1 - majority_correct(f.base_label, q)
        # correctness_ece: calibration of the raw confidence against majority agreement (as "ECE" in report v03)
        rows.append({"model": k, "n": len(f), **distribution_metrics(p, q),
                     "correctness_ece": error_metrics(1 - f.mean_confidence.to_numpy(), errors)["correctness_ece"],
                     "mean_confidence": float(f.mean_confidence.mean()),
                     "mean_epistemic_mi": float(f.epistemic_mi.mean()),
                     "label_flip_rate": float((f.answer_disagreement > 0).mean()),
                     "mean_credal_width": float(f.credal_width.mean()),
                     **{f"answer_share_{c}": float((f.base_label == i).mean()) for i, c in enumerate("enc")}})
    return pd.DataFrame(rows)


def paired_differences(subset, labels, cfg):
    """A: Apertus minus GPT and minus Claude, premise groups resampled (frozen bootstrap settings)."""
    reps, seed = cfg["bootstrap_repetitions"], cfg["seed"]
    th = None
    ap = subset["apertus"]
    q = aligned_human(ap, labels)
    groups = ap.group_id.to_numpy()
    rows = []
    for other in FROZEN_MODELS:
        o = subset[other]
        assert (o.uid.to_numpy() == ap.uid.to_numpy()).all()
        pa, po = ap[["mean_e", "mean_n", "mean_c"]].to_numpy(), o[["mean_e", "mean_n", "mean_c"]].to_numpy()
        agree = majority_correct(ap.base_label, q) - majority_correct(o.base_label, q)
        b = paired_group_bootstrap(agree, groups, reps, seed, direction=f"positive: Apertus agrees with the majority more often than {other}")
        rows.append({"comparison": f"apertus_minus_{other}", "metric": "majority_agreement", "estimate": b["mean_difference"],
                     "ci_95_low": b["ci_95_low"], "ci_95_high": b["ci_95_high"], "better_if": "positive"})
        js = js_divergence(pa, q) - js_divergence(po, q)
        b = paired_group_bootstrap(js, groups, reps, seed, direction="negative: Apertus closer to the human distribution")
        rows.append({"comparison": f"apertus_minus_{other}", "metric": "js_divergence", "estimate": b["mean_difference"],
                     "ci_95_low": b["ci_95_low"], "ci_95_high": b["ci_95_high"], "better_if": "negative"})
        same = float((ap.base_label.to_numpy() == o.base_label.to_numpy()).mean())
        rows.append({"comparison": f"apertus_vs_{other}", "metric": "same_answer_rate", "estimate": same,
                     "ci_95_low": None, "ci_95_high": None, "better_if": "descriptive"})
    g, c = subset["gpt"], subset["claude"]
    rows.append({"comparison": "gpt_vs_claude", "metric": "same_answer_rate",
                 "estimate": float((g.base_label.to_numpy() == c.base_label.to_numpy()).mean()),
                 "ci_95_low": None, "ci_95_high": None, "better_if": "descriptive"})
    return pd.DataFrame(rows)


def raw_dissociation(subset, labels, thresholds, cfg):
    """A: raw signals of each model, AUROC for debatable cases and for majority deviations on clear cases."""
    out = []
    for k, f in subset.items():
        q = aligned_human(f, labels)
        errors = 1 - majority_correct(f.base_label, q)
        clear = entropy(q) <= thresholds["clear_max_entropy"]
        signals = {"model_total_entropy": f.mean_entropy, "model_expected_entropy": f.expected_entropy,
                   "model_epistemic_mi": f.epistemic_mi, "model_credal_width": f.credal_width,
                   "model_label_flips": f.answer_disagreement, "model_one_minus_confidence": 1 - f.mean_confidence}
        t = dissociation_table(signals, q, errors, clear, thresholds["debatable_min_entropy"],
                               group_ids=f.group_id.to_numpy(), repetitions=cfg["bootstrap_repetitions"], seed=cfg["seed"])
        t.insert(0, "model", k)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def error_profile(subset, labels, thresholds):
    """C: on clear cases, at which answer label do the majority deviations happen (as Check 1)?"""
    rows = []
    for k, f in subset.items():
        q = aligned_human(f, labels)
        errors = 1 - majority_correct(f.base_label, q)
        clear = entropy(q) <= thresholds["clear_max_entropy"]
        majority = q.argmax(axis=1)
        for i, lab in enumerate(LABELS):
            m = clear & (f.base_label.to_numpy() == i)
            rows.append({"model": k, "answer": lab, "answered_on_clear": int(m.sum()), "deviations": int(errors[m].sum()),
                         "deviation_rate": float(errors[m].mean()) if m.any() else None,
                         "mean_confidence_when_deviating": float(f.mean_confidence[m & (errors == 1)].mean()) if (m & (errors == 1)).any() else None,
                         **{f"majority_was_{c}": int(((majority == j) & m & (errors == 1)).sum()) for j, c in enumerate("enc")}})
    return pd.DataFrame(rows)


def transfer_tables(bundles, subset, labels, cfg):
    """B: frozen GPT/Claude assessors applied unchanged to Apertus, next to their own model on the same cases."""
    reps, seed = cfg["bootstrap_repetitions"], cfg["seed"]
    results, summary = {}, []
    for src in FROZEN_MODELS:
        own = evaluate_bundle(bundles[src], subset[src], labels, None, reps, seed)
        tr = evaluate_bundle(bundles[src], subset["apertus"], labels, None, reps, seed)
        results[f"{src}_own"], results[f"{src}_to_apertus"] = own, tr

        def pick(r, table, method, col):
            t = r[table]
            v = t.loc[t.method == method, col]
            return float(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else None

        def risk(r, method, subset_name="clear_cases"):
            t = r["error_risk"]
            v = t.loc[(t.method == method) & (t.subset == subset_name), "error_auroc"]
            return float(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else None

        def dis(r, signal, col):
            t = r["dissociation"]
            v = t.loc[t.signal == signal, col]
            return float(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else None

        for target, r in (("own (" + src + ")", own), ("apertus", tr)):
            summary.append({"assessor_from": src, "applied_to": target,
                            "debatable_auroc_learned_disagreement": dis(r, "learned_human_entropy", "auroc_debatable"),
                            "debatable_auroc_raw_total_entropy": dis(r, "model_total_entropy", "auroc_debatable"),
                            "clear_error_auroc_learned_risk": risk(r, "epistemic_tree_full"),
                            "clear_error_auroc_label_confidence": risk(r, "epistemic_label_confidence_tree"),
                            "clear_error_auroc_raw_confidence": risk(r, "raw_confidence"),
                            "human_js_tree": pick(r, "distributions", "tree_full", "js_divergence"),
                            "human_js_prompt_mean": pick(r, "distributions", "prompt_mean", "js_divergence"),
                            "policy_bal_acc_decomposed": pick(r, "policies", "decomposed", "balanced_accuracy"),
                            "policy_bal_acc_single_score_risk": pick(r, "policies", "single_score_risk", "balanced_accuracy"),
                            "policy_bal_acc_raw": pick(r, "policies", "raw", "balanced_accuracy"),
                            "policy_bal_acc_best_always": max(pick(r, "policies", f"always_{a}", "balanced_accuracy")
                                                              for a in ("answer", "hedge", "alternatives"))})
    return results, pd.DataFrame(summary)


def transfer_vs_raw(results, subset, labels, thresholds, cfg):
    """B: does the transferred learned signal beat Apertus' own raw signal? Paired AUROC differences."""
    reps, seed = cfg["bootstrap_repetitions"], cfg["seed"]
    ap = subset["apertus"]
    q = aligned_human(ap, labels)
    errors = 1 - majority_correct(ap.base_label, q)
    clear = entropy(q) <= thresholds["clear_max_entropy"]
    debatable = (entropy(q) >= thresholds["debatable_min_entropy"]).astype(float)
    groups = ap.group_id.to_numpy()
    raw_total, raw_conf = ap.mean_entropy.to_numpy(), 1 - ap.mean_confidence.to_numpy()
    rows = []
    for src in FROZEN_MODELS:
        pred = results[f"{src}_to_apertus"]["_pred"]
        learned_dis = entropy(pred["distributions"]["tree_full"])
        learned_risk = pred["risks"]["epistemic_tree_full"]
        for target, x, y, xname, yname in (("debatable", learned_dis, raw_total, "learned_human_entropy", "model_total_entropy"),
                                           ("error_on_clear", learned_risk, raw_conf, "learned_epistemic_risk", "model_one_minus_confidence")):
            if target == "debatable":
                def diff(i, x=x, y=y):
                    u, v = safe_auroc(debatable[i], x[i]), safe_auroc(debatable[i], y[i])
                    return None if u is None or v is None else u - v
            else:
                def diff(i, x=x, y=y):
                    j = i[clear[i]]
                    u, v = safe_auroc(errors[j], x[j]), safe_auroc(errors[j], y[j])
                    return None if u is None or v is None else u - v
            r = group_bootstrap_metric(diff, groups, reps, seed)
            rows.append({"assessor_from": src, "target": target, "signal": f"{xname} ({src} assessor)",
                         "minus": f"{yname} (Apertus raw)", "auroc_difference": r["estimate"],
                         "ci_95_low": r["ci_95_low"], "ci_95_high": r["ci_95_high"], "valid_resamples": r["valid_resamples"]})
    return pd.DataFrame(rows)


def invalid_profile(invalid, inputs, labels, thresholds):
    """D: which cases lost answers, by human majority label and agreement level."""
    if invalid.empty:
        return pd.DataFrame()
    lab = labels.set_index("uid")
    rows = []
    for uid, g in invalid.groupby("uid"):
        q = normalize(np.asarray(lab.loc[uid, "human_distribution"], float))
        h = float(entropy(q))
        rows.append({"uid": uid, "invalid_variants": len(g), "all_labels_zero": int(g.all_labels_zero.sum()),
                     "human_majority": LABELS[int(q.argmax())], "human_entropy": h,
                     "human_status": "clear" if h <= thresholds["clear_max_entropy"] else
                                     ("debatable" if h >= thresholds["debatable_min_entropy"] else "between"),
                     "excluded": len(g) > 0})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ main

def evaluate_part(root, split, bundles, cfg, outputs=OUTPUTS, check_reproduction=True):
    inputs, labels = load_split(root, split), _labels_for(root, split)
    thresholds = bundles["gpt"]["human_thresholds"]
    records = load_records(SPLITS[split], outputs)
    ap, excluded = apertus_features(inputs, records)
    invalid = invalid_answers(SPLITS[split], records, outputs)
    if set(invalid.uid) != set(excluded):
        raise RuntimeError("Excluded cases and logged invalid answers do not match. Check outputs_v2.")
    full = {k: feature_frame(root, inputs, k) for k in FROZEN_MODELS}
    out = {"meta": {"split": split, "name": SPLIT_NAMES[split], "cases": len(inputs), "apertus_complete": len(ap),
                    "excluded": excluded, "valid_answers": sum(len(v) for v in records.values()),
                    "requests": len(inputs) * len(project_api.PROMPT_VARIANTS), "invalid_answers": len(invalid)}}
    if check_reproduction:
        joint = joint_feature_frame(root, inputs, list(FROZEN_MODELS))
        out["reproduction"] = reproduce_notebook03(root, split, full, joint, labels, bundles)
    keep = set(ap.uid)
    subset = {k: full[k][full[k].uid.isin(keep)].reset_index(drop=True) for k in FROZEN_MODELS}
    subset["apertus"] = ap.set_index("uid").loc[subset["gpt"].uid].reset_index()
    out["raw_comparison"] = raw_comparison(subset, labels, cfg)
    out["paired_differences"] = paired_differences(subset, labels, cfg)
    out["raw_dissociation"] = raw_dissociation(subset, labels, thresholds, cfg)
    out["error_profile"] = error_profile(subset, labels, thresholds)
    results, out["transfer_summary"] = transfer_tables(bundles, subset, labels, cfg)
    out["transfer_vs_raw"] = transfer_vs_raw(results, subset, labels, thresholds, cfg)
    for name, r in results.items():
        out[f"{name}_specialisation"] = r["specialisation"]
        out[f"{name}_policies"] = r["policies"]
        out[f"{name}_dissociation"] = r["dissociation"]
        out[f"{name}_policy_intervals"] = r["policy_intervals"]
    out["invalid_answers"] = invalid
    out["invalid_cases"] = invalid_profile(invalid, inputs, labels, thresholds)
    out["apertus_features"] = ap
    q = aligned_human(ap, labels)
    out["meta"].update({"clear_cases": int((entropy(q) <= thresholds["clear_max_entropy"]).sum()),
                        "debatable_cases": int((entropy(q) >= thresholds["debatable_min_entropy"]).sum()),
                        "apertus_clear_deviations": int(((1 - majority_correct(ap.base_label, q)) *
                                                         (entropy(q) <= thresholds["clear_max_entropy"])).sum())})
    return out


def save(results, folder=RESULTS):
    folder = Path(folder)
    for split, out in results.items():
        sub = folder / split
        sub.mkdir(parents=True, exist_ok=True)
        for name, table in out.items():
            if isinstance(table, pd.DataFrame):
                table.to_csv(sub / f"{name}.csv", index=False)
        write_json(sub / "meta.json", {**out["meta"], "written_at": now()})


def figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"gpt": "#2a7ab9", "claude": "#d95f02", "apertus": "#1b9e77"}
    metrics = [("majority_agreement", "Mehrheitsübereinstimmung ↑"), ("js_divergence", "JS zu Menschen ↓"),
               ("correctness_ece", "Kalibrierung (ECE) ↓"), ("entropy_spearman", "Spearman Entropie ↑")]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    for ax, (m, title) in zip(axes[:4], metrics):
        for j, split in enumerate(results):
            t = results[split]["raw_comparison"].set_index("model")
            for i, k in enumerate(ALL_MODELS):
                ax.bar(j + (i - 1) * 0.26, t.loc[k, m], width=0.24, color=colors[k], label=k if j == 0 else None)
        ax.set_xticks(range(len(results)), ["MNLI-Test", "SNLI"])
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    ax = axes[4]
    t = results["test"]["transfer_summary"]
    labels_x, vals, cols = [], [], []
    for src in FROZEN_MODELS:
        own = t[(t.assessor_from == src) & (t.applied_to != "apertus")].iloc[0]
        tr = t[(t.assessor_from == src) & (t.applied_to == "apertus")].iloc[0]
        labels_x += [f"{src}\neigen", f"{src}→\napertus"]
        vals += [own.clear_error_auroc_learned_risk, tr.clear_error_auroc_learned_risk]
        cols += [colors[src], colors["apertus"]]
    labels_x.append("apertus\nroh"); vals.append(t[t.applied_to == "apertus"].iloc[0].clear_error_auroc_raw_confidence)
    cols.append("#999999")
    ax.bar(range(len(vals)), vals, color=cols)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_xticks(range(len(vals)), labels_x, fontsize=8)
    ax.set_ylim(0.3, 1)
    ax.set_title("MNLI: Fehler auf klaren Fällen (AUROC)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Apertus (post-hoc) vs. GPT und Claude, gleiche Fälle", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(root=ROOT, quick=False, outputs=OUTPUTS, results_dir=RESULTS, check_reproduction=True):
    root = Path(root)
    cfg = dict(config(root))
    if quick:
        cfg["bootstrap_repetitions"] = 50
    bundles = {k: load_bundle(root, k) for k in FROZEN_MODELS}
    results = {split: evaluate_part(root, split, bundles, cfg, outputs, check_reproduction) for split in SPLITS}
    if check_reproduction:
        bad = pd.concat([r["reproduction"] for r in results.values()]).query("not ok")
        if len(bad):
            raise RuntimeError(f"Notebook-03 numbers NOT reproduced ({len(bad)} values). Environment differs; stop.\n{bad.head()}")
    save(results, results_dir)
    figure(results, Path(results_dir) / "apertus_vergleich.png")
    write_json(Path(results_dir) / "run_info.json",
               {"bootstrap_repetitions": cfg["bootstrap_repetitions"], "seed": cfg["seed"], "model": MODEL,
                "prompt_fingerprint": prompt_fingerprint(), "frozen_protocol": read_json(experiment_directory(root) / "frozen_protocol.json")["protocol_id"],
                "status": "post-hoc extension, exploratory, not in the frozen analysis plan", "written_at": now()})
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="50 bootstrap repetitions instead of 500")
    args = ap.parse_args()
    res = run(quick=args.quick)
    for split, out in res.items():
        print(f"\n=== {out['meta']['name']}: {out['meta']['apertus_complete']}/{out['meta']['cases']} Fälle komplett")
        print(out["raw_comparison"].round(3).to_string(index=False))
        print(out["transfer_summary"].round(3).to_string(index=False))
    print(f"\nGespeichert in {RESULTS}")

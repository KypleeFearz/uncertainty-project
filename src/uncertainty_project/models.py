"""Small, inspectable assessors learned from stored model outputs (the LLM itself is never retrained).

Per feature set, three shallow regression trees:
  distribution : predicts the human vote distribution        -> ALEATORIC estimate ("debatable?")
  epistemic    : predicts majority error on CLEAR cases only  -> EPISTEMIC estimate ("uncertain?")
  risk         : predicts majority error on all cases         -> comparison: risk not separated by source

Policy variants compared (all tuned on development data only):
  decomposed               two signals: predicted human entropy + epistemic tree (clear cases)
  risk_trained_all_cases   two signals: predicted human entropy + risk tree (all cases)
                           -> isolates the effect of training the risk signal on clear cases only
  single_score_risk        ONE score (risk tree, all cases) with two thresholds
  single_score_confidence  ONE score (1 - model confidence) with two thresholds
                           -> test whether two separate signals are needed at all
  raw / calibrated         two signals from the model's own probabilities only

"Error" always means: the model's answer is not a human majority label. On cases with high human
agreement this is an operational proxy for a model error, not proof that the answer is objectively wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.tree import DecisionTreeRegressor, export_text

from .features import FEATURE_SETS, FULL_FEATURES, LABEL_CONFIDENCE_FEATURES, entropy, majority_correct, normalize, probabilities
from .metrics import distribution_metrics, error_metrics, js_divergence
from .policy import apply_policy, reference_actions, tune_policy, tune_single_score_policy


def aligned_human(features, labels):
    if labels.uid.duplicated().any() or features.uid.duplicated().any():
        raise ValueError("Duplicate IDs in learning data.")
    return normalize(np.stack(labels.set_index("uid").loc[features.uid, "human_distribution"].to_numpy()))


def aligned_taxonomy(features, labels):
    return labels.set_index("uid").loc[features.uid, "taxonomy_categories"].tolist()


def temperature_scale(p, temperature):
    return softmax(np.log(np.clip(normalize(p), 1e-8, 1)) / temperature, axis=1)


def human_thresholds(train_human, cfg):
    """Declared from TRAIN human entropy only; reused unchanged for dev, test and every model."""
    h = entropy(train_human)
    d = cfg["decomposition"]
    return {"clear_max_entropy": float(np.quantile(h, d["clear_case_train_entropy_quantile"])),
            "debatable_min_entropy": float(np.quantile(h, d["debatable_train_entropy_quantile"])),
            "clarify_cue_threshold": float(d["clarify_cue_threshold"])}


def _grid(cfg):
    for depth in cfg["trees"]["max_depth"]:
        for leaf in cfg["trees"]["min_samples_leaf"]:
            yield dict(max_depth=depth, min_samples_leaf=leaf, random_state=cfg["seed"])


def fit_assessors(train_features, train_labels, dev_features, dev_labels, cfg, model_key):
    if set(train_features.group_id) & set(dev_features.group_id):
        raise ValueError("Train/development premise groups overlap.")
    if len(train_features) < 100 or len(dev_features) < 50:
        raise ValueError("The 50-example pilot checks the pipeline. Collect full train/development predictions before model fitting.")
    qt, qd = aligned_human(train_features, train_labels), aligned_human(dev_features, dev_labels)
    et = 1 - majority_correct(train_features.base_label, qt)
    ed = 1 - majority_correct(dev_features.base_label, qd)
    thresholds = human_thresholds(qt, cfg)
    clear_t = entropy(qt) <= thresholds["clear_max_entropy"]
    clear_d = entropy(qd) <= thresholds["clear_max_entropy"]
    result = {"model_key": model_key, "models": {}, "feature_sets": FEATURE_SETS, "train_prior": qt.mean(axis=0),
              "human_thresholds": thresholds, "selected_parameters": {},
              "training_rows": len(qt), "development_rows": len(qd),
              "training_clear_rows": int(clear_t.sum()), "training_clear_errors": int(et[clear_t].sum())}
    search_rows = []
    for name, columns in FEATURE_SETS.items():
        x_train, x_dev = train_features[columns].to_numpy(), dev_features[columns].to_numpy()
        best = {"distribution": (np.inf, None), "risk": (np.inf, None), "epistemic": (np.inf, None)}
        for params in _grid(cfg):
            trees = {"distribution": DecisionTreeRegressor(**params).fit(x_train, qt),
                     "risk": DecisionTreeRegressor(**params).fit(x_train, et),
                     "epistemic": DecisionTreeRegressor(**params).fit(x_train[clear_t], et[clear_t])}
            scores = {"distribution": float(js_divergence(trees["distribution"].predict(x_dev), qd).mean()),
                      "risk": float(np.mean((trees["risk"].predict(x_dev) - ed) ** 2)),
                      "epistemic": float(np.mean((trees["epistemic"].predict(x_dev[clear_d]) - ed[clear_d]) ** 2))}
            search_rows.append({"features": name, "max_depth": params["max_depth"], "min_samples_leaf": params["min_samples_leaf"],
                                "dev_js_divergence": scores["distribution"], "dev_error_brier": scores["risk"],
                                "dev_clear_error_brier": scores["epistemic"]})
            for target in best:
                if scores[target] < best[target][0]:
                    best[target] = (scores[target], trees[target])
        result["models"][name] = {target: model for target, (_, model) in best.items()}
        result["selected_parameters"][name] = {
            target: {k: model.get_params()[k] for k in ("max_depth", "min_samples_leaf")}
            for target, model in result["models"][name].items()}
    # Comparison assessors (same grid, train fit, dev selection):
    #   entropy_direct              predicts human vote ENTROPY directly. A distribution tree can produce high predicted
    #                               entropy by averaging unanimous cases with different labels in one leaf; this checks that.
    #   label_confidence(_clear)    risk from answer label + confidence only (all cases / clear cases only).
    extra_specs = {"entropy_direct": (FULL_FEATURES, entropy(qt), entropy(qd), None, None),
                   "label_confidence_risk": (LABEL_CONFIDENCE_FEATURES, et, ed, None, None),
                   "label_confidence_epistemic": (LABEL_CONFIDENCE_FEATURES, et, ed, clear_t, clear_d)}
    result["extra_models"], result["extra_features"] = {}, {}
    for target, (columns, yt, yd, mt, md) in extra_specs.items():
        xt, xd = train_features[columns].to_numpy(), dev_features[columns].to_numpy()
        mt = np.ones(len(yt), bool) if mt is None else mt
        md = np.ones(len(yd), bool) if md is None else md
        best = (np.inf, None)
        for params in _grid(cfg):
            tree = DecisionTreeRegressor(**params).fit(xt[mt], yt[mt])
            score = float(np.mean((tree.predict(xd[md]) - yd[md]) ** 2))
            search_rows.append({"features": target, "max_depth": params["max_depth"], "min_samples_leaf": params["min_samples_leaf"],
                                "dev_mse": score})
            if score < best[0]:
                best = (score, tree)
        result["extra_models"][target], result["extra_features"][target] = best[1], list(columns)
    pt = probabilities(train_features)
    fitted = minimize_scalar(lambda log_t: float(-(qt * np.log(np.clip(temperature_scale(pt, np.exp(log_t)), 1e-15, 1))).sum(axis=1).mean()),
                             bounds=(-2, 2), method="bounded")
    result["temperature"] = float(np.exp(fitted.x))
    result["risk_calibrator"] = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(pt.max(axis=1), et)
    result["label_error_rates"] = label_error_rates(train_features.base_label.to_numpy(), et, clear_t)
    # Policies are tuned on development data against reference actions.
    pred = predict_assessors(result, dev_features)
    ref, _ = reference_actions(qd, dev_features.base_label, thresholds["debatable_min_entropy"])
    result["policies"], policy_search = {}, []
    for name, inputs in policy_inputs(pred, dev_features).items():
        if isinstance(inputs, tuple):
            policy, grid = tune_policy(*inputs, dev_features.clarifiable_cue, ref, cfg["policy"], thresholds["clarify_cue_threshold"])
        else:
            policy, grid = tune_single_score_policy(inputs, dev_features.clarifiable_cue, ref, thresholds["clarify_cue_threshold"])
        result["policies"][name] = policy
        grid["method"] = name
        policy_search.append(grid)
    return result, pd.DataFrame(search_rows), pd.concat(policy_search, ignore_index=True)


def label_error_rates(base_labels, errors, clear):
    """Baseline: how often the model's answer label alone was a majority deviation on TRAIN.
    Tests whether learned risk adds anything beyond 'which label did the model pick'
    (e.g. a systematic preference for neutral). Labels never seen fall back to the overall rate."""
    base_labels, errors, clear = np.asarray(base_labels, int), np.asarray(errors, float), np.asarray(clear, bool)
    def rates(mask):
        overall = float(errors[mask].mean())
        return [float(errors[mask & (base_labels == k)].mean()) if np.any(mask & (base_labels == k)) else overall
                for k in range(3)]
    return {"clear": rates(clear), "all": rates(np.ones(len(errors), bool))}


def predict_assessors(bundle, features):
    p = probabilities(features)
    distributions = {"training_prior": np.tile(bundle["train_prior"], (len(features), 1)),
                     "single_prompt": probabilities(features, "single"), "prompt_mean": p,
                     "calibrated_prompt_mean": temperature_scale(p, bundle["temperature"])}
    labels = features.base_label.to_numpy(int)
    rates = bundle["label_error_rates"]
    risks = {"raw_confidence": 1 - p.max(axis=1),
             "calibrated_confidence": bundle["risk_calibrator"].predict(p.max(axis=1)),
             "answer_label_only": np.asarray(rates["all"])[labels],
             "epistemic_answer_label_only": np.asarray(rates["clear"])[labels]}
    for name, models in bundle["models"].items():
        x = features[bundle["feature_sets"][name]].to_numpy()
        distributions[f"tree_{name}"] = normalize(np.clip(models["distribution"].predict(x), 1e-6, 1))
        risks[f"tree_{name}"] = np.clip(models["risk"].predict(x), 0, 1)
        risks[f"epistemic_tree_{name}"] = np.clip(models["epistemic"].predict(x), 0, 1)
    extra = {k: m.predict(features[bundle["extra_features"][k]].to_numpy()) for k, m in bundle["extra_models"].items()}
    risks["label_confidence_tree"] = np.clip(extra["label_confidence_risk"], 0, 1)
    risks["epistemic_label_confidence_tree"] = np.clip(extra["label_confidence_epistemic"], 0, 1)
    return {"distributions": distributions, "risks": risks, "entropy_direct": np.clip(extra["entropy_direct"], 0, 1)}


def policy_inputs(pred, features):
    """Two-signal policies get (aleatoric distribution, epistemic risk); single-score policies get one array."""
    return {"decomposed": (pred["distributions"]["tree_full"], pred["risks"]["epistemic_tree_full"]),
            "risk_trained_all_cases": (pred["distributions"]["tree_full"], pred["risks"]["tree_full"]),
            "single_score_risk": np.asarray(pred["risks"]["tree_full"], float),
            "single_score_confidence": 1 - features.mean_confidence.to_numpy(float),
            "raw": (pred["distributions"]["prompt_mean"], pred["risks"]["raw_confidence"]),
            "calibrated": (pred["distributions"]["calibrated_prompt_mean"], pred["risks"]["calibrated_confidence"])}


def run_policies(bundle, features, pred=None):
    pred = pred if pred is not None else predict_assessors(bundle, features)
    top2 = np.argsort(-pred["distributions"]["tree_full"], axis=1, kind="stable")[:, :2]
    out = {}
    for name, inputs in policy_inputs(pred, features).items():
        actions, sets = apply_policy(bundle["policies"][name], inputs, features.clarifiable_cue)
        out[name] = (actions, sets if sets is not None else top2)
    return out


def evaluation_tables(bundle, features, labels):
    q = aligned_human(features, labels)
    errors = 1 - majority_correct(features.base_label, q)
    clear = entropy(q) <= bundle["human_thresholds"]["clear_max_entropy"]
    pred = predict_assessors(bundle, features)
    dist = pd.DataFrame([{"method": name, **distribution_metrics(p, q)} for name, p in pred["distributions"].items()])
    risk = pd.DataFrame([{"method": name, "subset": subset, "n": int(mask.sum()), "errors": int(errors[mask].sum()),
                          **error_metrics(r[mask], errors[mask])}
                         for name, r in pred["risks"].items()
                         for subset, mask in (("all", np.ones(len(q), bool)), ("clear_cases", clear))])
    return dist, risk, pred


def tree_rules(bundle, name="full"):
    rules = {target: export_text(model, feature_names=bundle["feature_sets"][name], decimals=3)
             for target, model in bundle["models"][name].items()}
    for target, model in bundle.get("extra_models", {}).items():
        rules[target] = export_text(model, feature_names=bundle["extra_features"][target], decimals=3)
    return rules

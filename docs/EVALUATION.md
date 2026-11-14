# Evaluation protocol

## Separation of data

Use one fixed seed. Allocate approximately 60% / 20% / 20% of premise groups to training,
development and test. Group by Unicode-normalised, whitespace-normalised, case-folded premise
so hypotheses sharing a premise cannot cross splits. Fractions apply to groups, so example
counts can differ slightly from exact percentages. Near-duplicate paraphrases are not guaranteed absent.

The pilot comes from training data. Fit trees, temperature and isotonic calibration on training
examples. Use development examples for tree size and action thresholds. Final test labels are
stored separately and read only after complete predictions and a frozen protocol exist.
The frozen file hashes configuration, source code, data, fitted models and training/development
response caches. This is a reproducibility mechanism, not external preregistration or a security barrier.

Do not revise a hypothesis based on the final test and report that same test as fresh evidence.
Exploratory follow-ups must be labelled as such. Public benchmark exposure during LLM pretraining
remains a limitation even with this project-level separation.

## Targets

- Human distribution: the empirical E/N/C frequencies among 100 responses.
- Human disagreement: normalised entropy of that distribution, range 0–1.
- Operational model error: ensemble-mean top label is not one of the human majority labels.
  All tied majority labels count as agreement. This reference does not make minority readings invalid.

The full assessor contains three shallow regressors per model: a three-output distribution tree,
an epistemic error tree fitted on clear cases and an error tree fitted on all cases. Leaf averages yield probabilities.
Their parameters are selected separately on development data (JS divergence, clear-case error Brier, error Brier). Learned distribution predictions
do not replace the fixed main-model answer in the error-risk target.

## Comparisons

Distribution: training-average prior; first prompt; five-prompt mean; temperature-scaled mean;
trees using only first-prompt scores, numerical stability signals, or all signals including cues.
Temperature is fitted against training human distributions, not test outcomes.

Error risk: one minus mean-model confidence; isotonic-calibrated risk; the three tree variants.
All error-risk comparisons use the same fixed main-model answer, so a change in answer accuracy
does not masquerade as an improvement in error detection.

## Metrics

| Measure | Meaning | Direction |
|---|---|---|
| Jensen–Shannon divergence, base 2 | Difference between predicted and human distributions; not square-root distance | Lower |
| Expected multiclass Brier | `1 + sum(p²) - 2 sum(p*q)`, expectation over an empirical human vote | Lower |
| Majority agreement | Predicted top class belongs to the empirical majority set | Higher |
| Human support ECE | Predicted top-class confidence versus fraction of humans choosing it, 10 fixed bins | Lower |
| Entropy Spearman | Ranking agreement between predicted and observed disagreement | Higher |
| Error Brier | Squared error of predicted majority-disagreement probability | Lower |
| Correctness ECE | Predicted chance of matching the majority versus observed rate | Lower |
| Error AUROC | Ranking errors above majority-matching answers | Higher |
| AURC | Mean risk across retained coverage levels | Lower |

Tied risk scores use the expected risk under random ordering within each tie group.
Report class support and sample size. Undefined correlations/AUROC are left missing, not replaced by zero.
Group bootstrap resamples entire premises and provides paired 95% percentile intervals for
the JS difference and the balanced-accuracy differences between policies. These intervals are conditional on the
one fitted model and do not capture all training-seed or API variability.

## Concepts

Map the three predicted cues to Lexical, Coreference and Probabilistic taxonomy categories.
Use round 1 `a1` and round 2 `agg`. Unannotated examples stay missing. Exclude `N/A` from cue
scoring rather than treating it as proof of absence. Use a fixed threshold 0.5 for F1, plus
average precision/AUROC where defined. Report how many examples came from each annotation round.
This is exploratory concept-quality evidence on a selected subset, not representative accuracy
on all language or proof that the tree explains the LLM's internal reasoning.

## Decomposition: uncertain vs debatable (core analysis)

Each prompt variant is one head (5 per model, 10 when GPT and Claude are pooled).
Ensemble decomposition per example: `H(mean p) = mean_k H(p_k) + MI`.
`expected_entropy` (mean head entropy) is the model's own estimate of aleatoric ambiguity,
`epistemic_mi` (disagreement between heads) the epistemic part. Credal-style signals use the
per-label interval `[min_k p_k, max_k p_k]`: mean/max width and whether one label's interval
dominates all others. Prompt variation is a measurable proxy, not a proven identification.

Human reference sets, declared from TRAIN human entropy only (config `decomposition`):
- clear cases: human entropy at or below the train 33% quantile. A majority deviation on these cases
  is the operational proxy for a model error (the *uncertain* target). It does not prove that the
  answer is objectively wrong; a minority reading can still be defensible.
- debatable cases: human entropy at or above the train median (the *debatable* target).

Assessors per model and feature set: `distribution` tree (human votes, aleatoric),
`epistemic` tree (majority error fitted on clear training cases only) and the `risk` tree
(majority error on all cases) as a comparison that does not separate sources.

Dissociation table: for each raw or learned signal, AUROC for `debatable` and AUROC for
majority deviation on clear cases, each with a premise-grouped bootstrap 95% interval.
The research claim is deliberately modest: do these signals help distinguish human disagreement
from majority deviations on high-agreement cases? The link to credal concept bottleneck models is an
inspired simplification (API queries as heads), not a reproduction with trained concept heads. A successful separation shows aleatoric signals informative for the
first and near 0.5 for the second, and epistemic signals the reverse. Report the number of
errors on clear cases: with few errors these AUROCs have wide uncertainty.

## Actions

Four actions: `answer`, `hedge` (state uncertainty or abstain), `alternatives` (top two labels)
and `clarify` (ask a question). Rule: debatable if predicted human entropy crosses a threshold;
then `clarify` if the mean predicted lexical/reference cue is at least 0.5, else `alternatives`.
Otherwise `hedge` if the epistemic risk crosses a threshold, else `answer`.

No utility costs are assumed. Each example receives a reference action from human votes and the
model's answer: debatable -> clarify/alternatives (split by taxonomy category where annotated:
Lexical or Coreference -> clarify, otherwise alternatives), clear and correct -> answer,
otherwise -> hedge. Main metric: balanced accuracy over answer/hedge/debatable. Also reported:
per-class recall, the rate of unwarranted direct answers, and for clarify vs alternatives the recall
on ALL annotated reference cases (missed cases count as wrong) plus, clearly labelled, the
conditional accuracy among cases where the system already chose one of the two.

Compared policies, thresholds tuned on development data only:
- decomposed: predicted human entropy + risk tree fitted on clear cases
- risk_trained_all_cases: predicted human entropy + risk tree fitted on all cases
  (isolates the effect of fitting the risk signal on clear cases)
- single_score_risk and single_score_confidence: ONE score with two thresholds
  (tests whether two separate signals are needed at all)
- raw and calibrated: the model's own probabilities only
- always answer / hedge / alternatives

Paired premise-grouped bootstrap intervals are computed for the main metric
(balanced accuracy of decomposed minus each comparison).
The reference is a declared operationalisation, not observed user benefit.

## Models and transfer

GPT and Claude answer the same examples with the same prompts. Model comparison reports the raw
ensemble-mean distributions against human votes and a premise-grouped paired bootstrap for the
difference in majority agreement. Transfer applies an assessor fitted on one model's train/dev
outputs to the other model's outputs without retuning. Joint (10-head) signals need both
providers and are reported separately from single-provider transfer.

## What this version does not evaluate

Individual annotator reliability, true out-of-distribution detection, autonomous retrieval,
the wording quality of clarification questions, end-user utility and a trained credal concept
bottleneck. They must not be claimed as completed by these notebooks.

# Analysis plan (fixed before the test)

Source of truth: `config/project.json` → `analysis_plan`, sealed by the freeze fingerprint. This file mirrors it.

## Primary (claims rest on these)

- RQ1: specialisation, paired AUROC difference on debatable cases: learned_human_entropy minus learned_epistemic_risk
- RQ1: specialisation, paired AUROC difference on majority deviations among clear cases: learned_epistemic_risk minus learned_human_entropy
- RQ1: added value of ensemble signals, AUROC on clear cases: learned_epistemic_risk minus baseline_label_confidence
- RQ2: balanced accuracy, decomposed minus single_score_risk (paired bootstrap)
- RQ3: transfer GPT <-> Claude: policy balanced accuracy, human JS divergence and clear-case error AUROC, transfer vs own

## Secondary (descriptive)

- Remaining policy comparisons: decomposed minus risk_trained_all_cases, single_score_confidence, calibrated
- Robustness: learned_entropy_direct minus learned_human_entropy on debatable cases
- Full dissociation table incl. raw signals, joint GPT+Claude signals and the answer-label baseline
- Distribution metrics (JS divergence, majority agreement of the distribution tree, reported as its own result, not as an improved model answer)
- Transfer to another NLI text source (SNLI sample): same tables, reported separately

## Exploratory

- clarify vs alternatives on annotated MNLI cases: reported in full although development results were weak; not evaluated on SNLI (no annotations)
- Language cues vs disagreement categories
- Hypothesis that neutral answers act as an evasive label

## Interpretation rules

- Signals are compared on two observable targets; this does not identify epistemic and aleatoric uncertainty. Prompt MI first measures prompt dependence.
- Cases between the thresholds (0.624 < entropy < 0.683) are not debatable: reference answer if the model matches the majority, else hedge.
- The evaluated model answer is always the top label of the five-prompt mean, also in transfer (reference actions use the target model's answer).
- Intervals are 95% percentile bootstraps over premise groups, not adjusted for multiple comparisons; only primary comparisons support claims, others are descriptive.
- Development numbers are optimistic because trees and thresholds were selected on the same data.

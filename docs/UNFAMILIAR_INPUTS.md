# Transfer to another NLI text source ("unfamiliar inputs", added 12 October 2026, before freezing)

Supervisor request (feedback e-mail, point 3): "assess calibration, error detection and behaviour on unfamiliar inputs".

- Source: ChaosNLI-SNLI (image captions; 1,514 pairs, 100 votes each), pinned file and SHA-256 in `config/data_sources.json`.
  The MNLI training, development and test data come from a different source (multi-genre text).
- Sample: 300 pairs, fixed seed (`config/project.json`, section `shift`), drawn with `prepare_shift` in notebook 02
  BEFORE freezing. The sample files are part of the frozen fingerprint.
- IDs carry the prefix `snli_`, so cached responses can never collide with MNLI IDs.
- Pairs whose normalised premise also occurs in the MNLI data are excluded before sampling (count reported by `prepare_shift`).
- Both ChaosNLI subsets were selected by the same low-agreement criterion and re-annotated with 100 votes: a limited transfer check within a related benchmark, not robustness to arbitrary inputs, and no evidence that the models never saw these items.
- Labels are stored separately (`shift_labels.jsonl`), as for the test split.
- Paid calls for these inputs are only allowed after freezing and may not be mixed with other requests.
- Evaluation: the frozen assessors, MNLI-derived human thresholds (0.624 / 0.683) and dev-selected policies are applied
  unchanged (`final_evaluation`, folder `outputs/experiments/.../shift`). No refitting, no threshold changes.
- No disagreement-category annotations exist for SNLI, so clarify vs. alternatives is not evaluated there.
- Limitation: same task and label set, so this is a mild shift in text source, not an out-of-task test.

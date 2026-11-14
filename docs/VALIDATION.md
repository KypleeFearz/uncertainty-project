# Preparation checks

> Status note (November 2026): this is a planning document from before the freeze. The study is complete; results and limitations are in the project report.

Version 2.1, 11 October 2026 (two providers, decomposition, four actions; review fixes: single-score baselines,
unconditional clarify/alternatives recall, bootstrap intervals for the main policy metric and the dissociation AUROCs,
fixed train threshold in the pilot view, more cautious wording).

- Eleven automated tests pass, including a complete artificial-data cycle with two models:
  fit, development report, transfer, freeze, final evaluation and the protection of the test split.
- The OpenAI Responses request and the Anthropic Messages request with `output_config`
  (JSON schema + effort) were exercised with offline HTTP mocks, including a schema violation
  that is retried and recorded rather than repaired. No genuine model output or account access was tested.
- Notebooks 01 to 04 were executed end to end in a Jupyter kernel on an artificial project.
  Notebook 00 is unchanged from version 1 (data download needs network access).
- Unit tests check the decomposition identity (total = expected entropy + mutual information),
  credal interval dominance, the four actions and the dissociation table on synthetic signals.

The tests' artificial responses remain in temporary test directories and are not scientific results.
The package ships without raw benchmark files, API keys, response caches or fitted assessors.

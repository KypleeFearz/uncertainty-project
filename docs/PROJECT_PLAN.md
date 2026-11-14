# Working research scope (version 2)

> Status note (November 2026): this is a planning document from before the freeze. The study is complete; results and limitations are in the project report.

Status: executable starting point, aligned with the supervisors' feedback (define goals before
architecture, compare models, simple baselines, clean dev/test separation). Not a completed study.

## Research questions

- **RQ1 (core): uncertain or debatable?** Do signals from a prompt ensemble (and a two-model ensemble)
  help distinguish human disagreement from majority deviations on cases with high human agreement?
  Evidence: dissociation table and plot with bootstrap intervals.
- **RQ2: response choice.** Does an assessor with two separate signals choose among answer, hedge,
  clarify and alternatives better than a single score, a risk signal fitted on all cases, or the
  model's own confidence?
- **RQ3: models and transfer.** How do GPT and Claude differ on identical examples, and does an
  assessor fitted on one model's outputs work on the other's without retuning?

## Contribution

A lightweight, API-based simplification inspired by the epistemic/aleatoric distinction of credal concept
bottleneck models (Mukherjee et al., 2026): prompt variants act as heads, label intervals across
heads act as credal sets, and human vote distributions supervise the aleatoric part. Evaluated against
calibrated baselines, with a test of whether the two signals actually dissociate, and connected to
a four-action response policy that feeds the thesis interface.

## Work packages

1. Smoke test (5 pairs) and pilot (50 pairs) for both models. Revise prompts on training examples only.
2. Collect all train/development responses for both models.
3. Fit assessors, inspect the dissociation, select thresholds on development data. Freeze.
4. Evaluate once on the held-out test: dissociation, policies, model comparison, transfer.
5. Demo: show_case logic in the thesis interface; 90-second video; final presentation.
6. Report: method, results, failure examples, limitations.

## Decisions to align with supervisors

- Is this API-based credal adaptation acceptable, or are trained concept heads expected?
- Is the reference-action evaluation (no invented costs) acceptable for the response policy?
- Is a small shift check (for example ambiguous NLI from AmbiEnt) wanted, or is the held-out test enough?

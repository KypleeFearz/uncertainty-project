# Uncertain or Genuinely Debatable? Teaching AI to Tell the Difference

Integrative Master's Project (IMP), Project 14, School of Computer Science, University of St. Gallen.

Author: Nikola Milosavljevic. Supervisor: Prof. Dr. Christina Niklaus. Co-supervisor: Thomas Huber.

A confidence score does not say why caution is needed: the model may be unreliable on an input that people
judge consistently, or people may themselves disagree about the input. This project tests whether the two
situations can be told apart and whether doing so improves the choice among four responses: **answer**,
**hedge**, **alternatives** and **clarify**.

The setting is natural language inference on ChaosNLI, where every sentence pair has 100 human judgements.
GPT (`gpt-6.1-sol`) and Claude (`claude-sonnet-5-5`) answer each pair under five prompt formulations. Small
decision trees learn a disagreement signal from the human vote distributions and an error-risk signal from
majority deviations on clear cases. The method was frozen with checksums before the held-out test.

## Main findings

- The models' own uncertainty detects human disagreement only weakly on MNLI (AUROC at most 0.57). A learned
  disagreement signal reaches 0.71 (GPT) and 0.66 (Claude).
- Error risk on clear cases is captured by the answer label and its confidence (AUROC 0.90 and 0.89). Prompt
  ensemble signals add nothing measurable on top.
- A two-signal policy beats a single learned risk score for GPT (+0.076 balanced accuracy) but not for Claude.
- The learned assessors do not travel. They deteriorate on SNLI, and the GPT risk assessor ranks the errors of
  a third model (Apertus) worse than chance, because the two models err behind opposite labels.

All numbers, intervals and limitations are in the project report.

## Repository structure

```
config/               experiment settings, data sources, sealed analysis plan
data/processed/       fixed train, development, test and SNLI splits with human votes
src/uncertainty_project/   research code (API calls, features, assessors, policy, metrics)
notebooks/            00 data, 01 model responses, 02 fitting and freeze, 03 final evaluation,
                      04 case inspection, 05 Apertus extension
outputs/runs/         stored model responses (9,495 per model)
outputs/experiments/  fitted assessors, frozen protocol, development, test and SNLI result tables
demo/                 local web demonstrator and its tests
extension_apertus/    post-hoc extension: a third model evaluated with the frozen assessors
extension_posthoc/    post-hoc robustness checks: definition of a clear case, source separation
tests/                tests of the research pipeline
tools/                verify_freeze.py
docs/                 evaluation design, analysis plan, data provenance, development checks
```

Sealed by the freeze (protocol `3942fc496262a9b3`, 12 October 2026): `src/`, `config/`, `data/processed/`,
`outputs/runs/` and the fitted assessors in `outputs/experiments/`. Everything added later lives outside
these folders and only reads them.

## Setup

Python 3.11 or newer (3.12 recommended).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Check that everything is intact

Neither command calls a model provider or costs anything.

```bash
python tools/verify_freeze.py
python -m pytest tests demo extension_apertus extension_posthoc -q
```

Expected: `OK: every sealed file is byte-identical to the frozen state.` and `47 passed`
(13 research, 21 demonstrator, 5 Apertus and 8 robustness tests).

## Run the demonstrator

```bash
python demo/app.py
```

The browser opens at http://127.0.0.1:8765. Stored cases need no API key. Live sentence pairs need keys
in `.env` (copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`). See
`demo/README.md` for details and limits.

## Reproduce the results

| Step | What it does | Needs |
|---|---|---|
| `notebooks/00` | downloads ChaosNLI and the disagreement taxonomy, builds the splits | internet |
| `notebooks/01` | collects the model responses | API keys, about USD 50 in total |
| `notebooks/02` | fits the assessors, tunes the policy on development data, freezes the method | stored responses |
| `notebooks/03` | final evaluation on the MNLI test and the SNLI sample | frozen method |
| `notebooks/04` | inspects single stored cases | fitted assessors |
| `notebooks/05`, `extension_apertus/evaluate_apertus.py` | third model with the frozen assessors | stored responses |
| `extension_posthoc/evaluate_posthoc.py` | robustness checks | stored responses |

All responses and fitted assessors are stored in this repository, so steps 02 to 05 and both extensions run
without new API calls. Refitting is blocked once test access has begun; the stored exports in
`outputs/experiments/` are the authoritative results. The notebooks are explained in German; code, documentation
and report are in English.

## Data and licence notes

ChaosNLI (Nie, Zhou and Bansal, 2020) is distributed by its authors under CC BY-NC 4.0. The disagreement
categories come from Jiang and de Marneffe (2022). `data/processed/` contains derived splits of these data for
non-commercial research use. Sources, mirror revision and checksums are in `docs/DATA_PROVENANCE.md` and
`config/data_sources.json`. Raw downloads (`data/raw/`) are not part of the repository.

## Scope

The claims are limited to English NLI sentence pairs and to operational proxies: a majority deviation on a
clear case stands in for a model error, and variation across prompts is not an identification of epistemic
uncertainty. The demonstrator is a functional prototype and not a validated classifier for arbitrary text.

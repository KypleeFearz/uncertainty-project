# Demo: Uncertain or Genuinely Debatable?

A local web app that shows the frozen system at work. Enter an English sentence and a statement; the system
decides whether to **answer**, **hedge** (flag its own uncertainty), show **alternatives** (several reasonable
interpretations to investigate) or **clarify** (ask back), and explains why.


## Start

From the project root (the folder with `config/` and `src/`):

```
python demo/app.py
```

The browser opens at <http://127.0.0.1:8765>. Stop with `Ctrl+C`. Options: `--port 8800`, `--no-browser`.

No new packages are needed beyond the project requirements.

## What it does

| Input | API calls | Source of the decision |
|---|---|---|
| Example case or random MNLI/SNLI case | none | stored research responses (identical to the evaluated results) |
| Your own sentence pair | 5 calls to the selected model | same system prompt, five prompts and model settings as the experiment |
| "Write the reply with the language model" | +1 call | phrases the frozen decision; wording is **not evaluated** |

Pipeline per input: five prompts → signals → frozen assessors and decomposed policy (protocol `3942fc49…`) →
action → wording (fixed template, or optional LLM wording that may not change the decision).

Every answer also shows **how simpler rules would decide** (RQ2): one learned risk score with two thresholds,
and model confidence only. Both are the frozen comparison policies from the experiment. For stored cases each
rule is marked ✓/✗ against the declared coarse reference (answer / hedge / debatable). The reference depends
on human votes AND the particular model's answer; people did not annotate which interface reaction they prefer.
Fine clarify/alternatives agreement is displayed separately where that reference exists.

**Compare GPT and Claude** (RQ3) runs the same input through both models and their own frozen assessors and
shows the two decisions side by side (stored cases: free; your own pair: 10 live calls).

**Apertus (post-hoc, added 2026-11-07).** For stored cases, the comparison shows a third, dashed column with the
stored answers of `swiss-ai/apertus-v1.5-70b` and the decisions of the frozen GPT and Claude assessors applied to
them unchanged. No assessor was trained on Apertus and there are no live Apertus calls. Pairs where Apertus gave no
valid answer for all five prompts show a note instead (they are excluded from the evaluation, see `extension_apertus/`).

The "Why this decision?" panel shows the decision path, the three signals against their thresholds, predicted
versus actual human votes (stored cases only), the comparison with the human reference reaction and all five
model outputs.

## Design

The visual language is taken from the thesis prototype (*Perspectival AI Interaction*, 2026): a single answer
mirrors condition A; the "alternatives" reaction mirrors condition B (overview, perspective cards, key
disagreement). The thesis studied *how* to present perspectives; this project decides *when*.

## Guarantees and limits

- The original freeze is verified at startup, including its identifier. The demo never writes to `outputs/runs` or any sealed file. Live responses are cached in `demo/cache/`
  (ignored by git). `python tools/verify_freeze.py` confirms the seal after using the demo.
- `demo/test_demo.py` checks that every stored decision of the adapter equals the frozen batch evaluation
  (627 pairs × 2 models), that the shown comparison rules equal the frozen comparison policies, that the
  model comparison works, that live calls only use the demo cache, that LLM wording keeps the decision,
  and the HTTP endpoints. Run: `pytest -q demo`.
- English sentence pairs only. The primary learned disagreement signal and response policy deteriorated on
  this SNLI sample. Other components retained some discrimination; this is not a universal failure of learned assessors.
- Disagreement is normalised entropy. Error risk and cues are estimates, not calibrated guarantees. Predicted
  human label proportions are distributions; they do not establish the existence or legitimacy of particular readings.
- Templates report classes and offer a generic clarification. Optional generated interpretations are proposals,
  not evaluated explanations. Structural validation does not guarantee semantic faithfulness to the fixed label.
  The frozen classification is shown independently. Failed wording falls back to the template.
- Selecting a stored case or switching its model never requests generated wording automatically. Click Check
  with the optional wording box selected to request it. Compare always uses templates.

## Curated cases

| Case | Model | Shows |
|---|---|---|
| 65353n | GPT | clear case → answer |
| 106411n | GPT | model disagrees with a clear majority → hedge |
| 66260c | GPT | humans split → alternatives |
| 13531e | Claude | lexical ambiguity ("felt like a rat"), humans split 53/26/21 → clarify |
| 95883e | GPT | failure: answers although humans split (Claude shows alternatives) |
| snli_3552435734.jpg#0r1e | GPT | limit: other text source, humans split, system answers |

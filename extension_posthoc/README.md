# Post-hoc robustness checks (10 November 2026)

Added after the freeze (protocol `3942fc496262a9b3`). The script only reads the sealed files and writes to
`extension_posthoc/results/`. No tree, threshold or policy was refitted. All results are exploratory.

```
python extension_posthoc/evaluate_posthoc.py           # about two minutes, 500 bootstrap repetitions
python -m pytest -q extension_posthoc/test_posthoc.py  # 8 checks, includes the seal
```

## A. Sensitivity to the definition of a clear case

The frozen definition (three-class entropy at or below 0.624) admits two-way splits such as 59/41/0. The check
repeats the clear-case analyses on the stricter subset with a human majority of at least 80 percent.

| File | Content |
|---|---|
| `clear_summary.csv` | How many frozen clear cases and deviations have a majority below 0.80, human support for deviating answers |
| `clear_composition.csv` | Frozen clear cases and deviations by majority share |
| `clear_sensitivity.csv` | Clear-case AUROC of the risk tree, the label-and-confidence tree, raw confidence and the disagreement signal under both definitions, with paired differences and 95 percent premise-bootstrap intervals |
| `clear_error_profile.csv` | Which answer label carries the deviations under both definitions |
| `clear_sensitivity_apertus.csv` | The frozen GPT and Claude risk assessors applied to Apertus under both definitions |

## B. Source separation (a first look at out-of-distribution detection)

`source_separation.csv`: AUROC of each frozen signal for telling SNLI inputs from MNLI test inputs, for all
cases and within the clear and debatable strata. Values below 0.5 mean the signal is lower on SNLI.

The rows that use the frozen definition reproduce the sealed exports of notebook 03 exactly; the tests check this.

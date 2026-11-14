# Relationship to the literature

This implementation is **inspired by the uncertainty/action distinction**, not a reproduction
of CREDENCE or a claim of an equivalent mathematical decomposition.

| Paper | Relevant idea | Implementation here |
|---|---|---|
| Mukherjee et al., ACL 2026, §3–4 | Epistemic uncertainty from disagreement among diverse heads (credal intervals); aleatoric uncertainty from a head trained on annotator disagreement; signals drive decisions | Prompt variants (and two providers) as heads; per-label [min, max] intervals and mutual information as epistemic signals; a tree trained on human vote distributions as the aleatoric estimate; an epistemic tree trained on clear cases only; four actions |
| Nie, Zhou and Bansal, EMNLP 2020 | Evaluate against distributions of human NLI judgments | Predict all three label frequencies and measure distribution fit |
| Jiang and de Marneffe, TACL 2022 | Existing categories of why NLI judgments differ | Check predicted lexical, reference and probabilistic-inference cues against the annotated subset |
| Koh et al., ICML 2020 | Predictions through an interpretable concept layer | Motivation for inspectable cues; the current assessor also uses numeric scores and is not a strict CBM |

CREDENCE trains concept predictors and propagates intervals. This project instead queries two
API models using five instructions each, treats the answers as heads, and fits decision trees. Those
ranges are prompt-sensitivity summaries, not calibrated credal bounds. Human vote entropy is
an empirical disagreement target, not proof that disagreement is legitimate or irreducible.

The contribution to test is whether this simpler construction provides useful, explainable
predictions and actions in NLI. If supervision requires a CREDENCE implementation, retain these
notebooks as the data and evaluation baseline and add a model with trained concept heads.
Confirm that expectation before promising a complete uncertainty decomposition.

## Sources

- https://aclanthology.org/2026.acl-long.1991/
- https://github.com/Tankiit/Credal_Sets/tree/ensemble-credal-cbm
- https://aclanthology.org/2020.emnlp-main.734/
- https://github.com/easonnie/ChaosNLI
- https://aclanthology.org/2022.tacl-1.78/
- https://github.com/njjiang/NLI_disagreement_taxonomy
- https://proceedings.mlr.press/v119/koh20a.html

The application code in this package is newly written. Paper repositories are references,
not downloaded executable dependencies.

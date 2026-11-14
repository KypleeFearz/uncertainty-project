# Data provenance

No benchmark files are redistributed in this package. Notebook 00 downloads three small files
from the pinned public URLs in `config/data_sources.json` and checks recorded SHA-256 digests.
Automatic schema checks ran against these files during preparation.

ChaosNLI-MNLI: 1,599 sentence pairs, 100 judgments each. The original authors' repository is
https://github.com/easonnie/ChaosNLI . Its documented Dropbox link did not return a usable
archive during the preliminary audit. The loader therefore uses the documented community
preservation mirror at https://huggingface.co/datasets/earino/chaosnli . This choice is explicit.

Mirror revision: `bf8d66a1d3434ac8b90ffc392b1b5e44bdf7e439`.
Its provenance record is linked in the source configuration. The preserved file is consistent
with the documented schema and size, and its hash matches the inspected copy. This does not
prove identity to an author-issued archive or author endorsement. Cite the original researchers
and record the mirror route in the report. The authors' distribution is under CC BY-NC 4.0;
consult its licence and the mirror's third-party notices for downstream use.

The disagreement-taxonomy files come from the authors' repository:
https://github.com/njjiang/NLI_disagreement_taxonomy
at commit `60268f44c2326d11a4842cdaa08aef395faead0c`.
Round 1 contains 400 records with `a1`; round 2 has 110 records with `agg` available.
450 IDs match ChaosNLI-MNLI. The remaining 60 taxonomy records are outside this main dataset.
Matching IDs are also checked against both text fields.
Seventeen text fields in the taxonomy omit one leading ASCII quotation mark present in
ChaosNLI. The join permits exactly that inspected formatting difference and records the
affected fields in the audit. It does not permit general fuzzy matching or change the input text.

Raw files are preserved unchanged. Derived files only normalise text for grouping, retain the
original displayed text, map fixed label order E/N/C, attach category annotations, and separate
training/development/test. The 100 individual ChaosNLI annotators cannot be tracked across
examples from these aggregate vote fields.

Sources:
- https://aclanthology.org/2020.emnlp-main.734/
- https://aclanthology.org/2022.tacl-1.78/
- https://huggingface.co/datasets/earino/chaosnli/blob/bf8d66a1d3434ac8b90ffc392b1b5e44bdf7e439/PROVENANCE.md

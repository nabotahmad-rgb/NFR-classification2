# Source datasets

The corpus in `corpus/nfr_corpus.csv` is derived from two publicly available datasets,
both included in this directory so the pipeline can be run end to end.

---

## `dcai24_src_dataset.xlsx`

A consolidated aggregation of three established requirements-engineering corpora:

- **PROMISE NFR** — the standard benchmark for NFR classification, originally distributed
  through the PROMISE Software Engineering Repository and widely redistributed since,
  including in the replication package for NoRBERT (Hey et al., RE 2020).
- **PURE** — a corpus of public requirements documents.
- **SecReq** — security requirements, contributing the `NONSE` (non-security) coding
  visible in the unmapped-label counts.

3,482 rows. Column `Requirement` holds the text; column `Specific_Type` holds the NFR
sub-category. Note that this file also carries a binary functional/non-functional flag;
`build_corpus.py` selects the sub-category column by evidence rather than by name,
because matching on name picks the binary flag and discards every NFR (see README).

**Contribution to the corpus:** 2,024 NFR records before deduplication.

---

## `Cornelius_FR_NFR_Curated_Mendeley_Dataset_2025.xlsx`

A curated functional/non-functional requirements collection published through Mendeley
Data. The workbook contains a `Complete_Dataset` sheet plus per-category sheets holding
the same records again; `build_corpus.py` reads the complete sheet only, since reading
all sheets would triple-count some categories.

1,145 rows in `Complete_Dataset`. Column `Description` holds the text; column
`NFR_Category` holds the label.

**Contribution to the corpus:** 406 NFR records before deduplication.

---

## Cross-source independence

We checked overlap between the two sources on a punctuation- and case-insensitive
fingerprint of the requirement text. **No record in the Mendeley collection matches any
record in the PROMISE/PURE/SecReq aggregation.** The two sources are genuinely
independent, and results are not inflated by the same requirements appearing twice.

One requirement text carries conflicting labels across the two sources. We retained the
first occurrence and report this in the manuscript as a construct-validity consideration
for aggregated corpora.

---

## Licensing

The MIT Licence in `LICENSE` covers this project's own contributions only. The two
spreadsheets in this directory are the work of their original authors and remain under
their own terms.

**Their licence terms have not been confirmed.** See `PUBLIC_RELEASE_NOTICE.md` in this
directory for what is and is not known, how to confirm it, and what to do if
redistribution turns out not to be permitted. Do not publish this directory until that
notice has been resolved and deleted.

`SOURCE_HASHES.txt` records the SHA-256 of both files as used to build the released
corpus, so a copy obtained from the original record can be checked against what this
pipeline actually consumed.

Our own contributions — the aggregation, deduplication, scope decisions, fold
assignments, code, and results — are released under the MIT Licence (see `LICENSE`).

---

## Rebuilding from these files

```bash
python code/build_corpus.py --base sources --inspect     # show detected columns
./reproduce.sh corpus                                     # rebuild and verify
```

The rebuild is deterministic. `reproduce.sh corpus` rebuilds into a temporary directory
and checks that the result is byte-identical to the released `corpus/nfr_corpus.csv` and
`corpus/folds.csv`, so you can confirm that the released corpus really is what this code
produces from these inputs.

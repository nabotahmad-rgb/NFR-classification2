# Obtaining the raw source datasets

The two upstream spreadsheets are not redistributed here, because their licence terms
could not be confirmed as permitting it. They are not needed to verify any result in
the paper: `corpus/nfr_corpus.csv` is released, as are all predictions and every
analysis output, and `code/verify_paper.py` recomputes the reported claims from those.

They are needed only for `./reproduce.sh corpus`, which rebuilds the corpus from the
raw inputs. To run that step, obtain both files and place them in this directory under
their original names, then verify them against the hashes in `SOURCE_HASHES.txt`.

- `dcai24_src_dataset.xlsx` — an aggregation of PROMISE NFR, PURE and SecReq. See
  `SOURCES.md` for what each contributes.
- `Cornelius_FR_NFR_Curated_Mendeley_Dataset_2025.xlsx` — a curated functional and
  non-functional requirements collection published through Mendeley Data.

`SOURCES.md` records the column layout and row counts each file must have, so a
substitute copy can be checked before use.

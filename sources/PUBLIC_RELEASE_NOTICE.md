# Read this before making the repository public

`sources/` holds two third-party spreadsheets that this project redistributes so the
pipeline runs end to end. **Their licence terms have not been confirmed.** Until they
are, the safe course is not to publish them.

| File | Original record | Licence | Confirmed |
|---|---|---|---|
| `dcai24_src_dataset.xlsx` | *(add URL)* | *(add licence)* | ☐ |
| `Cornelius_FR_NFR_Curated_Mendeley_Dataset_2025.xlsx` | *(add URL)* | *(add licence)* | ☐ |

## What we could and could not determine

The embedded document properties of the Mendeley file name its creator as "Engr. Cornel"
and its creation date as 2025-05-31. We could not identify the specific Mendeley Data
record from the public search interface, so the DOI and licence are still unknown.
Mendeley Data records most often carry CC BY 4.0, but that is a tendency, not this
record's terms, and it must not be written into the table as though it had been checked.

The `dcai24_src_dataset.xlsx` aggregation combines PROMISE NFR, PURE and SecReq. Those
have separate provenance and no single licence; PROMISE-derived data has circulated
through many replication packages for years, which is a redistribution practice rather
than a grant.

## How to confirm

1. Find the Mendeley Data record (search Mendeley Data for the dataset title or the
   creator name). Record its DOI and the licence shown on the record page.
2. Identify where `dcai24_src_dataset.xlsx` was obtained and record that URL, plus the
   licence each constituent corpus states.
3. Fill the table above, tick both boxes, and delete this section.

## If a licence prohibits redistribution

Run `./prepare_public_release.sh` from the repository root. It removes the raw
spreadsheets, writes `sources/HOW_TO_OBTAIN.md` in their place, and leaves everything
else intact. `corpus/nfr_corpus.csv`, every prediction file and every number in the
paper remain verifiable without the raw sources; only `reproduce.sh corpus`, which
rebuilds the corpus from scratch, needs them.

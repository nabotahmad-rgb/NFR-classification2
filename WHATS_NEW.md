# Changes since the 2026-07-26 snapshot

A package dated 2026-07-26 was published earlier. It is a self-consistent record of an
earlier, narrower study and its own verifier passes, but it predates every robustness
analysis and corresponds to a manuscript two titles old. Anyone holding that snapshot
should replace it with this one.

## What that snapshot was missing

| | 2026-07-26 | This release |
|---|---|---|
| Manuscript title | *Does Multi-Paradigm Composition Improve Non-Functional Requirements Classification?* | *Architectural Complexity in NFR Classification: Controlled Ablations, Robustness Analyses, and a Six-Label Component Follow-Up* |
| Tables in the manuscript | 5 | 11 |
| `robustness/` | absent | 9 bundles |
| `splits/` | absent | 19 files |
| Files | 255 | 628 |
| Verified claims | 53 | 118, plus 199 Stage09 checks |

The nine robustness bundles added here are the source holdout, equal-trial-budget
tuning, near-duplicate-grouped cross-validation, the contrastive positive-pair audit,
the pair-aware contrastive rerun, CPU threshold and weighting sensitivities, the
six-label endpoint comparison and the six-label component ablation. The last of these
produces the result that reverses the primary conclusion, so the earlier snapshot
contains no evidence for the paper's most consequential finding.

## What did not change

`corpus/nfr_corpus.csv` and `corpus/folds.csv` are byte-identical across both releases.
The corpus underneath every result has been stable since July.

## Verification state of this release

- `code/verify_paper.py` — 118 of 118 claims, exit 0
- `robustness/09_component_ablation/verify_stage09.py` — 199 checks, maximum deviation
  from the frozen summary 9.89e-17
- `CHECKSUMS.txt` — SHA-256 for every released data file
- Both verifiers pass with the raw `sources/` spreadsheets absent, so licence-driven
  removal of those files costs no verifiability.

## Before publishing this release

Read `sources/PUBLIC_RELEASE_NOTICE.md`. If the upstream licences are not confirmed, run
`./prepare_public_release.sh`, regenerate `CHECKSUMS.txt`, then commit.

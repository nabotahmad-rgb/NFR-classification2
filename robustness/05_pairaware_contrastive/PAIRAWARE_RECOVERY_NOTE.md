# Pair-aware predictions: why an audit reported them missing, and why they are not

A 2026-10-10 Google Drive recovery audit recorded a "material contradiction" for this
experiment: it found 30 of 50 expected runs (3 of 5 repeats) and computed
delta = +0.007222 with 95% CI [-0.005230, +0.019659], against the manuscript's
-0.0003 with 95% CI [-0.0090, +0.0087]. It listed this as a release blocker and
recommended reconciling the paper or the experiment before submission.

**No reconciliation is needed. The manuscript is correct.**

All 50 fold-level prediction files exist and always did. They were inside
`pairaware_contrastive_cv_FINAL.zip`, under `pairaware_contrastive_cv/predictions/`,
which the Drive search did not open; the loose copies on Drive were written to an
ephemeral `/kaggle/working/` path, so only a subset survived there. The 50 files are
now extracted to `predictions/folds/` and pooled per repeat into `predictions/pooled/`
so that no future audit depends on opening an archive.

## Independent recomputation from the raw predictions, all 5 repeats

| Quantity | Recomputed | Manuscript |
|---|---|---|
| Full macro F1 | 0.886923 +/- 0.009499 | 0.8869 +/- 0.0095 |
| No-contrastive macro F1 | 0.887175 +/- 0.010115 | 0.8872 +/- 0.0101 |
| Delta | -0.000252 | -0.00025 |
| 95% CI (4,000 record-level resamples) | [-0.0094, +0.0089] | [-0.0090, +0.0087] |
| Whole 95% CI inside +/-0.01 | yes | yes |

Every repeat carries 1,667 records with item IDs and gold labels aligned across the two
arms. The interval differs from the manuscript's only by bootstrap resampling noise and
reaches the same equivalence decision, so the paper's claim that this is the one analysis
establishing practical equivalence stands.

## Why three repeats gave a materially different answer

The per-repeat Full-minus-no-contrastive differences are:

| Repeat | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Delta | +0.001105 | +0.014329 | +0.006231 | **-0.010445** | **-0.012479** |

Both negative repeats are the two the audit lost. Truncating to repeats 0-2 therefore
discards every negative difference and biases the estimate strongly positive. The mean
over repeats 0, 1 and 2 is +0.007222, which reproduces the audit's figure to six decimal
places and identifies it unambiguously as a partial-data artifact rather than a
discrepancy in the experiment. No other three-repeat subset gives that value.

The episode is a reason to ship predictions as loose files rather than inside archives,
which this bundle now does.

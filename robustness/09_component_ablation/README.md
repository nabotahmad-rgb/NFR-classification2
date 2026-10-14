# MAGNET-NFR Stage09 recomputation inputs

This bundle contains the exact **20 pooled ablation CSVs** requested for Stage09 (4 ablations x 5 repeats), each with columns `item_id,gold,pred` and 1,719 rows. It also contains the 5 frozen Full pooled reference CSVs required for the paired Full-minus-ablation contrasts.

## Bootstrap draws
The original Stage09 archive did **not** preserve bootstrap index arrays as separate files. The original analysis script fixes the full draw process: `numpy.random.default_rng(20260916)`, 4,000 record-level draws of 1,719 positions, with variants processed in this order: `no_gcn`, `no_xl`, `no_contrastive`, `no_meta`.

`bootstrap_draw_indices_regenerated.npz` contains those regenerated index matrices (shape 4000 x 1719, stored as uint16 after generation). They were generated using the original call `RNG.integers(0,n,n)` and then cast only for compact storage. Recomputing the stored 90%, 95%, Bonferroni 98.75% intervals and raw bootstrap p-values from these arrays gives maximum absolute discrepancy 9.89e-17.

Thus the arrays are **regenerated from the frozen RNG specification**, not original saved draw files.

## Contents
- `pooled_ablation/`: 20 requested CSVs.
- `pooled_full_reference/`: 5 frozen Full pooled CSVs.
- `bootstrap_draw_indices_regenerated.npz`: exact regenerated draw indices by variant.
- `analysis/analyze_component_ablation.py`: original analysis script.
- `analysis/component_ablation_summary.csv`: frozen Stage09 summary.
- `analysis/component_ablation_details.json`: seed/N/estimand metadata.
- `analysis/bootstrap_regeneration_verification.csv`: numerical reproduction check.

---

## Independent verification performed on integration

Three checks beyond the bundled script, run before this bundle was added to the
replication package.

**1. Point estimates recomputed from the raw CSVs.** All five values match the
manuscript exactly: Full macro F1 0.8286, and ablation means 0.8061 (GCN),
0.8078 (contrastive), 0.8141 (meta-learning), 0.8191 (relative-position), giving
contrasts of +0.0225, +0.0207, +0.0144 and +0.0094.

**2. Intervals re-derived with an independent bootstrap seed.** The bundled draws
are regenerated from `default_rng(20260916)`, so matching them only proves
determinism. A fresh bootstrap under `default_rng(987654321)` reproduces every
interval to within 0.001 and preserves the sign of every bound, so no conclusion
in Table 11 depends on the choice of draws. Holm-adjusted p-values shift as
expected under new draws (GCN 0.0120 against 0.0165, meta-learning 0.2799 against
0.2269) while the resolve / does-not-resolve decision is identical for all four
components.

**3. Cross-experiment consistency.** The five frozen Full prediction files here are
byte-identical to those in `08_six_class_sensitivity/predictions/pooled/`, item
IDs, gold labels and predictions alike. The component follow-up and the six-label
endpoint comparison are therefore provably computed on the same Full predictions,
which the manuscript asserts but the bundled script does not itself check.

## Correction applied to the manuscript

The sum of the four marginal removal effects was stated as 0.0670, which is the sum
of the four values after rounding to four decimal places. The sum of the unrounded
contrasts is 0.0671449616, which rounds to **0.0671**. The manuscript now reports
0.0671. The ratio against the 0.0248 joint effect is 2.71, unchanged at the stated
precision.

## Change to the bundled script

`verify_stage09.py` is the author-supplied `verify_paper.py` with one robustness fix:
a missing frozen summary or draw archive previously raised `FileNotFoundError` after
its own existence check had already reported the problem. It now returns exit code 2
with a readable message, so a reviewer with an incomplete download sees a diagnosis
rather than a traceback. No statistical logic was altered.

The script resolves `--root` from its own location, so it must remain inside this
directory, or be invoked with an explicit `--root`.

## Bootstrap draw archive omitted by design

`bootstrap_draw_indices_regenerated.npz` (4 x 4000 x 1719 int32, ~42 MB) is not
shipped. Every index in it is determined by `numpy.random.default_rng(20260916)`
drawn in the order `no_gcn, no_xl, no_contrastive, no_meta`, which we confirmed
reproduces the original archive exactly, array for array. Distributing 42 MB of
regenerable integers would make the package harder to archive for no scientific gain.

`verify_stage09.py` regenerates the draws in memory when the archive is absent and
reports that it has done so. Verification without the file reproduces the frozen
summary to a maximum absolute difference of 9.89e-17, identical to the run with it.
To recreate the file on disk instead, run `python regenerate_draws.py`.

The two SHA-256 manifests have had their entries for the archive removed accordingly, and
their recorded hash for this README updated to match the text you are reading. Every other
listed file is byte-unchanged from the author's bundle and still verifies against the
original hash.

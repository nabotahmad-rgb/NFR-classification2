# MAGNET-NFR Stage09 independent recomputation verification

## Verdict
**PASS.** The four Stage09 Full-minus-ablation contrasts independently reproduce from the 20 pooled ablation CSVs, 5 frozen Full pooled CSVs, and the deterministically regenerated bootstrap draw indices.

Maximum absolute numerical discrepancy versus the frozen Stage09 summary: **9.89e-17**.

The bootstrap draw arrays in this package are **regenerated**, not original persisted draw files. They were reconstructed from the frozen analysis specification: `numpy.random.default_rng(20260916)`, 4,000 record-level draws per variant, 1,719 positions per draw, and the original variant processing order.

## Recomputed locked results

| Variant | Full mean | Ablation mean | Delta | 95% CI | Bonf. 98.75% CI | Raw p | Holm p | Equivalent ±0.01 |
|---|---:|---:|---:|---|---|---:|---:|---|
| no_gcn | 0.828580 | 0.806076 | +0.022504 | [0.007011, 0.040443] | [0.002270, 0.047397] | 0.005499 | 0.016496 | no |
| no_xl | 0.828580 | 0.819130 | +0.009450 | [-0.008163, 0.027989] | [-0.014340, 0.033394] | 0.313922 | 0.313922 | no |
| no_contrastive | 0.828580 | 0.807838 | +0.020742 | [0.009000, 0.033165] | [0.005408, 0.037099] | 0.001000 | 0.003999 | no |
| no_meta | 0.828580 | 0.814131 | +0.014449 | [-0.003868, 0.034225] | [-0.009712, 0.040650] | 0.113472 | 0.226943 | no |

## What `verify_paper.py` checks

1. SHA-256 integrity of every file listed in the original frozen recomputation manifest.
2. Exactly 20 pooled ablation CSVs and 5 Full reference CSVs.
3. Exact `item_id,gold,pred` schema, 1,719 unique items per file, and identical item/gold alignment across all repeats and variants.
4. Bootstrap archive keys, shape (4000×1719 per variant), integer type, and valid index bounds.
5. Full and ablation macro-F1 means/SDs and all four point contrasts.
6. 90%, 95%, and Bonferroni 98.75% bootstrap intervals.
7. Finite-sample-corrected two-sided bootstrap p-values and four-comparison Holm adjustment.
8. Frozen ±0.01 equivalence decisions.

## Reviewer command

```bash
python verify_paper.py
```

A successful run exits with code 0 and prints `PASS: Stage09 results independently reproduce...`.

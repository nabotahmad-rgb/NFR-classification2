# Record-level prediction coverage

Every experiment reported in the manuscript has its record-level predictions in this
package as loose CSV files. Nothing is held only inside an archive, and nothing is
missing. The table below is what an auditor should check against.

| Experiment | Location | Files | Verified against the manuscript |
|---|---|---|---|
| Primary four-label ablation | `predictions/pooled/`, `predictions/folds/` | 40 pooled, 175 fold | all 7 configs match at 4 d.p. |
| Source holdout (LOSO) | `robustness/01_source_aware/` | 18 | holdout scores match Table 5 |
| Equal-trial-budget tuning | `robustness/02_equal_budget_tuning/predictions/final_by_fold/` | 10 | Full 0.9628/0.9010, Encoder 0.9634/0.8923 |
| Near-duplicate-grouped CV | `robustness/03_near_duplicate_grouped_cv/predictions/` | 15 pooled, 27 fold | Full 0.8754±0.0194, Encoder 0.8816±0.0254, TF-IDF 0.8409±0.0077, accuracies also exact |
| Contrastive positive-pair audit | `robustness/04_contrastive_positive_pair_audit/` | n/a | analytic over fold label vectors; no predictions involved |
| Pair-aware contrastive | `robustness/05_pairaware_contrastive/predictions/` | 50 fold, 10 pooled | Full 0.886923±0.009499, no-contrastive 0.887175±0.010115, delta -0.000252 |
| CPU sensitivities | `robustness/06_reviewer2_cpu_sensitivities/` | n/a | classical pipeline recomputed from the corpus |
| Six-label endpoint | `robustness/08_six_class_sensitivity/predictions/pooled/` | 15 | Full 0.9499/0.8286, Encoder 0.9425/0.8038, TF-IDF 0.9393/0.7681 |
| Six-label component ablation | `robustness/09_component_ablation/` | 20 ablation, 5 Full | 199/199 verifier checks; max difference 9.89e-17 |

## Why this file exists

A 2026-10-10 Drive recovery audit reported the pair-aware experiment as 30 of 50 runs
and flagged a contradiction with the manuscript. The predictions were never missing;
they were inside `pairaware_contrastive_cv_FINAL.zip`, which the search did not open,
and the partial subset it did find excluded both repeats with negative differences
(see `05_pairaware_contrastive/PAIRAWARE_RECOVERY_NOTE.md`).

Two further bundles had the same exposure: the grouped-CV and equal-budget-tuning
predictions were also archive-only. Both have now been extracted as loose files and
re-verified against the manuscript, with every value reproducing exactly. The original
archives are retained alongside them for provenance.

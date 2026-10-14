# Robustness extensions

Five bundles, one per analysis, each independently inspectable.

    01_source_aware/        leave-one-source-out, neural configurations + splits
    02_equal_budget_tuning/ nested equal-budget search, 25 trials per architecture
    03_near_duplicate_grouped_cv/  grouped folds at cosine 0.90
    04_contrastive_positive_pair_audit/  per-class positive-pair availability
    05_pairaware_contrastive/      pair-aware rerun, Full vs no-contrastive
    06_reviewer2_cpu_sensitivities/  CPU threshold and weighting sensitivities

## Corrections applied to the delivered bundles

`02_equal_budget_tuning/equal_budget_outer_results.csv` originally carried nine rows;
the fold-4 `roberta_only` row was absent, which would suggest the encoder arm covered
only four of five outer folds. The run itself completed: the prediction file is present
in `tuning_equal_budget_rep0-*.zip`, and pooling all five encoder folds reproduces
macro F1 0.8923 on exactly 1,667 records. The row has been restored (test macro F1
0.8702) so the file matches the experiment.

## Verification performed

Every number reported in the manuscript from these bundles was recomputed from the
released outputs. The source-holdout means, the equal-budget contrast (recomputed from
raw fold predictions rather than the summary file), the grouped-CV means and bootstrap,
and the pair-aware contrast all reproduce.

One manuscript claim was corrected as a result. The 27-record CPN holdout returns macro
F1 = 0.180180 for all three neural configurations, identical to six decimal places,
which is exactly the value obtained by predicting Security for all 27 records. This is
majority-class collapse, not metric instability, and the manuscript now says so. TF-IDF
scored 0.407 on the same holdout and did not collapse.

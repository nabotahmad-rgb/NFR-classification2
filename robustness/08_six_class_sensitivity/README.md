# MAGNET-NFR Six-Class Sensitivity

## What this closes

The primary manuscript evaluates four labels because Maintainability (n=34) and Portability (n=18) were below the original per-class precision threshold. This experiment retains all six mapped labels and asks whether the central Full-vs-Encoder architectural conclusion changes.

## Frozen design

- Dataset: 1,719 globally deduplicated requirements
- Labels: Performance, Security, Usability, Reliability, Maintainability, Portability
- Repeated CV: 5 repeats × 5 stratified folds
- Neural comparison: Full MAGNET-NFR vs Encoder-only
- Neural fits: 50 total
- Backbone: distilroberta-base
- Training settings: identical to the manuscript primary operating point
- CPU reference: TF-IDF + class-weighted LinearSVC
- Main metric: macro F1
- Main contrast: Full − Encoder-only
- Bootstrap: 4,000 paired record resamples, interpreted conditionally on the fixed corpus/predictions

## Completed already

`preflight_six_class.py` passes. The six-class corpus and folds are frozen. The TF-IDF baseline has already been run and is stored under `predictions/pooled/`.

## Kaggle execution

Recommended runtime: **GPU T4 x2** with Internet enabled.

1. Upload `MAGNET_NFR_SixClass_Sensitivity_Experiment.zip` as a Kaggle dataset/input.
2. Upload/open `MAGNET_NFR_SixClass_Kaggle_T4x2.ipynb`.
3. Select **GPU T4 x2**.
4. Turn **Internet On** so `distilroberta-base` can be cached.
5. Run all cells.
6. Download `MAGNET_NFR_SIXCLASS_RESULTS.zip` from `/kaggle/working/`.
7. Return that results ZIP for manuscript integration and the final rejection test.

## Manual command after extraction

```bash
python robustness/08_six_class_sensitivity/preflight_six_class.py
python robustness/08_six_class_sensitivity/run_six_class_neural.py --gpus 0,1
python robustness/08_six_class_sensitivity/analyze_six_class.py
```

The neural runner is resume-safe: completed non-empty fold prediction files are skipped.

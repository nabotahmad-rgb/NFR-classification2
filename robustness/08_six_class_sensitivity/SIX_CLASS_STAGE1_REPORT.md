# Six-Class Sensitivity — Stage 1 Report

## Purpose

This robustness extension tests the main remaining task-definition threat in the manuscript: whether the four-class architectural conclusion changes when Maintainability and Portability are retained rather than removed for low support.

## Frozen six-class corpus

The corpus was rebuilt from the two original source spreadsheets using the released label mapping and deduplication logic, with the minimum-class threshold changed from 40 to 1. The resulting dataset contains 1,719 unique requirements across all six mapped labels:

| Label | n |
|---|---:|
| Security | 1,343 |
| Usability | 150 |
| Performance | 106 |
| Reliability | 68 |
| Maintainability | 34 |
| Portability | 18 |

The majority-class floor is 78.13%. Five repeated stratified five-fold assignments were generated. Every one of the 25 held-out folds contains all six labels, although Portability remains very small in each fold.

## CPU baseline completed

TF-IDF (word 1–2 grams, min_df=2) + class-weighted LinearSVC was evaluated on the frozen six-class folds.

- Accuracy: **0.9393 ± 0.0031**
- Macro F1: **0.7681 ± 0.0231**
- Margin above majority floor: **15.80 percentage points**

Mean class-wise F1 across the five repeats:

| Label | Mean F1 | SD |
|---|---:|---:|
| Security | 0.9732 | 0.0015 |
| Usability | 0.8504 | 0.0084 |
| Performance | 0.8600 | 0.0112 |
| Reliability | 0.7641 | 0.0207 |
| Maintainability | 0.5732 | 0.0526 |
| Portability | 0.5876 | 0.0621 |

Relative to the four-class TF-IDF result reported in the manuscript (macro F1 0.8730; accuracy 95.48%), adding the two low-support labels lowers macro F1 by approximately **0.1049** and accuracy by **1.55 percentage points**. This confirms that the six-class task is materially harder and that the sensitivity is scientifically necessary rather than cosmetic.

## Decisive neural comparison to run

The frozen neural sensitivity compares:

1. **Full MAGNET-NFR**
2. **Encoder-only (roberta_only)**

Each configuration is retrained on the six-class corpus under the same primary operating point for 5 repeats × 5 folds, giving **50 neural fits** in total. The primary estimand is Full minus Encoder-only macro F1 averaged across repeated out-of-fold prediction sets. A 4,000-resample paired record bootstrap is used only as conditional uncertainty on this fixed corpus; it is not described as population-iid inference.

This comparison directly addresses the strongest remaining reviewer question: whether excluding Maintainability and Portability created the apparent absence of benefit from architectural composition.

## Interpretation rule

- If Full materially exceeds Encoder-only on six classes and the interval excludes zero, the manuscript's architectural conclusion must be narrowed and the six-class result becomes a substantive exception.
- If the point difference remains small and the interval overlaps zero, the main negative architectural conclusion survives the broader label space, subject to wide uncertainty for Maintainability and Portability.
- Equivalence within ±0.01 should be claimed only if the 90% interval lies fully inside that margin; otherwise report the estimate and uncertainty without calling the models equivalent.

No neural result is inserted into the manuscript until the 50 fits are complete.

# Does Multi-Paradigm Composition Improve NFR Classification?

Replication package for a controlled ablation study of composed neural architectures for
non-functional requirements (NFR) classification.

**The headline result is negative, and quantified as such.** Removing any single paradigm
from a five-paradigm composition changes macro *F*₁ by at most 0.005, every confidence
interval contains zero, and the intervals are narrow enough to **exclude any
single-component contribution above 0.015 macro *F*₁**, and their joint contribution above
0.013. A fine-tuned encoder alone matches
the full architecture using 71% of the parameters and 47% of the training time. An
independently developed baseline, NoRBERT, lands within 0.0023 of both. Even a TF-IDF and
linear SVM pipeline is within 0.0254.

What does move performance is the data. A variance decomposition attributes **91.5%** of
the variation in per-category *F*₁ to category and **1.5%** to method. A matched-support
experiment isolates class support as one causal driver: capping the majority category's
training data at the minority category's level costs 0.083 *F*₁, or 43% of the gap between
them — the remaining 57% being intrinsic to the categories.

Everything needed to check those claims is in this repository.

---

## Verify the paper in one minute

```bash
pip install -r requirements.txt
./reproduce.sh verify
```

This recomputes all 111 quantitative claims in the manuscript from the released
predictions and compares each against the printed value. It needs no GPU and no
training. Exit status is 0 only if every claim matches.

`./reproduce.sh all` additionally regenerates every figure, re-runs the classical
baseline, and rebuilds the corpus from the raw spreadsheets to confirm it is
byte-identical to the released version — the full chain from source data to reported
number, in about three minutes on a laptop.

```
==================================================================
111 of 111 claims verified
==================================================================
Every quantitative claim in the paper matches the released data.
```

---

## What is in here

```
corpus/
  nfr_corpus.csv        1,667 requirements: item_id, text, label, source
  folds.csv             fold assignment for every record x 5 repeats
  provenance.csv        per-source contribution and column selection
  REPORT.txt            corpus construction log

predictions/
  folds/                175 files, one per (configuration, repeat, fold)
  pooled/               40 files, one per (configuration, repeat), all 1,667 records

results/
  summary.txt           full statistical output
  metrics.json          machine-readable
  table_*.tex           LaTeX tables as used in the paper
  runs.jsonl            parameter counts and wall-clock for all 150 MAGNET runs
  convergence_*.csv     per-epoch validation traces

sources/
  dcai24_src_dataset.xlsx                          PROMISE / PURE / SecReq aggregation
  Cornelius_FR_NFR_Curated_Mendeley_Dataset_2025.xlsx   curated Mendeley collection
  SOURCES.md            provenance, contribution, and licensing of each source

code/                   see "Scripts" below
figures/                the six figures, as PDF
  fig_corpus.pdf          Figure 1  corpus composition and class imbalance
  fig_architecture.pdf    Figure 2  architecture and ablation points
  fig_effects.pdf         Figure 3  effect sizes with bootstrap intervals
  fig_support.pdf         Figure 4  per-category F1 vs support, variance decomposition
  fig_matched_support.pdf Figure 5  matched-support experiment
  fig_convergence.pdf     Figure 6  training dynamics
manuscript/
  manuscript.tex          submission source (see BUILD.md)
  BUILD.md                how to compile, and how to switch to Springer's class
  CITATION_CHECKLIST.md   per-entry reference verification list
CHECKSUMS.txt           SHA-256 for every released data file
```

### Configurations

| Configuration | Prediction file stem | Params | Min/fold |
|---|---|---|---|
| Full MAGNET-NFR | `MAGNET-NFR-distilroberta-base` | 116.0M | 10.3 |
| − GCN | `MAGNET-NFR-no_gcn-distilroberta-base` | 114.1M | 10.6 |
| − Extended context | `MAGNET-NFR-no_xl-distilroberta-base` | 87.2M | 5.7 |
| − Contrastive loss | `MAGNET-NFR-no_contrastive-distilroberta-base` | 116.0M | 10.7 |
| − Meta-learning | `MAGNET-NFR-no_meta-distilroberta-base` | 116.0M | 8.9 |
| Encoder only | `MAGNET-NFR-roberta_only-distilroberta-base` | 82.8M | 4.8 |
| NoRBERT | `NoRBERT-bert-base-uncased` | 109.5M | 9.3 |
| TF-IDF + LinearSVC | `TFIDF-LinearSVC` | — | 0.1 |

Every prediction file has three columns: `item_id`, `gold`, `pred`. Files in
`predictions/pooled/` are named `<stem>__seed<repeat>.csv` and contain exactly one
out-of-fold prediction for each of the 1,667 records, produced by a model that never saw
that record during training.

---

## Scripts

| Script | Purpose | Hardware |
|---|---|---|
| `code/verify_paper.py` | recompute and check every reported number | CPU, 1 min |
| `code/make_figures.py` | regenerate Figures 1–5 | CPU, 1 min |
| `code/run_tfidf.py` | classical baseline | CPU, 1 min |
| `code/run_matched_support.py` | matched-support experiment isolating class support | CPU, 3 min |
| `code/run_lexical_analysis.py` | tests (and rejects) the lexical explanation for the residual | CPU, 10 s |
| `code/run_provenance.py` | recovers the documented sub-source composition of the corpus | CPU, 10 s |
| `code/build_corpus.py` | rebuild corpus and folds from source spreadsheets | CPU, 1 min |
| `code/train.py` | train one MAGNET-NFR variant on one fold | GPU, ~10 min |
| `code/run_norbert.py` | train NoRBERT on one fold | GPU, ~9 min |
| `code/pool_folds.py` | pool per-fold predictions into out-of-fold sets | CPU, seconds |
| `code/evaluate.py` | full statistical evaluation from predictions | CPU, 1 min |
| `code/preflight.py` | check splits and predictions for leakage before running | CPU, seconds |

`reproduce.sh` wraps these. Run `./reproduce.sh` with no argument for the verify target,
or see the header of that file for the others.

---

## Re-running the experiments

### Everything except training

```bash
./reproduce.sh all        # verify, regenerate figures, re-run the classical baseline
```

### The transformer runs

```bash
./reproduce.sh train      # 175 runs, ~25 h on one NVIDIA T4
```

Or a single run:

```bash
python code/train.py --variant no_gcn --seed 42 --rep 0 --fold 0 \
    --corpus corpus/nfr_corpus.csv --folds corpus/folds.csv --out predictions/folds/
```

Variants: `full`, `roberta_only`, `no_gcn`, `no_xl`, `no_contrastive`, `no_meta`.

Training runs are independent, so the loop can be interrupted and resumed — existing
output files are not recomputed if you filter for them, and the loop in `reproduce.sh`
simply overwrites. Total compute for the published results was 25.1 hours: 21.2 for the
150 MAGNET-NFR runs and 3.9 for the 25 NoRBERT runs.

### Rebuilding the corpus from raw sources

Both source spreadsheets are included in `sources/`, so the pipeline runs end to end
from raw data:

```bash
./reproduce.sh corpus
```

This rebuilds the corpus in a temporary directory and checks the result against the
released files:

```
  IDENTICAL  corpus/nfr_corpus.csv
  IDENTICAL  corpus/folds.csv

The released corpus is exactly what this code produces from these sources.
```

To inspect the sources or rebuild manually:

```bash
python code/build_corpus.py --base sources --inspect                     # show schemas
python code/build_corpus.py --base sources --build --min-class 40 --cv 5 --repeats 5
```

See `sources/SOURCES.md` for what each dataset contributes and under what terms.

`build_corpus.py` selects the label column **by evidence** rather than by name: several
of these files carry both a binary functional/non-functional flag and an NFR
sub-category column, and matching on the column name picks the wrong one. It also
deduplicates on a punctuation- and case-insensitive fingerprint, reports cross-source
overlap, and makes the scope reduction explicit.

Rebuilding is deterministic (`random_state=42`), so it reproduces `corpus/nfr_corpus.csv`
and `corpus/folds.csv` exactly.

---

## A note on the statistics

The primary inferential instrument is a **paired bootstrap over records** on the
difference in macro *F*₁, not McNemar's test. Two reasons, both documented in the paper:

1. **McNemar tests accuracy, not macro *F*₁.** On a corpus where one category holds 80.6%
   of records, the two move independently, so a test of the former does not license a
   claim about the latter.
2. **Repeated cross-validation produces correlated evaluations.** Pooling contingency
   counts across five repeats treats 8,335 paired observations as independent when there
   are only 1,667 records. In this data that inflation is material: pooling gives
   *p* = 0.002 for the classical-baseline comparison, whereas the correct record-level
   bootstrap gives *p* = 0.042.

`verify_paper.py` reports both figures side by side so the discrepancy is visible.

## Design decisions worth knowing about

**Cross-validation, not a holdout.** A 70/15/15 split of this corpus leaves Reliability
with 10 test instances and Performance with 16, from which per-category *F*₁ cannot be
estimated usefully. We use stratified 5-fold cross-validation repeated 5 times and pool
out-of-fold predictions, which evaluates every category on its full support.

**The knowledge graph is built from training data only.** Mining graph terms from the
full corpus would leak test information. `train.py` constructs the graph inside each
fold from that fold's training partition.

**Class weighting is on.** With an 80.56% majority class, an unweighted run collapses to
always-predict-Security and reports a meaningless 80.56% accuracy.

**Reported honestly, as run.** The encoder is `distilroberta-base`, not a full-size
RoBERTa. Stage 3 implements relative-position attention, not the segment-recurrence
mechanism of Transformer-XL. Stage 6 is first-order MAML, not second-order. Each of
these is stated in the manuscript.

**Known limits.** Two ISO/IEC 25010 categories, Maintainability (34 records) and
Portability (18), were excluded for insufficient support. All requirements are English.
No human-expert baseline exists for this task, so we cannot say how close macro *F*₁ of
0.898 is to what the task permits.

---

## Requirements

```
python >= 3.9
torch >= 2.0
transformers >= 4.40
scikit-learn >= 1.3
pandas, numpy, scipy, matplotlib, openpyxl
```

`pip install -r requirements.txt`. Only `train.py` and `run_norbert.py` use the GPU;
everything else runs on CPU.

---

## Citation

```bibtex
@article{nabot2026composition,
  title  = {Does Multi-Paradigm Composition Improve Non-Functional Requirements
            Classification? A Controlled Ablation Study},
  author = {Nabot, Ahmad and Nashwan, Shadi},
  year   = {2026}
}
```

## Licence

The MIT Licence (see `LICENSE`) covers this project's own contributions: the corpus
aggregation, scope and fold decisions, all code, the analysis and verification scripts,
and every derived result under `results/`, `predictions/`, `splits/` and `robustness/`.

It does **not** cover the third-party datasets under `sources/`, which remain under
their original authors' terms. See `sources/SOURCES.md` for provenance and the licence
status of each. `corpus/nfr_corpus.csv` contains requirement text derived from those
datasets; our aggregation and identifiers are MIT-licensed, the underlying text is not.

**Before making this repository public**, read `sources/PUBLIC_RELEASE_NOTICE.md`. If the
upstream licences have not been confirmed, run `./prepare_public_release.sh`, which
removes the raw spreadsheets and installs a pointer file in their place. Every result in
the paper stays verifiable without them.

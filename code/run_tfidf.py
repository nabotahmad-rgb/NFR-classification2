#!/usr/bin/env python3
"""
run_tfidf.py — classical machine learning baseline.

TF-IDF bigram features with a class-weighted linear SVM, evaluated on exactly the
same cross-validation folds as every neural configuration. This is the reference
against which the transformer advantage is measured in the paper.

Runs on CPU in well under a minute for all five repeats.

USAGE
    python code/run_tfidf.py \
        --corpus corpus/nfr_corpus.csv \
        --folds  corpus/folds.csv \
        --out    predictions/pooled/

OUTPUT
    predictions/pooled/TFIDF-LinearSVC__seed<rep>.csv    one per repeat

Each output file holds one out-of-fold prediction for every record in the corpus,
matching the layout produced by pool_folds.py for the neural configurations.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/nfr_corpus.csv")
    ap.add_argument("--folds", default="corpus/folds.csv")
    ap.add_argument("--out", default="predictions/pooled")
    ap.add_argument("--config-name", default="TFIDF-LinearSVC")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    corpus = pd.read_csv(args.corpus)
    folds = pd.read_csv(args.folds)
    gold = corpus.set_index("item_id").label

    print(f"corpus {len(corpus)} records, {corpus.label.nunique()} classes")
    floor = corpus.label.value_counts(normalize=True).max()
    print(f"majority-class floor {floor * 100:.2f}%\n")

    accs, f1s = [], []
    for rep in sorted(folds.rep.unique()):
        frep = folds[folds.rep == rep]
        pred = pd.Series(index=corpus.item_id, dtype=object)

        for k in sorted(frep.fold.unique()):
            test_ids = set(frep[frep.fold == k].item_id)
            te = corpus[corpus.item_id.isin(test_ids)]
            tr = corpus[~corpus.item_id.isin(test_ids)]

            # sanity: the fold must not leak
            assert not (set(tr.item_id) & set(te.item_id)), "fold leakage"

            clf = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                LinearSVC(class_weight="balanced", max_iter=5000),
            )
            clf.fit(tr.text.astype(str), tr.label)
            pred.loc[te.item_id] = clf.predict(te.text.astype(str))

        assert pred.notna().all(), "folds do not tile the corpus"

        out = pd.DataFrame(
            {"item_id": gold.index, "gold": gold.values, "pred": pred.loc[gold.index].values}
        )
        fp = os.path.join(args.out, f"{args.config_name}__seed{rep}.csv")
        out.to_csv(fp, index=False)

        a = accuracy_score(out.gold, out.pred)
        f = f1_score(out.gold, out.pred, average="macro", zero_division=0)
        accs.append(a)
        f1s.append(f)
        print(f"  rep {rep}: acc {a * 100:.2f}%  macro F1 {f:.4f}  -> {os.path.basename(fp)}")

    accs, f1s = np.array(accs), np.array(f1s)
    print(f"\naccuracy  {accs.mean() * 100:.2f} +/- {accs.std(ddof=1) * 100:.2f}")
    print(f"macro F1  {f1s.mean():.4f} +/- {f1s.std(ddof=1):.4f}")
    print(f"margin over floor  {(accs.mean() - floor) * 100:+.2f} pp\n")

    last = pd.read_csv(
        os.path.join(args.out, f"{args.config_name}__seed{sorted(folds.rep.unique())[0]}.csv")
    )
    print(classification_report(last.gold, last.pred, zero_division=0, digits=3))


if __name__ == "__main__":
    main()

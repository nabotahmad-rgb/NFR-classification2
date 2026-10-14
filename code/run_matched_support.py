#!/usr/bin/env python3
"""
run_matched_support.py — isolate class support from category identity.

The per-category F1 gradient (Security 0.982 at n=1343, Reliability 0.811 at n=68)
is associational: category identity confounds class support with whatever else
distinguishes one quality attribute from another.

This experiment separates the two. We cap Security's TRAINING support at successively
lower levels, leaving the evaluation set untouched, and measure Security's F1. If
support is the mechanism, Security's F1 should fall toward the minority categories'
values as its training data is reduced.

Result reported in the paper: it falls from 0.979 to 0.895 at n=68, which accounts for
43% of the Security-Reliability gap. The remaining 57% is intrinsic to the categories.

Uses the TF-IDF pipeline so the experiment is cheap (CPU, ~2 min) and so the conclusion
does not depend on any particular neural architecture.

USAGE
    python code/run_matched_support.py
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

LEVELS = [68, 106, 150, 300, 600, 1343]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/nfr_corpus.csv")
    ap.add_argument("--folds", default="corpus/folds.csv")
    ap.add_argument("--target", default="Security", help="majority category to cap")
    ap.add_argument("--reference", default="Reliability", help="minority category to match")
    a = ap.parse_args()

    c = pd.read_csv(a.corpus)
    folds = pd.read_csv(a.folds)
    gold = c.set_index("item_id").label

    n_ref = int((c.label == a.reference).sum())
    print(f"corpus {len(c)} records")
    print(f"  {a.target}: {int((c.label == a.target).sum())} records")
    print(f"  {a.reference}: {n_ref} records\n")
    print(f"Capping {a.target}'s TRAINING support; evaluation set untouched.\n")

    results = {}
    for lvl in LEVELS:
        f1s = []
        for rep in sorted(folds.rep.unique()):
            fr = folds[folds.rep == rep]
            preds = pd.Series(index=c.item_id, dtype=object)
            for k in sorted(fr.fold.unique()):
                ids = set(fr[fr.fold == k].item_id)
                te = c[c.item_id.isin(ids)]
                tr = c[~c.item_id.isin(ids)]
                maj = tr[tr.label == a.target]
                keep = maj.sample(min(lvl, len(maj)), random_state=rep * 100 + k)
                tr2 = pd.concat([tr[tr.label != a.target], keep])
                m = make_pipeline(
                    TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                    LinearSVC(class_weight="balanced", max_iter=5000),
                )
                m.fit(tr2.text.astype(str), tr2.label)
                preds.loc[te.item_id] = m.predict(te.text.astype(str))
            f = precision_recall_fscore_support(
                gold, preds.loc[gold.index], labels=[a.target], zero_division=0
            )[2][0]
            f1s.append(f)
        results[lvl] = (float(np.mean(f1s)), float(np.std(f1s, ddof=1)))
        print(f"  {a.target} training support {lvl:5d}:  F1 {np.mean(f1s):.3f} "
              f"+/- {np.std(f1s, ddof=1):.3f}")

    # reference category at its own support
    f1s = []
    for rep in sorted(folds.rep.unique()):
        fr = folds[folds.rep == rep]
        preds = pd.Series(index=c.item_id, dtype=object)
        for k in sorted(fr.fold.unique()):
            ids = set(fr[fr.fold == k].item_id)
            te = c[c.item_id.isin(ids)]
            tr = c[~c.item_id.isin(ids)]
            m = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                LinearSVC(class_weight="balanced", max_iter=5000),
            )
            m.fit(tr.text.astype(str), tr.label)
            preds.loc[te.item_id] = m.predict(te.text.astype(str))
        f1s.append(precision_recall_fscore_support(
            gold, preds.loc[gold.index], labels=[a.reference], zero_division=0)[2][0])
    ref_f1 = float(np.mean(f1s))

    full = results[max(LEVELS)][0]
    matched = results[n_ref][0] if n_ref in results else results[min(LEVELS)][0]

    print(f"\n  {a.reference} at its own support ({n_ref}):  F1 {ref_f1:.3f}")
    print("\n" + "=" * 62)
    print(f"GAP DECOMPOSITION: {a.target} vs {a.reference}")
    print("=" * 62)
    gap = full - ref_f1
    sup = full - matched
    intr = matched - ref_f1
    print(f"  total gap                        {gap:.3f}")
    print(f"  attributable to class support    {sup:.3f}  ({100 * sup / gap:.0f}%)")
    print(f"  residual, intrinsic to category  {intr:.3f}  ({100 * intr / gap:.0f}%)")
    print()
    print("  Support is causally implicated but does not explain the whole gradient.")
    print("  Reporting the variance decomposition as attributable to 'class support'")
    print("  alone would overstate what this experiment supports.")


if __name__ == "__main__":
    main()

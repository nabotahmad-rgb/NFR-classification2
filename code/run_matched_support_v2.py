#!/usr/bin/env python3
"""
run_matched_support_v2.py — repeated, source-stratified matched-support experiment.

Addresses two limitations of the original single-draw design:

  1. Each support level is now generated from many independent subsamples rather
     than one draw per fold, so the reported spread reflects *which records were
     retained* and not only cross-validation variance.
  2. Subsamples preserve the source composition of the capped category. Security
     is 79.7% one collection, so an unstratified draw perturbs source mixture and
     class support together and the two cannot be separated.

The experiment is also run for Performance and Usability, so the claim concerns
training support generally rather than Security alone.

CPU only, TF-IDF pipeline. Runtime ~25 min at B=40.
"""
import argparse, json
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


def pipe():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
        LinearSVC(class_weight="balanced", max_iter=5000))


def stratified_cap(pool, n, rng, src_col):
    """Draw n records from pool, preserving source proportions where feasible."""
    if n >= len(pool):
        return pool
    shares = pool[src_col].value_counts(normalize=True)
    take, out = {}, []
    for s, frac in shares.items():
        take[s] = int(np.floor(frac * n))
    short = n - sum(take.values())
    for s in shares.index[:short]:
        take[s] += 1
    for s, k in take.items():
        g = pool[pool[src_col] == s]
        k = min(k, len(g))
        if k:
            out.append(g.sample(k, random_state=int(rng.integers(1 << 31))))
    got = pd.concat(out) if out else pool.iloc[:0]
    if len(got) < n:                      # top up if rounding left a shortfall
        rest = pool.drop(index=got.index)
        got = pd.concat([got, rest.sample(min(n - len(got), len(rest)),
                                          random_state=int(rng.integers(1 << 31)))])
    return got


def curve(c, folds, gold, target, levels, B, src_col, stratify, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for lvl in levels:
        vals = []
        for b in range(B):
            rep = int(folds.rep.unique()[b % folds.rep.nunique()])
            fr = folds[folds.rep == rep]
            preds = pd.Series(index=c.item_id, dtype=object)
            for k in sorted(fr.fold.unique()):
                ids = set(fr[fr.fold == k].item_id)
                te, tr = c[c.item_id.isin(ids)], c[~c.item_id.isin(ids)]
                pool = tr[tr.label == target]
                keep = (stratified_cap(pool, lvl, rng, src_col) if stratify
                        else pool.sample(min(lvl, len(pool)),
                                         random_state=int(rng.integers(1 << 31))))
                tr2 = pd.concat([tr[tr.label != target], keep])
                m = pipe(); m.fit(tr2.text.astype(str), tr2.label)
                preds.loc[te.item_id] = m.predict(te.text.astype(str))
            vals.append(precision_recall_fscore_support(
                gold, preds.loc[gold.index], labels=[target], zero_division=0)[2][0])
        v = np.array(vals)
        out[lvl] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                        lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))
        print(f"    n={lvl:5d}  F1 {v.mean():.3f} +/- {v.std(ddof=1):.3f}   "
              f"95% range [{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/nfr_corpus.csv")
    ap.add_argument("--folds", default="corpus/folds.csv")
    ap.add_argument("--source", default="corpus/record_source.csv")
    ap.add_argument("-B", type=int, default=40, help="subsamples per level")
    a = ap.parse_args()

    c = pd.read_csv(a.corpus)
    try:
        c = c.merge(pd.read_csv(a.source), on="item_id"); src = "coll"
    except FileNotFoundError:
        c["coll"] = "unknown"; src = "coll"
    folds = pd.read_csv(a.folds); gold = c.set_index("item_id").label
    N = {k: int((c.label == k).sum()) for k in c.label.unique()}
    res = {}

    print("=" * 74)
    print(f"REPEATED SOURCE-STRATIFIED MATCHED SUPPORT  (B={a.B} subsamples per level)")
    print("=" * 74)

    print(f"\nSecurity  (n={N['Security']}), source-stratified:")
    res["Security_strat"] = curve(c, folds, gold, "Security",
                                  [68, 106, 150, 300, 600, N["Security"]], a.B, src, True)
    print(f"\nSecurity  (n={N['Security']}), unstratified, for contrast:")
    res["Security_unstrat"] = curve(c, folds, gold, "Security",
                                    [68, N["Security"]], a.B, src, False, seed=7)
    for cat in ("Usability", "Performance"):
        lv = sorted({34, 68, N[cat]})
        print(f"\n{cat}  (n={N[cat]}), source-stratified:")
        res[cat] = curve(c, folds, gold, cat, lv, a.B, src, True, seed=hash(cat) % 1000)

    # reference category at native support
    vals = []
    for rep in sorted(folds.rep.unique()):
        fr = folds[folds.rep == rep]; preds = pd.Series(index=c.item_id, dtype=object)
        for k in sorted(fr.fold.unique()):
            ids = set(fr[fr.fold == k].item_id)
            te, tr = c[c.item_id.isin(ids)], c[~c.item_id.isin(ids)]
            m = pipe(); m.fit(tr.text.astype(str), tr.label)
            preds.loc[te.item_id] = m.predict(te.text.astype(str))
        vals.append(precision_recall_fscore_support(
            gold, preds.loc[gold.index], labels=["Reliability"], zero_division=0)[2][0])
    ref = float(np.mean(vals)); res["Reliability_native"] = ref

    S = res["Security_strat"]
    full, matched = S[N["Security"]]["mean"], S[68]["mean"]
    gap, sup = full - ref, full - matched
    print("\n" + "=" * 74)
    print(f"  Reliability at native support (68):  {ref:.3f}")
    print(f"  total gap {gap:.3f}   support {sup:.3f} ({100*sup/gap:.0f}%)   "
          f"unexplained {matched-ref:.3f} ({100*(matched-ref)/gap:.0f}%)")
    res["decomposition"] = dict(total=gap, support=sup, unexplained=matched - ref,
                                support_pct=100 * sup / gap)
    json.dump(res, open("results/matched_support_v2.json", "w"), indent=1)
    print("\n  written to results/matched_support_v2.json")


if __name__ == "__main__":
    main()

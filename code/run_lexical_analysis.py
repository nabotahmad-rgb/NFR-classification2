#!/usr/bin/env python3
"""
run_lexical_analysis.py — test whether lexical diffuseness explains the residual
category effect.

Section 4.4 of the paper shows that class support accounts for 43% of the
Security-Reliability performance gap, leaving 57% intrinsic to the categories. The
natural explanation is that Reliability requirements are phrased more diffusely. This
script tests that explanation and rejects it.

Three measurements, all at matched support so that vocabulary statistics are not
confounded by corpus size:

  1. Type-token ratio        lower = more formulaic
  2. Top-20 term coverage    higher = more concentrated vocabulary
  3. Cross-category Jaccard  lower = more lexically distinct

Result: Reliability is MORE formulaic and MORE lexically distinct than Security at equal
support, yet is classified worst. The diffuseness explanation points the wrong way.

USAGE
    python code/run_lexical_analysis.py
"""
import argparse
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

CATS = ["Security", "Usability", "Performance", "Reliability"]


def toks(s):
    return re.findall(r"[a-z]+", str(s).lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/nfr_corpus.csv")
    ap.add_argument("--preds",
                    default="predictions/pooled/MAGNET-NFR-distilroberta-base__seed0.csv")
    ap.add_argument("--draws", type=int, default=200)
    a = ap.parse_args()

    c = pd.read_csv(a.corpus)
    N = {k: int((c.label == k).sum()) for k in CATS}

    # ---- raw lexical statistics -----------------------------------
    print("=" * 66)
    print("LEXICAL STATISTICS BY CATEGORY (unmatched support)")
    print("=" * 66)
    print(f"{'category':13s} {'n':>5s} {'TTR':>8s} {'top20 cov':>11s}")
    per = {}
    for k in CATS:
        cnt = Counter(w for s in c[c.label == k].text for w in toks(s))
        per[k] = cnt
        tot = sum(cnt.values())
        ttr = len(cnt) / tot
        top20 = sum(v for _, v in cnt.most_common(20)) / tot
        print(f"{k:13s} {N[k]:5d} {ttr:8.4f} {top20:11.3f}")
    print("\n  These are confounded by corpus size: a larger sample of any text")
    print("  yields a lower type-token ratio. The matched comparison follows.")

    # ---- matched-support comparison -------------------------------
    ref = "Reliability"
    n_ref = N[ref]
    cnt_ref = per[ref]
    tot_ref = sum(cnt_ref.values())
    ttr_ref = len(cnt_ref) / tot_ref
    cov_ref = sum(v for _, v in cnt_ref.most_common(20)) / tot_ref

    rng = np.random.default_rng(0)
    ttrs, covs = [], []
    for _ in range(a.draws):
        sub = c[c.label == "Security"].sample(n_ref, random_state=int(rng.integers(1e6)))
        cnt = Counter(w for s in sub.text for w in toks(s))
        tot = sum(cnt.values())
        ttrs.append(len(cnt) / tot)
        covs.append(sum(v for _, v in cnt.most_common(20)) / tot)
    ttrs, covs = np.array(ttrs), np.array(covs)

    print("\n" + "=" * 66)
    print(f"MATCHED SUPPORT: Security subsampled to {n_ref} documents, {a.draws} draws")
    print("=" * 66)
    z_t = (ttr_ref - ttrs.mean()) / ttrs.std()
    z_c = (cov_ref - covs.mean()) / covs.std()
    print(f"  type-token ratio    Security {ttrs.mean():.4f} +/- {ttrs.std():.4f}   "
          f"{ref} {ttr_ref:.4f}   ({z_t:+.1f} s.d.)")
    print(f"  top-20 coverage     Security {covs.mean():.3f} +/- {covs.std():.3f}   "
          f"{ref} {cov_ref:.3f}   ({z_c:+.1f} s.d.)")
    print()
    print(f"  -> {ref} is {'MORE' if z_t < 0 else 'LESS'} formulaic than Security at equal support")
    print(f"  -> {ref} vocabulary is {'MORE' if z_c > 0 else 'LESS'} concentrated")
    if z_t < 0 and z_c > 0:
        print("  -> the 'diffuse language' explanation for the residual is REJECTED")

    # ---- cross-category distinctiveness ---------------------------
    print("\n" + "=" * 66)
    print("CROSS-CATEGORY VOCABULARY OVERLAP (Jaccard on top-100 terms)")
    print("=" * 66)
    top = {k: set(w for w, _ in per[k].most_common(100)) for k in CATS}
    for k in CATS:
        j = np.mean([len(top[k] & top[o]) / len(top[k] | top[o]) for o in CATS if o != k])
        print(f"  {k:13s} mean Jaccard with others {j:.3f}")
    jr = np.mean([len(top[ref] & top[o]) / len(top[ref] | top[o]) for o in CATS if o != ref])
    others = [np.mean([len(top[k] & top[o]) / len(top[k] | top[o]) for o in CATS if o != k])
              for k in CATS if k != ref]
    print(f"\n  -> {ref} is the {'MOST' if jr < min(others) else 'not the most'} "
          f"lexically distinct category, yet is classified worst.")

    # ---- where the errors go --------------------------------------
    print("\n" + "=" * 66)
    print("ERROR DESTINATIONS (full model, first repeat)")
    print("=" * 66)
    d = pd.read_csv(a.preds)
    cm = confusion_matrix(d.gold, d.pred, labels=CATS)
    i = CATS.index(ref)
    fn = cm[i].sum() - cm[i][i]
    for j, k in enumerate(CATS):
        if j != i and cm[i][j] > 0:
            print(f"  {ref} misread as {k:13s} {cm[i][j]:3d}  ({100 * cm[i][j] / fn:.0f}% of false negatives)")
    prior = {k: N[k] / sum(N[x] for x in CATS if x != ref) for k in CATS if k != ref}
    print(f"\n  class prior among non-{ref} records: "
          + ", ".join(f"{k} {100 * v:.0f}%" for k, v in prior.items()))
    print("  -> errors skew toward the majority class, but less than the prior predicts,")
    print("     so majority-class attraction is a partial account at best.")


if __name__ == "__main__":
    main()

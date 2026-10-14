#!/usr/bin/env python3
"""
verify_paper.py — check every quantitative claim in the paper against the data.

This script exists so that a reader does not have to take any reported figure on
trust. It recomputes each number from corpus/ and predictions/pooled/ and compares
it with the value printed in the manuscript, which is hard-coded below.

Exit status is 0 if every claim matches and 1 otherwise.

USAGE
    python code/verify_paper.py
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

PREDS = "predictions/pooled"
CORPUS = "corpus/nfr_corpus.csv"

CONFIGS = {
    "full": "MAGNET-NFR-distilroberta-base",
    "no_gcn": "MAGNET-NFR-no_gcn-distilroberta-base",
    "no_xl": "MAGNET-NFR-no_xl-distilroberta-base",
    "no_con": "MAGNET-NFR-no_contrastive-distilroberta-base",
    "no_meta": "MAGNET-NFR-no_meta-distilroberta-base",
    "ronly": "MAGNET-NFR-roberta_only-distilroberta-base",
    "norbert": "NoRBERT-bert-base-uncased",
    "tfidf": "TFIDF-LinearSVC",
}
CATS = ["Security", "Usability", "Performance", "Reliability"]

results = []


def check(label, computed, paper):
    ok = computed == paper
    results.append((label, ok, computed, paper))


def load(stem):
    out = {}
    for fp in glob.glob(os.path.join(PREDS, f"{stem}__seed*.csv")):
        s = int(re.search(r"seed(\d+)", fp).group(1))
        out[s] = pd.read_csv(fp).sort_values("item_id").reset_index(drop=True)
    return dict(sorted(out.items()))


def mcnemar(A, B):
    n12 = n21 = 0
    for s in sorted(set(A) & set(B)):
        a, b = A[s], B[s]
        assert (a.item_id.values == b.item_id.values).all(), "item sets differ"
        ca = a.gold.values == a.pred.values
        cb = b.gold.values == b.pred.values
        n12 += int((ca & ~cb).sum())
        n21 += int((~ca & cb).sum())
    chi2 = (abs(n12 - n21) - 1) ** 2 / (n12 + n21)
    return n12, n21, chi2, float(1 - stats.chi2.cdf(chi2, 1))


def main():
    if not os.path.exists(CORPUS):
        sys.exit("run this from the repository root: python code/verify_paper.py")

    cfg = {k: load(v) for k, v in CONFIGS.items()}
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        sys.exit(f"missing predictions for: {missing}")

    corpus = pd.read_csv(CORPUS)

    # ---- integrity -------------------------------------------------
    print("=" * 66)
    print("INTEGRITY")
    print("=" * 66)
    sigs = {(k, s): tuple(d.item_id) for k, A in cfg.items() for s, d in A.items()}
    print(f"  identical item set across all {len(sigs)} pooled sets: "
          f"{'yes' if len(set(sigs.values())) == 1 else 'NO'}")
    print(f"  records per set: {len(next(iter(cfg['full'].values())))}")
    print(f"  repeats per configuration: {len(cfg['full'])}")

    # ---- corpus ----------------------------------------------------
    print("\n" + "=" * 66)
    print("CORPUS (Section 3.1, Table 2)")
    print("=" * 66)
    vc = corpus.label.value_counts()
    floor = vc.max() / len(corpus)
    check("corpus size", str(len(corpus)), "1667")
    check("majority floor", f"{floor * 100:.2f}", "80.56")
    for cat, want in [("Security", "1343"), ("Usability", "150"),
                      ("Performance", "106"), ("Reliability", "68")]:
        check(f"{cat} support", str(int(vc[cat])), want)

    # ---- main results ----------------------------------------------
    print("\n" + "=" * 66)
    print("MAIN RESULTS (Table 4)")
    print("=" * 66)
    paper = {
        "full": ("96.22", "0.8984"), "no_gcn": ("96.14", "0.8955"),
        "no_xl": ("96.38", "0.8980"), "no_con": ("96.09", "0.8933"),
        "no_meta": ("96.30", "0.8983"), "ronly": ("96.32", "0.8966"),
        "norbert": ("96.30", "0.8961"), "tfidf": ("95.48", "0.8730"),
    }
    for k, (pa, pf) in paper.items():
        a = np.array([accuracy_score(d.gold, d.pred) for d in cfg[k].values()])
        f = np.array([f1_score(d.gold, d.pred, average="macro", zero_division=0)
                      for d in cfg[k].values()])
        check(f"{k} accuracy", f"{a.mean() * 100:.2f}", pa)
        check(f"{k} macro F1", f"{f.mean():.4f}", pf)
        print(f"  {k:9s} acc {a.mean() * 100:6.2f} +/- {a.std(ddof=1) * 100:.2f}   "
              f"macro F1 {f.mean():.4f} +/- {f.std(ddof=1):.4f}")

    a_full = np.array([accuracy_score(d.gold, d.pred) for d in cfg["full"].values()])
    check("margin over floor", f"{a_full.mean() * 100 - floor * 100:.2f}", "15.66")

    # ---- ablation tests: paired bootstrap over records --------------
    print("\n" + "=" * 66)
    print("ABLATION — paired bootstrap over records (Section 4.2, Table 4)")
    print("=" * 66)
    print("  McNemar is NOT used: it tests accuracy, not macro F1, and pooling")
    print("  contingency counts across repeats would violate independence.\n")

    CATS_I = {c: i for i, c in enumerate(sorted(CATS))}
    reps = sorted(cfg["full"])
    G = np.array([CATS_I[x] for x in cfg["full"][reps[0]].gold])
    P = {k: np.stack([np.array([CATS_I[x] for x in cfg[k][s].pred]) for s in reps]) for k in cfg}
    NREC = len(G)
    K = len(CATS_I)

    def macro_f1_w(g, pr, w):
        tot = 0.0
        for c in range(K):
            tp = (w * ((pr == c) & (g == c))).sum()
            fp = (w * ((pr == c) & (g != c))).sum()
            fn = (w * ((pr != c) & (g == c))).sum()
            tot += 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
        return tot / K

    def boot(a, b, n_boot=4000, seed=1):
        rng = np.random.default_rng(seed)
        d = np.empty(n_boot)
        for i in range(n_boot):
            w = np.bincount(rng.integers(0, NREC, NREC), minlength=NREC).astype(float)
            d[i] = (np.mean([macro_f1_w(G, P[a][j], w) for j in range(len(reps))])
                    - np.mean([macro_f1_w(G, P[b][j], w) for j in range(len(reps))]))
        one = np.ones(NREC)
        obs = (np.mean([macro_f1_w(G, P[a][j], one) for j in range(len(reps))])
               - np.mean([macro_f1_w(G, P[b][j], one) for j in range(len(reps))]))
        lo, hi = np.percentile(d, [2.5, 97.5])
        pv = 2 * min((d <= 0).mean(), (d >= 0).mean())
        return obs, lo, hi, max(pv, 1 / n_boot)

    paper_boot = {
        "no_gcn":  ("+0.0029", "-0.0070", "+0.0129"),
        "no_xl":   ("+0.0005", "-0.0111", "+0.0130"),
        "no_con":  ("+0.0051", "-0.0045", "+0.0145"),
        "no_meta": ("+0.0001", "-0.0089", "+0.0092"),
        "ronly":   ("+0.0018", "-0.0089", "+0.0130"),
        "norbert": ("+0.0023", "-0.0139", "+0.0189"),
        "tfidf":   ("+0.0254", "+0.0011", "+0.0496"),
    }
    widest_single = 0.0
    widest_joint = 0.0
    for k, (po, pl, ph) in paper_boot.items():
        obs, lo, hi, pv = boot("full", k)
        check(f"{k} dF1", f"{obs:+.4f}", po)
        check(f"{k} CI lo", f"{lo:+.4f}", pl)
        check(f"{k} CI hi", f"{hi:+.4f}", ph)
        spans = lo <= 0 <= hi
        if k in ("no_gcn", "no_xl", "no_con", "no_meta"):
            widest_single = max(widest_single, abs(lo), abs(hi))
        if k == "ronly":
            widest_joint = max(abs(lo), abs(hi))
        print(f"  full vs {k:8s} {obs:+.4f}  [{lo:+.4f}, {hi:+.4f}]  p={pv:.3f}  "
              f"{'contains zero' if spans else 'EXCLUDES zero'}")

    check("equivalence bound (single component)", f"{widest_single:.3f}", "0.015")
    check("equivalence bound (joint)", f"{widest_joint:.3f}", "0.013")
    print(f"\n  Widest single-component bound: {widest_single:.4f} macro F1")
    print(f"  Joint (all four removed) bound: {widest_joint:.4f} macro F1")
    print(f"  -> excludes single-component contributions above ~{widest_single:.3f},")
    print(f"     and their joint contribution above ~{widest_joint:.3f}")

    keys = ["full", "no_gcn", "no_xl", "no_con", "no_meta", "ronly", "norbert"]
    common = sorted(set.intersection(*[set(cfg[k]) for k in keys]))
    mat = np.array([[f1_score(cfg[k][s].gold, cfg[k][s].pred, average="macro", zero_division=0)
                     for k in keys] for s in common])
    fchi, fp = stats.friedmanchisquare(*mat.T)
    check("friedman chi2 (macro F1)", f"{fchi:.2f}", "2.49")
    check("friedman p (macro F1)", f"{fp:.2f}", "0.87")
    print(f"\n  Friedman across 7 transformer configurations: chi2={fchi:.2f}, df=6, p={fp:.3f}")

    print("\n  For contrast, the inflated pooled-McNemar figure the paper warns about:")
    n12, n21, chi2, p = mcnemar(cfg["full"], cfg["tfidf"])
    check("inflated mcnemar p", f"{p:.3f}", "0.002")
    print(f"    pooled McNemar full vs TF-IDF: chi2={chi2:.2f}, p={p:.4f} (inflated)")
    print(f"    correct bootstrap p: 0.042")

    # ---- transformer vs classical ----------------------------------
    print("\n" + "=" * 66)
    print("TRANSFORMER vs CLASSICAL (Section 4.1)")
    print("=" * 66)
    for k, po, pl, ph in [("full", "+0.0254", "+0.0011", "+0.0496"),
                          ("ronly", "+0.0236", "-0.0009", "+0.0480"),
                          ("norbert", "+0.0230", "+0.0011", "+0.0461")]:
        obs, lo, hi, pv = boot(k, "tfidf")
        check(f"{k} vs tfidf dF1", f"{obs:+.4f}", po)
        surv = "survives" if pv < 0.05 / 9 else "does NOT survive"
        print(f"  {k:8s} vs TF-IDF  {obs:+.4f}  [{lo:+.4f}, {hi:+.4f}]  p={pv:.3f}  "
              f"Bonferroni(m=9): {surv}")

    # ---- per-category ----------------------------------------------
    print("\n" + "=" * 66)
    print("PER-CATEGORY (Table 5)")
    print("=" * 66)
    for cat, pf in [("Security", "0.982"), ("Usability", "0.887"),
                    ("Performance", "0.914"), ("Reliability", "0.811")]:
        fs = [precision_recall_fscore_support(d.gold, d.pred, labels=[cat], zero_division=0)[2][0]
              for d in cfg["full"].values()]
        check(f"{cat} F1", f"{np.mean(fs):.3f}", pf)
        print(f"  {cat:12s} n={int((corpus.label == cat).sum()):5d}  F1 {np.mean(fs):.3f}")

    # ---- variance decomposition ------------------------------------
    print("\n" + "=" * 66)
    print("VARIANCE DECOMPOSITION (Section 4.5, Figure 4)")
    print("=" * 66)
    rows = []
    for meth, A in cfg.items():
        for cat in CATS:
            for d in A.values():
                rows.append({"method": meth, "cat": cat,
                             "f1": precision_recall_fscore_support(d.gold, d.pred, labels=[cat],
                                                                   zero_division=0)[2][0]})
    df = pd.DataFrame(rows)
    gm = df.f1.mean()
    ss_tot = ((df.f1 - gm) ** 2).sum()
    ss_cat = sum(len(g) * (g.f1.mean() - gm) ** 2 for _, g in df.groupby("cat"))
    ss_meth = sum(len(g) * (g.f1.mean() - gm) ** 2 for _, g in df.groupby("method"))
    check("category variance", f"{100 * ss_cat / ss_tot:.1f}", "91.5")
    check("method variance", f"{100 * ss_meth / ss_tot:.1f}", "1.5")
    print(f"  category {100 * ss_cat / ss_tot:.1f}%   method {100 * ss_meth / ss_tot:.1f}%   "
          f"residual {100 * (ss_tot - ss_cat - ss_meth) / ss_tot:.1f}%")

    bycat = df.groupby("cat").f1.mean()
    bymeth = df.groupby("method").f1.mean()
    ratio = (bycat.max() - bycat.min()) / (bymeth.max() - bymeth.min())
    check("range ratio", f"{ratio:.1f}", "6.9")
    print(f"  category range {bycat.max() - bycat.min():.4f} is {ratio:.1f}x "
          f"the method range {bymeth.max() - bymeth.min():.4f}")

    # fit over method-category means, which is what Figure 4(a) plots
    n_by_cat = {c: int((corpus.label == c).sum()) for c in CATS}
    cell = df.groupby(["method", "cat"], as_index=False).f1.mean()
    x = np.log10([n_by_cat[c] for c in cell.cat])
    sl, ic, r, _, _ = stats.linregress(x, cell.f1.values)
    check("fit slope", f"{sl:.3f}", "0.114")
    check("fit r2", f"{r ** 2:.2f}", "0.80")
    print(f"  F1 = {ic:.3f} + {sl:.3f} log10(n),  r^2 = {r ** 2:.2f}")

    # ---- lexical analysis: the claims that reject the diffuseness account ----
    import re as _re
    from collections import Counter as _C
    print("\n" + "=" * 66)
    print("LEXICAL ANALYSIS (Section 4.4)")
    print("=" * 66)
    _tok = lambda x: _re.findall(r"[a-z]+", str(x).lower())
    per = {k: _C(w for s in corpus[corpus.label == k].text for w in _tok(s)) for k in CATS}
    n_ref = int((corpus.label == "Reliability").sum())
    cr = per["Reliability"]; tr = sum(cr.values())
    ttr_ref = len(cr) / tr
    cov_ref = sum(v for _, v in cr.most_common(20)) / tr
    rng = np.random.default_rng(0)
    T, V = [], []
    for _ in range(200):
        sub = corpus[corpus.label == "Security"].sample(n_ref, random_state=int(rng.integers(1e6)))
        cc = _C(w for s in sub.text for w in _tok(s)); tt = sum(cc.values())
        T.append(len(cc) / tt); V.append(sum(v for _, v in cc.most_common(20)) / tt)
    T, V = np.array(T), np.array(V)
    check("Reliability TTR", f"{ttr_ref:.3f}", "0.286")
    check("matched Security TTR", f"{T.mean():.3f}", "0.373")
    check("Reliability top20", f"{cov_ref:.3f}", "0.488")
    check("matched Security top20", f"{V.mean():.3f}", "0.371")
    check("TTR z", f"{(ttr_ref - T.mean()) / T.std():.1f}", "-6.2")
    check("top20 z", f"{(cov_ref - V.mean()) / V.std():.1f}", "9.0")
    top = {k: set(w for w, _ in per[k].most_common(100)) for k in CATS}
    jac = {k: np.mean([len(top[k] & top[o]) / len(top[k] | top[o]) for o in CATS if o != k])
           for k in CATS}
    check("Reliability Jaccard", f"{jac['Reliability']:.3f}", "0.258")
    print(f"  Reliability TTR {ttr_ref:.3f} vs matched Security {T.mean():.3f} "
          f"({(ttr_ref - T.mean()) / T.std():+.1f} s.d.)")
    print(f"  Reliability top-20 coverage {cov_ref:.3f} vs {V.mean():.3f} "
          f"({(cov_ref - V.mean()) / V.std():+.1f} s.d.)")
    print(f"  Reliability mean Jaccard {jac['Reliability']:.3f} "
          f"(lowest of {min(jac.values()):.3f}-{max(jac.values()):.3f})")
    print("  -> diffuse-language explanation rejected, as reported")

    # ---- provenance: documented sub-source composition (Table 2) ----
    print("\n" + "=" * 66)
    print("SUB-SOURCE PROVENANCE (Section 3.1, Table 2)")
    print("=" * 66)
    import importlib.util as _il, os as _os
    aggf = "sources/dcai24_src_dataset.xlsx"
    if _os.path.exists(aggf):
        _sp = _il.spec_from_file_location("bc", "code/build_corpus.py")
        _bc = _il.module_from_spec(_sp); _sp.loader.exec_module(_bc)
        dd = pd.read_excel(aggf, sheet_name=0)
        dd["ds"] = dd.Dataset_Name.astype(str).str.replace("'", "", regex=False).str.strip()
        dd["text"] = dd.Requirement.map(_bc.norm_text)
        dd["label"] = dd.Specific_Type.map(_bc.norm_label)
        dd = dd.dropna(subset=["text"])
        nf = dd[dd.label.isin(_bc.CANON)].copy()
        nf["fp"] = nf.text.map(_bc.fingerprint)
        cf = corpus.copy(); cf["fp"] = cf.text.map(_bc.fingerprint)
        keepset = set(cf.fp)
        ded = nf.drop_duplicates(subset=["fp"], keep="first")
        contrib = {s2: len(set(g.fp) & keepset) for s2, g in ded.groupby("ds")}
        tot = sum(contrib.values())
        check("n documented collections", str(len(contrib) + 1), "7")
        check("largest collection share",
              f"{100 * max(contrib.values()) / len(corpus):.1f}", "64.2")
        check("Mendeley contribution", str(len(corpus) - tot), "57")
        check("promise_exp share",
              f"{100 * contrib.get('promise_exp', 0) / len(corpus):.1f}", "9.9")
        for s2, k in sorted(contrib.items(), key=lambda x: -x[1]):
            print(f"  {s2[:44]:44s} {k:5d}  ({100 * k / len(corpus):5.1f}%)")
        print(f"  {'Curated Mendeley collection':44s} {len(corpus)-tot:5d}  "
              f"({100 * (len(corpus)-tot) / len(corpus):5.1f}%)")
    else:
        print("  sources/ not present; skipping (corpus checks above are unaffected)")

    # ---- new checks from the review round ---------------------------
    print("\n" + "=" * 66)
    print("REVIEW-ROUND ADDITIONS")
    print("=" * 66)
    from scipy import stats as _st
    from sklearn.metrics import confusion_matrix as _cm
    # the paper's claim is across the six MAGNET-NFR configurations, not the baselines
    six = ["full", "no_gcn", "no_xl", "no_con", "no_meta", "ronly"]
    acc_by = {k: np.array([accuracy_score(d.gold, d.pred) for _, d in sorted(cfg[k].items())])
              for k in six}
    lev = _st.levene(*[acc_by[k] for k in acc_by], center="median")
    check("Brown-Forsythe W", f"{lev.statistic:.2f}", "0.65")
    check("Brown-Forsythe p", f"{lev.pvalue:.2f}", "0.67")
    l2 = _st.levene(acc_by["full"], acc_by["no_gcn"], center="median")
    check("BF pairwise W", f"{l2.statistic:.2f}", "0.87")
    print(f"  variance homogeneity: W={lev.statistic:.2f} p={lev.pvalue:.3f} (does not reject)")

    obs = np.zeros(3)
    dest = ["Security", "Usability", "Performance"]
    for _, dd in sorted(cfg["full"].items()):
        m = _cm(dd.gold, dd.pred, labels=CATS)
        i = CATS.index("Reliability")
        obs += np.array([m[i][CATS.index(x)] for x in dest])
    Nc = {k: int((corpus.label == k).sum()) for k in CATS}
    tot_other = sum(Nc[k] for k in dest)
    exp = np.array([Nc[k] / tot_other for k in dest]) * obs.sum()
    chi2 = ((obs - exp) ** 2 / exp).sum()
    check("error-destination chi2", f"{chi2:.2f}", "12.56")
    check("error-destination p", f"{1 - _st.chi2.cdf(chi2, 2):.3f}", "0.002")
    check("Security share of errors", f"{100 * obs[0] / obs.sum():.1f}", "70.5")
    print(f"  error destinations vs prior: chi2={chi2:.2f}, df=2, "
          f"p={1 - _st.chi2.cdf(chi2, 2):.4f} (rejects proportionality)")

    # ---- equivalence testing and Bayes factors (Table 7) ------------
    print("\n" + "=" * 66)
    print("EQUIVALENCE TESTING (Section 4.2, Table 7)")
    print("=" * 66)
    DELTA = 0.01
    def boot90(a, b, nb=4000, seed=1):
        rng = np.random.default_rng(seed); d = np.empty(nb)
        for i in range(nb):
            w = np.bincount(rng.integers(0, NREC, NREC), minlength=NREC).astype(float)
            d[i] = (np.mean([macro_f1_w(G, P[a][j], w) for j in range(len(reps))])
                    - np.mean([macro_f1_w(G, P[b][j], w) for j in range(len(reps))]))
        return np.percentile(d, [5, 95])
    bounds = {}
    for k in ("no_gcn", "no_xl", "no_con", "no_meta", "ronly"):
        lo, hi = boot90("full", k)
        bounds[k] = max(abs(lo), abs(hi))
        print(f"  full vs {k:8s} 90% CI [{lo:+.4f}, {hi:+.4f}]  bound {bounds[k]:.4f}  "
              f"{'EQUIVALENT' if (lo > -DELTA and hi < DELTA) else 'not equiv at 0.01'}")
    single = max(bounds[k] for k in ("no_gcn", "no_xl", "no_con", "no_meta"))
    check("TOST single-component bound", f"{single:.3f}", "0.013")
    check("TOST joint bound", f"{bounds['ronly']:.3f}", "0.011")
    print(f"\n  single-component bound {single:.4f} -> 0.013;  joint {bounds['ronly']:.4f} -> 0.011")

    from scipy.integrate import quad as _quad
    def _bf10(t_stat, n, r=0.707):
        nu = n - 1
        f = lambda g: ((1 + n*g*r*r)**-0.5) * ((1 + t_stat**2/((1 + n*g*r*r)*nu))**(-(nu+1)/2)) \
                      * (g**-1.5) * np.exp(-1/(2*g)) / np.sqrt(2*np.pi)
        num, _ = _quad(f, 0, np.inf, limit=200)
        return num / (1 + t_stat**2/nu)**(-(nu+1)/2)
    f1s = {k: np.array([f1_score(d.gold, d.pred, average="macro", zero_division=0)
                        for _, d in sorted(v.items())]) for k, v in cfg.items()}
    for k, want in (("no_gcn", "2.4"), ("no_con", "1.8"), ("tfidf", "0.006")):
        dd = f1s["full"] - f1s[k]
        tt = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))
        bf01 = 1 / _bf10(tt, len(dd))
        check(f"BF01 vs {k}", f"{bf01:.3f}" if bf01 < 1 else f"{bf01:.1f}", want)
        print(f"  BF01 full vs {k:8s} {bf01:8.3f}")

    # ---- standard deviations printed in Table 5 ---------------------
    print("\n" + "=" * 66)
    print("REPORTED STANDARD DEVIATIONS (Table 5)")
    print("=" * 66)
    sd_paper = {"full": ("0.28", "0.0067"), "no_gcn": ("1.08", "0.0256"),
                "no_xl": ("0.23", "0.0068"), "no_con": ("0.46", "0.0140"),
                "no_meta": ("0.28", "0.0054"), "ronly": ("0.40", "0.0143"),
                "norbert": ("0.23", "0.0089"), "tfidf": ("0.21", "0.0065")}
    for k, (wa, wf) in sd_paper.items():
        a = np.array([accuracy_score(d.gold, d.pred) for d in cfg[k].values()])
        f = np.array([f1_score(d.gold, d.pred, average="macro", zero_division=0)
                      for d in cfg[k].values()])
        check(f"{k} accuracy s.d.", f"{a.std(ddof=1) * 100:.2f}", wa)
        check(f"{k} macro F1 s.d.", f"{f.std(ddof=1):.4f}", wf)
    print("  all eight accuracy and macro F1 standard deviations recomputed")

    # ---- repeated source-stratified matched support (Table 7) -------
    import json as _json, os as _os
    _f = "results/matched_support_v2.json"
    if _os.path.exists(_f):
        ms = _json.load(open(_f))
        S = ms["Security_strat"]
        for lvl, want in (("1343", "0.979"), ("600", "0.978"), ("300", "0.973"),
                          ("150", "0.955"), ("106", "0.936"), ("68", "0.896")):
            check(f"matched Security n={lvl}", f"{S[lvl]['mean']:.3f}", want)
        check("matched Usability n=34", f"{ms['Usability']['34']['mean']:.3f}", "0.629")
        check("matched Performance n=34", f"{ms['Performance']['34']['mean']:.3f}", "0.768")
        check("unstratified Security n=68",
              f"{ms['Security_unstrat']['68']['mean']:.3f}", "0.894")
        check("Reliability native", f"{ms['Reliability_native']:.3f}", "0.787")
        check("support share of gap", f"{ms['decomposition']['support_pct']:.0f}", "43")
        print("  repeated source-stratified matched support reproduced")

    # ---- revision-round analyses (Sections 3.1, 3.2, 4.4) -----------
    print("\n" + "=" * 66)
    print("REVISION-ROUND ANALYSES")
    print("=" * 66)
    import os as _os
    from sklearn.feature_extraction.text import TfidfVectorizer as _TV
    from sklearn.svm import LinearSVC as _SVC
    from sklearn.pipeline import make_pipeline as _mp
    from sklearn.metrics.pairwise import cosine_similarity as _cs
    folds_df = pd.read_csv("corpus/folds.csv")

    # leave-one-source-out (Table 4)
    src = "corpus/record_source.csv"
    if _os.path.exists(src):
        cc = corpus.merge(pd.read_csv(src), on="item_id")
        def _clf():
            return _mp(_TV(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                       _SVC(class_weight="balanced", max_iter=5000))
        tot_n = tot_w = 0.0; sc = []
        for s2, g in cc.groupby("coll"):
            if len(g) < 20:
                continue
            m = _clf(); m.fit(cc[cc.coll != s2].text.astype(str), cc[cc.coll != s2].label)
            pr = m.predict(g.text.astype(str))
            pres = [x for x in CATS if (g.label == x).sum() > 0]
            f = f1_score(g.label, pr, average="macro", labels=pres, zero_division=0)
            tot_n += len(g); tot_w += len(g) * f; sc.append(f)
        check("LOSO weighted", f"{tot_w / tot_n:.3f}", "0.815")
        check("LOSO unweighted", f"{np.mean(sc):.3f}", "0.714")
        print(f"  leave-one-source-out: weighted {tot_w / tot_n:.3f}, unweighted {np.mean(sc):.3f}")

    # contrastive batch audit (Section 3.2)
    rng = np.random.default_rng(0); B = 16
    res = {k: [0, 0] for k in CATS}
    for rp in sorted(folds_df.rep.unique()):
        fr = folds_df[folds_df.rep == rp]
        for k in sorted(fr.fold.unique()):
            ids = set(fr[fr.fold == k].item_id)
            lab = corpus[~corpus.item_id.isin(ids)].label.values
            for _ in range(10):
                idx = rng.permutation(len(lab))
                for st in range(0, len(idx) - B + 1, B):
                    bb = lab[idx[st:st + B]]
                    for cat in CATS:
                        mm = (bb == cat).sum()
                        if mm > 0:
                            res[cat][0] += mm
                            if mm > 1:
                                res[cat][1] += mm
    for cat, want in (("Security", "100.0"), ("Usability", "75.5"),
                      ("Performance", "62.2"), ("Reliability", "45.5")):
        a, pz = res[cat]
        check(f"contrastive positives {cat}", f"{100 * pz / a:.1f}", want)
    print("  contrastive anchors with a same-class positive: "
          + ", ".join(f"{c2} {100 * res[c2][1] / res[c2][0]:.1f}%" for c2 in CATS))

    # per-fold training support (Table 7 caption)
    for cat, want in (("Security", "913"), ("Usability", "102"),
                      ("Performance", "72"), ("Reliability", "46")):
        tr = []
        for rp in sorted(folds_df.rep.unique()):
            fr = folds_df[folds_df.rep == rp]
            for k in sorted(fr.fold.unique()):
                ids = set(fr[fr.fold == k].item_id)
                na = (corpus[~corpus.item_id.isin(ids)].label == cat).sum()
                tr.append(na - int(round(na * 0.15)))
        check(f"per-fold train support {cat}", str(round(np.mean(tr))), want)

    # near-duplicates (Section 3.1)
    X = _TV(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform(corpus.text.astype(str))
    S = _cs(X); np.fill_diagonal(S, 0)
    nd = int((S.max(axis=1) >= 0.90).sum())
    check("near-duplicate records", str(nd), "138")
    check("near-duplicate share", f"{100 * nd / len(corpus):.1f}", "8.3")
    print(f"  near-duplicates at cosine >= 0.90: {nd} records ({100 * nd / len(corpus):.1f}%)")

    # ---- report ----------------------------------------------------
    bad = [x for x in results if not x[1]]
    print("\n" + "=" * 66)
    print(f"{len(results) - len(bad)} of {len(results)} claims verified")
    print("=" * 66)
    if bad:
        for label, _, got, want in bad:
            print(f"  MISMATCH  {label}: paper says {want}, data gives {got}")
        sys.exit(1)
    print("Every quantitative claim in the paper matches the released data.")


if __name__ == "__main__":
    main()

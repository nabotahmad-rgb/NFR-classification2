#!/usr/bin/env python3
"""
MAGNET-NFR evaluation harness  (ECSER-compliant)

Consumes prediction CSVs, emits every statistic and LaTeX table the manuscript needs.
No training. No network. Reads only what your runs produced.

INPUT
  preds/<CONFIG>__seed<N>.csv   with columns: item_id,gold,pred
  (this is exactly the layout already in run_final_small/)

USAGE
  python evaluate.py --preds preds/ --ref MAGNET-NFR-distilroberta-base --out results/

OUTPUT
  results/summary.txt        human-readable, every number
  results/table_perf.tex     Table 5  (baselines)
  results/table_percat.tex   Table 6  (per-category)
  results/table_ablation.tex Table 10 (ablation)
  results/metrics.json       machine-readable
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, classification_report

SEEDS_EXPECTED = [42, 123, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


# ---------------------------------------------------------------- loading
def load(preds_dir):
    runs = defaultdict(dict)
    for fp in sorted(glob.glob(os.path.join(preds_dir, "*__seed*.csv"))):
        base = os.path.basename(fp)
        m = re.match(r"(.+)__seed(\d+)\.csv$", base)
        if not m:
            print(f"  ! skipping unparseable filename: {base}")
            continue
        cfg, seed = m.group(1), int(m.group(2))
        df = pd.read_csv(fp)
        missing = {"item_id", "gold", "pred"} - set(df.columns)
        if missing:
            sys.exit(f"ERROR {base}: missing columns {missing}")
        runs[cfg][seed] = df.sort_values("item_id").reset_index(drop=True)
    if not runs:
        sys.exit(f"ERROR: no prediction files found in {preds_dir}")
    return dict(runs)


def integrity(runs):
    """Checks that must pass before any number is trusted."""
    lines = ["=" * 72, "INTEGRITY CHECKS", "=" * 72]
    ok = True

    # identical evaluation set across every config and seed
    sigs = {}
    for cfg, seeds in runs.items():
        for s, df in seeds.items():
            sigs[(cfg, s)] = tuple(df.item_id)
    uniq = set(sigs.values())
    if len(uniq) == 1:
        lines.append(f"  [PASS] all {len(sigs)} runs evaluated on an identical item set "
                     f"(n = {len(next(iter(uniq)))})")
    else:
        ok = False
        lines.append(f"  [FAIL] runs do not share one evaluation set ({len(uniq)} distinct sets). "
                     "McNemar is invalid across non-identical sets.")

    # gold must be constant
    golds = {(c, s): tuple(d.gold) for c, seeds in runs.items() for s, d in seeds.items()}
    lines.append(f"  [{'PASS' if len(set(golds.values())) == 1 else 'FAIL'}] "
                 f"gold labels constant across all runs")
    ok &= len(set(golds.values())) == 1

    # seed coverage
    for cfg, seeds in runs.items():
        have = sorted(seeds)
        if have == SEEDS_EXPECTED:
            lines.append(f"  [PASS] {cfg}: all 10 ECSER seeds")
        else:
            lines.append(f"  [WARN] {cfg}: {len(have)} seeds {have}")

    # degenerate predictions
    for cfg, seeds in runs.items():
        for s, d in seeds.items():
            if d.pred.nunique() == 1:
                ok = False
                lines.append(f"  [FAIL] {cfg} seed {s}: predicts a single class "
                             f"({d.pred.iloc[0]}) — collapsed run")

    # majority-class floor
    d0 = next(iter(next(iter(runs.values())).values()))
    floor = d0.gold.value_counts(normalize=True).max()
    maj = d0.gold.value_counts().idxmax()
    lines.append("")
    lines.append(f"  MAJORITY-CLASS FLOOR: always-predict-{maj} = {floor*100:.2f}%")
    lines.append(f"  Every accuracy below must be read against this, not against zero.")

    # split sanity — the leakage tell
    n_eval = len(d0)
    lines.append("")
    lines.append(f"  Evaluation set size: {n_eval}")
    lines.append(f"  >> If your corpus is ~1,531 and this is a 70/15/15 holdout, expect ~230.")
    lines.append(f"  >> {n_eval} rows means either out-of-fold CV predictions (fine) or")
    lines.append(f"     scoring on training data (fatal). Confirm before reporting anything.")
    return "\n".join(lines), ok, floor


# ---------------------------------------------------------------- metrics
def per_seed(seeds):
    acc = np.array([accuracy_score(d.gold, d.pred) for _, d in sorted(seeds.items())])
    f1 = np.array([f1_score(d.gold, d.pred, average="macro", zero_division=0)
                   for _, d in sorted(seeds.items())])
    return acc, f1


def boot_ci(df, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    ok = (df.gold.values == df.pred.values)
    N = len(ok)
    bs = np.array([ok[rng.integers(0, N, N)].mean() for _ in range(n)])
    return np.percentile(bs, 2.5) * 100, np.percentile(bs, 97.5) * 100


def mcnemar(a_seeds, b_seeds):
    """Pooled over seeds. Returns n12, n21, chi2, p."""
    n12 = n21 = 0
    for s in sorted(set(a_seeds) & set(b_seeds)):
        a, b = a_seeds[s], b_seeds[s]
        ca = (a.gold.values == a.pred.values)
        cb = (b.gold.values == b.pred.values)
        n12 += int((ca & ~cb).sum())
        n21 += int((~ca & cb).sum())
    if n12 + n21 == 0:
        return n12, n21, 0.0, 1.0
    chi2 = (abs(n12 - n21) - 1) ** 2 / (n12 + n21)
    return n12, n21, chi2, float(1 - stats.chi2.cdf(chi2, 1))


def cohens_d(a, b):
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def band(d):
    d = abs(d)
    return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"


def friedman(runs):
    cfgs = sorted(runs)
    common = sorted(set.intersection(*[set(runs[c]) for c in cfgs]))
    if len(cfgs) < 3:
        return None
    mat = np.array([[accuracy_score(runs[c][s].gold, runs[c][s].pred) for c in cfgs]
                    for s in common])
    st, p = stats.friedmanchisquare(*mat.T)
    ranks = stats.rankdata(-mat, axis=1).mean(axis=0)
    return {"chi2": float(st), "p": float(p), "df": len(cfgs) - 1,
            "ranks": dict(zip(cfgs, np.round(ranks, 2).tolist()))}


# ---------------------------------------------------------------- latex
def tex_escape(s):
    return s.replace("_", r"\_").replace("%", r"\%")


def table_ablation(rows, ref):
    L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"\textbf{Configuration} & \textbf{Accuracy} & \textbf{$\Delta$ vs full (pp)} & "
         r"\textbf{$\chi^2$} & \textbf{$p$} & \textbf{Cohen's $d$} & \textbf{Effect}\\", r"\midrule"]
    for r in rows:
        if r["config"] == ref:
            L.append(f"Full model & {r['acc']:.2f} $\\pm$ {r['sd']:.2f} & --- & --- & --- & --- & ---\\\\")
        else:
            sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "n.s."
            L.append(f"{tex_escape(r['config'])} & {r['acc']:.2f} $\\pm$ {r['sd']:.2f} & "
                     f"{r['delta']:+.2f} & {r['chi2']:.2f} & {r['p']:.3f}{sig} & "
                     f"{r['d']:.2f} & {r['band']}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_percat(df, floor):
    rep = classification_report(df.gold, df.pred, output_dict=True, zero_division=0)
    L = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"\textbf{Category} & \textbf{Test ($n$)} & \textbf{Precision} & \textbf{Recall} & "
         r"\textbf{F1} & \textbf{Viable?}\\", r"\midrule"]
    for cat in sorted(c for c in rep if c not in ("accuracy", "macro avg", "weighted avg")):
        r = rep[cat]
        n = int(r["support"])
        viable = "yes" if n >= 20 else r"\textbf{no --- $n<20$}"
        L.append(f"{cat} & {n} & {r['precision']:.3f} & {r['recall']:.3f} & {r['f1-score']:.3f} & {viable}\\\\")
    L.append(r"\midrule")
    m = rep["macro avg"]
    L.append(f"\\textbf{{Macro average}} & {int(m['support'])} & {m['precision']:.3f} & "
             f"{m['recall']:.3f} & \\textbf{{{m['f1-score']:.3f}}} & ---\\\\")
    L.append(f"\\multicolumn{{6}}{{@{{}}l@{{}}}}{{\\footnotesize Majority-class floor "
             f"(accuracy) = {floor*100:.2f}\\%}}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--ref", required=True, help="reference config (the full model)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--bootstrap", type=int, default=10000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = load(a.preds)
    if a.ref not in runs:
        sys.exit(f"ERROR: --ref '{a.ref}' not found. Available:\n  " + "\n  ".join(sorted(runs)))

    out = []
    chk, ok, floor = integrity(runs)
    out.append(chk)

    # per-config summary
    out += ["", "=" * 72, "PER-CONFIG PERFORMANCE", "=" * 72]
    summary = {}
    for cfg in sorted(runs):
        acc, f1 = per_seed(runs[cfg])
        lo, hi = boot_ci(runs[cfg][sorted(runs[cfg])[0]], a.bootstrap)
        sw = stats.shapiro(acc).pvalue if len(acc) >= 3 else float("nan")
        summary[cfg] = {"acc_mean": acc.mean() * 100, "acc_sd": acc.std(ddof=1) * 100,
                        "f1_mean": f1.mean(), "f1_sd": f1.std(ddof=1),
                        "ci": [lo, hi], "shapiro_p": float(sw), "n_seeds": len(acc),
                        "acc_per_seed": (acc * 100).round(2).tolist()}
        out.append(f"\n  {cfg}  ({len(acc)} seeds)")
        out.append(f"    accuracy   {acc.mean()*100:6.2f}% +/- {acc.std(ddof=1)*100:.2f}   "
                   f"95% CI [{lo:.2f}, {hi:.2f}]")
        out.append(f"    macro F1   {f1.mean():.4f} +/- {f1.std(ddof=1):.4f}")
        out.append(f"    margin over majority floor: {acc.mean()*100 - floor*100:+.2f} pp")
        out.append(f"    Shapiro-Wilk p = {sw:.3f}  "
                   f"({'normal' if sw > 0.05 else 'non-normal'} at alpha=0.05)")

    # comparisons vs reference
    out += ["", "=" * 72, f"COMPARISONS vs {a.ref}", "=" * 72]
    ref_acc, ref_f1 = per_seed(runs[a.ref])
    rows = [{"config": a.ref, "acc": summary[a.ref]["acc_mean"], "sd": summary[a.ref]["acc_sd"]}]
    for cfg in sorted(runs):
        if cfg == a.ref:
            continue
        acc, f1 = per_seed(runs[cfg])
        n12, n21, chi2, p = mcnemar(runs[a.ref], runs[cfg])
        d = cohens_d(ref_acc, acc)
        w = stats.wilcoxon(ref_acc, acc)
        wf = stats.wilcoxon(ref_f1, f1)
        delta = (acc.mean() - ref_acc.mean()) * 100
        rows.append({"config": cfg, "acc": acc.mean() * 100, "sd": acc.std(ddof=1) * 100,
                     "delta": delta, "chi2": chi2, "p": p, "d": d, "band": band(d)})
        verdict = "SIGNIFICANT" if p < 0.05 else ">>> NOT SIGNIFICANT <<<"
        out.append(f"\n  {a.ref}  vs  {cfg}")
        out.append(f"    delta accuracy {delta:+.2f} pp   delta macroF1 {f1.mean()-ref_f1.mean():+.4f}")
        out.append(f"    McNemar  n12={n12} n21={n21}  chi2={chi2:.3f}  p={p:.4f}   {verdict}")
        out.append(f"    Wilcoxon acc p={w.pvalue:.4f}   macroF1 p={wf.pvalue:.4f}")
        out.append(f"    Cohen's d = {d:.3f} ({band(d)})")
        if p >= 0.05:
            out.append(f"    >> Removing this component costs nothing measurable.")
            out.append(f"    >> Report it as such. A null here is a finding, not a failure.")

    fr = friedman(runs)
    if fr:
        out += ["", "=" * 72, "FRIEDMAN (all configs)", "=" * 72,
                f"  chi2 = {fr['chi2']:.3f}  df = {fr['df']}  p = {fr['p']:.4f}",
                f"  mean ranks: {fr['ranks']}"]

    # bonferroni
    m = len(runs) - 1
    out += ["", "=" * 72, "MULTIPLE-TESTING CORRECTION", "=" * 72,
            f"  m = {m} comparisons vs reference -> alpha' = {0.05/m:.4f}" if m else "  n/a"]
    if m:
        k = len(runs)
        m_all = k * (k - 1) // 2
        out.append(f"  m = {m_all} all-pairs           -> alpha' = {0.05/m_all:.4f}")
        for r in rows[1:]:
            keep = "survives" if r["p"] < 0.05 / m else "does NOT survive"
            out.append(f"    {r['config']}: p={r['p']:.4f} {keep} Bonferroni")

    # per-category on reference
    d0 = runs[a.ref][sorted(runs[a.ref])[0]]
    out += ["", "=" * 72, f"PER-CATEGORY ({a.ref}, first seed)", "=" * 72, "",
            classification_report(d0.gold, d0.pred, zero_division=0, digits=3)]
    small = [c for c, n in d0.gold.value_counts().items() if n < 20]
    if small:
        out.append(f"  !! categories with n<20 test support: {small}")
        out.append(f"     Their F1 has a CI spanning most of the unit interval.")
        out.append(f"     Exclude from the macro average or from the study.")

    # write
    txt = "\n".join(out)
    open(os.path.join(a.out, "summary.txt"), "w").write(txt)
    open(os.path.join(a.out, "table_ablation.tex"), "w").write(table_ablation(rows, a.ref))
    open(os.path.join(a.out, "table_percat.tex"), "w").write(table_percat(d0, floor))
    json.dump({"summary": summary, "comparisons": rows, "friedman": fr,
               "majority_floor": floor * 100, "integrity_ok": bool(ok)},
              open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    print(txt)
    print(f"\n\nWritten to {a.out}/: summary.txt, table_ablation.tex, table_percat.tex, metrics.json")


if __name__ == "__main__":
    main()

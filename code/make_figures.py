#!/usr/bin/env python3
"""
make_figures.py — regenerate every figure in the paper from the released data.

Figure 1 (architecture) is a schematic drawn from measured constants.
Figures 2-6 are drawn entirely from corpus/, predictions/pooled/, and results/.

USAGE
    python code/make_figures.py
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support

plt.rcParams.update({
    "font.family": "serif", "font.size": 9.5,
    "axes.linewidth": 0.8, "axes.labelsize": 10.0,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.2, "savefig.dpi": 400,
})

FULLW = 6.38   # \textwidth at A4 with 2.4 cm margins
HALFW = 4.45   # 0.70 \textwidth


def SAVE(path):
    plt.savefig(path)          # exact canvas size; no tight-bbox rescaling


CLR = {"full": "#1a5490", "ablation": "#93b0c9", "encoder": "#2e7d5b",
       "norbert": "#8e5aa8", "tfidf": "#c0392b", "warn": "#c0392b",
       "neutral": "#555555", "grid": "#e8e8e8"}

CONFIGS = {
    "Full MAGNET-NFR": "MAGNET-NFR-distilroberta-base",
    "- GCN": "MAGNET-NFR-no_gcn-distilroberta-base",
    "- Ext. context": "MAGNET-NFR-no_xl-distilroberta-base",
    "- Contrastive": "MAGNET-NFR-no_contrastive-distilroberta-base",
    "- Meta-learning": "MAGNET-NFR-no_meta-distilroberta-base",
    "Encoder only": "MAGNET-NFR-roberta_only-distilroberta-base",
    "NoRBERT": "NoRBERT-bert-base-uncased",
    "TF-IDF + SVM": "TFIDF-LinearSVC",
}
CATS = ["Reliability", "Performance", "Usability", "Security"]


def load(preds, stem):
    out = {}
    for fp in glob.glob(os.path.join(preds, f"{stem}__seed*.csv")):
        s = int(re.search(r"seed(\d+)", fp).group(1))
        out[s] = pd.read_csv(fp).sort_values("item_id").reset_index(drop=True)
    if not out:
        raise SystemExit(f"no predictions for {stem}")
    return dict(sorted(out.items()))


def despine(ax, keep_left=True):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)


def percat(A, cl):
    return np.mean([precision_recall_fscore_support(d.gold, d.pred, labels=[cl],
                                                    zero_division=0)[2][0]
                    for d in A.values()])


# =====================================================================
def fig1_architecture(out):
    """Seven-stage pipeline. Ablation tags occupy a dedicated line inside each box,
    so they cannot collide with the title, and the data path runs strictly downward."""
    fig, ax = plt.subplots(figsize=(FULLW, 4.60))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, title, sub, fc, ec, abl=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    fc=fc, ec=ec, lw=1.3))
        if abl:
            # three stacked lines with equal breathing room: title, detail, ablation tag.
            # The tag is inset well clear of the lower border so it reads as belonging to
            # the box rather than floating against its edge.
            ax.text(x + w / 2, y + h * 0.775, title, ha="center", va="center",
                    fontsize=9.0, weight="bold")
            ax.text(x + w / 2, y + h * 0.525, sub, ha="center", va="center",
                    fontsize=8.0, color="#333")
            ax.plot([x + w * 0.14, x + w * 0.86], [y + h * 0.345, y + h * 0.345],
                    color=ec, lw=0.7, ls=(0, (2, 2)), alpha=0.6)
            ax.text(x + w / 2, y + h * 0.185, f"ablated in  {abl}", ha="center",
                    va="center", fontsize=8.0, color=ec, family="monospace")
        else:
            ax.text(x + w / 2, y + h * (0.66 if sub else 0.5), title, ha="center",
                    va="center", fontsize=9.0, weight="bold")
            if sub:
                ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                        fontsize=8.0, color="#333")

    def arr(p1, p2):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", color=CLR["neutral"],
                                     lw=1.2, mutation_scale=12))

    # input
    box(35, 92.0, 30, 6.6, "Requirement text", None, "#f5f5f5", "#999")

    # three parallel representation branches, the two ablatable ones taller
    box(0.5, 66.5, 31.5, 19.5, "Stage 1 $\\cdot$ Encoder",
        "transformer, 256 tokens\n82.8M parameters", "#dce8f2", CLR["full"])
    box(34.2, 68.0, 31.5, 17.0, "Stage 2 $\\cdot$ GCN",
        "ISO/IEC 25010 graph, +1.9M", "#e1efe7", "#2e7d5b", "no_gcn")
    box(68.0, 66.5, 31.5, 19.5, "Stage 3 $\\cdot$ Ext. context",
        "relative position, +28.9M", "#f7eade", "#b5651d", "no_xl")
    for x in (16.2, 50.0, 83.8):
        arr((x, 91.8), (x, 86.4))

    # fusion, then straight down the page
    box(20, 53.0, 60, 10.0, "Stage 4 $\\cdot$ Cross-attention fusion",
        "12 heads; queries from Stage 1, keys/values from Stages 2 and 3",
        "#eee9f5", "#6a4c93")
    arr((16.2, 66.3), (30.0, 63.4))
    arr((50.0, 66.3), (50.0, 63.4))
    arr((83.8, 66.3), (70.0, 63.4))

    # training-time objectives, side by side, no crossing arrows
    box(5, 29.5, 41, 19.0, "Stage 5 $\\cdot$ Contrastive",
        "supervised, $\\tau = 0.07$", "#fdf1e4", "#c77d1a", "no_contrastive")
    box(54, 29.5, 41, 19.0, "Stage 6 $\\cdot$ FOMAML",
        "episodic adaptation", "#fce9ed", "#a4133c", "no_meta")
    arr((35, 52.8), (25.5, 48.7))
    arr((65, 52.8), (74.5, 48.7))
    ax.text(50, 39.0, "applied during\ntraining only", ha="center", va="center",
            fontsize=8.0, color="#666", style="italic")

    # head and output
    box(28, 14.5, 44, 10.0, "Stage 7 $\\cdot$ Classification head",
        r"$768 \rightarrow 512 \rightarrow 256 \rightarrow k$, dropout 0.3",
        "#e7f1e9", "#37623f")
    arr((25.5, 29.3), (38.0, 24.9))
    arr((74.5, 29.3), (62.0, 24.9))
    box(28, 0.8, 44, 11.8, r"$k = 4$ evaluated categories",
        "Security  ·  Usability  ·  Performance  ·  Reliability", "#f5f5f5", "#999")
    arr((50, 14.3), (50, 12.9))

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    SAVE(f"{out}/fig_architecture.pdf")


# =====================================================================
def fig2_corpus(corpus_dir, out):
    c = pd.read_csv(f"{corpus_dir}/nfr_corpus.csv")
    vc = c.label.value_counts()
    ev = [(k, int(vc[k])) for k in ["Security", "Usability", "Performance", "Reliability"]]
    ex = [("Maintainability", 34), ("Portability", 18)]

    fig, ax = plt.subplots(1, 2, figsize=(FULLW, 2.85),
                           gridspec_kw={"width_ratios": [1.3, 1.0], "wspace": 0.34})
    names = [n for n, _ in ev + ex]
    vals = [v for _, v in ev + ex]
    ax[0].barh(range(len(names)), vals, height=0.66,
               color=[CLR["full"]] * 4 + [CLR["warn"]] * 2)
    ax[0].set_yticks(range(len(names)))
    ax[0].set_yticklabels(names)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("Records (log scale)")
    ax[0].set_xscale("log"); ax[0].set_xlim(10, 3600)
    for i, v in enumerate(vals):
        ax[0].text(v * 1.20, i, f"{v:,}", va="center", fontsize=8.2)
    ax[0].axhline(3.5, color=CLR["warn"], lw=0.9, ls="--")
    ax[0].text(11.5, 4.60, "excluded: $n < 40$", fontsize=8.0,
               color=CLR["warn"], style="italic")
    ax[0].set_title("(a) Records per category", fontsize=9.5, loc="left", pad=6)
    despine(ax[0])

    sh = [100 * v / len(c) for _, v in ev]
    floor = max(sh)
    ax[1].bar(range(4), sh, width=0.62,
              color=[CLR["full"], "#5d7a96", "#8fa8bd", "#c3d3de"])
    ax[1].axhline(floor, color=CLR["warn"], lw=1.2, ls="--")
    ax[1].text(3.45, floor + 2.5, "majority-class\nfloor 80.56%",
               fontsize=8.0, color=CLR["warn"], ha="right", va="bottom")
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels(["Sec.", "Usa.", "Perf.", "Rel."])
    ax[1].set_ylabel("Share of corpus (%)")
    ax[1].set_ylim(0, 100)
    for i, v in enumerate(sh):
        ax[1].text(i, v + 2.2, f"{v:.1f}", ha="center", fontsize=8.2)
    ax[1].set_title("(b) Class imbalance", fontsize=9.5, loc="left", pad=6)
    despine(ax[1])
    fig.subplots_adjust(left=0.163, right=0.985, top=0.895, bottom=0.185)
    SAVE(f"{out}/fig_corpus.pdf")


# =====================================================================
def fig3_effects(boot, out):
    """Forest plot. Numeric detail lives in Table 4, so nothing is placed outside the axes:
    an annotation column positioned in data coordinates breaks if the figure is rescaled."""
    lbl = {"Full vs -GCN": "$-$ GCN",
           "Full vs -ExtCtx": "$-$ Extended context",
           "Full vs -Contrastive": "$-$ Contrastive loss",
           "Full vs -Meta": "$-$ Meta-learning",
           "Full vs Encoder": "Encoder only",
           "Full vs NoRBERT": "NoRBERT",
           "Full vs TF-IDF": "TF-IDF + SVM"}
    order = ["Full vs -GCN", "Full vs -ExtCtx", "Full vs -Contrastive",
             "Full vs -Meta", "Full vs Encoder", "Full vs NoRBERT", "Full vs TF-IDF"]
    col = {"Full vs Encoder": CLR["encoder"], "Full vs NoRBERT": CLR["norbert"],
           "Full vs TF-IDF": CLR["tfidf"]}
    B = {r[0]: r for r in boot}

    fig, ax = plt.subplots(figsize=(FULLW, 3.25))
    y = np.arange(len(order))[::-1]
    ax.axvspan(-0.015, 0.015, color="#eef2f7", zorder=0, lw=0)
    ax.axvline(0, color="#333", lw=1.0, zorder=1)

    for i, k in enumerate(order):
        _, obs, lo, hi, p = B[k]
        c = col.get(k, CLR["ablation"])
        ax.plot([lo, hi], [y[i], y[i]], color=c, lw=2.2, solid_capstyle="butt", zorder=2)
        for e in (lo, hi):
            ax.plot([e, e], [y[i] - 0.16, y[i] + 0.16], color=c, lw=1.4, zorder=2)
        ax.scatter([obs], [y[i]], s=42, color=c, zorder=4,
                   edgecolors="white", linewidths=0.8)

    ax.text(0.0, -1.02,
            r"shaded: single-component contributions excluded by these 95% intervals ($\pm$0.015)",
            ha="center", fontsize=8.2, style="italic", color="#4a5b6b")
    ax.set_yticks(y)
    ax.set_yticklabels([lbl[k] for k in order])
    ax.set_xlabel("Change in macro $F_1$ when the component is removed or substituted\n"
                  "(positive favours the full composition; 95% bootstrap intervals)")
    ax.set_xlim(-0.030, 0.056)
    ax.set_ylim(-1.5, len(order) - 0.4)
    ax.set_xticks([-0.02, 0.0, 0.02, 0.04])
    despine(ax, keep_left=False)
    fig.subplots_adjust(left=0.245, right=0.985, top=0.985, bottom=0.255)
    SAVE(f"{out}/fig_effects.pdf")


# =====================================================================
def fig4_support(cfg, corpus_dir, out):
    c = pd.read_csv(f"{corpus_dir}/nfr_corpus.csv")
    N = [int((c.label == k).sum()) for k in CATS]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(FULLW, 3.55),
                                  gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.36})
    series = [("Full MAGNET-NFR", "o", CLR["full"], 1.7, 1.0, 5.0),
              ("- GCN", "v", CLR["ablation"], 0.9, 0.7, 3.2),
              ("- Ext. context", "^", CLR["ablation"], 0.9, 0.7, 3.2),
              ("- Contrastive", "<", CLR["ablation"], 0.9, 0.7, 3.2),
              ("- Meta-learning", ">", CLR["ablation"], 0.9, 0.7, 3.2),
              ("Encoder only", "s", CLR["encoder"], 1.7, 1.0, 4.6),
              ("NoRBERT", "D", CLR["norbert"], 1.7, 1.0, 4.2),
              ("TF-IDF + SVM", "X", CLR["tfidf"], 1.7, 1.0, 5.2)]
    xs, ys_all = [], []
    for key, mk, co, lw, al, ms in series:
        ys = [percat(cfg[key], cl) for cl in CATS]
        xs += N; ys_all += ys
        ax.plot(N, ys, marker=mk, color=co, ms=ms, lw=lw, alpha=al,
                label=key if lw > 1.2 else None, zorder=3 if lw > 1.2 else 2)
    ax.plot([], [], marker="v", color=CLR["ablation"], ms=3.2, lw=0.9,
            label="four ablations")

    sl, ic, r, _, _ = stats.linregress(np.log10(xs), ys_all)
    xx = np.logspace(np.log10(52), np.log10(2200), 60)
    ax.plot(xx, ic + sl * np.log10(xx), "--", color="#444", lw=1.0, zorder=1)
    # upper-left is the only region free of both the data and the legend
    ax.text(60, 0.995,
            f"$F_1 = {ic:.3f} + {sl:.3f}\\,\\log_{{10}} n$   $r^2 = {r ** 2:.2f}$",
            fontsize=8.2, color="#333", va="top", ha="left")

    ax.set_xscale("log")
    # Reliability(68), Performance(106) and Usability(150) sit close together on a log
    # axis; stagger their labels vertically so none collide.
    ax.set_xticks(N)
    ax.set_xticklabels([])
    ax.minorticks_off()
    ax.set_xlim(52, 2200)
    # offsets in points; the middle label drops a full two-line height to clear
    # its neighbours, which sit only ~3 pt apart horizontally at this scale
    offs = [-4.0, -26.0, -4.0, -4.0]
    for n, cl, dy in zip(N, CATS, offs):
        ax.annotate(f"{cl[:4]}.\n$n$={n}", xy=(n, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, dy), textcoords="offset points",
                    ha="center", va="top", fontsize=8.0, annotation_clip=False)
        if dy < -10:  # leader line so the staggered label is unambiguous
            ax.annotate("", xy=(n, 0), xytext=(0, dy + 20),
                        xycoords=("data", "axes fraction"), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.6),
                        annotation_clip=False)
    ax.set_ylabel("Per-category $F_1$")
    ax.set_ylim(0.75, 1.01)
    for n in N:
        ax.axvline(n, color=CLR["grid"], lw=0.7, zorder=0)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=3, handlelength=1.4, columnspacing=1.1, fontsize=7.6)
    ax.set_title("(a) $F_1$ rises with class support, for every method",
                 fontsize=9.5, loc="left", pad=6)
    despine(ax)

    rows = []
    for meth, A in cfg.items():
        for cl in CATS:
            for d in A.values():
                rows.append({"m": meth, "c": cl,
                             "f1": precision_recall_fscore_support(
                                 d.gold, d.pred, labels=[cl], zero_division=0)[2][0]})
    df = pd.DataFrame(rows)
    gm = df.f1.mean()
    sst = ((df.f1 - gm) ** 2).sum()
    ssc = sum(len(g) * (g.f1.mean() - gm) ** 2 for _, g in df.groupby("c"))
    ssm = sum(len(g) * (g.f1.mean() - gm) ** 2 for _, g in df.groupby("m"))
    vals = [100 * ssc / sst, 100 * ssm / sst, 100 * (sst - ssc - ssm) / sst]
    ax2.bar(range(3), vals, width=0.6, color=[CLR["full"], CLR["tfidf"], "#c8c8c8"])
    for i, v in enumerate(vals):
        ax2.text(i, v + 2.8, f"{v:.1f}%", ha="center", fontsize=8.6, weight="bold")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["Category", "Method", "Other"], fontsize=8.5)
    ax2.set_ylabel("Variance share (%)", fontsize=9.5)
    ax2.set_ylim(0, 112)
    ax2.set_title("(b) Variance decomposition", fontsize=9.5, loc="left", pad=6)
    despine(ax2)
    fig.subplots_adjust(left=0.112, right=0.985, top=0.915, bottom=0.315)
    SAVE(f"{out}/fig_support.pdf")
    return vals, (ic, sl, r ** 2)


# =====================================================================
def fig5_matched(lc, out, ref_f1=0.786, ref_n=68):
    lv = sorted(int(k) for k in lc)
    mu = np.array([lc[str(k)][0] for k in lv])
    sd = np.array([lc[str(k)][1] for k in lv])

    fig, ax = plt.subplots(figsize=(HALFW, 3.05))
    ax.fill_between(lv, mu - sd, mu + sd, color=CLR["full"], alpha=0.16, lw=0)
    ax.plot(lv, mu, marker="o", ms=5.0, color=CLR["full"], lw=1.7, zorder=3,
            label="Security, training support capped")
    ax.axhline(ref_f1, color=CLR["tfidf"], ls="--", lw=1.2, zorder=2)
    ax.scatter([ref_n], [ref_f1], marker="X", s=75, color=CLR["tfidf"], zorder=5,
               edgecolors="white", linewidths=0.8,
               label=f"Reliability at its own support ($n$={ref_n})")

    matched = mu[lv.index(ref_n)]
    full = mu[-1]
    ax.annotate("", xy=(ref_n, matched), xytext=(ref_n, ref_f1),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.1))
    ax.text(ref_n * 1.14, (matched + ref_f1) / 2,
            f"{matched - ref_f1:.3f}\nunexplained after\nsupport matching (57%)",
            fontsize=8.0, color="#333", va="center")
    ax.annotate("", xy=(1700, full), xytext=(1700, matched),
                arrowprops=dict(arrowstyle="<->", color=CLR["full"], lw=1.1))
    ax.text(1560, (full + matched) / 2, f"{full - matched:.3f}\nsupport (43%)",
            fontsize=8.0, color=CLR["full"], va="center", ha="right")

    ax.set_xscale("log")
    ax.set_xlabel("Security training records (log scale)")
    ax.set_ylabel("Security $F_1$")
    ax.set_xticks(lv)
    ax.set_xticklabels([str(v) for v in lv], fontsize=8.2)
    ax.minorticks_off()
    ax.set_xlim(56, 2600)
    ax.set_ylim(0.76, 1.005)
    ax.legend(frameon=False, loc="lower right", fontsize=8.0)
    despine(ax)
    fig.subplots_adjust(left=0.165, right=0.980, top=0.955, bottom=0.165)
    SAVE(f"{out}/fig_matched_support.pdf")


# =====================================================================
def fig6_convergence(results_dir, corpus_dir, out):
    """Training dynamics. The legend sits below the axes and the peak annotation is tied to
    the peak marker by a leader, so no text floats unanchored in the plot area."""
    hits = glob.glob(f"{results_dir}/convergence_MAGNET-NFR-distilroberta-base*.csv")
    if not hits:
        print("  (no convergence log; skipping)")
        return
    d = pd.read_csv(sorted(hits)[0])
    c = pd.read_csv(f"{corpus_dir}/nfr_corpus.csv")
    floor = c.label.value_counts(normalize=True).max()

    fig, ax = plt.subplots(figsize=(HALFW, 3.05))
    ax.plot(d.epoch, d.val_acc, marker="o", ms=4.2, color=CLR["full"], lw=1.5,
            label="Validation accuracy")
    ax.plot(d.epoch, d.val_f1, marker="s", ms=4.2, color=CLR["tfidf"], lw=1.5,
            label="Validation macro $F_1$")

    # majority-class floor, labelled hard against the right axis
    ax.axhline(floor, color="#888", ls="--", lw=1.0)
    ax.text(d.epoch.max() + 0.15, floor, f"floor\n{floor * 100:.1f}%", fontsize=7.6,
            color="#666", ha="left", va="center")

    # peak, marked on the curve and tied to its label by a leader
    pk = int(d.loc[d.val_f1.idxmax(), "epoch"])
    pv = float(d.val_f1.max())
    ax.scatter([pk], [pv], s=60, facecolors="none", edgecolors=CLR["tfidf"], lw=1.3, zorder=5)
    ax.annotate(f"macro $F_1$ peak, epoch {pk}", xy=(pk, pv), xytext=(pk + 1.4, pv - 0.155),
                fontsize=8.0, color="#333", va="center",
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.8,
                                shrinkA=2, shrinkB=6))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation score")
    ax.set_xticks(range(1, int(d.epoch.max()) + 1))
    ax.set_xlim(0.6, d.epoch.max() + 1.05)
    ax.set_ylim(0.28, 1.02)
    ax.legend(frameon=False, fontsize=8.0, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), ncol=2, handlelength=1.6, columnspacing=1.4)
    despine(ax)
    fig.subplots_adjust(left=0.150, right=0.845, top=0.975, bottom=0.325)
    SAVE(f"{out}/fig_convergence.pdf")


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="predictions/pooled")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cfg = {k: load(a.preds, v) for k, v in CONFIGS.items()}
    boot = json.load(open(f"{a.results}/bootstrap.json"))
    lc = json.load(open(f"{a.results}/matched_support.json"))

    fig1_architecture(a.out);                        print("  fig_architecture.pdf")
    fig2_corpus(a.corpus, a.out);                    print("  fig_corpus.pdf")
    fig3_effects(boot, a.out);                       print("  fig_effects.pdf")
    vals, fit = fig4_support(cfg, a.corpus, a.out);  print("  fig_support.pdf")
    fig5_matched(lc, a.out);                         print("  fig_matched_support.pdf")
    fig6_convergence(a.results, a.corpus, a.out);    print("  fig_convergence.pdf")

    print(f"\nvariance: category {vals[0]:.1f}%  method {vals[1]:.1f}%  residual {vals[2]:.1f}%")
    print(f"fit: F1 = {fit[0]:.4f} + {fit[1]:.4f} log10(n),  r^2 = {fit[2]:.4f}")


if __name__ == "__main__":
    main()

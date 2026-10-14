#!/usr/bin/env python3
"""
run_source_holdout.py — leave-one-source-out evaluation for the neural configurations.

WHY THIS EXPERIMENT EXISTS
--------------------------
Section 4.4 reports leave-one-source-out for the TF-IDF pipeline only, on the
grounds that the ablation showed architecture does not separate configurations.
That reasoning is circular: architecture may be irrelevant under random
record-level splits and still matter under distribution shift, and structured
knowledge and meta-learning are precisely the mechanisms one would expect to
behave differently when the source changes.

This script runs the same holdout protocol over the neural configurations, so
that the architectural null can be tested under source shift rather than assumed
to hold there.

DESIGN
------
For each source collection with at least 20 records, train on every other
collection and test on the held-out one. Configurations, by default:

    full        the complete composition
    ronly       encoder only
    norbert     the independently developed baseline

That is 3 configurations x 6 collections = 18 runs, roughly 2.5 hours on a
single T4. A validation split of 15% is carved from the training collections,
stratified by label, so early stopping never sees the held-out source.

Predictions are written one file per (config, held-out source) in the same
column format as the main experiment: item_id, gold, pred. evaluate_source
_holdout() then scores them exactly as Section 4.4 scores the classical
pipeline, reporting macro F1 for multi-class holdouts and class recall for
single-class holdouts, without aggregating the two.

USAGE
-----
    # one run
    python code/run_source_holdout.py --prepare          # writes splits/
    python code/train.py --variant full \
        --train splits/loso_promise_exp_train.csv \
        --val   splits/loso_promise_exp_val.csv \
        --test  splits/loso_promise_exp_test.csv \
        --out predictions/loso --seed 0

    # everything, sequentially
    python code/run_source_holdout.py --run-all

    # score whatever is present
    python code/run_source_holdout.py --score
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

CONFIGS = ["full", "ronly", "norbert"]
MIN_RECORDS = 20
VAL_FRAC = 0.15


def load_corpus(corpus, source):
    c = pd.read_csv(corpus)
    if not os.path.exists(source):
        sys.exit(f"missing {source}; run code/run_provenance.py first")
    return c.merge(pd.read_csv(source), on="item_id")


def prepare(c, outdir, seed=0):
    """Write train/val/test csvs for every eligible held-out collection."""
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    written = []
    for coll, g in c.groupby("coll"):
        if len(g) < MIN_RECORDS:
            continue
        slug = "".join(ch if ch.isalnum() else "_" for ch in coll)[:40]
        rest = c[c.coll != coll]
        # stratified validation split from the training collections only
        val_idx = []
        for lab, gl in rest.groupby("label"):
            k = max(1, int(round(VAL_FRAC * len(gl))))
            val_idx += list(rng.choice(gl.index, size=min(k, len(gl)), replace=False))
        val = rest.loc[val_idx]
        tr = rest.drop(index=val_idx)
        for name, df in (("train", tr), ("val", val), ("test", g)):
            df[["item_id", "text", "label"]].to_csv(
                f"{outdir}/loso_{slug}_{name}.csv", index=False)
        written.append((coll, slug, len(tr), len(val), len(g),
                        sorted(g.label.unique())))
        print(f"  {coll[:38]:38s} train {len(tr):5d}  val {len(val):4d}  "
              f"test {len(g):5d}  classes {len(g.label.unique())}")
    json.dump([{"collection": a, "slug": b, "n_train": ct, "n_val": cv,
                "n_test": cte, "classes": cl}
               for a, b, ct, cv, cte, cl in written],
              open(f"{outdir}/manifest.json", "w"), indent=1)
    return written


def run_all(written, splits, outdir, backbone="distilroberta-base"):
    os.makedirs(outdir, exist_ok=True)
    total = len(written) * len(CONFIGS)
    i = 0
    for coll, slug, *_ in written:
        for cfgname in CONFIGS:
            i += 1
            print(f"\n[{i}/{total}] {cfgname} holding out {coll}")
            script = "code/run_norbert.py" if cfgname == "norbert" else "code/train.py"
            cmd = [sys.executable, script]
            if cfgname != "norbert":
                cmd += ["--variant", cfgname, "--backbone", backbone]
            cmd += ["--train", f"{splits}/loso_{slug}_train.csv",
                    "--val", f"{splits}/loso_{slug}_val.csv",
                    "--test", f"{splits}/loso_{slug}_test.csv",
                    "--out", outdir, "--seed", "0"]
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"  !! run failed for {cfgname} / {coll}; continuing")


def score(preddir, splits):
    """Score whatever prediction files exist, using Section 4.4's conventions."""
    man = json.load(open(f"{splits}/manifest.json"))
    by_slug = {m["slug"]: m for m in man}
    rows = []
    for fp in sorted(glob.glob(f"{preddir}/*.csv")):
        d = pd.read_csv(fp)
        if not {"item_id", "gold", "pred"} <= set(d.columns):
            continue
        slug = next((s for s in by_slug if s in os.path.basename(fp)), None)
        if slug is None:
            continue
        m = by_slug[slug]
        cfgname = ("norbert" if "NoRBERT" in fp else
                   "ronly" if "roberta_only" in fp else
                   "full" if "MAGNET-NFR-distil" in fp else "other")
        present = sorted(d.gold.unique())
        if len(present) == 1:
            rows.append(dict(config=cfgname, collection=m["collection"],
                             n=len(d), classes=1, metric="recall",
                             value=recall_score(d.gold, d.pred, average="macro",
                                                labels=present, zero_division=0)))
        else:
            rows.append(dict(config=cfgname, collection=m["collection"],
                             n=len(d), classes=len(present), metric="macro_f1",
                             value=f1_score(d.gold, d.pred, average="macro",
                                            labels=present, zero_division=0)))
    if not rows:
        print("  no scoreable prediction files found in " + preddir)
        return
    df = pd.DataFrame(rows)
    print("\n" + "=" * 74)
    print("LEAVE-ONE-SOURCE-OUT, NEURAL CONFIGURATIONS")
    print("=" * 74)
    mc = df[df.metric == "macro_f1"]
    if len(mc):
        print("\nMulti-class holdouts (macro F1):")
        print(mc.pivot_table(index="collection", columns="config",
                             values="value").round(3).to_string())
        print("\nMean over multi-class holdouts:")
        print(mc.groupby("config").value.mean().round(3).to_string())
    sc = df[df.metric == "recall"]
    if len(sc):
        print("\nSingle-class holdouts (class recall):")
        print(sc.pivot_table(index="collection", columns="config",
                             values="value").round(3).to_string())
    df.to_csv("results/source_holdout_neural.csv", index=False)
    print("\n  written to results/source_holdout_neural.csv")
    print("\n  Compare the multi-class means against the classical pipeline's 0.607")
    print("  (Table 5). If the configurations remain within roughly 0.02 of each")
    print("  other, the architectural null survives source shift; if the full")
    print("  composition separates, Section 5 needs revising.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/nfr_corpus.csv")
    ap.add_argument("--source", default="corpus/record_source.csv")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--preds", default="predictions/loso")
    ap.add_argument("--backbone", default="distilroberta-base")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()

    if a.score and not (a.prepare or a.run_all):
        score(a.preds, a.splits)
        return

    c = load_corpus(a.corpus, a.source)
    print(f"corpus {len(c)} records across {c.coll.nunique()} collections\n")
    written = prepare(c, a.splits)
    print(f"\n  {len(written)} holdouts x {len(CONFIGS)} configurations "
          f"= {len(written) * len(CONFIGS)} runs")
    if a.run_all:
        run_all(written, a.splits, a.preds, a.backbone)
        score(a.preds, a.splits)


if __name__ == "__main__":
    main()

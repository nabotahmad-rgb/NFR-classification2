#!/usr/bin/env python3
"""
run_grouped_duplicate_cv.py

Near-duplicate-grouped cross-validation sensitivity analysis for MAGNET-NFR.

Purpose
-------
Reproduce the manuscript's near-duplicate audit (character 3-5 gram TF-IDF,
cosine >= 0.90) and keep every connected near-duplicate/template cluster wholly
inside one outer CV fold. The inner validation split is also group-safe.

Primary comparison
------------------
  * full MAGNET-NFR
  * roberta_only (encoder-only)
  * TF-IDF + class-weighted LinearSVC

The default is 5 repeated 5-fold group-stratified assignments so the sensitivity
analysis matches the main paper's repeated-CV depth. Neural runs are resumable
because every (repeat, fold, architecture) prediction is stored separately.

Typical use from the replication-package root
---------------------------------------------
  python code/run_grouped_duplicate_cv.py --work /path/to/work --prepare
  python code/run_grouped_duplicate_cv.py --work /path/to/work --run-tfidf
  python code/run_grouped_duplicate_cv.py --work /path/to/work --run-neural --reps 0
  python code/run_grouped_duplicate_cv.py --work /path/to/work --run-neural --reps 1,2,3,4
  python code/run_grouped_duplicate_cv.py --work /path/to/work --score --bootstrap 4000

A one-repeat first pass is deliberate: rep 0 alone needs only 10 neural fits and
provides a fast validity check. Re-running with the remaining reps appends results.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

ARCHS = ("full", "roberta_only")
LABELS = ("Performance", "Reliability", "Security", "Usability")


def _atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _parse_ints(s: str | None, maximum: int):
    if s is None or str(s).strip() == "":
        return list(range(maximum))
    out = []
    for part in str(s).split(","):
        x = int(part.strip())
        if not (0 <= x < maximum):
            raise ValueError(f"repeat {x} outside 0..{maximum-1}")
        out.append(x)
    return sorted(set(out))


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent, a, b):
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[rb] = ra


def build_clusters(corpus: pd.DataFrame, threshold: float = 0.90):
    """Return cluster mapping, similarity pairs and an audit dict.

    The settings intentionally reproduce the manuscript audit:
      analyzer='char_wb', ngram_range=(3,5), min_df=2, default TF-IDF scaling.
    """
    texts = corpus.text.astype(str).tolist()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    X = vec.fit_transform(texts)
    S = cosine_similarity(X)
    np.fill_diagonal(S, 0.0)

    ii, jj = np.where(np.triu(S >= threshold, k=1))
    parent = list(range(len(corpus)))
    for a, b in zip(ii.tolist(), jj.tolist()):
        _union(parent, a, b)

    roots = [_find(parent, i) for i in range(len(corpus))]
    root_to_members = defaultdict(list)
    for i, r in enumerate(roots):
        root_to_members[r].append(i)

    # Stable cluster IDs ordered by first corpus position.
    ordered = sorted(root_to_members.values(), key=lambda xs: min(xs))
    idx_to_gid = {}
    for gi, members in enumerate(ordered):
        gid = f"G{gi:04d}"
        for i in members:
            idx_to_gid[i] = gid

    rows = []
    mixed = 0
    multi = 0
    max_size = 1
    for members in ordered:
        gid = idx_to_gid[members[0]]
        size = len(members)
        labs = sorted(corpus.iloc[members].label.astype(str).unique().tolist())
        if size > 1:
            multi += 1
            max_size = max(max_size, size)
            if len(labs) > 1:
                mixed += 1
        for i in members:
            rows.append({
                "item_id": corpus.iloc[i].item_id,
                "group_id": gid,
                "cluster_size": size,
                "near_duplicate_cluster": bool(size > 1),
                "cluster_labels": "|".join(labs),
            })
    clusters = pd.DataFrame(rows)

    pairs = []
    for a, b in zip(ii.tolist(), jj.tolist()):
        pairs.append({
            "item_id_a": corpus.iloc[a].item_id,
            "item_id_b": corpus.iloc[b].item_id,
            "label_a": corpus.iloc[a].label,
            "label_b": corpus.iloc[b].label,
            "similarity": float(S[a, b]),
            "group_id": idx_to_gid[a],
        })
    pairs = pd.DataFrame(pairs)

    n_records_multi = int(clusters.loc[clusters.near_duplicate_cluster, "item_id"].nunique())
    audit = {
        "method": {
            "vectorizer": "TfidfVectorizer",
            "analyzer": "char_wb",
            "ngram_range": [3, 5],
            "min_df": 2,
            "sublinear_tf": False,
            "cosine_threshold": threshold,
            "clustering": "connected components over all pairs at or above threshold",
        },
        "n_records": int(len(corpus)),
        "n_similarity_pairs_ge_threshold": int(len(pairs)),
        "n_records_with_neighbor_ge_threshold": n_records_multi,
        "n_groups_total": int(len(ordered)),
        "n_multi_record_clusters": int(multi),
        "largest_cluster": int(max_size),
        "n_mixed_label_multi_clusters": int(mixed),
    }
    return clusters, pairs, audit


def _distribution_score(df_all, idx, target_fraction):
    sel = df_all.iloc[idx]
    nterm = abs(len(sel) / len(df_all) - target_fraction)
    overall = df_all.label.value_counts(normalize=True)
    got = sel.label.value_counts(normalize=True)
    lterm = sum(abs(float(got.get(l, 0.0)) - float(overall.get(l, 0.0))) for l in overall.index)
    return nterm + lterm


def choose_group_safe_val(rest: pd.DataFrame, seed: int, test_size: float = 0.15, candidates: int = 300):
    """Pick a group-safe validation split that approximates 15% and label proportions."""
    splitter = GroupShuffleSplit(n_splits=candidates, test_size=test_size, random_state=seed)
    best = None
    Xdummy = np.zeros((len(rest), 1))
    y = rest.label.to_numpy()
    groups = rest.group_id.to_numpy()
    all_labels = set(rest.label.unique())
    for tr_idx, va_idx in splitter.split(Xdummy, y=y, groups=groups):
        tr = rest.iloc[tr_idx]
        va = rest.iloc[va_idx]
        if set(tr.label.unique()) != all_labels or set(va.label.unique()) != all_labels:
            continue
        score = _distribution_score(rest, va_idx, test_size)
        if best is None or score < best[0]:
            best = (score, tr_idx, va_idx)
    if best is None:
        raise RuntimeError("Could not create group-safe validation split containing every class.")
    return best[1], best[2]


def prepare(args):
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_csv(args.corpus)
    need = {"item_id", "text", "label"}
    if not need.issubset(corpus.columns):
        raise SystemExit(f"corpus missing columns: {sorted(need - set(corpus.columns))}")
    if corpus.item_id.duplicated().any():
        raise SystemExit("duplicate item_id in corpus")

    clusters, pairs, audit = build_clusters(corpus, args.threshold)
    _atomic_csv(clusters, work / "duplicate_clusters.csv")
    _atomic_csv(pairs, work / "near_duplicate_pairs.csv")

    merged = corpus.merge(clusters[["item_id", "group_id"]], on="item_id", how="left", validate="one_to_one")

    # Reproduce the manuscript's original-fold leakage audit when folds.csv is available.
    folds_path = Path(args.original_folds)
    cross_by_rep = {}
    if folds_path.exists() and len(pairs):
        old = pd.read_csv(folds_path)
        for rep in sorted(old.rep.unique()):
            m = dict(zip(old.loc[old.rep == rep, "item_id"], old.loc[old.rep == rep, "fold"]))
            n_cross = sum(m.get(a) != m.get(b) for a, b in zip(pairs.item_id_a, pairs.item_id_b))
            cross_by_rep[str(int(rep))] = {
                "n_pairs_crossing_fold": int(n_cross),
                "n_pairs": int(len(pairs)),
                "fraction": float(n_cross / len(pairs)),
            }
    audit["original_random_fold_pair_crossing"] = cross_by_rep

    fold_rows = []
    manifest_rows = []
    split_root = work / "splits"
    split_root.mkdir(exist_ok=True)

    Xdummy = np.zeros((len(merged), 1))
    for rep in range(args.repeats):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed + rep)
        assigned = np.full(len(merged), -1, dtype=int)
        for fold, (_, te_idx) in enumerate(sgkf.split(Xdummy, merged.label, merged.group_id)):
            assigned[te_idx] = fold
        if (assigned < 0).any():
            raise RuntimeError("outer folds did not tile corpus")

        # Hard invariant: one group -> one outer fold.
        chk = pd.DataFrame({"group_id": merged.group_id, "fold": assigned}).groupby("group_id").fold.nunique()
        if int(chk.max()) != 1:
            raise RuntimeError("near-duplicate group split across outer folds")

        for item_id, fold in zip(merged.item_id, assigned):
            fold_rows.append({"item_id": item_id, "rep": rep, "fold": int(fold)})

        for fold in range(5):
            te = merged.loc[assigned == fold].copy()
            rest = merged.loc[assigned != fold].copy().reset_index(drop=True)
            tr_idx, va_idx = choose_group_safe_val(rest, args.seed + 1000 * rep + fold,
                                                   test_size=0.15, candidates=args.val_candidates)
            tr = rest.iloc[tr_idx].copy()
            va = rest.iloc[va_idx].copy()

            gtr, gva, gte = set(tr.group_id), set(va.group_id), set(te.group_id)
            if gtr & gva or gtr & gte or gva & gte:
                raise RuntimeError(f"group leakage in rep {rep} fold {fold}")
            for dname, d in (("train", tr), ("val", va), ("test", te)):
                if set(d.label.unique()) != set(LABELS):
                    raise RuntimeError(f"missing class in {dname}, rep {rep} fold {fold}: {sorted(d.label.unique())}")
                out = d.drop(columns=["group_id"]).copy()
                _atomic_csv(out, split_root / f"rep{rep}_fold{fold}_{dname}.csv")

            row = {
                "rep": rep,
                "fold": fold,
                "n_train": len(tr),
                "n_val": len(va),
                "n_test": len(te),
                "groups_train": len(gtr),
                "groups_val": len(gva),
                "groups_test": len(gte),
                "group_leakage": False,
            }
            for dname, d in (("train", tr), ("val", va), ("test", te)):
                cnt = d.label.value_counts()
                for lab in LABELS:
                    row[f"{dname}_{lab}"] = int(cnt.get(lab, 0))
            manifest_rows.append(row)

    folds = pd.DataFrame(fold_rows)
    manifest = pd.DataFrame(manifest_rows)
    _atomic_csv(folds, work / "grouped_folds.csv")
    _atomic_csv(manifest, work / "split_manifest.csv")

    # Audit that every >=threshold pair is co-folded in every grouped repeat.
    grouped_cross = {}
    for rep in range(args.repeats):
        m = dict(zip(folds.loc[folds.rep == rep, "item_id"], folds.loc[folds.rep == rep, "fold"]))
        n_cross = sum(m[a] != m[b] for a, b in zip(pairs.item_id_a, pairs.item_id_b)) if len(pairs) else 0
        grouped_cross[str(rep)] = int(n_cross)
    audit["grouped_fold_pair_crossing_counts"] = grouped_cross
    audit["repeats"] = int(args.repeats)
    audit["outer_folds_per_repeat"] = 5
    audit["inner_validation_group_safe"] = True
    _atomic_json(audit, work / "duplicate_group_audit.json")

    print("PREPARE COMPLETE")
    print(json.dumps(audit, indent=2))
    print("\nSplit sizes / class minima:")
    print(manifest[["rep", "fold", "n_train", "n_val", "n_test",
                    "train_Reliability", "val_Reliability", "test_Reliability"]].to_string(index=False))
    print(f"\nOutputs: {work}")


def _load_split(work: Path, rep: int, fold: int, kind: str):
    p = work / "splits" / f"rep{rep}_fold{fold}_{kind}.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}; run --prepare first")
    return pd.read_csv(p), p


def run_tfidf(args):
    work = Path(args.work)
    reps = _parse_ints(args.reps, args.repeats)
    pred_root = work / "tfidf_predictions"
    pred_root.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for rep in reps:
        for fold in range(5):
            tr, _ = _load_split(work, rep, fold, "train")
            va, _ = _load_split(work, rep, fold, "val")
            te, _ = _load_split(work, rep, fold, "test")
            fp = pred_root / f"TFIDF-LinearSVC__rep{rep}fold{fold}.csv"
            if fp.exists():
                old = pd.read_csv(fp)
                if len(old) == len(te) and set(old.item_id) == set(te.item_id):
                    print(f"TFIDF rep {rep} fold {fold}: RESUME/SKIP")
                    pred = old
                else:
                    fp.unlink()
                    pred = None
            else:
                pred = None
            if pred is None:
                fit = pd.concat([tr, va], ignore_index=True)
                clf = make_pipeline(
                    TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                    LinearSVC(class_weight="balanced", max_iter=5000),
                )
                clf.fit(fit.text.astype(str), fit.label)
                pp = clf.predict(te.text.astype(str))
                pred = pd.DataFrame({"item_id": te.item_id, "gold": te.label, "pred": pp})
                _atomic_csv(pred, fp)
            all_metrics.append({
                "rep": rep, "fold": fold, "arch": "tfidf",
                "accuracy": accuracy_score(pred.gold, pred.pred),
                "macro_f1": f1_score(pred.gold, pred.pred, average="macro", zero_division=0),
                "n_test": len(pred),
            })
            print(f"TFIDF rep {rep} fold {fold}: macroF1={all_metrics[-1]['macro_f1']:.4f}")

    _atomic_csv(pd.DataFrame(all_metrics), work / "tfidf_fold_metrics.csv")
    print("TFIDF COMPLETE")


def _expected_prediction_name(arch: str, backbone: str, seed: int):
    tag = "MAGNET-NFR" if arch == "full" else f"MAGNET-NFR-{arch}"
    return f"{tag}-{backbone}__seed{seed}.csv"


def run_neural(args):
    work = Path(args.work)
    reps = _parse_ints(args.reps, args.repeats)
    pred_root = work / "neural_predictions"
    log_root = work / "logs"
    pred_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    jobs = [(rep, fold, arch) for rep in reps for fold in range(5) for arch in ARCHS]
    print(f"NEURAL JOBS REQUESTED: {len(jobs)}")
    for ji, (rep, fold, arch) in enumerate(jobs, 1):
        tr, trp = _load_split(work, rep, fold, "train")
        va, vap = _load_split(work, rep, fold, "val")
        te, tep = _load_split(work, rep, fold, "test")
        outdir = pred_root / f"rep{rep}" / f"fold{fold}" / arch
        outdir.mkdir(parents=True, exist_ok=True)
        expected = outdir / _expected_prediction_name(arch, args.backbone, args.train_seed)

        if expected.exists():
            old = pd.read_csv(expected)
            if len(old) == len(te) and set(old.item_id) == set(te.item_id):
                print(f"[{ji}/{len(jobs)}] rep {rep} fold {fold} {arch}: RESUME/SKIP")
                continue
            expected.unlink()

        print(f"\n[{ji}/{len(jobs)}] rep {rep} fold {fold} {arch}")
        cmd = [
            sys.executable, "code/train.py",
            "--variant", arch,
            "--seed", str(args.train_seed),
            "--train", str(trp),
            "--val", str(vap),
            "--test", str(tep),
            "--out", str(outdir),
            "--backbone", args.backbone,
            "--epochs", str(args.epochs),
        ]
        t0 = time.time()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        mins = (time.time() - t0) / 60.0
        logp = log_root / f"rep{rep}_fold{fold}_{arch}.log"
        logp.write_text(proc.stdout, encoding="utf-8")
        # Show useful tail immediately even under buffered notebook execution.
        tail = "\n".join(proc.stdout.splitlines()[-12:])
        print(tail)
        if proc.returncode != 0:
            print(f"!! FAILED rep {rep} fold {fold} {arch}; see {logp}; continuing")
            continue
        if not expected.exists():
            print(f"!! expected prediction missing: {expected}")
            continue
        pred = pd.read_csv(expected)
        if len(pred) != len(te) or set(pred.item_id) != set(te.item_id):
            print(f"!! invalid prediction coverage: {expected}")
            expected.unlink(missing_ok=True)
            continue
        f = f1_score(pred.gold, pred.pred, average="macro", zero_division=0)
        print(f"DONE rep {rep} fold {fold} {arch}: macroF1={f:.4f}, {mins:.1f} min")

    print("NEURAL REQUEST COMPLETE (failed/missing jobs, if any, can be resumed with same command)")


def _collect_neural(work: Path, repeats: int, backbone: str, train_seed: int):
    rows = []
    pooled = {}
    for rep in range(repeats):
        for arch in ARCHS:
            parts = []
            for fold in range(5):
                p = work / "neural_predictions" / f"rep{rep}" / f"fold{fold}" / arch / _expected_prediction_name(arch, backbone, train_seed)
                if not p.exists():
                    continue
                d = pd.read_csv(p)
                parts.append(d)
                rows.append({
                    "rep": rep, "fold": fold, "arch": arch,
                    "accuracy": accuracy_score(d.gold, d.pred),
                    "macro_f1": f1_score(d.gold, d.pred, average="macro", zero_division=0),
                    "n_test": len(d),
                })
            if len(parts) == 5:
                q = pd.concat(parts, ignore_index=True)
                if q.item_id.duplicated().any() or len(q) != 1667:
                    raise RuntimeError(f"bad pooled coverage for rep {rep} {arch}: n={len(q)} dup={q.item_id.duplicated().sum()}")
                pooled[(rep, arch)] = q
    return pd.DataFrame(rows), pooled


def _collect_tfidf(work: Path, repeats: int):
    rows = []
    pooled = {}
    for rep in range(repeats):
        parts = []
        for fold in range(5):
            p = work / "tfidf_predictions" / f"TFIDF-LinearSVC__rep{rep}fold{fold}.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p)
            parts.append(d)
            rows.append({
                "rep": rep, "fold": fold, "arch": "tfidf",
                "accuracy": accuracy_score(d.gold, d.pred),
                "macro_f1": f1_score(d.gold, d.pred, average="macro", zero_division=0),
                "n_test": len(d),
            })
        if len(parts) == 5:
            q = pd.concat(parts, ignore_index=True)
            if q.item_id.duplicated().any() or len(q) != 1667:
                raise RuntimeError(f"bad pooled TFIDF coverage for rep {rep}")
            pooled[(rep, "tfidf")] = q
    return pd.DataFrame(rows), pooled


def _paired_bootstrap(pooled, repeats, n_boot, seed):
    common_reps = [r for r in range(repeats) if (r, "full") in pooled and (r, "roberta_only") in pooled]
    if not common_reps:
        return None
    base = pooled[(common_reps[0], "full")][["item_id", "gold"]].copy()
    ids = base.item_id.tolist()
    gold = base.gold.to_numpy()
    pred_full = []
    pred_enc = []
    for r in common_reps:
        f = pooled[(r, "full")].set_index("item_id").loc[ids]
        e = pooled[(r, "roberta_only")].set_index("item_id").loc[ids]
        if not np.array_equal(f.gold.to_numpy(), gold) or not np.array_equal(e.gold.to_numpy(), gold):
            raise RuntimeError("gold mismatch across pooled predictions")
        pred_full.append(f.pred.to_numpy())
        pred_enc.append(e.pred.to_numpy())

    def avg_f1(preds, idx):
        return float(np.mean([f1_score(gold[idx], p[idx], average="macro", zero_division=0) for p in preds]))

    all_idx = np.arange(len(ids))
    obs = avg_f1(pred_full, all_idx) - avg_f1(pred_enc, all_idx)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(ids), size=len(ids))
        vals[b] = avg_f1(pred_full, idx) - avg_f1(pred_enc, idx)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    p2 = 2 * min(float(np.mean(vals <= 0)), float(np.mean(vals >= 0)))
    return {
        "comparison": "full_minus_roberta_only",
        "n_complete_repeats": len(common_reps),
        "delta_macro_f1": float(obs),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "bootstrap_p_two_sided": float(min(1.0, p2)),
        "n_boot": int(n_boot),
        "bootstrap_unit": "record; macro-F1 computed per repeat then averaged across repeats",
        "seed": int(seed),
    }


def score(args):
    work = Path(args.work)
    fold_neural, pooled_neural = _collect_neural(work, args.repeats, args.backbone, args.train_seed)
    fold_tfidf, pooled_tfidf = _collect_tfidf(work, args.repeats)
    pooled = {**pooled_neural, **pooled_tfidf}
    fold_metrics = pd.concat([fold_neural, fold_tfidf], ignore_index=True) if len(fold_tfidf) else fold_neural
    if len(fold_metrics):
        _atomic_csv(fold_metrics, work / "grouped_fold_metrics.csv")

    rep_rows = []
    pooldir = work / "pooled"
    pooldir.mkdir(exist_ok=True)
    for (rep, arch), d in sorted(pooled.items()):
        rep_rows.append({
            "rep": rep,
            "arch": arch,
            "n": len(d),
            "accuracy": accuracy_score(d.gold, d.pred),
            "macro_f1": f1_score(d.gold, d.pred, average="macro", zero_division=0),
        })
        _atomic_csv(d, pooldir / f"{arch}__rep{rep}.csv")
    rep_metrics = pd.DataFrame(rep_rows)
    if len(rep_metrics):
        _atomic_csv(rep_metrics, work / "grouped_rep_metrics.csv")

    summary_rows = []
    if len(rep_metrics):
        for arch, g in rep_metrics.groupby("arch"):
            summary_rows.append({
                "arch": arch,
                "n_complete_repeats": len(g),
                "accuracy_mean": g.accuracy.mean(),
                "accuracy_sd": g.accuracy.std(ddof=1) if len(g) > 1 else np.nan,
                "macro_f1_mean": g.macro_f1.mean(),
                "macro_f1_sd": g.macro_f1.std(ddof=1) if len(g) > 1 else np.nan,
            })
    summary = pd.DataFrame(summary_rows)
    if len(summary):
        _atomic_csv(summary, work / "grouped_summary.csv")

    boot = _paired_bootstrap(pooled, args.repeats, args.bootstrap, args.seed + 90000)
    if boot is not None:
        _atomic_json(boot, work / "grouped_bootstrap_full_vs_encoder.json")

    expected_neural = args.repeats * 5 * len(ARCHS)
    completed_neural = len(fold_neural)
    expected_tfidf = args.repeats * 5
    completed_tfidf = len(fold_tfidf)
    completion = {
        "neural_fold_runs_complete": int(completed_neural),
        "neural_fold_runs_expected": int(expected_neural),
        "tfidf_fold_runs_complete": int(completed_tfidf),
        "tfidf_fold_runs_expected": int(expected_tfidf),
        "complete_neural_repeats_full": sum((r, "full") in pooled for r in range(args.repeats)),
        "complete_neural_repeats_encoder": sum((r, "roberta_only") in pooled for r in range(args.repeats)),
        "complete_tfidf_repeats": sum((r, "tfidf") in pooled for r in range(args.repeats)),
    }
    _atomic_json(completion, work / "completion.json")

    print("\nCOMPLETION")
    print(json.dumps(completion, indent=2))
    if len(rep_metrics):
        print("\nPER-REPEAT METRICS")
        print(rep_metrics.to_string(index=False))
    if len(summary):
        print("\nSUMMARY")
        print(summary.to_string(index=False))
    if boot is not None:
        print("\nPAIRED BOOTSTRAP: FULL - ENCODER")
        print(json.dumps(boot, indent=2))
    print(f"\nOutputs: {work}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--corpus", default="corpus/nfr_corpus.csv")
    ap.add_argument("--original-folds", default="corpus/folds.csv")
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260907, help="fold-generation/bootstrap base seed")
    ap.add_argument("--train-seed", type=int, default=42, help="neural training seed, matching main experiment")
    ap.add_argument("--backbone", default="distilroberta-base")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--val-candidates", type=int, default=300)
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--reps", help="comma-separated repeats to run, e.g. 0 or 1,2,3,4; default all")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run-tfidf", action="store_true")
    ap.add_argument("--run-neural", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()

    if not any((args.prepare, args.run_tfidf, args.run_neural, args.score)):
        ap.error("choose at least one action: --prepare, --run-tfidf, --run-neural, --score")
    if args.prepare:
        prepare(args)
    if args.run_tfidf:
        run_tfidf(args)
    if args.run_neural:
        run_neural(args)
    if args.score:
        score(args)


if __name__ == "__main__":
    main()

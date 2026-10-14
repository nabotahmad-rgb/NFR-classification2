#!/usr/bin/env python3
import argparse, json, os, random, re, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

ARCHS = ["full", "roberta_only"]


def log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def make_candidates(n, seed):
    """Matched base search points across architectures; trial 0 is the published default."""
    rng = np.random.default_rng(seed)
    rows = [{
        "trial": 0,
        "lr": 2e-5,
        "batch_size": 16,
        "lambda_con": 0.1,
        "maml_inner_lr": 1e-3,
    }]
    for t in range(1, n):
        rows.append({
            "trial": t,
            "lr": log_uniform(rng, 1e-5, 5e-5),
            "batch_size": int(rng.choice([8, 16])),
            "lambda_con": log_uniform(rng, 0.03, 0.30),
            "maml_inner_lr": log_uniform(rng, 3e-4, 3e-3),
        })
    return rows


def parse_last_metric(text, metric="macroF1"):
    # train.py final line: test acc 37.04%  macroF1 0.1802 ...
    vals = re.findall(r"macroF1\s+([0-9.]+)", text)
    return float(vals[-1]) if vals else np.nan


def run_cmd(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    log_path.write_text(p.stdout, encoding="utf-8")
    return p.returncode, p.stdout, elapsed


def split_outer(corpus, folds, rep, fold, split_dir, seed):
    f = folds[folds.rep == rep]
    test_ids = set(f[f.fold == fold].item_id)
    te = corpus[corpus.item_id.isin(test_ids)].reset_index(drop=True)
    rest = corpus[~corpus.item_id.isin(test_ids)].reset_index(drop=True)
    tr, va = train_test_split(
        rest,
        test_size=0.15,
        stratify=rest.label,
        random_state=seed + fold,
    )
    tr, va = tr.reset_index(drop=True), va.reset_index(drop=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    tr.to_csv(split_dir / "train.csv", index=False)
    va.to_csv(split_dir / "val.csv", index=False)
    te.to_csv(split_dir / "test.csv", index=False)
    return tr, va, te


def train_command(train_py, arch, backbone, seed, epochs, params, tr, va, te, outdir):
    cmd = [
        sys.executable, str(train_py),
        "--variant", arch,
        "--backbone", backbone,
        "--train", str(tr),
        "--val", str(va),
        "--test", str(te),
        "--out", str(outdir),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--batch-size", str(params["batch_size"]),
        "--lr", str(params["lr"]),
    ]
    if arch == "full":
        cmd += [
            "--lambda-con", str(params["lambda_con"]),
            "--maml-inner-lr", str(params["maml_inner_lr"]),
        ]
    return cmd


def locate_prediction(outdir, arch, backbone, seed):
    tag = "MAGNET-NFR" if arch == "full" else f"MAGNET-NFR-{arch}"
    p = outdir / f"{tag}-{backbone}__seed{seed}.csv"
    return p


def paired_bootstrap(full_df, enc_df, n_boot=4000, seed=20260905):
    a = full_df.sort_values("item_id").reset_index(drop=True)
    b = enc_df.sort_values("item_id").reset_index(drop=True)
    if not a.item_id.equals(b.item_id) or not a.gold.equals(b.gold):
        raise RuntimeError("Pooled predictions are not aligned between architectures")
    y = a.gold.to_numpy()
    pf = a.pred.to_numpy()
    pe = b.pred.to_numpy()
    n = len(y)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = (
            f1_score(y[idx], pf[idx], average="macro", zero_division=0)
            - f1_score(y[idx], pe[idx], average="macro", zero_division=0)
        )
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p_two = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {
        "delta_full_minus_encoder": float(
            f1_score(y, pf, average="macro", zero_division=0)
            - f1_score(y, pe, average="macro", zero_division=0)
        ),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "bootstrap_p_two_sided": float(p_two),
        "n_boot": int(n_boot),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/nfr_corpus.csv")
    ap.add_argument("--folds", default="corpus/folds.csv")
    ap.add_argument("--train-py", default="code/train.py")
    ap.add_argument("--work", default="tuning_equal_budget")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--epochs-tune", type=int, default=6)
    ap.add_argument("--epochs-final", type=int, default=10)
    ap.add_argument("--backbone", default="distilroberta-base")
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--outer-folds", default="0,1,2,3,4",
                    help="comma-separated outer folds; use 0 for a smoke test")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_csv(args.corpus)
    folds = pd.read_csv(args.folds)
    train_py = Path(args.train_py)

    print("EQUAL-BUDGET NESTED TUNING")
    print(f"outer rep       : {args.rep}")
    outer_folds = [int(x) for x in args.outer_folds.split(",") if x.strip()]
    if not outer_folds or any(f not in range(5) for f in outer_folds):
        raise SystemExit("--outer-folds must be a comma-separated subset of 0,1,2,3,4")
    print(f"outer folds     : {outer_folds}")
    print(f"trials/arch/fold: {args.trials}")
    print(f"tune epochs     : {args.epochs_tune}")
    print(f"final epochs    : {args.epochs_final}")
    print(f"architectures   : {ARCHS}")
    print("Outer test data are NEVER used during hyperparameter selection.\n")

    # Resume safely from CSV checkpoints if the Colab runtime disconnects.
    tuning_csv = work / "tuning_trials.csv"
    selected_csv = work / "selected_hyperparameters.csv"
    outer_csv = work / "equal_budget_outer_results.csv"
    tuning_rows = pd.read_csv(tuning_csv).to_dict("records") if tuning_csv.exists() else []
    selected_rows = pd.read_csv(selected_csv).to_dict("records") if selected_csv.exists() else []
    outer_rows = pd.read_csv(outer_csv).to_dict("records") if outer_csv.exists() else []
    pooled = {a: [] for a in ARCHS}

    for fold in outer_folds:
        print("=" * 78)
        print(f"OUTER FOLD {fold}")
        split_dir = work / "splits" / f"fold{fold}"
        tr, va, te = split_outer(corpus, folds, args.rep, fold, split_dir, args.seed)
        print(f"sizes: train={len(tr)} val={len(va)} test={len(te)}")
        print("test labels:", te.label.value_counts().to_dict())

        candidates = make_candidates(args.trials, args.seed + 100 * fold)
        pd.DataFrame(candidates).to_csv(work / f"candidates_fold{fold}.csv", index=False)
        final_seed = args.seed + fold

        best_by_arch = {}
        for arch in ARCHS:
            print(f"\n-- TUNING {arch} --")
            arch_rows = []
            for prm in candidates:
                trial = prm["trial"]
                # Resume: reuse a successful completed trial with the same fold/arch/trial.
                prior = [r for r in tuning_rows
                         if int(r.get("outer_rep", -1)) == args.rep
                         and int(r.get("outer_fold", -1)) == fold
                         and r.get("arch") == arch
                         and int(r.get("trial", -1)) == trial
                         and r.get("status") == "ok"
                         and int(r.get("epochs_tune", -1)) == args.epochs_tune]
                if prior:
                    row = prior[-1]
                    arch_rows.append(row)
                    print(f"fold {fold} {arch} trial {trial}: RESUME/SKIP, val macroF1={float(row['val_macro_f1']):.4f}")
                    continue

                trial_dir = work / "tune_predictions" / f"fold{fold}" / arch / f"trial{trial}"
                log_path = work / "logs" / f"fold{fold}_{arch}_trial{trial}.log"
                trial_dir.mkdir(parents=True, exist_ok=True)
                # IMPORTANT: test=validation during tuning, so the outer test is untouched.
                cmd = train_command(
                    train_py, arch, args.backbone, final_seed, args.epochs_tune, prm,
                    split_dir / "train.csv", split_dir / "val.csv", split_dir / "val.csv",
                    trial_dir,
                )
                print(
                    f"fold {fold} {arch} trial {trial}/{args.trials-1}: "
                    f"lr={prm['lr']:.3g} bs={prm['batch_size']} "
                    + (f"lambda={prm['lambda_con']:.3g} maml_lr={prm['maml_inner_lr']:.3g}" if arch == "full" else "")
                )
                rc, out, elapsed = run_cmd(cmd, log_path)
                score = parse_last_metric(out)
                status = "ok" if rc == 0 and np.isfinite(score) else "failed"
                row = {
                    "outer_rep": args.rep, "outer_fold": fold, "arch": arch,
                    **prm, "seed": final_seed, "epochs_tune": args.epochs_tune,
                    "val_macro_f1": score, "seconds": elapsed, "status": status,
                    "log": str(log_path),
                }
                # Drop any stale failed record for this exact key before appending.
                tuning_rows = [r for r in tuning_rows if not (
                    int(r.get("outer_rep", -1)) == args.rep
                    and int(r.get("outer_fold", -1)) == fold
                    and r.get("arch") == arch
                    and int(r.get("trial", -1)) == trial
                    and int(r.get("epochs_tune", -1)) == args.epochs_tune
                )]
                tuning_rows.append(row); arch_rows.append(row)
                pd.DataFrame(tuning_rows).to_csv(tuning_csv, index=False)
                print(f"  -> val macroF1={score:.4f} status={status} time={elapsed/60:.1f} min")

            ok = [r for r in arch_rows if r["status"] == "ok"]
            if not ok:
                raise RuntimeError(f"No successful tuning trials for fold {fold} / {arch}")
            best = max(ok, key=lambda r: (r["val_macro_f1"], -r["trial"]))
            best_by_arch[arch] = best
            selected_rows = [r for r in selected_rows if not (
                int(r.get("outer_rep", -1)) == args.rep
                and int(r.get("outer_fold", -1)) == fold
                and r.get("arch") == arch
            )]
            selected_rows.append(best.copy())
            pd.DataFrame(selected_rows).to_csv(selected_csv, index=False)
            print(f"BEST {arch}: trial {best['trial']} val macroF1={float(best['val_macro_f1']):.4f}")

        # Exactly one touch of the outer test per architecture, after selection.
        for arch in ARCHS:
            best = best_by_arch[arch]
            prm = {k: best[k] for k in ["trial", "lr", "batch_size", "lambda_con", "maml_inner_lr"]}
            outdir = work / "final_predictions" / f"fold{fold}" / arch
            log_path = work / "logs" / f"fold{fold}_{arch}_FINAL.log"
            outdir.mkdir(parents=True, exist_ok=True)
            pred_path = locate_prediction(outdir, arch, args.backbone, final_seed)
            prior_final = [r for r in outer_rows
                           if int(r.get("outer_rep", -1)) == args.rep
                           and int(r.get("outer_fold", -1)) == fold
                           and r.get("arch") == arch
                           and int(r.get("selected_trial", -999)) == int(best["trial"])
                           and pred_path.exists()]
            if prior_final:
                row = prior_final[-1]
                d = pd.read_csv(pred_path)
                pooled[arch].append(d)
                print(f"\nFINAL outer-test run: fold {fold} {arch}: RESUME/SKIP, macroF1={float(row['test_macro_f1']):.4f}")
                continue

            cmd = train_command(
                train_py, arch, args.backbone, final_seed, args.epochs_final, prm,
                split_dir / "train.csv", split_dir / "val.csv", split_dir / "test.csv",
                outdir,
            )
            print(f"\nFINAL outer-test run: fold {fold} {arch}, selected trial {best['trial']}")
            rc, out, elapsed = run_cmd(cmd, log_path)
            score = parse_last_metric(out)
            if rc != 0 or not np.isfinite(score):
                raise RuntimeError(f"Final run failed: fold {fold} {arch}. See {log_path}")
            if not pred_path.exists():
                raise RuntimeError(f"Expected prediction missing: {pred_path}")
            d = pd.read_csv(pred_path)
            pooled[arch].append(d)
            outer_rows = [r for r in outer_rows if not (
                int(r.get("outer_rep", -1)) == args.rep
                and int(r.get("outer_fold", -1)) == fold
                and r.get("arch") == arch
            )]
            outer_rows.append({
                "outer_rep": args.rep, "outer_fold": fold, "arch": arch,
                "selected_trial": int(best["trial"]), "lr": best["lr"],
                "batch_size": int(best["batch_size"]),
                "lambda_con": best["lambda_con"], "maml_inner_lr": best["maml_inner_lr"],
                "val_macro_f1_selected": best["val_macro_f1"],
                "test_macro_f1": score, "seconds": elapsed,
                "prediction": str(pred_path), "log": str(log_path),
            })
            pd.DataFrame(outer_rows).to_csv(outer_csv, index=False)
            print(f"  -> outer-test macroF1={score:.4f} time={elapsed/60:.1f} min")

    # For a partial fold list (smoke test), stop after preserving fold-level outputs.
    if set(outer_folds) != set(range(5)):
        print("\nPARTIAL/SMOKE RUN COMPLETE")
        print(f"Completed outer folds: {outer_folds}")
        print(f"Outputs: {work.resolve()}")
        return

    # Pool one out-of-fold prediction per record for each architecture.
    metrics = []
    pooled_dfs = {}
    for arch in ARCHS:
        d = pd.concat(pooled[arch], ignore_index=True)
        if d.item_id.duplicated().any() or len(d) != len(corpus):
            raise RuntimeError(f"Pooled {arch} does not contain exactly one prediction per corpus item")
        path = work / f"pooled_{arch}.csv"
        d.to_csv(path, index=False)
        pooled_dfs[arch] = d
        metrics.append({
            "arch": arch,
            "n": len(d),
            "accuracy": accuracy_score(d.gold, d.pred),
            "macro_f1": f1_score(d.gold, d.pred, average="macro", zero_division=0),
        })

    pd.DataFrame(metrics).to_csv(work / "equal_budget_metrics.csv", index=False)
    boot = paired_bootstrap(
        pooled_dfs["full"], pooled_dfs["roberta_only"],
        n_boot=args.bootstrap, seed=args.seed + 999,
    )
    (work / "equal_budget_bootstrap.json").write_text(json.dumps(boot, indent=2), encoding="utf-8")

    total_tune = pd.DataFrame(tuning_rows)
    compute = total_tune.groupby("arch").seconds.sum().to_dict()
    summary = {
        "design": "5-fold outer CV, fixed inner validation per outer fold; outer test untouched during tuning",
        "outer_rep": args.rep,
        "trials_per_arch_per_fold": args.trials,
        "total_tuning_trials_per_arch": args.trials * len(outer_folds),
        "epochs_tune": args.epochs_tune,
        "epochs_final": args.epochs_final,
        "search_space": {
            "lr": "log-uniform[1e-5,5e-5]",
            "batch_size": [8, 16],
            "full.lambda_con": "log-uniform[0.03,0.30]",
            "full.maml_inner_lr": "log-uniform[3e-4,3e-3]",
            "trial_0": "published defaults",
        },
        "tuning_seconds_by_arch": compute,
        "metrics": metrics,
        "paired_bootstrap": boot,
    }
    (work / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("EQUAL-BUDGET TUNING COMPLETE")
    print(pd.DataFrame(metrics).to_string(index=False))
    print("Paired bootstrap Full - Encoder:")
    print(json.dumps(boot, indent=2))
    print(f"\nOutputs: {work.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independent Stage09 verification for MAGNET-NFR.

Recomputes the four six-label Full-minus-ablation contrasts from the 20 pooled
ablation CSVs and 5 frozen Full pooled CSVs, using the deterministically
regenerated bootstrap draw arrays bundled with this package. It verifies input
integrity, schema/alignment, point estimates, 90/95/98.75% intervals, raw
bootstrap p-values, Holm adjustment, and the frozen +/-0.01 equivalence rule.

The bootstrap indices in this package are regenerated from the original frozen
RNG specification (default_rng(20260916)); they are not original persisted draw
files. Their numerical equivalence to the frozen analysis is verified here.
"""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd

VARIANTS = ("no_gcn", "no_xl", "no_contrastive", "no_meta")
LABELS = ["Maintainability", "Performance", "Portability", "Reliability", "Security", "Usability"]
L2I = {x: i for i, x in enumerate(LABELS)}
K = len(LABELS)
N = 1719
NBOOT = 4000
TOL = 5e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def encode(values):
    try:
        return np.fromiter((L2I[v] for v in values), dtype=np.int16, count=len(values))
    except KeyError as e:
        raise ValueError(f"Unknown label in predictions: {e.args[0]!r}") from e


def f1s_from_codes(g, p):
    c = np.bincount(g.astype(np.int64) * K + p.astype(np.int64), minlength=K*K).reshape(K, K)
    tp = np.diag(c).astype(float)
    fp = c.sum(0) - tp
    fn = c.sum(1) - tp
    den = 2*tp + fp + fn
    return np.divide(2*tp, den, out=np.zeros(K, dtype=float), where=den > 0)


def macro_f1(gold_code, pred_code):
    return float(f1s_from_codes(gold_code, pred_code).mean())


def holm_adjust(pvals):
    names = list(pvals)
    order = sorted(range(len(names)), key=lambda i: pvals[names[i]])
    adj = [0.0] * len(names)
    running = 0.0
    m = len(names)
    for rank, i in enumerate(order):
        val = min(1.0, (m-rank) * pvals[names[i]])
        running = max(running, val)
        adj[i] = running
    return {names[i]: adj[i] for i in range(m)}


def check(condition, message, failures):
    if condition:
        print(f"PASS  {message}")
    else:
        print(f"FAIL  {message}")
        failures.append(message)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--no-hash-check", action="store_true", help="Skip SHA256SUMS.json verification")
    args = ap.parse_args()
    root = args.root.resolve()
    failures = []

    print("MAGNET-NFR Stage09 independent verification")
    print("Root:", root)
    print()

    # 1. Hash integrity for the frozen recomputation package inputs.
    if not args.no_hash_check:
        manifest_path = root / "SHA256SUMS.json"
        check(manifest_path.exists(), "SHA256 manifest exists", failures)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            bad = []
            missing = []
            for rel, expected in manifest.items():
                p = root / rel
                if not p.exists():
                    missing.append(rel)
                elif sha256(p) != expected:
                    bad.append(rel)
            check(not missing, f"all {len(manifest)} manifest-listed files exist", failures)
            check(not bad, "all manifest-listed files match SHA-256", failures)
            if missing:
                print("  Missing:", missing)
            if bad:
                print("  Hash mismatch:", bad)

    abl_dir = root / "pooled_ablation"
    full_dir = root / "pooled_full_reference"
    summary_path = root / "analysis" / "component_ablation_summary.csv"
    draws_path = root / "bootstrap_draw_indices_regenerated.npz"

    # 2. Expected files.
    abl_files = sorted(abl_dir.glob("*.csv"))
    full_files = sorted(full_dir.glob("*.csv"))
    check(len(abl_files) == 20, f"20 pooled ablation CSVs present (found {len(abl_files)})", failures)
    check(len(full_files) == 5, f"5 frozen Full pooled CSVs present (found {len(full_files)})", failures)
    check(summary_path.exists(), "frozen Stage09 summary exists", failures)
    print(f"  bootstrap draw archive present: {draws_path.exists()}" + ("" if draws_path.exists() else " (will regenerate from seed)"))

    if not summary_path.exists():
        print("\nCannot continue: analysis/component_ablation_summary.csv is missing.")
        print("Re-download the full bundle and retry.")
        return 2

    # 3. Load Full references and establish canonical alignment.
    full = []
    ids = gold = gold_code = None
    full_pred = []
    for rep in range(5):
        p = full_dir / f"MAGNET-NFR-distilroberta-base__seed{rep}.csv"
        check(p.exists(), f"Full repeat {rep} exists", failures)
        if not p.exists():
            continue
        d = pd.read_csv(p).sort_values("item_id").reset_index(drop=True)
        check(list(d.columns) == ["item_id", "gold", "pred"], f"Full repeat {rep} schema item_id,gold,pred", failures)
        check(len(d) == N and d.item_id.nunique() == N, f"Full repeat {rep} has {N} unique rows", failures)
        if ids is None:
            ids = d.item_id.to_numpy()
            gold = d.gold.to_numpy()
            gold_code = encode(gold)
        else:
            check(np.array_equal(d.item_id.to_numpy(), ids), f"Full repeat {rep} item alignment", failures)
            check(np.array_equal(d.gold.to_numpy(), gold), f"Full repeat {rep} gold alignment", failures)
        full.append(d)
        full_pred.append(encode(d.pred.to_numpy()))

    if len(full) != 5 or ids is None:
        print("Cannot continue without all Full references.")
        return 2

    # 4. Load every variant and verify repeat alignment.
    variants = {}
    variant_pred = {}
    for v in VARIANTS:
        ds, ps = [], []
        for rep in range(5):
            p = abl_dir / f"MAGNET-NFR-{v}-distilroberta-base__seed{rep}.csv"
            check(p.exists(), f"{v} repeat {rep} exists", failures)
            if not p.exists():
                continue
            d = pd.read_csv(p).sort_values("item_id").reset_index(drop=True)
            check(list(d.columns) == ["item_id", "gold", "pred"], f"{v} repeat {rep} schema item_id,gold,pred", failures)
            check(len(d) == N and d.item_id.nunique() == N, f"{v} repeat {rep} has {N} unique rows", failures)
            check(np.array_equal(d.item_id.to_numpy(), ids), f"{v} repeat {rep} item alignment", failures)
            check(np.array_equal(d.gold.to_numpy(), gold), f"{v} repeat {rep} gold alignment", failures)
            ds.append(d)
            ps.append(encode(d.pred.to_numpy()))
        variants[v] = ds
        variant_pred[v] = ps

    if any(len(variants[v]) != 5 for v in VARIANTS):
        print("Cannot continue without all 20 ablation pooled files.")
        return 2

    # 5. Verify regenerated draw arrays.
    if draws_path.exists():
        draws = np.load(draws_path)
        check(set(draws.files) == set(VARIANTS),
              "bootstrap archive has exactly the four frozen variants", failures)
    else:
        print("  bootstrap archive absent; regenerating in memory from default_rng(20260916)")
        g = np.random.default_rng(20260916)
        draws = {v: g.integers(0, N, size=(NBOOT, N), dtype=np.int32) for v in VARIANTS}
        check(True, "bootstrap draws regenerated from the frozen RNG specification", failures)
    for v in VARIANTS:
        a = draws[v]
        check(a.shape == (NBOOT, N), f"{v} bootstrap draws shape = {NBOOT} x {N}", failures)
        check(np.issubdtype(a.dtype, np.integer), f"{v} bootstrap draws are integer indices", failures)
        check(int(a.min()) >= 0 and int(a.max()) < N, f"{v} bootstrap indices are in [0,{N-1}]", failures)

    # 6. Recompute all locked statistics.
    frozen = pd.read_csv(summary_path).set_index("variant")
    full_rep = np.array([macro_f1(gold_code, p) for p in full_pred])
    recomputed = []
    raw_p = {}

    for v in VARIANTS:
        vr = np.array([macro_f1(gold_code, p) for p in variant_pred[v]])
        point = float(np.mean(full_rep - vr))
        bd = np.empty(NBOOT, dtype=float)
        for b, ix in enumerate(draws[v]):
            ix = ix.astype(np.int64, copy=False)
            g = gold_code[ix]
            ds = []
            for rep in range(5):
                ds.append(macro_f1(g, full_pred[rep][ix]) - macro_f1(g, variant_pred[v][rep][ix]))
            bd[b] = np.mean(ds)
        lo95, hi95 = np.quantile(bd, [.025, .975])
        lo90, hi90 = np.quantile(bd, [.05, .95])
        loB, hiB = np.quantile(bd, [.00625, .99375])
        left = (np.sum(bd <= 0) + 1) / (NBOOT + 1)
        right = (np.sum(bd >= 0) + 1) / (NBOOT + 1)
        p_raw = float(min(1.0, 2 * min(left, right)))
        raw_p[v] = p_raw
        equival = bool(lo90 > -0.01 and hi90 < 0.01)
        recomputed.append({
            "variant": v,
            "full_macro_f1_mean": float(full_rep.mean()),
            "full_macro_f1_sd": float(full_rep.std(ddof=1)),
            "ablation_macro_f1_mean": float(vr.mean()),
            "ablation_macro_f1_sd": float(vr.std(ddof=1)),
            "delta_macro_f1": point,
            "ci95_lo": float(lo95), "ci95_hi": float(hi95),
            "ci90_lo": float(lo90), "ci90_hi": float(hi90),
            "ci98_75_bonf_lo": float(loB), "ci98_75_bonf_hi": float(hiB),
            "bootstrap_p_raw": p_raw,
            "equivalent_pm_0_01": equival,
        })

    adj = holm_adjust(raw_p)
    for r in recomputed:
        r["bootstrap_p_holm"] = adj[r["variant"]]

    out = pd.DataFrame(recomputed).set_index("variant")
    numeric_cols = [
        "full_macro_f1_mean", "full_macro_f1_sd", "ablation_macro_f1_mean",
        "ablation_macro_f1_sd", "delta_macro_f1", "ci95_lo", "ci95_hi",
        "ci90_lo", "ci90_hi", "ci98_75_bonf_lo", "ci98_75_bonf_hi",
        "bootstrap_p_raw", "bootstrap_p_holm"
    ]

    print("\n=== LOCKED STATISTIC COMPARISON ===")
    max_diff = 0.0
    for v in VARIANTS:
        for c in numeric_cols:
            diff = abs(float(out.loc[v, c]) - float(frozen.loc[v, c]))
            max_diff = max(max_diff, diff)
            check(diff <= TOL, f"{v}: {c} matches frozen summary (|diff|={diff:.3g})", failures)
        check(bool(out.loc[v, "equivalent_pm_0_01"]) == bool(frozen.loc[v, "equivalent_pm_0_01"]),
              f"{v}: equivalence decision matches frozen summary", failures)

    # 7. Save machine-readable verification outputs without changing frozen files.
    analysis_dir = root / "analysis"
    out_csv = analysis_dir / "stage09_recomputed_summary.csv"
    report_json = analysis_dir / "stage09_verify_report.json"
    out.reset_index().to_csv(out_csv, index=False)
    report = {
        "status": "PASS" if not failures else "FAIL",
        "n_failures": len(failures),
        "failures": failures,
        "max_abs_numeric_difference_vs_frozen": max_diff,
        "bootstrap_indices_provenance": "regenerated deterministically from frozen RNG specification; not original saved arrays",
        "bootstrap_seed": 20260916,
        "bootstrap_n": NBOOT,
        "records": N,
        "variants": list(VARIANTS),
        "pooled_ablation_csvs": len(abl_files),
        "pooled_full_reference_csvs": len(full_files),
    }
    report_json.write_text(json.dumps(report, indent=2))

    print("\n=== SUMMARY ===")
    print(out.reset_index().to_string(index=False))
    print(f"\nMaximum absolute numeric difference vs frozen summary: {max_diff:.3g}")
    print("Report:", report_json)
    print("Recomputed summary:", out_csv)
    print("\n" + ("PASS: Stage09 results independently reproduce from the pooled CSVs and bundled regenerated draws."
                   if not failures else f"FAIL: {len(failures)} verification checks failed."))
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())

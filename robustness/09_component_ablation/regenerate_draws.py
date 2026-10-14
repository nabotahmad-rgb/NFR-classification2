#!/usr/bin/env python3
"""Recreate bootstrap_draw_indices_regenerated.npz from the frozen RNG specification.

The archive holds 4 x 4000 x 1719 int32 resampling indices (~42 MB uncompressed),
every value of which is determined by numpy's default_rng(20260916) drawn in the
order no_gcn, no_xl, no_contrastive, no_meta. Shipping it is therefore redundant,
and this package omits it to stay archivable. Run this script to recreate it
byte-for-byte before running verify_stage09.py, or let that script regenerate the
draws in memory automatically.

    python regenerate_draws.py
"""
import numpy as np
from pathlib import Path

SEED, N, NBOOT = 20260916, 1719, 4000
VARIANTS = ("no_gcn", "no_xl", "no_contrastive", "no_meta")


def build():
    g = np.random.default_rng(SEED)
    return {v: g.integers(0, N, size=(NBOOT, N), dtype=np.int32) for v in VARIANTS}


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "bootstrap_draw_indices_regenerated.npz"
    np.savez_compressed(out, **build())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB compressed)")

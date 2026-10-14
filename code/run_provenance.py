#!/usr/bin/env python3
"""
run_provenance.py — recover the documented sub-source composition of the corpus.

The PROMISE/PURE/SecReq aggregation carries a `Dataset_Name` column naming the
collection each requirement came from. This script recovers that breakdown and traces
each sub-collection through deduplication and scope reduction into the released corpus.

The result matters for external validity: the corpus draws on seven documented
collections rather than the two source files suggest, but is dominated by one of them.

USAGE
    python code/run_provenance.py
"""
import argparse
import importlib.util
import os

import pandas as pd


def load_build_corpus(path="code/build_corpus.py"):
    spec = importlib.util.spec_from_file_location("bc", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="sources")
    ap.add_argument("--corpus", default="corpus/nfr_corpus.csv")
    a = ap.parse_args()

    bc = load_build_corpus()
    agg = os.path.join(a.sources, "dcai24_src_dataset.xlsx")
    if not os.path.exists(agg):
        raise SystemExit(f"missing {agg}")

    d = pd.read_excel(agg, sheet_name=0)
    if "Dataset_Name" not in d.columns:
        raise SystemExit("this file carries no Dataset_Name column")

    d["ds"] = d.Dataset_Name.astype(str).str.replace("'", "", regex=False).str.strip()
    d["text"] = d.Requirement.map(bc.norm_text)
    d["label"] = d.Specific_Type.map(bc.norm_label)
    d = d.dropna(subset=["text"])
    nfr = d[d.label.isin(bc.CANON)].copy()
    nfr["fp"] = nfr.text.map(bc.fingerprint)

    corpus = pd.read_csv(a.corpus)
    corpus["fp"] = corpus.text.map(bc.fingerprint)
    keep = set(corpus.fp)

    print("=" * 74)
    print("DOCUMENTED SUB-SOURCES OF THE AGGREGATION")
    print("=" * 74)
    print(f"{'sub-collection':44s} {'rows':>6s} {'NFR':>6s} {'unique':>8s}")
    for s, g in sorted(d.groupby("ds")):
        n = nfr[nfr.ds == s]
        print(f"  {s[:42]:42s} {len(g):6d} {len(n):6d} {n.fp.nunique():8d}")
    print(f"  {'total':42s} {len(d):6d} {len(nfr):6d} {nfr.fp.nunique():8d}")

    print("\n" + "=" * 74)
    print("CONTRIBUTION TO THE RELEASED CORPUS")
    print("=" * 74)
    ded = nfr.drop_duplicates(subset=["fp"], keep="first")
    rows, tot = [], 0
    for s, g in ded.groupby("ds"):
        k = len(set(g.fp) & keep)
        tot += k
        rows.append((s, k))
    rows.append(("Curated Mendeley collection", len(corpus) - tot))
    rows.sort(key=lambda x: -x[1])
    for s, k in rows:
        print(f"  {s[:44]:44s} {k:5d}  ({100 * k / len(corpus):5.1f}%)")
    print(f"  {'total':44s} {len(corpus):5d}  (100.0%)")

    top = rows[0]
    print(f"\n  {len(rows)} documented collections contribute.")
    print(f"  The largest, {top[0]}, supplies {100 * top[1] / len(corpus):.1f}% of records.")
    print("  The corpus is therefore multi-source but concentrated, and results should")
    print("  be read as pertaining to that mixture rather than to any single collection.")


if __name__ == "__main__":
    main()

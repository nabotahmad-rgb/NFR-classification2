#!/usr/bin/env python3
"""
build_corpus.py v2 — Phase 1. Rebuild the NFR corpus from source, honestly.

v2 FIX: label columns are chosen BY EVIDENCE, not by name.

Several of these source files carry two label-ish columns: a binary
Functional/Non-Functional flag, and the actual NFR sub-category. v1 matched on column
name, grabbed the binary flag, and threw away every NFR in the file (6,086 of 6,117
"functional"). v2 scores every candidate column by how many of its values actually map
to an ISO/IEC 25010 category, and keeps the winner.

Reads ONLY original source spreadsheets. Derived files are skipped by name.

  INSPECT   python build_corpus.py --base . --inspect
  BUILD     python build_corpus.py --base . --build --min-class 40

OUTPUT
  corpus/nfr_corpus.csv      item_id,text,label,source
  corpus/provenance.csv      per-source contribution, and which columns were used
  corpus/splits/{train,val,test}.csv
  corpus/REPORT.txt          Table 2 + section 3.1 material
"""
import argparse, glob, hashlib, os, re, sys
import pandas as pd

CANON = ['Performance', 'Security', 'Usability', 'Reliability', 'Maintainability', 'Portability']

LABEL_MAP = {
    'performance': 'Performance', 'pe': 'Performance', 'performance efficiency': 'Performance',
    'efficiency': 'Performance', 'time behaviour': 'Performance', 'speed': 'Performance',
    'security': 'Security', 'se': 'Security', 'sec': 'Security', 'privacy': 'Security',
    'usability': 'Usability', 'us': 'Usability', 'ui': 'Usability', 'look and feel': 'Usability',
    'lf': 'Usability', 'operability': 'Usability', 'accessibility': 'Usability',
    'reliability': 'Reliability', 'rel': 'Reliability', 'availability': 'Reliability',
    'a': 'Reliability', 'fault tolerance': 'Reliability', 'ft': 'Reliability',
    'maintainability': 'Maintainability', 'mn': 'Maintainability', 'maintenance': 'Maintainability',
    'ma': 'Maintainability', 'modifiability': 'Maintainability',
    'portability': 'Portability', 'po': 'Portability', 'pt': 'Portability',
    'compatibility': 'Portability', 'adaptability': 'Portability', 'installability': 'Portability',
}
NON_NFR = {'f', 'functional', 'fr', 'functional requirement', 'functional requirements',
           'nfr', 'non-functional', 'non functional', 'nonfunctional', 'nf',
           'non-functional requirement', 'other', 'others', 'na', 'n/a', 'none', '-'}


def norm_label(x):
    if pd.isna(x):
        return None
    k = re.sub(r'\s+', ' ', str(x).strip().lower())
    if k in NON_NFR:
        return 'FUNCTIONAL'
    return LABEL_MAP.get(k)


def norm_text(x):
    if pd.isna(x):
        return None
    t = re.sub(r'\s+', ' ', str(x)).strip()
    return t if len(t) >= 10 else None


def fingerprint(t):
    return hashlib.md5(re.sub(r'[^a-z0-9]', '', t.lower()).encode()).hexdigest()


def find_sources(base):
    out, seen = [], set()
    for p in ('**/*.xlsx', '**/*.xls', '**/*promise*.csv', '**/*PROMISE*.csv'):
        for f in glob.glob(os.path.join(base, p), recursive=True):
            b = os.path.basename(f)
            if any(s in b.lower() for s in ('nfr_dataset_2847', 'magnet_nfr_real', 'corpus',
                                            '~$')):
                continue
            if b in seen:
                continue
            seen.add(b); out.append(f)
    return sorted(out)


def read_any(fp):
    if fp.endswith(('.xlsx', '.xls')):
        return pd.read_excel(fp, sheet_name=None)
    return {'csv': pd.read_csv(fp)}


def score_label_col(s):
    """Count values in this column that map to a real ISO 25010 category."""
    try:
        return int(s.map(norm_label).isin(CANON).sum())
    except Exception:
        return 0


def pick_label_col(df, override=None):
    """Choose by evidence: most canonical NFR labels wins."""
    if override and override in df.columns:
        return override, score_label_col(df[override])
    best, best_n = None, 0
    for c in df.columns:
        try:
            if df[c].dtype == object or df[c].nunique(dropna=True) < 40:
                n = score_label_col(df[c])
                if n > best_n:
                    best, best_n = c, n
        except Exception:
            continue
    return best, best_n


def pick_text_col(df, override=None):
    if override and override in df.columns:
        return override
    named = ['text', 'requirement', 'requirement_text', 'requirementtext', 'description',
             'sentence', 'statement', 'req', 'requirements']
    low = {str(c).lower().strip(): c for c in df.columns}
    for n in named:
        if n in low:
            s = df[low[n]].dropna().astype(str)
            if len(s) and s.str.len().mean() > 20:
                return low[n]
    best, bl = None, 0
    for c in df.columns:
        s = df[c].dropna().astype(str)
        if len(s) and s.str.len().mean() > bl:
            bl, best = s.str.len().mean(), c
    return best if bl > 25 else None


def choose_sheets(sheets):
    """
    Multi-sheet workbooks repeat rows: a Complete_Dataset sheet plus per-category
    sheets. Reading every sheet inflates the counts. Prefer one complete sheet.
    """
    names = list(sheets)
    if len(names) == 1:
        return sheets, []
    for pat in ('complete', 'full', 'all_', 'dataset'):
        hits = [n for n in names if pat in n.lower()]
        if hits:
            pick = max(hits, key=lambda n: len(sheets[n]))
            return {pick: sheets[pick]}, [n for n in names if n != pick]
    return sheets, []


def inspect(base):
    srcs = find_sources(base)
    if not srcs:
        sys.exit(f"No source spreadsheets found under {base}")
    print("=" * 74); print("SOURCE INSPECTION"); print("=" * 74)
    for fp in srcs:
        print(f"\n### {os.path.basename(fp)}  ({os.path.getsize(fp):,} bytes)")
        try:
            sheets = read_any(fp)
        except Exception as e:
            print(f"    ERROR: {e}"); continue
        for name, df in sheets.items():
            print(f"\n  sheet '{name}': {len(df)} rows")
            tc = pick_text_col(df)
            lc, n = pick_label_col(df)
            print(f"    -> text  : {tc!r}")
            print(f"    -> label : {lc!r}   ({n} rows map to ISO 25010 categories)")
            for c in df.columns:
                try:
                    u = df[c].nunique(dropna=True)
                    sc = score_label_col(df[c]) if (df[c].dtype == object or u < 40) else 0
                except Exception:
                    continue
                tag = f"   [{sc} NFR]" if sc else ""
                if u <= 20 and u:
                    print(f"      {c!r}: {u} distinct -> {list(df[c].dropna().unique()[:20])}{tag}")
                else:
                    ex = df[c].dropna().astype(str).iloc[0][:60] if len(df[c].dropna()) else ''
                    print(f"      {c!r}: {u} distinct, e.g. {ex!r}{tag}")
    print("\n" + "=" * 74)
    print("Next: python build_corpus.py --base . --build --min-class 40")


def build(base, args):
    srcs = find_sources(base)
    frames, prov = [], []
    print("=" * 74); print("READING SOURCES"); print("=" * 74)

    for fp in srcs:
        tag = re.sub(r'\.(xlsx|xls|csv)$', '', os.path.basename(fp))[:38]
        try:
            sheets = read_any(fp)
        except Exception as e:
            print(f"  SKIP {tag}: {e}"); continue

        use, skipped = choose_sheets(sheets)
        if skipped:
            print(f"\n  {tag}: using sheet {list(use)[0]!r}; ignoring {len(skipped)} "
                  f"redundant sheet(s)")

        for sname, df in use.items():
            tc = pick_text_col(df, args.text_col)
            lc, nmap = pick_label_col(df, args.label_col)
            if not tc or not lc or nmap == 0:
                print(f"  SKIP {tag}[{sname}]: text={tc!r} label={lc!r} "
                      f"({nmap} NFR labels) — nothing usable")
                continue

            d = pd.DataFrame({'text': df[tc].map(norm_text), 'raw': df[lc], 'source': tag})
            d['label'] = d.raw.map(norm_label)
            n_raw = len(d)
            d = d.dropna(subset=['text'])
            n_func = int((d.label == 'FUNCTIONAL').sum())
            unmapped = d[d.label.isna()].raw.dropna().astype(str)
            n_un = len(unmapped)
            d = d[d.label.isin(CANON)]

            print(f"\n  {tag}[{sname}]: {n_raw} rows -> {len(d)} NFR")
            print(f"      text={tc!r}   label={lc!r}")
            print(f"      dropped {n_func} functional, {n_un} unmapped")
            if n_un:
                print(f"      unmapped values: {dict(unmapped.value_counts().head(8))}")
                print(f"      -> if any are real NFR categories, add them to LABEL_MAP")
            prov.append({'source': tag, 'sheet': sname, 'text_col': tc, 'label_col': lc,
                         'raw_rows': n_raw, 'nfr_rows': len(d),
                         'functional_dropped': n_func, 'unmapped_dropped': n_un})
            if len(d):
                frames.append(d[['text', 'label', 'source']])

    if not frames:
        print("\n" + "=" * 74)
        print("NO USABLE ROWS.")
        print("Run:  python build_corpus.py --base . --inspect")
        print("Send me that output and I will map the columns explicitly.")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    print(f"\n  pooled before dedup: {len(df)}")

    print("\n" + "=" * 74); print("DEDUPLICATION"); print("=" * 74)
    df['fp'] = df.text.map(fingerprint)
    seen_src = list(df.source.unique())
    overlap_rows = []
    for a in seen_src:
        mine = set(df[df.source == a].fp)
        for b in seen_src:
            if a == b:
                continue
            ov = len(mine & set(df[df.source == b].fp))
            overlap_rows.append((a, b, ov, len(mine), 100 * ov / len(mine) if mine else 0))
    if overlap_rows:
        print("\n  cross-source overlap (of A's records, how many also appear in B):")
        for a, b, ov, tot, pct in sorted(overlap_rows, key=lambda x: -x[4]):
            flag = "   <-- NOT AN INDEPENDENT SOURCE" if pct > 50 else ""
            print(f"    {a[:30]:30s} in {b[:30]:30s}: {ov:5d}/{tot:5d} ({pct:5.1f}%){flag}")

    n_conf = int((df.groupby('fp').label.nunique() > 1).sum())
    before = len(df)
    df = df.drop_duplicates(subset=['fp'], keep='first').reset_index(drop=True)
    print(f"\n  {before} -> {len(df)} after dedup ({before-len(df)} removed)")
    if n_conf:
        print(f"  !! {n_conf} identical texts carry CONFLICTING labels across sources.")
        print(f"     Kept first. This is cross-corpus label inconsistency — a real")
        print(f"     construct-validity threat for section 7.1.1, and one nobody working")
        print(f"     with aggregated NFR corpora has reported.")

    df['item_id'] = [f'NFR_{i+1:05d}' for i in range(len(df))]
    df = df[['item_id', 'text', 'label', 'source']]

    print("\n" + "=" * 74); print("CLASS DISTRIBUTION  (this is Table 2)"); print("=" * 74)
    vc = df.label.value_counts()
    for c in CANON:
        n = int(vc.get(c, 0))
        mark = '' if n >= args.min_class else f'   <-- below --min-class {args.min_class}'
        print(f"    {c:18s} {n:5d}   {100*n/len(df):5.1f}%{mark}")
    floor = vc.max() / len(df)
    print(f"\n    TOTAL {len(df)}")
    print(f"    majority-class floor: {floor*100:.2f}%  (always-predict-{vc.idxmax()})")
    print(f"    >> every accuracy is measured against this, not against zero")

    keep = [c for c in CANON if vc.get(c, 0) >= args.min_class]
    drop = [c for c in CANON if 0 < vc.get(c, 0) < args.min_class]
    print("\n" + "=" * 74); print("SCOPE DECISION"); print("=" * 74)
    for c in drop:
        print(f"    DROP {c} (n={int(vc.get(c,0))}) — cannot support a per-class F1")
    if drop:
        print(f'\n    Section 3.1: "we evaluate {len(keep)} categories; '
              f'{", ".join(drop)} lacked')
        print(f'    the support for a per-category F1 with a meaningful confidence interval."')
    df = df[df.label.isin(keep)].reset_index(drop=True)
    print(f"\n    evaluating {len(keep)} classes: {keep}")
    print(f"    corpus: {len(df)} records")

    from sklearn.model_selection import train_test_split
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=42)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.label, random_state=42)

    os.makedirs('corpus/splits', exist_ok=True)
    df.to_csv('corpus/nfr_corpus.csv', index=False)
    pd.DataFrame(prov).to_csv('corpus/provenance.csv', index=False)
    tr.to_csv('corpus/splits/train.csv', index=False)
    va.to_csv('corpus/splits/val.csv', index=False)
    te.to_csv('corpus/splits/test.csv', index=False)

    print("\n" + "=" * 74); print("SPLITS"); print("=" * 74)
    print(f"    train {len(tr)}  val {len(va)}  test {len(te)}")
    assert not (set(tr.item_id) & set(te.item_id)), "LEAKAGE: item_id"
    assert not (set(tr.text) & set(te.text)), "LEAKAGE: text"
    print("    [PASS] no train/test overlap on item_id or text")
    print("\n    test support per class (need >=20 for a usable per-class F1):")
    for c, n in te.label.value_counts().items():
        print(f"      {c:18s} {n:4d}  {'ok' if n >= 20 else '<-- TOO FEW'}")

    with open('corpus/REPORT.txt', 'w') as f:
        f.write(f"corpus: {len(df)} records, {len(keep)} classes\n")
        f.write(f"classes kept: {keep}\ndropped: {drop}\n")
        f.write(f"majority floor: {floor*100:.2f}% ({vc.idxmax()})\n")
        f.write(f"splits: train {len(tr)} val {len(va)} test {len(te)}\n")
        f.write(f"conflicting-label texts across sources: {n_conf}\n")
        f.write(f"cv: {args.cv}-fold x {args.repeats} repeats\n\n")
        f.write("distribution:\n" + df.label.value_counts().to_string())
        f.write("\n\ntest-set support:\n" + te.label.value_counts().to_string())
        f.write("\n\nper-source:\n" + pd.DataFrame(prov).to_string())
        f.write("\n\ncross-source overlap:\n")
        for a, b, ov, tot, pct in sorted(overlap_rows, key=lambda x: -x[4]):
            f.write(f"  {a} in {b}: {ov}/{tot} ({pct:.1f}%)\n")

    # ---- stratified k-fold, repeated ------------------------------------
    from sklearn.model_selection import StratifiedKFold
    print("\n" + "=" * 74); print("CROSS-VALIDATION FOLDS"); print("=" * 74)
    print(f"    A 15% holdout leaves minority classes with single-digit test support.")
    print(f"    k-fold gives every record exactly one out-of-fold prediction instead.")
    rows = []
    for rep in range(args.repeats):
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=100 + rep)
        for k, (_, te_idx) in enumerate(skf.split(df, df.label)):
            for i in te_idx:
                rows.append({'item_id': df.item_id.iloc[i], 'rep': rep, 'fold': k})
    folds = pd.DataFrame(rows)
    folds.to_csv('corpus/folds.csv', index=False)
    print(f"\n    {args.cv}-fold x {args.repeats} repeats -> {args.cv*args.repeats} "
          f"fold-level evaluations per config")
    print(f"    corpus/folds.csv: {len(folds)} assignments")
    print(f"\n    out-of-fold test support per class (pooled over one repeat):")
    for c, n in df.label.value_counts().items():
        print(f"      {c:18s} {n:4d}  {'ok' if n >= 20 else '<-- still too few'}")
    print(f"\n    vs the holdout, which gave:")
    for c, n in te.label.value_counts().items():
        print(f"      {c:18s} {n:4d}")

    print("\n" + "=" * 74)
    print("Written: corpus/nfr_corpus.csv, corpus/provenance.csv, corpus/splits/, corpus/REPORT.txt")
    print("Send me corpus/REPORT.txt")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='.')
    ap.add_argument('--inspect', action='store_true')
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--min-class', type=int, default=40)
    ap.add_argument('--cv', type=int, default=5, help='k for stratified k-fold')
    ap.add_argument('--repeats', type=int, default=2, help='fold-assignment repeats')
    ap.add_argument('--text-col')
    ap.add_argument('--label-col')
    a = ap.parse_args()
    if a.inspect:
        inspect(a.base)
    elif a.build:
        build(a.base, a)
    else:
        ap.print_help()

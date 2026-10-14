#!/usr/bin/env python3
"""
pool_folds.py — combine per-fold predictions into out-of-fold prediction sets.

train.py in CV mode writes one file per (config, rep, fold):
    preds/<CONFIG>__rep0fold0.csv ... __rep0fold4.csv

Each record is predicted exactly once per repeat, by a model that never saw it.
This pools each repeat into a single file evaluate.py understands:

    preds_pooled/<CONFIG>__seed0.csv      (rep 0, all folds)
    preds_pooled/<CONFIG>__seed1.csv      (rep 1, all folds)

The "seed" naming is for evaluate.py's parser. Each is an independent out-of-fold
evaluation over the FULL corpus — which is what fixes the minority-class support
problem a 15% holdout creates.

USAGE
    python pool_folds.py --preds preds/ --out preds_pooled/
    python evaluate.py --preds preds_pooled/ --ref MAGNET-NFR-distilroberta-base --out results/
"""
import argparse, glob, os, re, sys
from collections import defaultdict
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', required=True)
    ap.add_argument('--out', default='preds_pooled')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    groups = defaultdict(list)
    for fp in sorted(glob.glob(os.path.join(a.preds, '*__rep*fold*.csv'))):
        m = re.match(r'(.+)__rep(\d+)fold(\d+)\.csv$', os.path.basename(fp))
        if not m:
            continue
        groups[(m.group(1), int(m.group(2)))].append((int(m.group(3)), fp))

    if not groups:
        sys.exit(f'No *__rep*fold*.csv in {a.preds}. '
                 f'(Holdout-mode files need no pooling — point evaluate.py at them directly.)')

    configs = sorted({c for c, _ in groups})
    print(f'{len(groups)} (config, repeat) groups across {len(configs)} configs\n')

    sizes, ok = {}, True
    for (cfg, rep), files in sorted(groups.items()):
        dfs = [pd.read_csv(f) for _, f in sorted(files)]
        pooled = pd.concat(dfs, ignore_index=True).sort_values('item_id').reset_index(drop=True)

        dup = pooled.item_id.duplicated().sum()
        if dup:
            ok = False
            print(f'  [FAIL] {cfg} rep{rep}: {dup} items predicted more than once '
                  f'-> folds overlap, the CV split is broken')

        fp = os.path.join(a.out, f'{cfg}__seed{rep}.csv')
        pooled.to_csv(fp, index=False)
        sizes[(cfg, rep)] = len(pooled)
        acc = (pooled.gold == pooled.pred).mean()
        print(f'  {cfg}  rep{rep}: {len(files)} folds -> {len(pooled)} predictions  '
              f'acc {acc*100:.2f}%  -> {os.path.basename(fp)}')

    # every config must cover the same items, or McNemar is invalid
    print()
    uniq = set(sizes.values())
    if len(uniq) == 1:
        print(f'  [PASS] every pooled set covers {uniq.pop()} records')
    else:
        ok = False
        print(f'  [FAIL] pooled sets differ in size: {sorted(uniq)}')
        print(f'         evaluate.py will reject this — configs must share one item set')

    d0 = pd.read_csv(sorted(glob.glob(os.path.join(a.out, '*__seed*.csv')))[0])
    print(f'\n  out-of-fold test support per class:')
    for c, n in d0.gold.value_counts().items():
        print(f'    {c:18s} {n:5d}  {"ok" if n >= 20 else "<-- still too few"}')
    floor = d0.gold.value_counts(normalize=True).max()
    print(f'\n  majority-class floor: {floor*100:.2f}%')

    print(f'\n{"Next:" if ok else "Fix the failures above, then:"}')
    print(f'  python evaluate.py --preds {a.out} --ref MAGNET-NFR-distilroberta-base --out results/')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

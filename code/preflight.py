#!/usr/bin/env python3
"""
preflight.py — run this BEFORE the ablations.

Checks the things that, if wrong, make every subsequent run worthless.
Takes seconds. Saves an hour.

USAGE
  # if you have split files:
  python preflight.py --train splits/train.csv --val splits/val.csv --test splits/test.csv

  # if you only have the corpus + existing predictions:
  python preflight.py --corpus nfr_real_v2_full.csv --preds run_final_small/preds/
"""
import argparse, glob, os, re, sys
import pandas as pd

OK   = "[PASS]"
BAD  = "[FAIL]"
WARN = "[WARN]"
fail = 0


def say(tag, msg):
    global fail
    if tag == BAD:
        fail += 1
    print(f"  {tag} {msg}")


def check_splits(tr_p, va_p, te_p):
    print("\n--- SPLIT INTEGRITY ---")
    tr, va, te = (pd.read_csv(p) for p in (tr_p, va_p, te_p))

    for nm, df in (("train", tr), ("val", va), ("test", te)):
        need = {"item_id", "label"} - set(df.columns)
        if need:
            say(BAD, f"{nm}: missing columns {need}")
            return
    n = len(tr) + len(va) + len(te)
    print(f"        train {len(tr)}  val {len(va)}  test {len(te)}  total {n}")
    print(f"        ratio {len(tr)/n:.0%}/{len(va)/n:.0%}/{len(te)/n:.0%}")

    # THE check
    for a, b, an, bn in ((tr, te, "train", "test"), (tr, va, "train", "val"), (va, te, "val", "test")):
        ov = set(a.item_id) & set(b.item_id)
        say(OK if not ov else BAD,
            f"{an}/{bn} item_id overlap: {len(ov)}" + ("" if not ov else "  <-- LEAKAGE, fix before running"))

    # duplicate text across splits (leakage that item_id misses)
    if "text" in tr.columns and "text" in te.columns:
        dup = set(tr.text.astype(str).str.strip()) & set(te.text.astype(str).str.strip())
        say(OK if not dup else BAD,
            f"train/test duplicate requirement TEXT: {len(dup)}" +
            ("" if not dup else "  <-- same text, different id = still leakage"))

    # stratification + viability
    print("\n--- CLASS VIABILITY (test split) ---")
    vc = te.label.value_counts()
    floor = vc.max() / len(te)
    for c, k in vc.items():
        tag = OK if k >= 20 else BAD
        note = "" if k >= 20 else "  <-- n<20: F1 CI spans most of [0,1]. Exclude the class or the study."
        say(tag, f"{c:18s} n={k:5d}  ({100*k/len(te):5.1f}%){note}")
    print(f"\n        majority-class floor = {floor*100:.2f}%  (always-predict-{vc.idxmax()})")
    print(f"        any accuracy you report is read against THIS, not against zero.")

    # train must contain every test class
    missing = set(te.label) - set(tr.label)
    say(OK if not missing else BAD, f"classes in test but not train: {sorted(missing) or 'none'}")


def check_preds(preds_dir, corpus_p):
    print("\n--- EXISTING PREDICTIONS ---")
    files = sorted(glob.glob(os.path.join(preds_dir, "*__seed*.csv")))
    if not files:
        say(BAD, f"no *__seed*.csv in {preds_dir}")
        return
    cfgs = sorted({re.match(r"(.+)__seed\d+\.csv$", os.path.basename(f)).group(1) for f in files})
    print(f"        {len(files)} files, {len(cfgs)} configs:")
    for c in cfgs:
        seeds = sorted(int(re.search(r"seed(\d+)", f).group(1))
                       for f in files if os.path.basename(f).startswith(c + "__seed"))
        print(f"          {c}  ({len(seeds)} seeds)")

    d = pd.read_csv(files[0])
    n_eval = len(d)
    print(f"\n        evaluation set size: {n_eval}")

    if corpus_p and os.path.exists(corpus_p):
        corpus = pd.read_csv(corpus_p)
        pct = 100 * n_eval / len(corpus)
        print(f"        corpus size        : {len(corpus)}")
        print(f"        predictions cover  : {pct:.1f}% of corpus")
        if pct > 50:
            say(BAD, f"predictions cover {pct:.0f}% of the corpus")
            print(f"         -> A 70/15/15 holdout would be ~{len(corpus)*0.15:.0f} rows, not {n_eval}.")
            print(f"         -> Either these are out-of-fold CV predictions (legitimate — say so")
            print(f"            in the paper and ship the fold map), or the model was scored on")
            print(f"            data it trained on (fatal — every number is void).")
            print(f"         -> RESOLVE THIS BEFORE RUNNING THE ABLATIONS.")
        else:
            say(OK, f"evaluation set is {pct:.0f}% of corpus — consistent with a holdout")

    # what's still missing for Table 10
    print("\n--- ABLATION COVERAGE (Table 10 needs 6 rows) ---")
    want = {"full": None, "roberta_only": None, "no_gcn": None,
            "no_xl": None, "no_contrastive": None, "no_meta": None}
    for c in cfgs:
        for k in want:
            if k == "full":
                if not any(v in c for v in ("roberta_only", "no_gcn", "no_xl", "no_contrastive", "no_meta")):
                    want["full"] = c
            elif k in c:
                want[k] = c
    for k, v in want.items():
        say(OK if v else WARN, f"{k:16s} {'-> ' + v if v else 'MISSING — needs a run'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train"); ap.add_argument("--val"); ap.add_argument("--test")
    ap.add_argument("--corpus"); ap.add_argument("--preds")
    a = ap.parse_args()

    print("=" * 68)
    print("PREFLIGHT")
    print("=" * 68)

    if a.train and a.val and a.test:
        check_splits(a.train, a.val, a.test)
    else:
        say(WARN, "no split files given — cannot verify leakage")
        print("         If your pipeline does not emit train/val/test files, that is")
        print("         itself the problem: you cannot demonstrate the absence of")
        print("         leakage, and neither can a reviewer. Make it emit them.")

    if a.preds:
        check_preds(a.preds, a.corpus)

    print("\n" + "=" * 68)
    if fail:
        print(f"{fail} BLOCKING ISSUE(S). Do not start the ablations.")
        sys.exit(1)
    print("Preflight clean. Proceed to Step 1.")


if __name__ == "__main__":
    main()

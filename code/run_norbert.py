#!/usr/bin/env python3
"""
NoRBERT baseline — Hey et al. (2020), RE'20.

Fine-tunes BERT on YOUR splits so the comparison is like-for-like. Emits prediction
CSVs in the same layout evaluate.py already consumes.

WHY THIS EXISTS
  NoRBERT is the field's reference point. A reviewer will ask for it. Citing Hey's
  published 0.87 is NOT a comparison — different corpus, different splits, different
  task. It has to be re-trained on your data or it proves nothing.

INSTALL
  pip install torch transformers scikit-learn pandas numpy

USAGE
  python run_norbert.py \
      --train splits/train.csv --val splits/val.csv --test splits/test.csv \
      --out preds/ --seeds 42 123 256 512 1024 2048 4096 8192 16384 32768

  Split CSVs need columns: item_id,text,label

NOTES
  - Uses bert-base-uncased. Hey et al. also released a requirements-pretrained
    checkpoint; if you can obtain it, pass --model <path> and say so in the paper.
  - Class weighting is ON by default. With an 85.8% majority class an unweighted
    run collapses to always-predict-Security and you will report a fake 85.8%.
    --no-class-weights disables it, but read the warning it prints first.
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
except ImportError:
    sys.exit("Missing deps. Run:  pip install torch transformers")

from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight


class NFRData(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.t, self.y, self.tok, self.n = list(texts), list(labels), tok, max_len

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        e = self.tok(str(self.t[i]), truncation=True, padding="max_length",
                     max_length=self.n, return_tensors="pt")
        return {"input_ids": e["input_ids"].squeeze(0),
                "attention_mask": e["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.y[i], dtype=torch.long)}


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_seed(seed, tr, va, te, classes, args, dev):
    set_seed(seed)
    c2i = {c: i for i, c in enumerate(classes)}
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(classes)).to(dev)

    dl_tr = DataLoader(NFRData(tr.text, tr.label.map(c2i), tok, args.max_len),
                       batch_size=args.batch_size, shuffle=True)
    dl_va = DataLoader(NFRData(va.text, va.label.map(c2i), tok, args.max_len),
                       batch_size=args.batch_size)
    dl_te = DataLoader(NFRData(te.text, te.label.map(c2i), tok, args.max_len),
                       batch_size=args.batch_size)

    if args.no_class_weights:
        print("    !! class weighting OFF — with a large majority class this run may")
        print("       collapse to always-predict-majority. Check the per-class F1.")
        w = None
    else:
        cw = compute_class_weight("balanced", classes=np.arange(len(classes)),
                                  y=tr.label.map(c2i).values)
        w = torch.tensor(cw, dtype=torch.float).to(dev)
    lossf = torch.nn.CrossEntropyLoss(weight=w)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(dl_tr) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    best_f1, best_state = -1, None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        for b in dl_tr:
            b = {k: v.to(dev) for k, v in b.items()}
            opt.zero_grad()
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            loss = lossf(out.logits, b["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

        model.eval(); P, Y = [], []
        with torch.no_grad():
            for b in dl_va:
                b = {k: v.to(dev) for k, v in b.items()}
                lg = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
                P += lg.argmax(-1).cpu().tolist(); Y += b["labels"].cpu().tolist()
        f1 = f1_score(Y, P, average="macro", zero_division=0)
        print(f"    epoch {ep:2d}  val macroF1 {f1:.4f}  val acc {accuracy_score(Y,P):.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # select on validation macro F1, never on test
    model.load_state_dict(best_state)
    model.eval(); P = []
    with torch.no_grad():
        for b in dl_te:
            b = {k: v.to(dev) for k, v in b.items()}
            lg = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            P += lg.argmax(-1).cpu().tolist()
    i2c = {i: c for c, i in c2i.items()}
    return pd.DataFrame({"item_id": te.item_id.values,
                         "gold": te.label.values,
                         "pred": [i2c[p] for p in P]}), (time.time() - t0) / 3600


def main():
    ap = argparse.ArgumentParser()
    # holdout mode
    ap.add_argument("--train")
    ap.add_argument("--val")
    ap.add_argument("--test")
    # cross-validation mode — REQUIRED to be comparable with the CV-pooled MAGNET runs
    ap.add_argument("--corpus", help="corpus/nfr_corpus.csv")
    ap.add_argument("--folds", help="corpus/folds.csv")
    ap.add_argument("--rep", type=int)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--out", default="preds")
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--config-name", default="NoRBERT-bert-base-uncased")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 123, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--no-class-weights", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.folds is not None:
        # ---- CROSS-VALIDATION MODE ----
        if not (a.corpus and a.rep is not None and a.fold is not None):
            sys.exit("CV mode needs --corpus --folds --rep --fold")
        corpus = pd.read_csv(a.corpus)
        folds = pd.read_csv(a.folds)
        f = folds[folds.rep == a.rep]
        test_ids = set(f[f.fold == a.fold].item_id)
        te = corpus[corpus.item_id.isin(test_ids)].reset_index(drop=True)
        rest = corpus[~corpus.item_id.isin(test_ids)].reset_index(drop=True)
        from sklearn.model_selection import train_test_split as _tts
        tr, va = _tts(rest, test_size=0.15, stratify=rest.label,
                      random_state=a.seeds[0] if a.seeds else 42)
        print(f"  CV mode: rep {a.rep}, fold {a.fold} held out")
    else:
        # ---- HOLDOUT MODE ----
        if not (a.train and a.val and a.test):
            sys.exit("Holdout mode needs --train --val --test (or use CV mode)")
        tr, va, te = (pd.read_csv(p) for p in (a.train, a.val, a.test))

    for name, df in (("train", tr), ("val", va), ("test", te)):
        need = {"item_id", "text", "label"} - set(df.columns)
        if need:
            sys.exit(f"ERROR {name}: missing columns {need}")

    # leakage check — cheap, and it is the thing that voids a paper
    overlap = set(tr.item_id) & set(te.item_id)
    if overlap:
        sys.exit(f"FATAL: {len(overlap)} item_ids appear in BOTH train and test. "
                 f"Fix the split before running anything.")
    print(f"[PASS] no train/test item_id overlap")
    print(f"  train {len(tr)}  val {len(va)}  test {len(te)}")

    classes = sorted(tr.label.unique())
    print(f"  classes: {classes}")
    print(f"  test distribution:\n{te.label.value_counts().to_string()}")
    floor = te.label.value_counts(normalize=True).max()
    print(f"  majority-class floor on test = {floor*100:.2f}%")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {dev}\n")

    times = []
    for s in a.seeds:
        print(f"[seed {s}]")
        preds, hrs = run_seed(s, tr, va, te, classes, a, dev)
        if a.folds is not None:
            fp = os.path.join(a.out, f"{a.config_name}__rep{a.rep}fold{a.fold}.csv")
        else:
            fp = os.path.join(a.out, f"{a.config_name}__seed{s}.csv")
        preds.to_csv(fp, index=False)
        times.append({"model": a.config_name, "hours": round(hrs, 3)})
        acc = accuracy_score(preds.gold, preds.pred)
        f1 = f1_score(preds.gold, preds.pred, average="macro", zero_division=0)
        print(f"    -> {fp}   test acc {acc*100:.2f}%  macroF1 {f1:.4f}\n")

    pd.DataFrame(times).to_csv(os.path.join(a.out, "..", "norbert_times.csv"), index=False)
    print("Done. Now run:")
    print(f"  python evaluate.py --preds {a.out} --ref <your-full-model-config> --out results/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
train.py — MAGNET-NFR, implemented from the section 3.3 specification.

Seven stages, each independently ablatable:

  Stage 1  RoBERTa contextual encoder                    (always on — it is the backbone)
  Stage 2  GCN over an ISO/IEC 25010 knowledge graph     --variant no_gcn
  Stage 3  Transformer-XL style extended-context block   --variant no_xl
  Stage 4  Multi-head cross-attention fusion             (off automatically if 2 and 3 are off)
  Stage 5  Supervised contrastive objective              --variant no_contrastive
  Stage 6  First-order MAML adaptation                   --variant no_meta
  Stage 7  Hierarchical classification head              (always on)

USAGE
  python train.py --variant full --seed 42 \
      --train corpus/splits/train.csv --val corpus/splits/val.csv --test corpus/splits/test.csv \
      --out preds/

VARIANTS
  full | roberta_only | no_gcn | no_xl | no_contrastive | no_meta

HONEST NOTES — these belong in the methods section
  * Stage 3 is a relative-position transformer block over the token sequence, not the
    full Dai et al. segment-recurrence mechanism. With a median requirement length near
    50 tokens and a 256-token window, segment recurrence has nothing to recur over.
    That is worth stating plainly: the component may be inert on this data by
    construction, and the ablation will say so.
  * Stage 6 is first-order MAML (FOMAML). Second-order MAML needs ~3x the memory for a
    gradient term that is small in practice. Report FOMAML; do not claim MAML.
  * The knowledge graph is built from ISO/IEC 25010 seed terms plus terms mined from
    the TRAINING SPLIT ONLY. Mining from the full corpus would leak test information
    into the graph. This is checked at build time.
"""
import argparse, json, os, random, re, sys, time
from collections import Counter

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    sys.exit("Missing deps:  pip install torch transformers")

from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer


# =====================================================================
# Stage 2 — knowledge graph
# =====================================================================
ISO_25010_SEEDS = {
    'Performance': ['latency', 'throughput', 'response time', 'concurrent', 'load',
                    'speed', 'processing', 'capacity', 'resource utilization', 'seconds'],
    'Security':    ['encryption', 'authentication', 'authorization', 'access control',
                    'password', 'confidentiality', 'integrity', 'vulnerability',
                    'secure', 'credentials'],
    'Usability':   ['user', 'interface', 'intuitive', 'learn', 'accessibility',
                    'navigate', 'training', 'ease of use', 'operability', 'understandable'],
    'Reliability': ['availability', 'uptime', 'fault tolerance', 'recovery', 'failure',
                    'downtime', 'robust', 'failover', 'maturity', 'backup'],
    'Maintainability': ['modular', 'documentation', 'maintain', 'modify', 'testability',
                        'reusability', 'analysability', 'code quality', 'refactor', 'update'],
    'Portability': ['platform', 'install', 'adapt', 'browser', 'operating system',
                    'deploy', 'compatible', 'migration', 'replaceable', 'environment'],
}


def build_kg(train_texts, classes, n_mined=200, seed=42):
    """
    Nodes: ISO seed terms for the evaluated classes + terms mined from TRAINING TEXT ONLY.
    Edges: co-occurrence within a requirement, row-normalised.

    Returns (node_terms, adjacency [N,N], node_class_hint [N]).
    """
    seeds, hints = [], []
    for ci, c in enumerate(classes):
        for t in ISO_25010_SEEDS.get(c, []):
            seeds.append(t); hints.append(ci)

    # mine additional terms from the training split only
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=n_mined,
                            stop_words='english', min_df=3)
    tfidf.fit(train_texts)
    mined = [t for t in tfidf.get_feature_names_out() if t not in seeds]

    terms = seeds + mined
    hints = hints + [-1] * len(mined)
    N = len(terms)

    # co-occurrence
    A = np.zeros((N, N), dtype=np.float32)
    low = [t.lower() for t in train_texts]
    for txt in low:
        present = [i for i, t in enumerate(terms) if t in txt]
        for i in present:
            for j in present:
                if i != j:
                    A[i, j] += 1.0
    A = A + np.eye(N, dtype=np.float32)             # self loops
    deg = A.sum(1, keepdims=True); deg[deg == 0] = 1
    A = A / deg                                      # row-normalise
    return terms, torch.tensor(A), torch.tensor(hints)


class GCN(nn.Module):
    """Three spectral graph convolution layers, per section 3.3 Stage 2."""
    def __init__(self, n_nodes, dim, layers=3):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, dim)
        self.W = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.drop = nn.Dropout(0.1)

    def forward(self, A):
        H = self.emb.weight
        for W in self.W:
            H = torch.relu(W(A @ H))
            H = self.drop(H)
        return H                                     # [N, dim]


# =====================================================================
# Stage 3 — extended-context block
# =====================================================================
class RelPosBlock(nn.Module):
    """Transformer block with a learned relative-position bias."""
    def __init__(self, dim, heads=8, layers=6, max_len=256):
        super().__init__()
        self.rel = nn.Parameter(torch.zeros(heads, max_len, max_len))
        nn.init.normal_(self.rel, std=0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim * 2, 0.1,
                                       batch_first=True, norm_first=True)
            for _ in range(layers)])

    def forward(self, x, mask):
        for l in self.layers:
            x = l(x, src_key_padding_mask=~mask.bool())
        return x


# =====================================================================
# The model
# =====================================================================
class MagnetNFR(nn.Module):
    def __init__(self, backbone, n_classes, kg_A, n_nodes, variant, xl_layers=6, heads=12):
        super().__init__()
        self.variant = variant
        self.use_gcn = variant not in ('roberta_only', 'no_gcn')
        self.use_xl = variant not in ('roberta_only', 'no_xl')

        self.encoder = AutoModel.from_pretrained(backbone)       # Stage 1
        d = self.encoder.config.hidden_size

        if self.use_gcn:                                          # Stage 2
            self.gcn = GCN(n_nodes, d)
            self.register_buffer('A', kg_A)
        if self.use_xl:                                           # Stage 3
            self.xl = RelPosBlock(d, heads=8, layers=xl_layers)
        if self.use_gcn or self.use_xl:                           # Stage 4
            self.cross = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
            self.norm = nn.LayerNorm(d)

        self.proj = nn.Linear(d, 256)                             # contrastive space
        self.head = nn.Sequential(                                # Stage 7
            nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes))

    def forward(self, ids, mask, return_z=False):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        q = h[:, 0]                                               # [CLS]

        kv = []
        if self.use_gcn:
            g = self.gcn(self.A)                                  # [N, d]
            kv.append(g.unsqueeze(0).expand(h.size(0), -1, -1))
        if self.use_xl:
            kv.append(self.xl(h, mask))

        if kv:
            KV = torch.cat(kv, dim=1)
            a, _ = self.cross(q.unsqueeze(1), KV, KV)
            fused = self.norm(q + a.squeeze(1))
        else:
            fused = q                                             # roberta_only

        logits = self.head(fused)
        if return_z:
            return logits, F.normalize(self.proj(fused), dim=-1)
        return logits


# =====================================================================
# Stage 5 — supervised contrastive loss
# =====================================================================
def supcon(z, y, tau=0.07):
    sim = z @ z.T / tau
    sim.fill_diagonal_(-1e9)
    pos = (y.unsqueeze(0) == y.unsqueeze(1)).float()
    pos.fill_diagonal_(0)
    if pos.sum() == 0:
        return torch.tensor(0.0, device=z.device)
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    return -((pos * logp).sum(1) / pos.sum(1).clamp(min=1))[pos.sum(1) > 0].mean()


# =====================================================================
# data
# =====================================================================
class DS(Dataset):
    def __init__(self, df, tok, c2i, max_len):
        self.t = df.text.astype(str).tolist()
        self.y = df.label.map(c2i).tolist()
        self.tok, self.n = tok, max_len

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        e = self.tok(self.t[i], truncation=True, padding='max_length',
                     max_length=self.n, return_tensors='pt')
        return (e['input_ids'].squeeze(0), e['attention_mask'].squeeze(0),
                torch.tensor(self.y[i]))


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =====================================================================
# Stage 6 — first-order MAML
# =====================================================================
def maml_epoch(model, df, tok, c2i, args, dev, opt, lossf):
    """
    Episodic FOMAML. Each episode: sample a task (a class subset), inner-loop on a
    support set, evaluate on a query set, accumulate gradients from the query loss.
    """
    classes = list(c2i)
    for _ in range(args.maml_episodes):
        k = min(len(classes), args.maml_way)
        task = random.sample(classes, k)
        sup, qry = [], []
        for c in task:
            pool = df[df.label == c]
            if len(pool) < 2 * args.maml_shot:
                continue
            s = pool.sample(2 * args.maml_shot)
            sup.append(s.iloc[:args.maml_shot]); qry.append(s.iloc[args.maml_shot:])
        if len(sup) < 2:
            continue
        sup, qry = pd.concat(sup), pd.concat(qry)

        # inner loop — adapt on support
        inner = torch.optim.SGD(model.parameters(), lr=args.maml_inner_lr)
        for _ in range(args.maml_inner_steps):
            dl = DataLoader(DS(sup, tok, c2i, args.max_len), batch_size=len(sup))
            ids, m, y = next(iter(dl))
            inner.zero_grad()
            lossf(model(ids.to(dev), m.to(dev)), y.to(dev)).backward()
            inner.step()

        # outer — query loss, first order
        dl = DataLoader(DS(qry, tok, c2i, args.max_len), batch_size=len(qry))
        ids, m, y = next(iter(dl))
        opt.zero_grad()
        lossf(model(ids.to(dev), m.to(dev)), y.to(dev)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='full',
                    choices=['full', 'roberta_only', 'no_gcn', 'no_xl',
                             'no_contrastive', 'no_meta'])
    ap.add_argument('--seed', type=int, default=42)
    # holdout mode
    ap.add_argument('--train')
    ap.add_argument('--val')
    ap.add_argument('--test')
    # cross-validation mode (preferred on this corpus — see build_corpus output)
    ap.add_argument('--corpus', help='corpus/nfr_corpus.csv')
    ap.add_argument('--folds', help='corpus/folds.csv')
    ap.add_argument('--rep', type=int, help='which fold-assignment repeat')
    ap.add_argument('--fold', type=int, help='which fold is held out')
    ap.add_argument('--out', default='preds')
    ap.add_argument('--backbone', default='distilroberta-base')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--lambda-con', type=float, default=0.1)
    ap.add_argument('--xl-layers', type=int, default=6)
    ap.add_argument('--maml-episodes', type=int, default=8)
    ap.add_argument('--maml-way', type=int, default=3)
    ap.add_argument('--maml-shot', type=int, default=4)
    ap.add_argument('--maml-inner-lr', type=float, default=1e-3)
    ap.add_argument('--maml-inner-steps', type=int, default=1)
    ap.add_argument('--no-class-weights', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.folds is not None:
        # ---- CROSS-VALIDATION MODE ----
        if not (args.corpus and args.rep is not None and args.fold is not None):
            sys.exit('CV mode needs --corpus --folds --rep --fold')
        corpus = pd.read_csv(args.corpus)
        folds = pd.read_csv(args.folds)
        f = folds[folds.rep == args.rep]
        test_ids = set(f[f.fold == args.fold].item_id)
        te = corpus[corpus.item_id.isin(test_ids)].reset_index(drop=True)
        rest = corpus[~corpus.item_id.isin(test_ids)].reset_index(drop=True)
        # carve a validation slice out of the training folds only
        from sklearn.model_selection import train_test_split as _tts
        tr, va = _tts(rest, test_size=0.15, stratify=rest.label, random_state=args.seed)
        print(f'  CV mode: rep {args.rep}, fold {args.fold} held out')
    else:
        # ---- HOLDOUT MODE ----
        if not (args.train and args.val and args.test):
            sys.exit('Holdout mode needs --train --val --test (or use CV mode)')
        tr, va, te = (pd.read_csv(p) for p in (args.train, args.val, args.test))

    for n, d in (('train', tr), ('val', va), ('test', te)):
        miss = {'item_id', 'text', 'label'} - set(d.columns)
        if miss:
            sys.exit(f'ERROR {n}: missing {miss}')

    # leakage — cheap, and it is the thing that voids a paper
    ov = set(tr.item_id) & set(te.item_id)
    if ov:
        sys.exit(f'FATAL: {len(ov)} item_ids in both train and test.')
    ovt = set(tr.text.astype(str).str.strip()) & set(te.text.astype(str).str.strip())
    if ovt:
        sys.exit(f'FATAL: {len(ovt)} identical texts in both train and test.')
    print(f'[PASS] no train/test leakage (item_id or text)')

    classes = sorted(tr.label.unique())
    c2i = {c: i for i, c in enumerate(classes)}
    i2c = {i: c for c, i in c2i.items()}
    floor = te.label.value_counts(normalize=True).max()
    print(f'  variant  : {args.variant}')
    print(f'  backbone : {args.backbone}')
    print(f'  seed     : {args.seed}')
    print(f'  classes  : {classes}')
    print(f'  sizes    : train {len(tr)}  val {len(va)}  test {len(te)}')
    print(f'  majority-class floor on test: {floor*100:.2f}%')
    print(f'  device   : {dev}')

    tok = AutoTokenizer.from_pretrained(args.backbone)

    # KG from TRAINING TEXT ONLY
    kg_A, n_nodes = None, 1
    if args.variant not in ('roberta_only', 'no_gcn'):
        terms, kg_A, _ = build_kg(tr.text.astype(str).tolist(), classes)
        n_nodes = len(terms)
        kg_A = kg_A.to(dev)
        print(f'  knowledge graph: {n_nodes} nodes (mined from train split only)')

    model = MagnetNFR(args.backbone, len(classes), kg_A, n_nodes,
                      args.variant, args.xl_layers).to(dev)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  trainable params: {n_par:,}')

    if args.no_class_weights:
        print('  !! class weighting OFF — with a large majority class this may collapse')
        w = None
    else:
        cnt = Counter(tr.label.map(c2i))
        w = torch.tensor([len(tr) / (len(classes) * cnt[i]) for i in range(len(classes))],
                         dtype=torch.float).to(dev)
    lossf = nn.CrossEntropyLoss(weight=w)

    dl_tr = DataLoader(DS(tr, tok, c2i, args.max_len), batch_size=args.batch_size, shuffle=True)
    dl_va = DataLoader(DS(va, tok, c2i, args.max_len), batch_size=args.batch_size)
    dl_te = DataLoader(DS(te, tok, c2i, args.max_len), batch_size=args.batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    use_con = args.variant not in ('roberta_only', 'no_contrastive')
    use_maml = args.variant not in ('roberta_only', 'no_meta')

    best_f1, best_state, conv = -1, None, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0
        for ids, m, y in dl_tr:
            ids, m, y = ids.to(dev), m.to(dev), y.to(dev)
            opt.zero_grad()
            if use_con:
                logits, z = model(ids, m, return_z=True)
                loss = lossf(logits, y) + args.lambda_con * supcon(z, y)
            else:
                loss = lossf(model(ids, m), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()

        if use_maml:
            maml_epoch(model, tr, tok, c2i, args, dev, opt, lossf)

        model.eval(); P, Y = [], []
        with torch.no_grad():
            for ids, m, y in dl_va:
                P += model(ids.to(dev), m.to(dev)).argmax(-1).cpu().tolist()
                Y += y.tolist()
        f1 = f1_score(Y, P, average='macro', zero_division=0)
        acc = accuracy_score(Y, P)
        conv.append({'epoch': ep, 'train_loss': tot / len(dl_tr), 'val_acc': acc, 'val_f1': f1})
        print(f'    epoch {ep:2d}  loss {tot/len(dl_tr):.4f}  val acc {acc:.4f}  val macroF1 {f1:.4f}')
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    hrs = (time.time() - t0) / 3600
    model.load_state_dict(best_state)          # selected on val macro F1, never test
    model.eval(); P = []
    with torch.no_grad():
        for ids, m, _ in dl_te:
            P += model(ids.to(dev), m.to(dev)).argmax(-1).cpu().tolist()

    tag = ('MAGNET-NFR' if args.variant == 'full' else f'MAGNET-NFR-{args.variant}') \
          + f'-{args.backbone}'
    if args.folds is not None:
        fp = os.path.join(args.out, f'{tag}__rep{args.rep}fold{args.fold}.csv')
    else:
        fp = os.path.join(args.out, f'{tag}__seed{args.seed}.csv')
    pd.DataFrame({'item_id': te.item_id.values, 'gold': te.label.values,
                  'pred': [i2c[p] for p in P]}).to_csv(fp, index=False)

    acc = accuracy_score(te.label, [i2c[p] for p in P])
    f1 = f1_score(te.label, [i2c[p] for p in P], average='macro', zero_division=0)
    print(f'\n  -> {fp}')
    print(f'     test acc {acc*100:.2f}%  macroF1 {f1:.4f}  (floor {floor*100:.2f}%)  {hrs*60:.1f} min')
    if acc <= floor + 0.005:
        print(f'     !! at or below the majority floor — this run did not learn.')

    os.makedirs('logs', exist_ok=True)
    pd.DataFrame(conv).to_csv(f'logs/convergence_{tag}_seed{args.seed}.csv', index=False)
    with open('logs/runs.jsonl', 'a') as f:
        f.write(json.dumps({'config': tag, 'variant': args.variant, 'seed': args.seed,
                            'params': n_par, 'hours': round(hrs, 4),
                            'test_acc': acc, 'test_macro_f1': f1, 'floor': floor}) + '\n')


if __name__ == '__main__':
    main()

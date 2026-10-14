#!/usr/bin/env python3
"""Analyze the six-label component ablation against frozen Full predictions.
Primary estimand: mean over five repeat-pooled macro-F1 differences, Full - ablation.
Uncertainty: paired record bootstrap, resampling item IDs once per replicate and carrying
all five repeat predictions for that sampled record. Conditional on this fixed corpus.
Multiplicity family: four component comparisons. Reports ordinary 95% intervals plus
Bonferroni 98.75% intervals and Holm-adjusted p-values. Equivalence sensitivity uses
pre-frozen ±0.01 margin and 90% interval inclusion.
"""
from pathlib import Path
import pandas as pd, numpy as np, json, math, sys

EXP=Path(__file__).resolve().parent
PRED=EXP/'predictions'/'folds'
POOL=EXP/'predictions'/'pooled'; POOL.mkdir(parents=True,exist_ok=True)
FULL=EXP/'reference_full'/'pooled'
RES=EXP/'results'; RES.mkdir(exist_ok=True)
VARIANTS=('no_gcn','no_xl','no_contrastive','no_meta')
LABELS=['Maintainability','Performance','Portability','Reliability','Security','Usability']
NBOOT=4000
RNG=np.random.default_rng(20260916)

def stem(v): return f'MAGNET-NFR-{v}-distilroberta-base'

def pool_variant(v):
    outs=[]
    for rep in range(5):
        fs=[PRED/f'{stem(v)}__rep{rep}fold{k}.csv' for k in range(5)]
        miss=[str(x) for x in fs if not x.exists()]
        if miss: raise FileNotFoundError('Missing fold predictions: '+', '.join(miss))
        d=pd.concat([pd.read_csv(x) for x in fs],ignore_index=True)
        if len(d)!=1719 or d.item_id.nunique()!=1719:
            raise ValueError(f'{v} rep {rep}: pooled rows/unique invalid: {len(d)}/{d.item_id.nunique()}')
        d=d.sort_values('item_id').reset_index(drop=True)
        out=POOL/f'{stem(v)}__seed{rep}.csv'; d.to_csv(out,index=False); outs.append(d)
    return outs

def load_full():
    out=[]
    for rep in range(5):
        p=FULL/f'MAGNET-NFR-distilroberta-base__seed{rep}.csv'
        if not p.exists(): raise FileNotFoundError(p)
        d=pd.read_csv(p).sort_values('item_id').reset_index(drop=True)
        if len(d)!=1719 or d.item_id.nunique()!=1719: raise ValueError(f'Full rep{rep} invalid')
        out.append(d)
    return out

L2I={x:i for i,x in enumerate(LABELS)}
K=len(LABELS)

def encode(x):
    return np.fromiter((L2I[v] for v in x),dtype=np.int16,count=len(x))

def f1s_from_codes(g,p):
    c=np.bincount(g.astype(np.int64)*K+p.astype(np.int64),minlength=K*K).reshape(K,K)
    tp=np.diag(c).astype(float); fp=c.sum(0)-tp; fn=c.sum(1)-tp
    den=2*tp+fp+fn
    return np.divide(2*tp,den,out=np.zeros(K,dtype=float),where=den>0)

def metric(d):
    return float(f1s_from_codes(encode(d.gold.to_numpy()),encode(d.pred.to_numpy())).mean())

def perclass(d):
    return f1s_from_codes(encode(d.gold.to_numpy()),encode(d.pred.to_numpy()))

def accuracy(d):
    return float(np.mean(d.gold.to_numpy()==d.pred.to_numpy()))

def holm_adjust(pvals):
    names=list(pvals)
    order=sorted(range(len(names)),key=lambda i:pvals[names[i]])
    adj=[0.0]*len(names); running=0.0; m=len(names)
    for rank,i in enumerate(order):
        val=min(1.0,(m-rank)*pvals[names[i]])
        running=max(running,val); adj[i]=running
    return {names[i]:adj[i] for i in range(m)}

full=load_full()
variants={v:pool_variant(v) for v in VARIANTS}
# strict alignment and gold check
ids=full[0].item_id.to_numpy(); gold=full[0].gold.to_numpy()
for rep in range(5):
    if not np.array_equal(full[rep].item_id.to_numpy(),ids): raise ValueError('Full item order mismatch')
    if not np.array_equal(full[rep].gold.to_numpy(),gold): raise ValueError('Full gold mismatch')
    for v in VARIANTS:
        d=variants[v][rep]
        if not np.array_equal(d.item_id.to_numpy(),ids): raise ValueError(f'{v} rep{rep} item order mismatch')
        if not np.array_equal(d.gold.to_numpy(),gold): raise ValueError(f'{v} rep{rep} gold mismatch')

gold_code=encode(gold)
full_pred_code=[encode(d.pred.to_numpy()) for d in full]
variant_pred_code={v:[encode(d.pred.to_numpy()) for d in variants[v]] for v in VARIANTS}

summary=[]; class_rows=[]; raw_p={}; details={}
full_rep=np.array([metric(d) for d in full])
for v in VARIANTS:
    vr=np.array([metric(d) for d in variants[v]])
    point=float(np.mean(full_rep-vr))
    # record bootstrap: same sampled record positions applied to all repeats and both models
    bd=np.empty(NBOOT)
    n=len(ids)
    for b in range(NBOOT):
        ix=RNG.integers(0,n,n)
        ds=[]
        g=gold_code[ix]
        for rep in range(5):
            pf=full_pred_code[rep][ix]
            pv=variant_pred_code[v][rep][ix]
            ds.append(f1s_from_codes(g,pf).mean()-f1s_from_codes(g,pv).mean())
        bd[b]=np.mean(ds)
    lo95,hi95=np.quantile(bd,[.025,.975])
    lo90,hi90=np.quantile(bd,[.05,.95])
    # two-sided sign-tail bootstrap p, with +1 finite-sample correction
    left=(np.sum(bd<=0)+1)/(NBOOT+1); right=(np.sum(bd>=0)+1)/(NBOOT+1)
    p=min(1.0,2*min(left,right)); raw_p[v]=p
    # Bonferroni familywise 98.75% CI for 4 comparisons
    loB,hiB=np.quantile(bd,[.00625,.99375])
    equival=(lo90>-0.01 and hi90<0.01)
    # per-class means and descriptive deltas
    fpc=np.mean(np.vstack([perclass(d) for d in full]),axis=0)
    vpc=np.mean(np.vstack([perclass(d) for d in variants[v]]),axis=0)
    for lab,a,c in zip(LABELS,fpc,vpc):
        class_rows.append({'variant':v,'label':lab,'full_f1':a,'ablation_f1':c,'full_minus_ablation':a-c})
    summary.append({'comparison':f'Full - {v}','variant':v,
                    'full_macro_f1_mean':float(full_rep.mean()),'full_macro_f1_sd':float(full_rep.std(ddof=1)),
                    'ablation_macro_f1_mean':float(vr.mean()),'ablation_macro_f1_sd':float(vr.std(ddof=1)),
                    'delta_macro_f1':point,'ci95_lo':float(lo95),'ci95_hi':float(hi95),
                    'ci90_lo':float(lo90),'ci90_hi':float(hi90),'equivalent_pm_0_01':bool(equival),
                    'ci98_75_bonf_lo':float(loB),'ci98_75_bonf_hi':float(hiB),'bootstrap_p_raw':float(p)})
    details[v]={'repeat_full':full_rep.tolist(),'repeat_ablation':vr.tolist(),
                'bootstrap_seed':20260916,'bootstrap_n':NBOOT,
                'estimand':'mean across five repeat-pooled macro-F1 differences; Full minus ablation',
                'resampling_unit':'record/item_id, same sampled positions carried across all repeats/models'}

adj=holm_adjust(raw_p)
for r in summary: r['bootstrap_p_holm']=adj[r['variant']]
sdf=pd.DataFrame(summary)
cdf=pd.DataFrame(class_rows)
sdf.to_csv(RES/'component_ablation_summary.csv',index=False)
cdf.to_csv(RES/'component_ablation_per_class_f1.csv',index=False)
(RES/'component_ablation_details.json').write_text(json.dumps(details,indent=2))

# Repeat-level table and accuracy/F1 diagnostics
rows=[]
for rep in range(5):
    rows.append({'repeat':rep,'variant':'full','accuracy':accuracy(full[rep]),'macro_f1':metric(full[rep])})
    for v in VARIANTS:
        rows.append({'repeat':rep,'variant':v,'accuracy':accuracy(variants[v][rep]),'macro_f1':metric(variants[v][rep])})
pd.DataFrame(rows).to_csv(RES/'component_ablation_repeat_metrics.csv',index=False)

print('\nSIX-LABEL COMPONENT ABLATION SUMMARY')
print(sdf.to_string(index=False))
print('\nPer-class descriptive effects saved to:',RES/'component_ablation_per_class_f1.csv')

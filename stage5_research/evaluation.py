"""Evaluator-only fixed comparisons, stability slices and descriptive diagnostics."""
import numpy as np
import pandas as pd
from sbrt.metric import age_row, aggregate
from sbrt.stage2 import summarize_ages
from sbrt.comparison import gate_conditions
from .contract import GATE, IDS


def gate(delta,early,partitions,*,valid=True,replicates=None,child=False,child_delta=None,child_replicates=None):
    conditions=dict(valid_complete=bool(valid),**gate_conditions(delta,GATE),
        historical_early_guard=early is not None and early>=-.005,
        partition_A=partitions.get("A") is not None and partitions["A"]>0,
        partition_B=partitions.get("B") is not None and partitions["B"]>0)
    q=None; interval=None
    if replicates is not None:
        if len(replicates)!=999 or not np.isfinite(replicates).all(): raise ValueError("Require 999 finite replicates")
        q=float(np.quantile(replicates,.01,method="linear"))
    if child_replicates is not None:
        if len(child_replicates)!=999 or not np.isfinite(child_replicates).all(): raise ValueError("Require 999 finite child replicates")
        interval=np.quantile(child_replicates,[.025,.975],method="linear").tolist()
    child_pooled=not child or (child_delta is not None and child_delta>0)
    child_ci=not child or (interval is not None and interval[0]>0)
    serious=all(conditions.values())
    return dict(conditions=conditions,family_size=4,bootstrap_required=bool(serious and child_pooled),
        Q_0_01=q,bootstrap_positive=q is not None and q>0,child_required=child,
        child_pooled_positive=child_pooled,child_interval_95=interval,child_lower_positive=child_ci,
        historical_early_delta=early,advances=bool(serious and q is not None and q>0 and child_pooled and child_ci))


def winner(results):
    candidates=[v for k,v in results.items() if k in IDS and v["status"]=="valid" and v["gate"]["advances"]]
    return min(candidates,key=lambda r:(-r["pooled_oof_ts_auc"],r["model_input_count"],r["experiment_id"]))["experiment_id"] if candidates else "E14"


def frame_score(frame,column):
    rows=[age_row(int(age),g.target,g[column]) for age,g in frame.groupby("time_online",sort=True)]
    score,ages=aggregate(rows)
    buckets,combined=summarize_ages(ages)
    return dict(ts_auc=score,pair_weight=int(ages.weight.sum()),age_buckets=buckets,combined=combined,
                series=int(frame.id.nunique()),rows=len(frame))


def partition_scores(frame,predictions,reference,partition):
    f=frame[["id","time_online","target"]].copy()
    f["candidate"]=predictions; f["reference"]=reference
    f["partition"]=f.id.map(dict(zip(partition.id,partition.partition)))
    if f.partition.isna().any(): raise ValueError("Incomplete partition join")
    out={}
    for name in ("A","B"):
        g=f.loc[f.partition==name]
        a,b=frame_score(g,"candidate"),frame_score(g,"reference")
        if a["pair_weight"]!=b["pair_weight"]: raise AssertionError("Partition pair weights differ")
        out[name]=dict(candidate=a,reference=b,delta=a["ts_auc"]-b["ts_auc"])
    return out


def tie_rate(frame,column="prediction"):
    n=frame.groupby("time_online").size().to_numpy(dtype=np.int64)
    g=frame.groupby(["time_online",column],sort=False).size().to_numpy(dtype=np.int64)
    denom=int(np.sum(n*(n-1)//2)); num=int(np.sum(g*(g-1)//2))
    return dict(tied_unordered_pairs=num,total_unordered_pairs=denom,rate=num/denom if denom else None)


def correlations(frame,columns):
    age=frame.time_online.to_numpy(dtype=np.int64)
    target=frame.target.to_numpy(dtype=np.float64)
    count=np.bincount(age); pos=np.bincount(age,weights=target); neg=count-pos
    weight=pos*neg
    mass=weight/weight.sum()
    row_weight=mass[age]/count[age]
    def centered(x):
        mean=np.divide(np.bincount(age,weights=x),count,out=np.zeros(len(count)),where=count>0)
        return x-mean[age]
    def corr(x,y):
        mx=float(row_weight@x); my=float(row_weight@y)
        a=x-mx; b=y-my
        vx=float(row_weight@(a*a)); vy=float(row_weight@(b*b))
        return float((row_weight@(a*b))/np.sqrt(vx*vy)) if vx>0 and vy>0 else None
    result=[]
    for column in columns:
        x=frame[column].to_numpy(dtype=float); cx=centered(x)
        for against in ("I","J_s","P"):
            y=frame[against].to_numpy(dtype=float)
            ordinary=corr(x,y); conditional=corr(cx,centered(y))
            result.append(dict(feature=column,against=against,raw_weighted=ordinary,age_centered_weighted=conditional,
                possible_near_duplication=conditional is not None and abs(conditional)>=.98))
    return dict(row_weight_total=float(row_weight.sum()),weighting="W_t/(Z*N_active(t)); evaluator only",values=result)


def saturation(frame,columns):
    return {k:dict(at_zero=float(np.mean(frame[k].to_numpy()==0)),at_one=float(np.mean(frame[k].to_numpy()==1)),
                   ge_099=float(np.mean(frame[k].to_numpy()>=.99))) for k in columns}

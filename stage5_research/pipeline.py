"""One preregistered causal export; four fits; fresh replay; fixed G5 decision."""
from collections import deque
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from sbrt.data import iter_series, sha256
from sbrt.replay import replay
from sbrt.logistic import training_weights, objective, InvalidFit
from sbrt.stage2 import score_outputs, load_result, difference
from sbrt.comparison import paired_bootstrap
from sbrt.provenance import ROOT, write_json, canonical_hash, finalize_artifacts, verify_artifacts, quantiles, peak_memory_bytes
from stage4_research.models import Model as E14Model
from .contract import IDS, RAW, BASE, EXTRA, INPUTS, PARENTS, GATE, MOMENTS, LIKELIHOOD, TRAJECTORY, CALIBRATED
from .features import FeatureState, Detector
from .models import Model, fit_fold
from .provenance import admission, provenance, evaluator, protected, source
from .evaluation import gate, winner, frame_score, partition_scores, correlations, saturation, tie_rate

KEYS=["id","time","time_online","target","fold"]
NEW=list(RAW[len(BASE):])


def distribution(values):
    values=[v for v in values if v is not None]
    levels=[0,.05,.25,.5,.75,.95,1]
    return dict(count=len(values),levels=levels,values=np.quantile(values,levels,method="linear").tolist() if values else None)


def add_counts(records):
    if not records: return {}
    out={}
    for key,value in records[0].items():
        if isinstance(value,dict): out[key]=add_counts([r[key] for r in records])
        elif isinstance(value,list): out[key]=np.sum([r[key] for r in records],axis=0).tolist()
        else: out[key]=sum(r[key] for r in records)
    return out


class Exporter:
    def __init__(self): self.completed=deque()

    def infer(self,datasets):
        yield
        for history,online in datasets:
            tick=time.perf_counter()
            state=FeatureState(history,"export",null_audit=True)
            init=time.perf_counter()-tick
            null=state.null_rows; state.null_rows=None
            size=len(state.dumps()); rows=[]; updates=0.
            for value in online:
                tick=time.perf_counter()
                row=state.observe(value); state.commit()
                updates+=time.perf_counter()-tick
                rows.append([row[k] for k in RAW])
                yield .5  # Unscored handshake sentinel; never a selectable predictor.
            record=dict(initialization_seconds=init,updates_seconds=updates,serialized_state_bytes=max(size,len(state.dumps())),
                array_bytes=state.array_bytes(),historical=state.historical,counts=dict(transform=state.counts,
                new=state.statistics.counts,scale=state.base.scales["slow"].counts,old_dependence=state.base.blocks.dependence.counts))
            self.completed.append((rows,null,record))


def export_features(root):
    root=Path(root); frozen,manifest=admission(root); out=root/"features"
    if out.exists():
        verify_artifacts(out)
        old=json.loads((out/"metrics.json").read_text())
        if old["provenance"]!=provenance(frozen): raise ValueError("Incompatible export")
        return old
    out.mkdir(); start=time.perf_counter(); con=evaluator(root)
    observer=Exporter(); lookup=dict(zip(manifest.id,manifest.fold))
    records=[]; frames=[]; nulls=[]; writers={}; count=0
    def flush():
        for key,buffers in (("raw",frames),("historical_null",nulls)):
            if not buffers: continue
            table=pa.Table.from_pandas(pd.concat(buffers,ignore_index=True),preserve_index=False)
            if key not in writers: writers[key]=pq.ParquetWriter(out/(key+".parquet"),table.schema,compression="zstd")
            writers[key].write_table(table); buffers.clear()
    try:
        for record,predictions,_ in replay(iter_series(con),infer_fn=observer.infer):
            rows,null,detail=observer.completed.popleft()
            values=np.asarray(rows,dtype=np.float64)
            if values.shape!=(len(predictions),len(RAW)) or not np.isfinite(values).all(): raise AssertionError("Invalid causal export")
            frame=pd.DataFrame(values,columns=RAW)
            frames.append(frame.assign(id=record.id,time=record.time,time_online=np.arange(len(frame)),target=record.target,fold=lookup[record.id]))
            nulls.append(pd.DataFrame(null).loc[:,NEW].assign(id=record.id,position_C=np.arange(len(null))))
            detail["id"]=record.id; records.append(detail); count+=len(frame)
            if len(frames)>=64: flush()
            if len(records)%500==0: print(f"Stage5 causal export: {len(records)}/10000",flush=True)
        flush()
        for writer in writers.values(): writer.close()
        writers.clear()
        if count!=5036517 or len(records)!=10000: raise AssertionError("Incomplete feature coverage")
        con.read_parquet(str(out/"raw.parquet")).create_view("s5_export",replace=True)
        con.read_parquet(str(ROOT/"stage4_results/features/raw.parquet")).create_view("s5_frozen",replace=True)
        clauses=" OR ".join(f"a.{k} IS DISTINCT FROM b.{k}" for k in (*BASE,"time","target","fold"))
        mismatch=con.execute(f"SELECT count(*) FROM s5_export a FULL JOIN s5_frozen b USING(id,time_online) WHERE a.id IS NULL OR b.id IS NULL OR {clauses}").fetchone()[0]
        if mismatch: raise AssertionError("Export changed frozen E14 features/keys")
        write_json(out/"series_diagnostics.json",records)
        history=[dict(id=r["id"],**r["historical"]) for r in records]
        write_json(out/"historical_calibration.json",history)
        historical={k:distribution([r[k] for r in history]) for k in ("M","u_min","u_max","z_mean","z_sd","m_z","v0")}
        historical["moments"]={f"M{j+1}":{k:distribution([r[k][j] for r in history]) for k in ("channel_mean","channel_sd","channel_sd_before_floor")} for j in range(5)}
        historical["HAC"]={k:distribution([r["calibration"][k] for r in history]) for k in ("gamma0","V_raw","V","eta")}
        init=sum(r["initialization_seconds"] for r in records); updates=sum(r["updates_seconds"] for r in records)
        result=dict(provenance=provenance(frozen),rows=count,series=len(records),columns=list(RAW),R_I_P_D_J_s_exact=True,
            raw_sha256=sha256(out/"raw.parquet"),historical_null_sha256=sha256(out/"historical_null.parquet"),
            initialization_seconds=init,updates_seconds=updates,export_wall_seconds=time.perf_counter()-start,
            feature_updates_per_second=count/updates,state_array_bytes=quantiles([r["array_bytes"] for r in records]),
            serialized_state_bytes=quantiles([r["serialized_state_bytes"] for r in records]),support_counts=add_counts([r["counts"] for r in records]),
            historical_distributions=historical,peak_memory_bytes=peak_memory_bytes(),
            timing_note="Shared all-block export, including in-sample historical replay; not production throughput")
        admission(root); write_json(out/"metrics.json",result); finalize_artifacts(out)
        return result
    finally:
        for writer in writers.values(): writer.close()
        con.close()


def load_frame(root,experiment):
    verify_artifacts(Path(root)/"features")
    f=pd.read_parquet(Path(root)/"features/raw.parquet",columns=KEYS+list(BASE)+list(EXTRA[experiment]))
    return f.sort_values(["id","time_online"],kind="stable").reset_index(drop=True)


def embedding_audit(root,frame,fold,model,weights):
    original=E14Model.loads((ROOT/f"stage4_results/E14/models/fold{fold}.json").read_bytes())
    old_weights=json.loads((ROOT/f"stage4_results/E14/models/fold{fold}_weights.json").read_text())
    if weights!=old_weights or model.pre.base.as_dict()!=original.pre.as_dict(): raise AssertionError("Changed E14 training transforms")
    def audit(parent,count):
        nested=Model(model.pre,parent.intercept,np.r_[parent.beta,np.zeros(len(model.beta)-count)])
        error=0.
        for start in range(0,len(frame),65536):
            f=frame.iloc[start:start+65536]
            error=max(error,float(np.max(np.abs(nested.predict(f)-parent.predict(f)))))
        train=frame.loc[frame.fold!=fold]
        eligible,omega,_=training_weights(train.time_online.to_numpy(),train.target.to_numpy())
        train=train.loc[eligible]; x=model.pre.transform(train); y=train.target.to_numpy(dtype=float)
        theta=np.r_[nested.intercept,nested.beta]
        with threadpool_limits(limits=1):
            loss_error=abs(objective(theta,x,y,omega)[0]-objective(theta[:count+1],x[:,:count],y,omega)[0])
        if error>1e-14 or loss_error>1e-14: raise AssertionError("Nested model equivalence failed")
        return dict(prediction_error=error,objective_error=loss_error)
    result=dict(E14_preprocessing_exact=True,E14_weights_exact=True,E14=audit(original,5))
    if model.pre.experiment=="E20":
        path=Path(root)/f"E19/models/fold{fold}.json"
        if path.exists():
            parent=Model.loads(path.read_bytes())
            if not np.array_equal(parent.pre.mean,model.pre.mean[:2]) or not np.array_equal(parent.pre.std,model.pre.std[:2]):
                raise AssertionError("E20 changed E19 preprocessing")
            result["E19"]=audit(parent,7)
    return result


def streaming_audit(root,frame,models,predictions):
    con=evaluator(root); active={}; details=deque()
    lookup={sid:(int(g.fold.iloc[0]),g.index.to_numpy()) for sid,g in frame.groupby("id",sort=False)}
    def records():
        for record in iter_series(con):
            active["model"]=models[lookup[record.id][0]]
            yield record
    def infer_fn(datasets):
        yield
        for history,online in datasets:
            tick=time.perf_counter(); state=Detector(history,active["model"]); init=time.perf_counter()-tick
            size=len(state.dumps()); updates=0.
            for value in online:
                tick=time.perf_counter(); prediction=state.update(value); updates+=time.perf_counter()-tick
                yield prediction
            details.append(dict(initialization_seconds=init,update_seconds=updates,array_bytes=state.features.array_bytes(),
                                serialized_bytes=max(size,len(state.dumps()))))
    start=time.perf_counter(); rows=0; records_out=[]
    try:
        for record,values,_ in replay(records(),infer_fn=infer_fn):
            if not np.array_equal(values,predictions[lookup[record.id][1]]): raise AssertionError("Cached/fresh causal prediction mismatch")
            rows+=len(values); records_out.append(details.popleft())
            if len(records_out)%2000==0: print(f"{models[0].pre.experiment} full causal inference {len(records_out)}/10000",flush=True)
        if rows!=len(frame) or len(records_out)!=len(lookup): raise AssertionError("Incomplete streaming coverage")
        init=sum(r["initialization_seconds"] for r in records_out); update=sum(r["update_seconds"] for r in records_out)
        return dict(series=len(records_out),rows=rows,all_predictions_exact=True,initialization_seconds=init,
            update_and_logistic_seconds=update,end_to_end_inference_seconds=init+update,end_to_end_predictions_per_second=rows/(init+update),
            update_predictions_per_second=rows/update,guarded_replay_wall_seconds=time.perf_counter()-start,
            array_bytes=quantiles([r["array_bytes"] for r in records_out]),serialized_state_bytes=quantiles([r["serialized_bytes"] for r in records_out]),
            note="Full fresh AR/scale/CDF/history initialization + causal updates + logistic; evaluator I/O and serialization excluded from compute timing, included in guarded wall")
    finally: con.close()


def run_one(root,experiment):
    root=Path(root)
    if experiment not in IDS: raise ValueError("Only frozen E18-E21")
    frozen,manifest=admission(root); out=root/experiment
    if out.exists():
        verify_artifacts(out); old=json.loads((out/"metrics.json").read_text())
        if old["provenance"]!=provenance(frozen): raise ValueError("Incompatible completed run")
        return old
    out.mkdir(); start=time.perf_counter()
    try:
        frame=load_frame(root,experiment)
        if len(frame)!=5036517 or frame.id.nunique()!=10000 or frame[["id","time_online"]].duplicated().any() or frame.groupby("id").fold.nunique().max()!=1:
            raise AssertionError("CV/OOF coverage mismatch")
        predictions=np.full(len(frame),np.nan); seen=np.zeros(len(frame),dtype=bool)
        models=[]; training=[]; hashes=[]; inference=0.
        for fold in range(5):
            print(f"{experiment}: fitting outer fold {fold}",flush=True)
            model,detail,weights=fit_fold(frame,fold,experiment)
            detail["nested_audit"]=embedding_audit(root,frame,fold,model,weights)
            path=out/"models"/f"fold{fold}.json"; path.parent.mkdir(exist_ok=True)
            with path.open("xb") as f: f.write(model.dumps())
            restored=Model.loads(path.read_bytes())
            if restored.dumps()!=model.dumps(): raise AssertionError("Model byte roundtrip mismatch")
            models.append(restored); hashes.append(sha256(path))
            indices=np.flatnonzero(frame.fold.to_numpy()==fold)
            train_ids=sorted(frame.loc[frame.fold!=fold,"id"].unique()); valid_ids=sorted(frame.loc[frame.fold==fold,"id"].unique())
            if set(train_ids)&set(valid_ids): raise AssertionError("Outer CV leakage")
            for begin in range(0,len(indices),65536):
                ix=indices[begin:begin+65536]; tick=time.perf_counter(); values=restored.predict(frame.iloc[ix]); inference+=time.perf_counter()-tick
                if seen[ix].any(): raise AssertionError("Duplicate OOF key")
                predictions[ix]=values; seen[ix]=True
            detail.update(training_ids_sha256=canonical_hash(train_ids),validation_ids_sha256=canonical_hash(valid_ids),
                training_series=len(train_ids),validation_series=len(valid_ids),model_sha256=hashes[-1])
            write_json(out/"models"/f"fold{fold}_training.json",detail); write_json(out/"models"/f"fold{fold}_weights.json",weights)
            training.append(detail)
        if not seen.all() or not np.isfinite(predictions).all(): raise AssertionError("Incomplete OOF")
        streamed=streaming_audit(root,frame,models,predictions)
        result_frame=frame[KEYS].copy(); result_frame["prediction"]=predictions
        result_frame.to_parquet(out/"oof.parquet",index=False,compression="zstd")
        con=evaluator(root)
        try: measured=score_outputs(con,out/"oof.parquet",manifest,out)
        finally: con.close()
        reference=pd.read_parquet(ROOT/"stage4_results/E14/oof.parquet",columns=["id","time_online","prediction"]).sort_values(["id","time_online"],kind="stable").reset_index(drop=True)
        if not reference[["id","time_online"]].equals(frame[["id","time_online"]]): raise AssertionError("Reference key mismatch")
        partitions=partition_scores(frame,predictions,reference.prediction.to_numpy(),pd.read_parquet(root/"partition/manifest.parquet"))
        refs={"E14":load_result(ROOT/"stage4_results/E14"),"E08":load_result(ROOT/"stage2_phase2/E08"),"E04":load_result(ROOT/"stage2_results/E04")}
        parent=refs["E14"] if PARENTS[experiment]=="E14" else (json.loads((root/"E19/metrics.json").read_text()) if (root/"E19/metrics.json").exists() else None)
        result=dict(**measured,status="valid",experiment_id=experiment,parent_id=PARENTS[experiment],provenance=provenance(frozen),
            model_inputs=INPUTS[experiment],model_input_count=len(INPUTS[experiment]),new_feature_count=len(EXTRA[experiment]),
            training_convergence_by_fold=training,training_seconds_by_fold=[d["training_seconds"] for d in training],
            training_seconds_total=sum(d["training_seconds"] for d in training),model_sha256=hashes,
            shared_model_bytes_by_fold=[len(m.dumps()) for m in models],streaming=streamed,cached_inference_seconds=inference,
            feature_cache_sha256=sha256(root/"features/artifacts.sha256.json"),OOF_sha256=sha256(out/"oof.parquet"),
            **{f"delta_vs_{k}":difference(measured,v) for k,v in refs.items()},
            delta_vs_parent=difference(measured,parent) if parent and parent["status"]=="valid" else None,
            partitions=partitions,correlations=correlations(frame,EXTRA[experiment]),feature_saturation=saturation(frame,EXTRA[experiment]),
            prediction_ties=tie_rate(result_frame),peak_memory_bytes=peak_memory_bytes(),total_wall_seconds=time.perf_counter()-start)
        admission(root); write_json(out/"metrics.json",result); finalize_artifacts(out)
        print(f"{experiment}: valid TS-AUC={measured['pooled_oof_ts_auc']:.10f}; delta E14={result['delta_vs_E14']['pooled']:+.10f}",flush=True)
        return result
    except Exception as exc:
        failure=dict(status="invalid",experiment_id=experiment,parent_id=PARENTS[experiment],provenance=provenance(frozen),
            exception_type=type(exc).__name__,message=str(exc),failure_category="numerical convergence" if isinstance(exc,InvalidFit) else "implementation/admission failure",
            total_wall_seconds=time.perf_counter()-start)
        if isinstance(exc,InvalidFit): failure["convergence"]=exc.diagnostics
        write_json(out/"metrics.json",failure); finalize_artifacts(out)
        raise


def standalone(root):
    frame=pd.read_parquet(Path(root)/"features/raw.parquet",columns=KEYS+NEW)
    frame["moment_omnibus"]=sum(frame[k] for k in MOMENTS)/5.
    frame["likelihood_max"]=np.maximum(frame.L_plus,frame.L_minus)
    frame["calibrated_max"]=np.maximum(frame.L_plus_calibrated,frame.L_minus_calibrated)
    out={}
    for key in NEW+["moment_omnibus","likelihood_max","calibrated_max"]:
        out[key]=dict(**frame_score(frame,key),selectable=False)
    return out


def null_diagnostics(con,root):
    con.read_parquet(str(Path(root)/"features/historical_null.parquet")).create_view("s5_null",replace=True)
    result={}
    for key in NEW:
        values=con.execute(f'''SELECT count(*), quantile_cont({key},[0,.05,.25,.5,.75,.95,1]),
            avg(CAST({key}=0 AS DOUBLE)),avg(CAST({key}=1 AS DOUBLE)),avg(CAST({key}>=.99 AS DOUBLE)) FROM s5_null''').fetchone()
        result[key]=dict(rows=values[0],levels=[0,.05,.25,.5,.75,.95,1],quantiles=values[1],at_zero=values[2],at_one=values[3],ge_099=values[4])
    return dict(description="In-sample historical-reference replay on chronological z_C with accumulators reset; inclusive CDF and references use same C. Not a held-out null test.",features=result)


def compare(root):
    root=Path(root); frozen,manifest=admission(root); out=root/"comparisons"
    if out.exists(): verify_artifacts(out); return json.loads((out/"stage5_report.json").read_text())
    out.mkdir(); start=time.perf_counter(); con=evaluator(root); results={}
    try:
        e01=load_result(ROOT/"stage2_results/E01_squared")
        for name in IDS:
            verify_artifacts(root/name); entry=json.loads((root/name/"metrics.json").read_text())
            if entry["provenance"]!=provenance(frozen): raise ValueError("Candidate provenance drift")
            if entry["status"]=="valid":
                early=difference(entry,e01)["combined"]["le64"]
                parts={k:v["delta"] for k,v in entry["partitions"].items()}
                child=name in ("E20","E21")
                child_delta=entry["delta_vs_parent"]["pooled"] if entry["delta_vs_parent"] else None
                decision=gate(entry["delta_vs_E14"],early,parts,child=child,child_delta=child_delta)
                boot=childboot=None
                if decision["bootstrap_required"]:
                    boot=paired_bootstrap(con,root/name/"oof.parquet",ROOT/"stage4_results/E14/oof.parquet",manifest,GATE)
                    if child and child_delta is not None:
                        childboot=paired_bootstrap(con,root/name/"oof.parquet",root/"E19/oof.parquet",manifest,GATE)
                        if boot["multiplicities_sha256"]!=childboot["multiplicities_sha256"]: raise AssertionError("Child bootstrap multiplicities drift")
                    decision=gate(entry["delta_vs_E14"],early,parts,replicates=boot["delta_replicates"],child=child,
                        child_delta=child_delta,child_replicates=childboot["delta_replicates"] if childboot else None)
                    boot["Q_0_01"]=decision["Q_0_01"]
                entry.update(gate=decision,bootstrap=boot,child_bootstrap=childboot,
                             bootstrap_seconds=sum(b["seconds"] for b in (boot,childboot) if b))
            results[name]=entry; write_json(out/f"{name}.json",entry)
            print(f"{name}: {entry['status']}; G5 advances={entry.get('gate',{}).get('advances',False)}",flush=True)
        standalone_results=standalone(root); write_json(out/"standalone.json",standalone_results)
        null=null_diagnostics(con,root); write_json(out/"historical_null.json",null)
        value=dict(provenance=provenance(frozen),experiments=results,family_size=4,final_reference=winner(results),
            standalone=standalone_results,historical_null=null,feature_export=json.loads((root/"features/metrics.json").read_text()),
            partition=frozen["partition"],comparison_wall_seconds=time.perf_counter()-start,
            total_bootstrap_seconds=sum(r.get("bootstrap_seconds",0.) for r in results.values()),protected_after=protected(),
            source_unchanged=source()==frozen["source"],stopping_point="G5 report and reference decision only; no production/cloud/combined candidates/Stage 6")
        admission(root); write_json(out/"stage5_report.json",value); finalize_artifacts(out)
        return value
    finally: con.close()


def report(root):
    root=Path(root); admission(root); verify_artifacts(root/"comparisons")
    value=json.loads((root/"comparisons/stage5_report.json").read_text())
    lines=["# Frozen Stage-5 G5 report", "",f"Selected reference: **{value['final_reference']}**.",
        "A/B are reused-data stability slices, not untouched holdouts or independent replications.",
        "All reported bootstrap intervals are conditional on fixed OOF predictions; no refitting.", ""]
    for name,r in value["experiments"].items():
        lines += [f"## {name}", "", f"Status: {r['status']}"]
        if r["status"]!="valid": lines += [r["message"],""]; continue
        lines += [f"Pooled TS-AUC: {r['pooled_oof_ts_auc']:.12f}; delta E14: {r['delta_vs_E14']['pooled']:+.12f}; advances: {r['gate']['advances']}.",
            "", "Complete fold, age, partition, convergence, coefficient, runtime, gate and hash details:", "", "```json",json.dumps(r,indent=2),"```", ""]
    lines += ["## Diagnostics and provenance", "", "See comparisons/standalone.json, comparisons/historical_null.json, features/series_diagnostics.json and freeze/freeze.json.",
        "", "No production refit, cloud submission, candidate combination, exploratory modeling or Stage 6 performed."]
    (root/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    files={p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(root.rglob("*")) if p.is_file() and "scratch" not in p.relative_to(root).parts and p.name!="delivery_manifest.json"}
    write_json(root/"delivery_manifest.json",dict(files=files,sha256=canonical_hash(files)))
    return value


def batch(root):
    root=Path(root); start=time.perf_counter(); admission(root); export_features(root); codes={}
    for name in IDS:
        codes[name]=subprocess.run([sys.executable,str(ROOT/"run_stage5.py"),"run","--experiment",name,"--out",str(root)],cwd=ROOT,check=False).returncode
    result=compare(root)
    write_json(root/"batch_execution.json",dict(return_codes=codes,total_wall_seconds=time.perf_counter()-start,final_reference=result["final_reference"]))
    report(root)
    return result

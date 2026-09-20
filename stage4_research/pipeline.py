"""One causal export, four frozen fits, guarded streaming audit and G4 report."""
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
from sbrt.data import iter_series, iter_frames, sha256
from sbrt.replay import replay
from sbrt.logistic import LogisticModel, training_weights, objective, InvalidFit
from sbrt.stage2 import score_outputs, load_result, difference, summarize_ages
from sbrt.metric import age_row, aggregate
from sbrt.comparison import gate_conditions, paired_bootstrap
from sbrt.provenance import ROOT, write_json, finalize_artifacts, verify_artifacts, canonical_hash, quantiles, peak_memory_bytes
from .contract import IDS, RAW, BASE, INPUTS, PARENTS, STANDALONE, GATE
from .features import FeatureState, Detector, contrasts
from .models import Model, fit_fold
from .provenance import admission, provenance, evaluator, protected, source


class Exporter:
    """Inference-facing export has no labels, keys, folds or horizons."""
    def __init__(self):
        self.completed = deque()

    def infer(self,datasets):
        yield
        for history,online in datasets:
            tick = time.perf_counter()
            state = FeatureState(history,"export")
            init = time.perf_counter()-tick
            initial_size = len(state.dumps())
            rows, diagnostic = [], []
            updates = 0.0
            for value in online:
                tick = time.perf_counter()
                row = state.observe(value)
                state.commit()
                updates += time.perf_counter()-tick
                rows.append([row[k] for k in RAW])
                diagnostic.append([state.diagnostic[k] for k in ("e_squared","v_s","v_f","v_r")])
                yield .5  # Guard handshake only; never a scored candidate.
            record = dict(initialization_seconds=init,updates_seconds=updates,
                original_squared_acf=state.original_squared_acf,serialized_state_bytes=max(initial_size,len(state.dumps())),
                array_bytes=state.array_bytes(),scales={})
            for kind,s in state.scales.items():
                record["scales"][kind] = dict(b_raw=s.b_raw,b=s.b,v_min=s.v_min,cap=s.cap,
                    clipped_mean=s.clipped_mean,factor=s.factor,robust_supported=s.robust_supported,
                    counts=s.counts,reference_squared_acf=s.reference_squared_acf,mu=s.mu,sd=s.sd)
            self.completed.append((rows,diagnostic,record))


def export_features(root):
    root = Path(root)
    frozen,manifest = admission(root)
    out = root/"features"
    if out.exists():
        verify_artifacts(out)
        result = json.loads((out/"metrics.json").read_text())
        if result["provenance"]!=provenance(frozen):
            raise ValueError("Incompatible export")
        return result
    out.mkdir()
    start = time.perf_counter()
    con = evaluator(root)
    writer = diag_writer = None
    records, frames, diag_frames = [], [], []
    exported = 0
    lookup = dict(zip(manifest.id,manifest.fold))
    observer = Exporter()
    try:
        for s,predictions,_ in replay(iter_series(con),infer_fn=observer.infer):
            values,diagnostic,detail = observer.completed.popleft()
            values = np.asarray(values,dtype=np.float64)
            if values.shape!=(len(predictions),len(RAW)) or not np.isfinite(values).all():
                raise ValueError("Invalid feature export")
            frame = pd.DataFrame(values,columns=RAW)
            frame = frame.assign(id=s.id,time=s.time,time_online=np.arange(len(predictions)),target=s.target,fold=lookup[s.id])
            diag = pd.DataFrame(diagnostic,columns=["e_squared","v_s","v_f","v_r"])
            diag = diag.assign(id=s.id,time_online=np.arange(len(predictions)))
            frames.append(frame)
            diag_frames.append(diag)
            detail["id"] = s.id
            records.append(detail)
            exported += len(frame)
            if len(frames)>=64:
                table = pa.Table.from_pandas(pd.concat(frames,ignore_index=True),preserve_index=False)
                diagnostic_table = pa.Table.from_pandas(pd.concat(diag_frames,ignore_index=True),preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out/"raw.parquet",table.schema,compression="zstd")
                    diag_writer = pq.ParquetWriter(out/"evaluator_scale.parquet",diagnostic_table.schema,compression="zstd")
                writer.write_table(table)
                diag_writer.write_table(diagnostic_table)
                frames.clear(); diag_frames.clear()
            if len(records)%500==0:
                print(f"Stage4 causal export: {len(records)}/10000 series",flush=True)
        if frames:
            table = pa.Table.from_pandas(pd.concat(frames,ignore_index=True),preserve_index=False)
            diagnostic_table = pa.Table.from_pandas(pd.concat(diag_frames,ignore_index=True),preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out/"raw.parquet",table.schema,compression="zstd")
                diag_writer = pq.ParquetWriter(out/"evaluator_scale.parquet",diagnostic_table.schema,compression="zstd")
            writer.write_table(table); diag_writer.write_table(diagnostic_table)
        writer.close(); writer = None
        diag_writer.close(); diag_writer = None
        if exported!=frozen["config"]["expected_rows"] or len(records)!=frozen["config"]["expected_series"]:
            raise AssertionError("Export population coverage mismatch")
        con.read_parquet(str(out/"raw.parquet")).create_view("s4_export",replace=True)
        con.read_parquet(str(ROOT/"stage2_results/E07/oof.parquet")).create_view("s4_e07",replace=True)
        clauses = " OR ".join(f"a.{k} IS DISTINCT FROM b.{k}" for k in (*BASE,"time","target","fold"))
        mismatches = con.execute(f"SELECT count(*) FROM s4_export a FULL JOIN s4_e07 b USING(id,time_online) WHERE a.id IS NULL OR b.id IS NULL OR {clauses}").fetchone()[0]
        if mismatches or con.execute("SELECT count(*) FROM (SELECT id,time_online FROM s4_export GROUP BY id,time_online HAVING count(*)!=1)").fetchone()[0]:
            raise AssertionError("Authoritative full-population R/I/P/D audit failed")
        write_json(out/"series_diagnostics.json",records)
        context = pd.DataFrame([dict(id=r["id"],rho=r["original_squared_acf"],
            group="unsupported" if r["original_squared_acf"] is None else "le0.1" if r["original_squared_acf"]<=.1 else "gt0.1") for r in records])
        context.to_parquet(out/"historical_context.parquet",index=False)
        support, acf = {}, {}
        for kind in ("slow","fast","robust"):
            support[kind] = {key:sum(r["scales"][kind]["counts"][key] for r in records) for key in records[0]["scales"][kind]["counts"]}
            before = [r["original_squared_acf"] for r in records if r["original_squared_acf"] is not None]
            after = [r["scales"][kind]["reference_squared_acf"] for r in records if r["scales"][kind]["reference_squared_acf"] is not None]
            changes = [r["scales"][kind]["reference_squared_acf"]-r["original_squared_acf"] for r in records if r["original_squared_acf"] is not None and r["scales"][kind]["reference_squared_acf"] is not None]
            acf[kind] = {key: distribution(v) for key,v in (("before",before),("after",after),("after_minus_before",changes))}
        init = sum(r["initialization_seconds"] for r in records)
        updates = sum(r["updates_seconds"] for r in records)
        result = dict(provenance=provenance(frozen),rows=exported,series=len(records),raw_columns=list(RAW),
            R_I_P_D_exact=True,raw_sha256=sha256(out/"raw.parquet"),evaluator_scale_sha256=sha256(out/"evaluator_scale.parquet"),
            initialization_seconds=init,updates_seconds=updates,feature_updates_per_second=exported/updates,
            export_wall_seconds=time.perf_counter()-start,state_array_bytes=quantiles([r["array_bytes"] for r in records]),
            serialized_state_bytes=quantiles([r["serialized_state_bytes"] for r in records]),support_counts=support,
            historical_squared_acf=acf,robust_factor=distribution([r["scales"]["robust"]["factor"] for r in records if r["scales"]["robust"]["factor"] is not None]),
            timing_note="One shared three-stream export; not single-candidate production throughput")
        admission(root)
        write_json(out/"metrics.json",result)
        finalize_artifacts(out)
        return result
    finally:
        if writer is not None: writer.close()
        if diag_writer is not None: diag_writer.close()
        con.close()


def distribution(values):
    a = np.asarray(values,dtype=np.float64)
    levels = [0,.05,.25,.5,.75,.95,1]
    return dict(count=len(a),levels=levels,values=np.quantile(a,levels,method="linear").tolist() if len(a) else None)


def load_frame(root,experiment):
    verify_artifacts(Path(root)/"features")
    needed = ["J_s","P_s"] if experiment=="E17" else [STANDALONE[experiment]]
    frame = pd.read_parquet(Path(root)/"features/raw.parquet",columns=["id","time","time_online","target","fold",*BASE,*needed])
    return frame.sort_values(["id","time_online"],kind="stable").reset_index(drop=True)


def embedding_audit(root,frame,fold,model,weights):
    old = LogisticModel.loads((ROOT/f"stage2_phase2/E08/models/fold{fold}.json").read_bytes())
    recorded = json.loads((ROOT/f"stage2_phase2/E08/models/fold{fold}_weights.json").read_text())
    if weights!=recorded or not np.array_equal(old.mean,model.pre.mean) or not np.array_equal(old.std,model.pre.std):
        raise AssertionError("Frozen E08 weights/base preprocessing changed")
    train = frame.loc[frame.fold!=fold]
    eligible,omega,_ = training_weights(train.time_online.to_numpy(),train.target.to_numpy())
    train = train.loc[eligible]
    x = model.pre.transform(train)
    theta = np.r_[old.intercept,old.beta,np.zeros(x.shape[1]-4)]
    y = train.target.to_numpy(dtype=np.float64)
    with threadpool_limits(limits=1):
        loss_error = abs(objective(theta,x,y,omega)[0]-objective(theta[:5],x[:,:4],y,omega)[0])
    embedded = Model(model.pre,old.intercept,theta[1:])
    error = 0.0
    for begin in range(0,len(frame),65536):
        chunk = frame.iloc[begin:begin+65536]
        error = max(error,float(np.max(np.abs(embedded.predict(chunk)-old.predict(chunk.loc[:,list(BASE)].to_numpy(dtype=float))))))
    audit = dict(E08_base_exact=True,E08_weights_exact=True,E08_objective_error=loss_error,E08_prediction_error=error)
    if loss_error>1e-14 or error>1e-14:
        raise AssertionError("E08 nested extension regression failed")
    if model.pre.experiment=="E17" and (Path(root)/f"E14/models/fold{fold}.json").exists():
        previous = Model.loads((Path(root)/f"E14/models/fold{fold}.json").read_bytes())
        if not np.array_equal(previous.pre.extra_mean,model.pre.extra_mean[:1]) or not np.array_equal(previous.pre.extra_std,model.pre.extra_std[:1]):
            raise AssertionError("E17 changed E14 contrast preprocessing")
        nested = Model(model.pre,previous.intercept,np.r_[previous.beta,0.])
        theta = np.r_[previous.intercept,previous.beta,0.]
        with threadpool_limits(limits=1):
            loss_error = abs(objective(theta,x,y,omega)[0]-objective(theta[:-1],x[:,:-1],y,omega)[0])
        error = 0.
        for begin in range(0,len(frame),65536):
            chunk = frame.iloc[begin:begin+65536]
            error = max(error,float(np.max(np.abs(nested.predict(chunk)-previous.predict(chunk)))))
        if loss_error>1e-14 or error>1e-14:
            raise AssertionError("E14 nested E17 regression failed")
        audit.update(E14_contrast_preprocessing_exact=True,E14_objective_error=loss_error,E14_prediction_error=error)
    return audit


def streaming_audit(root,frame,models,predictions):
    """Full-population fresh causal inference, not cached block timing."""
    con = evaluator(root)
    active = {}
    details = deque()
    lookup = {sid:(int(g.fold.iloc[0]),g.index.to_numpy()) for sid,g in frame.groupby("id",sort=False)}
    def records():
        for record in iter_series(con):
            active["model"] = models[lookup[record.id][0]]  # evaluator-only outer-fold routing
            yield record
    def infer_fn(datasets):
        yield
        for history,online in datasets:
            tick = time.perf_counter()
            state = Detector(history,active["model"])
            init = time.perf_counter()-tick
            initial_size = len(state.dumps())
            updates = 0.
            for value in online:
                tick = time.perf_counter()
                prediction = state.update(value)
                updates += time.perf_counter()-tick
                yield prediction
            details.append(dict(initialization_seconds=init,update_seconds=updates,
                state_array_bytes=state.features.array_bytes(),state_serialized_bytes=max(initial_size,len(state.dumps()))))
    start = time.perf_counter()
    rows,series,records_out = 0,0,[]
    try:
        for record,values,_ in replay(records(),infer_fn=infer_fn):
            indices = lookup[record.id][1]
            if not np.array_equal(values,predictions[indices]):
                raise AssertionError(f"Cached/causal full replay disagreement: {record.id}")
            rows += len(values); series += 1
            records_out.append(details.popleft())
            if series%2000==0:
                print(f"{models[0].pre.experiment} guarded full inference: {series}/10000",flush=True)
        if rows!=len(frame) or series!=len(lookup):
            raise AssertionError("Incomplete guarded inference")
        init = sum(r["initialization_seconds"] for r in records_out)
        updates = sum(r["update_seconds"] for r in records_out)
        return dict(series=series,rows=rows,all_predictions_exact=True,initialization_seconds=init,
            update_and_logistic_seconds=updates,end_to_end_inference_seconds=init+updates,
            end_to_end_predictions_per_second=rows/(init+updates),update_predictions_per_second=rows/updates,
            guarded_replay_wall_seconds=time.perf_counter()-start,
            state_array_bytes=quantiles([r["state_array_bytes"] for r in records_out]),
            serialized_state_bytes=quantiles([r["state_serialized_bytes"] for r in records_out]),
            timing_note="Full 10000-series fresh AR/scale/CDF initialization + causal updates + logistic; excludes evaluator I/O, serialization and verification. Guarded wall also reported.")
    finally:
        con.close()


def run_one(root,experiment):
    root = Path(root)
    if experiment not in IDS:
        raise ValueError("Only E14-E17 are selectable")
    frozen,manifest = admission(root)
    out = root/experiment
    if out.exists():
        verify_artifacts(out)
        old = json.loads((out/"metrics.json").read_text())
        if old["provenance"]!=provenance(frozen): raise ValueError("Incompatible completed result")
        return old
    out.mkdir()
    start = time.perf_counter()
    try:
        frame = load_frame(root,experiment)
        if len(frame)!=frozen["config"]["expected_rows"] or frame.id.nunique()!=frozen["config"]["expected_series"] or frame[["id","time_online"]].duplicated().any() or frame.groupby("id").fold.nunique().max()!=1:
            raise AssertionError("Invalid CV/full population coverage")
        predictions = np.full(len(frame),np.nan)
        seen = np.zeros(len(frame),dtype=bool)
        models,training,hashes = [],[],[]
        inference = 0.
        for fold in range(5):
            print(f"{experiment}: fitting outer fold {fold}",flush=True)
            model,detail,weights = fit_fold(frame,fold,experiment)
            detail["nested_audit"] = embedding_audit(root,frame,fold,model,weights)
            path = out/"models"/f"fold{fold}.json"
            path.parent.mkdir(exist_ok=True)
            with path.open("xb") as f: f.write(model.dumps())
            restored = Model.loads(path.read_bytes())
            if restored.dumps()!=model.dumps(): raise AssertionError("Model roundtrip mismatch")
            models.append(restored); hashes.append(sha256(path))
            positions = np.flatnonzero(frame.fold.to_numpy()==fold)
            valid_ids = sorted(frame.loc[frame.fold==fold,"id"].unique())
            train_ids = sorted(frame.loc[frame.fold!=fold,"id"].unique())
            if set(valid_ids)&set(train_ids): raise AssertionError("CV leakage")
            for begin in range(0,len(positions),65536):
                indices = positions[begin:begin+65536]
                tick = time.perf_counter()
                values = restored.predict(frame.iloc[indices])
                inference += time.perf_counter()-tick
                if seen[indices].any(): raise AssertionError("Duplicate OOF key")
                predictions[indices],seen[indices] = values,True
            detail.update(training_ids_sha256=canonical_hash(train_ids),validation_ids_sha256=canonical_hash(valid_ids),
                training_series=len(train_ids),validation_series=len(valid_ids),model_sha256=hashes[-1])
            write_json(out/"models"/f"fold{fold}_training.json",detail)
            write_json(out/"models"/f"fold{fold}_weights.json",weights)
            training.append(detail)
        if not seen.all() or not np.isfinite(predictions).all(): raise AssertionError("Incomplete OOF")
        # Audit the real inference path before revealing any candidate score.
        streamed = streaming_audit(root,frame,models,predictions)
        result_frame = frame[["id","time","time_online","target","fold"]].copy()
        result_frame["prediction"] = predictions
        result_frame.to_parquet(out/"oof.parquet",index=False,compression="zstd")
        con = evaluator(root)
        try:
            measured = score_outputs(con,out/"oof.parquet",manifest,out)
        finally: con.close()
        ref,ar = load_result(ROOT/"stage2_phase2/E08"),load_result(ROOT/"stage2_results/E04")
        parent_path = ROOT/"stage2_phase2/E08" if PARENTS[experiment]=="E08" else root/PARENTS[experiment]
        parent = json.loads((parent_path/"metrics.json").read_text()) if (parent_path/"metrics.json").exists() else None
        extra = contrasts(experiment,frame)
        correlations = []
        for j,against in enumerate(["I","P"] if experiment=="E17" else ["I"]):
            a,b = extra[:,j],frame[against].to_numpy()
            correlations.append(dict(extension=INPUTS[experiment][j+4],against=against,
                distribution=distribution(a),correlation=float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None))
        result = dict(**measured,status="valid",experiment_id=experiment,parent_id=PARENTS[experiment],
            provenance=provenance(frozen),model_inputs=INPUTS[experiment],model_input_count=len(INPUTS[experiment]),
            primitive_feature_count=30 if experiment=="E17" else 22,training_convergence_by_fold=training,
            training_seconds_by_fold=[v["training_seconds"] for v in training],
            training_seconds_total=sum(v["training_seconds"] for v in training),
            shared_model_bytes_by_fold=[len(m.dumps()) for m in models],streaming=streamed,
            cached_inference_seconds=inference,cached_rows_per_second=len(frame)/inference,
            feature_cache_sha256=sha256(root/"features/artifacts.sha256.json"),model_sha256=hashes,
            OOF_sha256=sha256(out/"oof.parquet"),delta_vs_E08=difference(measured,ref),delta_vs_E04=difference(measured,ar),
            delta_vs_parent=difference(measured,parent) if parent and parent["status"]=="valid" else None,
            extension_descriptive=correlations,peak_memory_bytes=peak_memory_bytes(),total_wall_seconds=time.perf_counter()-start)
        admission(root)
        write_json(out/"metrics.json",result)
        finalize_artifacts(out)
        print(f"{experiment}: valid TS-AUC={measured['pooled_oof_ts_auc']:.10f}; E08 delta={result['delta_vs_E08']['pooled']:+.10f}",flush=True)
        return result
    except Exception as exc:
        failure = dict(status="invalid",experiment_id=experiment,parent_id=PARENTS[experiment],provenance=provenance(frozen),
            exception_type=type(exc).__name__,message=str(exc),failure_category="numerical convergence" if isinstance(exc,InvalidFit) else "implementation/admission failure",
            total_wall_seconds=time.perf_counter()-start)
        if isinstance(exc,InvalidFit): failure["convergence"] = exc.diagnostics
        write_json(out/"metrics.json",failure)
        finalize_artifacts(out)
        raise


def gate(delta,early,valid=True,replicates=None):
    conditions = dict(valid_complete=bool(valid),**gate_conditions(delta,GATE),
        historical_early_guard=early is not None and early>=GATE["minimum_age_le64_vs_e01_delta"])
    q = None
    if replicates is not None:
        if len(replicates)!=999 or not np.isfinite(replicates).all(): raise ValueError("999 finite paired replicates required")
        q = float(np.quantile(replicates,.01,method="linear"))
    return dict(conditions=conditions,bootstrap_required=all(conditions.values()),family_size=4,Q_0_01=q,
        historical_early_delta=early,advances=bool(all(conditions.values()) and q is not None and q>0))


def winner(results):
    candidates = [r for r in results.values() if r["status"]=="valid" and r["gate"]["advances"]]
    return min(candidates,key=lambda r:(-r["pooled_oof_ts_auc"],r["model_input_count"],r["experiment_id"]))["experiment_id"] if candidates else "E08"


def subgroup_diagnostics(con,root,experiment,out,parent_path=None):
    con.read_parquet(str(Path(root)/experiment/"oof.parquet")).create_view("s4_candidate",replace=True)
    con.read_parquet(str(parent_path or ROOT/"stage2_phase2/E08/oof.parquet")).create_view("s4_reference",replace=True)
    con.read_parquet(str(Path(root)/"features/historical_context.parquet")).create_view("s4_context",replace=True)
    result = {}
    for group in ("le0.1","gt0.1","unsupported"):
        count = con.execute('SELECT count(*) FROM s4_context WHERE "group"=?',[group]).fetchone()[0]
        rows,reference = [],[]
        query = f'''SELECT a.id,a.time_online,a.target,a.prediction,b.prediction AS reference FROM s4_candidate a
            JOIN s4_reference b USING(id,time_online) JOIN s4_context c USING(id)
            WHERE c."group"='{group}' ORDER BY a.time_online,a.id'''
        for g in iter_frames(con,query,key="time_online"):
            rows.append(age_row(int(g.time_online.iloc[0]),g.target,g.prediction))
            reference.append(age_row(int(g.time_online.iloc[0]),g.target,g.reference))
        if rows:
            score,ages = aggregate(rows); other,_ = aggregate(reference)
            ages.to_csv(out/f"{experiment}_{group}_per_age.csv",index=False)
            result[group] = dict(series=count,pair_weight=int(ages.weight.sum()),ts_auc=score,reference=other,delta=score-other)
        else:
            result[group] = dict(series=count,pair_weight=0,ts_auc=None,reference=None,delta=None)
    return dict(subsets=result,note="Historical-only descriptive subsets, not model features or pooled AUC decomposition")


def absorption(con,root):
    con.read_parquet(str(Path(root)/"features/evaluator_scale.parquet")).create_view("s4_scale",replace=True)
    per_series = {k:{b:[] for b in ("1-32","33-128","129-256","257+")} for k in ("s","f","r")}
    query = """SELECT d.*,i.tau_index FROM s4_scale d JOIN series_index i USING(id)
        WHERE i.tau_index>=0 AND d.time_online>=i.tau_index ORDER BY d.id,d.time_online"""
    for g in iter_frames(con,query):
        elapsed = g.time_online.to_numpy()-int(g.tau_index.iloc[0])+1
        for kind in per_series:
            v = g["v_"+kind].to_numpy(); base = float(v[0]); e2 = g.e_squared.to_numpy()
            for name,lo,hi in (("1-32",1,32),("33-128",33,128),("129-256",129,256),("257+",257,None)):
                mask = (elapsed>=lo)&((elapsed<=hi) if hi else True)
                if mask.any():
                    per_series[kind][name].append(dict(id=str(g.id.iloc[0]),observations=int(mask.sum()),
                        energy_over_break=float(np.mean(e2[mask]/base)),energy_over_predictive=float(np.mean(e2[mask]/v[mask])),
                        predictive_over_break=float(np.mean(v[mask]/base))))
    summary = {}
    for kind,buckets in per_series.items():
        summary[kind] = {}
        for name,records in buckets.items():
            summary[kind][name] = dict(series=len(records),observations=sum(r["observations"] for r in records),
                statistics={key:dict(zip(("q25","median","q75"),np.quantile([r[key] for r in records],[.25,.5,.75],method="linear").tolist())) if records else None
                    for key in ("energy_over_break","energy_over_predictive","predictive_over_break")})
    return dict(summary=summary,series_values=per_series,evaluator_only=True,uses_partial_available_bins=True)


def standalone_diagnostics(con,root,manifest,out):
    con.read_parquet(str(Path(root)/"features/raw.parquet")).create_view("s4_raw",replace=True)
    result = {}
    for block in ("J_s","J_f","J_r","P_s","P"):
        directory = out/"standalone"/block
        directory.mkdir(parents=True)
        path = directory/"oof.parquet"
        con.execute(f"COPY (SELECT id,time,time_online,target,fold,{block} AS prediction FROM s4_raw ORDER BY id,time_online) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",[str(path)])
        result[block] = dict(**score_outputs(con,path,manifest,directory),selectable=False,artifact_sha256=sha256(path))
    e04 = load_result(ROOT/"stage2_results/E04")
    for block in ("J_s","J_f","J_r"):
        result[block]["delta_vs_E04"] = difference(result[block],e04)
    result["P_s"]["delta_vs_P"] = difference(result["P_s"],result["P"])
    return result


def compare(root):
    root = Path(root)
    frozen,manifest = admission(root)
    out = root/"comparisons"
    if out.exists():
        verify_artifacts(out)
        return json.loads((out/"stage4_report.json").read_text())
    out.mkdir()
    start = time.perf_counter()
    con = evaluator(root)
    results = {}
    try:
        e01 = load_result(ROOT/"stage2_results/E01_squared")
        for name in IDS:
            verify_artifacts(root/name)
            entry = json.loads((root/name/"metrics.json").read_text())
            if entry["provenance"]!=provenance(frozen): raise ValueError("Candidate provenance changed")
            if entry["status"]=="valid":
                early = difference(entry,e01)["combined"]["le64"]
                result_gate = gate(entry["delta_vs_E08"],early)
                boot = None
                if result_gate["bootstrap_required"]:
                    boot = paired_bootstrap(con,root/name/"oof.parquet",ROOT/"stage2_phase2/E08/oof.parquet",manifest,GATE)
                    result_gate = gate(entry["delta_vs_E08"],early,replicates=boot["delta_replicates"])
                    boot["Q_0_01"] = result_gate["Q_0_01"]
                    boot["multiplicity_guard"] = "Frozen one-sided 1% percentile guard, family four; conditional fixed OOF only"
                entry.update(gate=result_gate,bootstrap=boot,bootstrap_seconds=boot["seconds"] if boot else 0.,
                    subgroup_diagnostics=subgroup_diagnostics(con,root,name,out))
            results[name] = entry
            write_json(out/f"{name}.json",entry)
            print(f"{name}: {entry['status']}; G4 advances={entry.get('gate',{}).get('advances',False)}",flush=True)
        standalone = standalone_diagnostics(con,root,manifest,out)
        write_json(out/"standalone.json",standalone)
        absorbed = absorption(con,root)
        write_json(out/"break_absorption.json",absorbed)
        report = dict(provenance=provenance(frozen),experiments=results,family_size=4,final_reference=winner(results),
            standalone=standalone,break_absorption=absorbed["summary"],
            feature_export=json.loads((root/"features/metrics.json").read_text()),
            comparison_wall_seconds=time.perf_counter()-start,total_bootstrap_seconds=sum(r.get("bootstrap_seconds",0.) for r in results.values()),
            protected_after=protected(),source_unchanged=source()==frozen["source"],
            stopping_point="G4 report complete. No production refit, cloud submission or additional research.")
        admission(root)
        write_json(out/"stage4_report.json",report)
        finalize_artifacts(out)
        return report
    finally:
        con.close()


def batch(root):
    start = time.perf_counter()
    admission(root)
    export_features(root)
    codes = {}
    for name in IDS:
        command = [sys.executable,str(ROOT/"run_stage4.py"),"run","--experiment",name,"--out",str(root)]
        codes[name] = subprocess.run(command,cwd=ROOT,check=False).returncode
    result = compare(root)
    write_json(Path(root)/"batch_execution.json",dict(return_codes=codes,total_wall_seconds=time.perf_counter()-start,
        final_reference=result["final_reference"]))
    return result

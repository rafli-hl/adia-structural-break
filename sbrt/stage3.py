"""Immutable admission, one causal export, five independent fits, frozen G3."""
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits, threadpool_info
from .data import connect, iter_series, iter_frames, sha256
from .folds import load_v2
from .replay import replay
from .stage2 import check_cache, score_outputs, validate_coverage, load_result, difference, summarize_ages
from .comparison import gate_conditions, paired_bootstrap
from .metric import age_row, aggregate
from .logistic import LogisticModel, objective, training_weights, InvalidFit
from .provenance import (ROOT, canonical_hash, write_json, source_manifest, versions,
    finalize_artifacts, verify_artifacts, quantiles, peak_memory_bytes)
from .stage3_features import RAW, INPUT_LISTS, raw_values, context_q
from .stage3_models import TREE_PARAMS, Model, fit_fold

SPEC = ROOT / "research_specs/stage3_v1.json"
DOCUMENT_HASH = "35b21bc6e1f3c09f45a7ab49a31dd216872a4669c516ca34a40efc0ea2c1601e"
IDS = tuple(INPUT_LISTS)
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def load_spec():
    spec = json.loads(SPEC.read_text())
    if spec["authoritative_sha256"] != DOCUMENT_HASH or sha256(ROOT/spec["authoritative_document"]) != DOCUMENT_HASH:
        raise ValueError("Authoritative Stage-3 specification changed")
    if spec["execution_order"] != list(IDS) or spec["experiments"]["E13"]["parameters"] != TREE_PARAMS:
        raise ValueError("Resolved estimator contract mismatch")
    for name in IDS:
        if spec["experiments"][name]["inputs"] != INPUT_LISTS[name]:
            raise ValueError("Input allowlist mismatch")
    return spec


def protected(root):
    baseline = json.loads((Path(root)/"before/baseline.json").read_text())
    expected = baseline["protected_files"]
    if canonical_hash(expected) != baseline["protected_sha256"]:
        raise ValueError("Baseline inventory changed")
    for relative, digest in expected.items():
        if sha256(ROOT/relative) != digest:
            raise ValueError(f"Protected artifact modified: {relative}")
    for folder in baseline["protected_roots"]:
        actual = {p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob("*") if p.is_file()}
        if actual != {p for p in expected if p.startswith(folder+"/")}:
            raise ValueError(f"Protected artifact inventory changed: {folder}")
    for name, digest in baseline["source"]["files"].items():
        if name != "sbrt/cli.py" and sha256(ROOT/name) != digest:
            raise ValueError(f"Trusted numerical source changed: {name}")
    return dict(files=len(expected), sha256=baseline["protected_sha256"], unchanged=True)


def freeze(root, cache, folds, validation):
    root = Path(root)
    if (root/"freeze").exists():
        raise ValueError("Freeze already exists; never overwrite")
    spec = load_spec()
    manifest, settings = load_v2(cache, folds)
    source = source_manifest()
    verify_artifacts(validation)
    tests = json.loads((Path(validation)/"validation.json").read_text())
    log = (Path(validation)/"unittest.log").read_text()
    if (not tests["successful"] or tests["source_sha256"] != source["sha256"]
            or "test_stage3" not in log or tests["tests_run"] <= 55):
        raise ValueError("Require passing Stage-3 tests of current source")
    if settings["fold_manifest_sha256"] != spec["cv2_sha256"]:
        raise ValueError("Wrong immutable CV-v2")
    parent = load_result(ROOT/"stage2_phase2/E08")
    if sha256(ROOT/"stage2_phase2/E08/oof.parquet") != spec["parent_oof_sha256"] or parent["pooled_oof_ts_auc"] != spec["parent_score"]:
        raise ValueError("Wrong E08 parent")
    if any(os.environ.get(key) != "1" for key in THREADS):
        raise ValueError("Native environment threads must be 1 before imports")
    integrity = protected(root)
    record = dict(spec=spec, config_sha256=canonical_hash(spec), source=source,
        authoritative_spec_sha256=DOCUMENT_HASH, cv2=settings, versions=versions(),
        native_threads=threadpool_info(), protected=integrity,
        validation=str(Path(validation).resolve()), validation_sha256=sha256(Path(validation)/"artifacts.sha256.json"),
        dataset_sha256={k:v["sha256"] for k,v in check_cache(cache).items()},
        cache=str(Path(cache).resolve()), folds=str(Path(folds).resolve()))
    write_json(root/"freeze/freeze.json", record)
    finalize_artifacts(root/"freeze")
    return record


def check_evaluation_patch(root):
    """Audited SQL/report-only correction; never authorizes another model fit."""
    import ast
    incident = json.loads((Path(root)/"recovery/incident.json").read_text())
    current = source_manifest()
    before = incident["original_source"]
    changed = sorted(k for k in set(before["files"]) | set(current["files"])
                     if before["files"].get(k) != current["files"].get(k))
    if set(changed) != {"sbrt/stage3.py", "tests/test_stage3_pipeline.py"}:
        raise ValueError("Correction changed non-evaluator source")
    text = incident["stage3_source_before"]
    if hashlib.sha256(text.encode()).hexdigest() != before["files"]["sbrt/stage3.py"]:
        raise ValueError("Original source snapshot changed")
    allowed = {"check_evaluation_patch", "admission", "compare", "subgroup_diagnostics"}
    def unaffected(text):
        tree = ast.parse(text)
        tree.body = [node for node in tree.body if not (isinstance(node,ast.FunctionDef) and node.name in allowed)]
        return ast.dump(tree,include_attributes=False)
    if unaffected(text) != unaffected((ROOT/"sbrt/stage3.py").read_text()):
        raise ValueError("Correction altered model/export/bootstrap/gate execution")
    return dict(changed_files=changed, model_feature_gate_bootstrap_code_unchanged=True,
                original_source_sha256=before["sha256"], evaluation_source_sha256=current["sha256"])


def admission(root, evaluation_only=False):
    root = Path(root)
    verify_artifacts(root/"freeze")
    frozen = json.loads((root/"freeze/freeze.json").read_text())
    if load_spec() != frozen["spec"]:
        raise ValueError("Configuration changed since freeze")
    if source_manifest() != frozen["source"]:
        if not evaluation_only:
            raise ValueError("Source changed since final tests; additional model execution prohibited")
        correction_path = root/"recovery/evaluation_freeze"
        verify_artifacts(correction_path)
        correction = json.loads((correction_path/"freeze.json").read_text())
        if correction["source"] != source_manifest() or correction["original_source_sha256"] != frozen["source"]["sha256"]:
            raise ValueError("Evaluator correction source mismatch")
        audit = check_evaluation_patch(root)
        if correction["audit"] != audit:
            raise ValueError("Evaluator correction audit mismatch")
        verify_artifacts(correction["validation"])
        test = json.loads((Path(correction["validation"])/"validation.json").read_text())
        if not test["successful"] or test["source_sha256"] != correction["source"]["sha256"]:
            raise ValueError("Evaluator correction lacks passing tests")
    if versions() != frozen["versions"] or any(os.environ.get(key) != "1" for key in THREADS):
        raise ValueError("Dependency/thread environment drift")
    if sha256(Path(frozen["validation"])/"artifacts.sha256.json") != frozen["validation_sha256"]:
        raise ValueError("Validation artifact changed")
    verify_artifacts(frozen["validation"])
    protected(root)
    manifest, settings = load_v2(frozen["cache"], frozen["folds"])
    if settings != frozen["cv2"] or {k:v["sha256"] for k,v in check_cache(frozen["cache"]).items()} != frozen["dataset_sha256"]:
        raise ValueError("Dataset/folds changed")
    return frozen, manifest


def provenance(frozen):
    return dict(source_sha256=frozen["source"]["sha256"], config_sha256=frozen["config_sha256"],
        authoritative_spec_sha256=DOCUMENT_HASH, fold_manifest_sha256=frozen["cv2"]["fold_manifest_sha256"],
        dataset_sha256=frozen["dataset_sha256"], parent_oof_sha256=frozen["spec"]["parent_oof_sha256"])


class ExportObserver:
    def __init__(self):
        self.completed = deque()

    def __call__(self, event, state):
        if event == "initialize":
            self.current = dict(rows=[], r=state.ar.mse_ratio, q=context_q(state.ar.mse_ratio),
                                initial_state_bytes=len(state.dumps()))
        elif event == "update":
            if self.current["r"] != state.ar.mse_ratio:
                raise AssertionError("Historical context mutated online")
            values = raw_values(state)
            self.current["rows"].append([values[k] for k in RAW])
        else:
            self.current.update(serialized_state_bytes=max(self.current.pop("initial_state_bytes"),len(state.dumps())),
                state_array_bytes=state.array_bytes(), dependence_counts=state.dependence.counts,
                historical_dependence_disabled=state.dependence.reference_correlation is None)
            self.completed.append(self.current)


def export_features(root):
    root = Path(root)
    frozen, manifest = admission(root)
    out = root/"features"
    if out.exists():
        verify_artifacts(out)
        result = json.loads((out/"metrics.json").read_text())
        if result["provenance"] != provenance(frozen):
            raise ValueError("Feature cache incompatible")
        return result
    out.mkdir()
    started = time.perf_counter()
    con, writer = connect(frozen["cache"]), None
    observer, records, buffers, seen = ExportObserver(), [], [], set()
    init = updates = 0.0
    rows = ticks = 0
    try:
        for s, predictions, timing in replay(iter_series(con), detector="E07", state_observer=observer):
            if s.id in seen:
                raise AssertionError("Duplicate exported series")
            seen.add(s.id)
            record = observer.completed.popleft()
            values = np.asarray(record.pop("rows"), dtype=float)
            n = len(predictions)
            if values.shape != (n, len(RAW)) or not np.isfinite(values).all():
                raise AssertionError("Invalid exported feature values")
            record.update(id=s.id, unsupported=record["r"] is None, **timing)
            records.append(record)
            init += timing["initialization_seconds"]
            updates += timing["update_seconds"]
            ticks += max(n-1, 0)
            rows += n
            frame = pd.DataFrame(values, columns=RAW)
            frame.insert(0,"time_online",np.arange(n,dtype=np.int64))
            frame.insert(0,"time",s.time)
            frame.insert(0,"id",s.id)
            buffers.append(frame)
            if len(buffers) >= 32:
                table = pa.Table.from_pandas(pd.concat(buffers,ignore_index=True),preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out/"raw.partial.parquet",table.schema,compression="zstd")
                writer.write_table(table)
                buffers = []
            if len(seen)%1000 == 0:
                print(f"Stage-3 causal export: {len(seen)}/{len(manifest)}",flush=True)
        if buffers:
            table = pa.Table.from_pandas(pd.concat(buffers,ignore_index=True),preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out/"raw.partial.parquet",table.schema,compression="zstd")
            writer.write_table(table)
        writer.close()
        writer = None
        replay_wall = time.perf_counter()-started
        if seen != set(manifest.id) or rows != int(manifest.online_length.sum()) or observer.completed:
            raise AssertionError("Incomplete export")
        raw = out/"raw.parquet"
        (out/"raw.partial.parquet").replace(raw)
        con.read_parquet(str(raw)).create_view("stage3_raw",replace=True)
        con.read_parquet(str(ROOT/"stage2_results/E07/oof.parquet")).create_view("stage3_old",replace=True)
        checks = " OR ".join(f"a.{k} IS DISTINCT FROM b.{k}" for k in ("time",*RAW[:4],*RAW[8:]))
        bad = con.execute(f"SELECT count(*) FROM stage3_raw a FULL JOIN stage3_old b USING(id,time_online) WHERE a.id IS NULL OR b.id IS NULL OR {checks}").fetchone()[0]
        duplicates = con.execute("SELECT count(*) FROM (SELECT id,time_online FROM stage3_raw GROUP BY ALL HAVING count(*)!=1)").fetchone()[0]
        energy_error = con.execute("SELECT max(abs(I-(I32+I128+I256+Icumulative)/4)) FROM stage3_raw").fetchone()[0]
        con.read_parquet(str(ROOT/"stage2_phase2/E08/oof.parquet")).create_view("stage3_e08",replace=True)
        old_error = con.execute("SELECT count(*) FROM stage3_raw a JOIN stage3_e08 b USING(id,time_online) WHERE "+" OR ".join(f"a.{k} IS DISTINCT FROM b.{k}" for k in RAW[:4])).fetchone()[0]
        if bad or duplicates or old_error or energy_error > 1e-14:
            raise AssertionError("Export differs from immutable Stage-2 components")
        old = {v["id"]:v["ar_mse_ratio"] for v in json.loads((ROOT/"stage2_results/E04/series_diagnostics.json").read_text())}
        if any(row["r"] != old[row["id"]] for row in records):
            raise AssertionError("Historical AR context differs from E04")
        pd.DataFrame([{k:r[k] for k in ("id","r","q","unsupported")} for r in records]).to_parquet(out/"context.parquet",index=False)
        write_json(out/"series_diagnostics.json",records)
        support = {str(w):{key:sum(v["dependence_counts"].get(str(w),{}).get(key,0) for v in records)
            for key in ("supported","insufficient_pairs","online_degenerate","historical_disabled")} for w in (128,256)}
        result = dict(provenance=provenance(frozen), rows=rows, series=len(seen), raw_columns=list(RAW),
            raw_sha256=sha256(raw), context_sha256=sha256(out/"context.parquet"),
            R_I_P_D_exact=True, rank_subblocks_exact=True, energy_max_error=energy_error, context_exact=True,
            unsupported_context=sum(r["unsupported"] for r in records), context_q=quantiles([r["q"] for r in records]),
            context_q_quantile_levels=[0,.05,.25,.5,.75,.95,1],
            context_q_quantiles=np.quantile([r["q"] for r in records],[0,.05,.25,.5,.75,.95,1]).tolist(),
            context_q_zero_series=sum(r["q"]==0 for r in records),
            initialization_seconds=init, guarded_update_seconds=updates, guarded_update_rows=ticks,
            guarded_feature_updates_per_second=ticks/updates, export_replay_wall_seconds=replay_wall,
            feature_export_wall_seconds=time.perf_counter()-started,
            feature_state_serialized_bytes=quantiles([r["serialized_state_bytes"] for r in records]),
            feature_state_array_bytes=quantiles([r["state_array_bytes"] for r in records]), support_counts=support,
            historical_dependence_disabled=sum(r["historical_dependence_disabled"] for r in records),
            throughput_note="Guarded causal feature export, not end-to-end Stage-3 model inference; includes observer capture, excludes first tick.")
        admission(root)
        write_json(out/"metrics.json",result)
        finalize_artifacts(out)
        return result
    finally:
        if writer is not None:
            writer.close()
        con.close()


def load_frame(con, root, experiment):
    verify_artifacts(Path(root)/"features")
    con.read_parquet(str(Path(root)/"features/raw.parquet")).create_view("s3_raw",replace=True)
    con.read_parquet(str(ROOT/"stage2_phase2/E08/oof.parquet")).create_view("s3_parent",replace=True)
    con.read_parquet(str(Path(root)/"features/context.parquet")).create_view("s3_context",replace=True)
    extra = list(RAW[4:8]) if experiment == "E10" else list(RAW[8:]) if experiment == "E11" else []
    columns = ",".join("a."+k for k in [*RAW[:4],*extra])
    return con.execute(f"SELECT a.id,a.time,a.time_online,b.target,b.fold,{columns},c.q FROM s3_raw a JOIN s3_parent b USING(id,time_online) JOIN s3_context c USING(id) ORDER BY a.id,a.time_online").df()


def audit_e08_embedding(frame, fold, model, weights):
    old = LogisticModel.loads((ROOT/f"stage2_phase2/E08/models/fold{fold}.json").read_bytes())
    recorded = json.loads((ROOT/f"stage2_phase2/E08/models/fold{fold}_weights.json").read_text())
    if any(recorded[k] != weights[k] for k in recorded):
        raise AssertionError("E08 weights/eligibility changed")
    if not np.array_equal(old.mean,model.pre.mean) or not np.array_equal(old.std,model.pre.std):
        raise AssertionError("E08 base transformation changed")
    train = frame.loc[frame.fold != fold]
    eligible, omega, _ = training_weights(train.time_online.to_numpy(),train.target.to_numpy())
    train = train.loc[eligible]
    x = model.pre.transform(train)
    theta = np.r_[old.intercept,old.beta,np.zeros(x.shape[1]-4)]
    with threadpool_limits(limits=1):
        old_value = objective(theta[:5], x[:,:4], train.target.to_numpy(dtype=float),omega)[0]
        new_value = objective(theta,x,train.target.to_numpy(dtype=float),omega)[0]
    prediction_error = 0.0
    embedded = None if model.pre.experiment == "E13" else Model(model.pre,old.intercept,theta[1:])
    if embedded is not None:
        for begin in range(0,len(frame),65536):
            chunk = frame.iloc[begin:begin+65536]
            prediction_error = max(prediction_error,float(np.max(np.abs(embedded.predict(chunk)-old.predict(chunk.loc[:,list(RAW[:4])].to_numpy(dtype=float))))))
    if abs(old_value-new_value)>1e-14 or prediction_error>1e-14:
        raise AssertionError("Nested E08 equivalence failed")
    return dict(base_mean_std_exact=True, weights_exact=True, objective_absolute_error=abs(old_value-new_value), prediction_max_error=prediction_error)


def run_one(root, experiment):
    root = Path(root)
    if experiment not in IDS:
        raise ValueError("Only frozen E09-E13")
    frozen, manifest = admission(root)
    out = root/experiment
    if out.exists():
        verify_artifacts(out)
        result = json.loads((out/"metrics.json").read_text())
        if result["provenance"] != provenance(frozen):
            raise ValueError("Incompatible completed result")
        return result
    out.mkdir()
    started = time.perf_counter()
    con = connect(frozen["cache"])
    try:
        export = json.loads((root/"features/metrics.json").read_text())
        frame = load_frame(con,root,experiment)
        if len(frame)!=frozen["spec"]["expected_rows"] or frame.id.nunique()!=frozen["spec"]["expected_series"]:
            raise AssertionError("Full-population training coverage mismatch")
        if frame[["id","time_online"]].duplicated().any() or frame.groupby("id").fold.nunique().max()!=1:
            raise AssertionError("Duplicate keys or CV leakage")
        predictions = np.empty(len(frame))
        seen = np.zeros(len(frame),dtype=bool)
        diagnostics, hashes = [], []
        inference = replay_wall = 0.0
        for fold in range(5):
            print(f"{experiment}: fitting outer fold {fold}",flush=True)
            model, detail, weights = fit_fold(frame,fold,experiment)
            detail["e08_base_audit"] = audit_e08_embedding(frame,fold,model,weights)
            payload = model.dumps()
            path = out/"models"/f"fold{fold}.{'pkl' if experiment=='E13' else 'json'}"
            path.parent.mkdir(exist_ok=True)
            with path.open("xb") as f:
                f.write(payload)
            hashes.append(sha256(path))
            restored = Model.loads(path.read_bytes(), native=experiment=="E13")
            positions = np.flatnonzero(frame.fold.to_numpy()==fold)
            valid_ids = sorted(frame.loc[frame.fold==fold,"id"].unique())
            train_ids = sorted(frame.loc[frame.fold!=fold,"id"].unique())
            if set(valid_ids)&set(train_ids):
                raise AssertionError("CV leakage")
            detail.update(training_ids_sha256=canonical_hash(train_ids),validation_ids_sha256=canonical_hash(valid_ids),
                          training_series=len(train_ids),validation_series=len(valid_ids), model_sha256=hashes[-1])
            tick_replay = time.perf_counter()
            for begin in range(0,len(positions),65536):
                indices = positions[begin:begin+65536]
                chunk = frame.iloc[indices]
                tick = time.perf_counter()
                values = restored.predict(chunk)
                inference += time.perf_counter()-tick
                if not np.array_equal(values,model.predict(chunk)):
                    raise AssertionError("Save/load predictions changed")
                if seen[indices].any():
                    raise AssertionError("Duplicate OOF prediction")
                predictions[indices], seen[indices] = values, True
            replay_wall += time.perf_counter()-tick_replay
            write_json(out/"models"/f"fold{fold}_training.json",detail)
            write_json(out/"models"/f"fold{fold}_weights.json",weights)
            diagnostics.append(detail)
        if not seen.all():
            raise AssertionError("Incomplete OOF coverage")
        result_frame = frame[["id","time","time_online","target","fold"]].copy()
        result_frame["prediction"] = predictions
        result_frame.to_parquet(out/"oof.parquet",index=False,compression="zstd")
        measured = score_outputs(con,out/"oof.parquet",manifest,out)
        # Explicit score tie diagnostic: fraction of within-age unordered row pairs tied.
        pairs = tied = 0
        positive_negative_pairs = positive_negative_ties = 0
        duplicate_rows = 0
        for _, group in result_frame.groupby("time_online",sort=True):
            counts = group.groupby("prediction").target.agg(["size","sum"])
            n = len(group)
            pairs += n*(n-1)//2
            tied += int((counts["size"]*(counts["size"]-1)//2).sum())
            duplicate_rows += n-len(counts)
            pos = int(group.target.sum())
            positive_negative_pairs += pos*(n-pos)
            positive_negative_ties += int((counts["sum"]*(counts["size"]-counts["sum"])).sum())
        fold_age_diagnostics = []
        for fold in range(5):
            candidate_age = pd.read_csv(out/f"per_age_fold{fold}.csv")
            parent_age = pd.read_csv(ROOT/f"stage2_phase2/E08/per_age_fold{fold}.csv")
            cb,cc = summarize_ages(candidate_age)
            pb,pc = summarize_ages(parent_age)
            fold_age_diagnostics.append(dict(fold=fold, candidate_buckets=cb, parent_buckets=pb,
                bucket_deltas={a["age_range"]:a["ts_auc"]-b["ts_auc"] if a["ts_auc"] is not None and b["ts_auc"] is not None else None for a,b in zip(cb,pb)},
                candidate_combined=cc,parent_combined=pc,
                combined_deltas={k:cc[k]["ts_auc"]-pc[k]["ts_auc"] if cc[k]["ts_auc"] is not None and pc[k]["ts_auc"] is not None else None for k in cc}))
        parent = load_result(ROOT/"stage2_phase2/E08")
        ar = load_result(ROOT/"stage2_results/E04")
        result = dict(**measured, provenance=provenance(frozen), status="valid", experiment_id=experiment,parent_id="E08",
            model_inputs=INPUT_LISTS[experiment],model_input_count=len(INPUT_LISTS[experiment]),primitive_feature_count=18,
            fold_age_diagnostics=fold_age_diagnostics,
            resolved_config=frozen["spec"], training_convergence_by_fold=diagnostics,
            training_seconds_by_fold=[d["training_seconds"] for d in diagnostics],
            training_seconds_total=sum(d["training_seconds"] for d in diagnostics),
            cached_inference_seconds=inference,cached_model_rows_per_second=len(frame)/inference,cached_replay_wall_seconds=replay_wall,
            shared_model_bytes_by_fold=[d["serialized_model_bytes"] for d in diagnostics],
            shared_model_bytes_total=sum(d["serialized_model_bytes"] for d in diagnostics),
            per_series_context_bytes=8 if experiment=="E09" else 0,
            feature_export=export, feature_cache_sha256=sha256(root/"features/artifacts.sha256.json"),
            model_sha256=hashes,prediction_artifact_sha256=sha256(out/"oof.parquet"),
            within_age_row_pair_tie_rate=tied/pairs,within_age_positive_negative_pair_tie_rate=positive_negative_ties/positive_negative_pairs,
            within_age_duplicate_row_fraction=duplicate_rows/len(frame),
            delta_vs_parent=difference(measured,parent),delta_vs_E04=difference(measured,ar)["pooled"],
            native_environment=versions(),peak_memory_bytes=peak_memory_bytes(),
            total_wall_seconds=time.perf_counter()-started,bootstrap_seconds=0.,advancement="pending comparisons",
            timing_notes="Training, cached prediction and scoring measured here. Causal feature export is shared once, separately reported; no end-to-end model streaming throughput claimed.")
        admission(root)
        write_json(out/"metrics.json",result)
        finalize_artifacts(out)
        print(f"{experiment}: valid TS-AUC={result['pooled_oof_ts_auc']:.10f}; E08 delta={result['delta_vs_parent']['pooled']:+.10f}",flush=True)
        return result
    except Exception as exc:
        failure = dict(status="invalid",experiment_id=experiment,parent_id="E08",provenance=provenance(frozen),
            exception_type=type(exc).__name__,message=str(exc),total_wall_seconds=time.perf_counter()-started,
            failure_category="numerical convergence" if isinstance(exc,InvalidFit) else "implementation/admission failure")
        if isinstance(exc,InvalidFit):
            failure["convergence"] = exc.diagnostics
        write_json(out/"metrics.json",failure)
        finalize_artifacts(out)
        raise
    finally:
        con.close()


def gate(delta, early_vs_e01, settings, replicates=None):
    conditions = gate_conditions(delta,settings)
    conditions["historical_early_guard"] = early_vs_e01 is not None and early_vs_e01 >= settings["minimum_age_le64_vs_e01_delta"]
    q = None
    if replicates is not None:
        if len(replicates)!=999 or not np.isfinite(replicates).all():
            raise ValueError("Exactly 999 finite paired replicates required")
        q = float(np.quantile(replicates,.01,method="linear"))
    return dict(conditions_1_to_5=conditions, bootstrap_required=all(conditions.values()),
        historical_early_delta=early_vs_e01, family_size=5,Q_0_01=q,
        uncertainty_passes=q is not None and q>0,
        advances=bool(all(conditions.values()) and q is not None and q>0))


def winner(results):
    candidates = [r for r in results.values() if r["status"]=="valid" and r["gate"]["advances"]]
    return min(candidates,key=lambda r:(-r["pooled_oof_ts_auc"],r["model_input_count"],r["experiment_id"]))["experiment_id"] if candidates else "E08"


def subgroup_diagnostics(con,root,out,parent_path=None):
    con.read_parquet(str(root/"E09/oof.parquet")).create_view("s3_candidate",replace=True)
    con.read_parquet(str(parent_path or ROOT/"stage2_phase2/E08/oof.parquet")).create_view("s3_parent",replace=True)
    con.read_parquet(str(root/"features/context.parquet")).create_view("s3_context",replace=True)
    results = {}
    for name, condition in (("r_lt_0.9","r<0.9"),("r_0.9_to_1","r>=0.9 AND r<=1"),("r_gt_1","r>1"),("unsupported","unsupported")):
        count = con.execute(f"SELECT count(*) FROM s3_context WHERE {condition}").fetchone()[0]
        if not count:
            results[name] = dict(series=0,pair_weight=0,ts_auc=None,reference=None,delta=None)
            continue
        rows, reference = [], []
        qualified = condition.replace("unsupported","c.unsupported").replace("r<","c.r<").replace("r>","c.r>")
        query = f"SELECT a.time_online,a.target,a.prediction,b.prediction AS reference FROM s3_candidate a JOIN s3_parent b USING(id,time_online) JOIN s3_context c USING(id) WHERE {qualified} ORDER BY a.time_online,a.id"
        for g in iter_frames(con,query,key="time_online"):
            rows.append(age_row(int(g.time_online.iloc[0]),g.target,g.prediction))
            reference.append(age_row(int(g.time_online.iloc[0]),g.target,g.reference))
        score, ages = aggregate(rows)
        other, _ = aggregate(reference)
        ages.to_csv(out/f"E09_{name}_per_age.csv",index=False)
        results[name] = dict(series=count,pair_weight=int(ages.weight.sum()),ts_auc=score,reference=other,delta=score-other)
    return dict(subsets=results,note="Descriptive historical-only subsets; subset AUCs do not decompose pooled AUC")


def compare(root):
    root = Path(root)
    frozen, manifest = admission(root,evaluation_only=True)
    out = root/"comparisons"
    if out.exists():
        verify_artifacts(out)
        return json.loads((out/"stage3_report.json").read_text())
    out.mkdir()
    started = time.perf_counter()
    con = connect(frozen["cache"])
    results = {}
    try:
        e01 = load_result(ROOT/"stage2_results/E01_squared")
        for name in IDS:
            verify_artifacts(root/name)
            entry = json.loads((root/name/"metrics.json").read_text())
            if entry["provenance"]!=provenance(frozen):
                raise ValueError("Candidate source/config mismatch")
            if entry["status"]=="valid":
                early = difference(entry,e01)["combined"]["le64"]
                result_gate = gate(entry["delta_vs_parent"],early,frozen["spec"]["gate"])
                boot = None
                if result_gate["bootstrap_required"]:
                    boot = paired_bootstrap(con,root/name/"oof.parquet",ROOT/"stage2_phase2/E08/oof.parquet",manifest,frozen["spec"]["gate"])
                    result_gate = gate(entry["delta_vs_parent"],early,frozen["spec"]["gate"],boot["delta_replicates"])
                    boot["Q_0_01"] = result_gate["Q_0_01"]
                    boot["multiplicity_guard"] = "approximate one-sided 0.05/5 conditional percentile guard; no refitting or global research-history adjustment"
                entry.update(gate=result_gate,bootstrap=boot,bootstrap_seconds=boot["seconds"] if boot else 0.)
                entry["total_wall_seconds"] += entry["bootstrap_seconds"]
                if name=="E09":
                    entry["subgroup_diagnostics"] = subgroup_diagnostics(con,root,out)
            results[name] = entry
            write_json(out/f"{name}.json",entry)
            print(f"{name}: status={entry['status']}; advances={entry.get('gate',{}).get('advances',False)}",flush=True)
        report = dict(provenance=provenance(frozen),experiments=results,final_reference=winner(results),
            family_size=5,comparison_wall_seconds=time.perf_counter()-started,
            total_bootstrap_seconds=sum(r.get("bootstrap_seconds",0.) for r in results.values()),
            feature_export=json.loads((root/"features/metrics.json").read_text()),
            measured_model_source_sha256=frozen["source"]["sha256"],
            evaluation_source_sha256=source_manifest()["sha256"],
            evaluator_correction=check_evaluation_patch(root) if source_manifest()!=frozen["source"] else None,
            protected_after=protected(root),
            stopping_point="Stage-3 complete; no combined extensions, tuning, pairwise ranking or Stage-4 work")
        admission(root,evaluation_only=True)
        write_json(out/"stage3_report.json",report)
        finalize_artifacts(out)
        return report
    finally:
        con.close()


def batch(root):
    started = time.perf_counter()
    admission(root)
    export_features(root)
    codes = {}
    for name in IDS:
        command = [sys.executable,str(ROOT/"run_research.py"),"stage3-run","--out",str(root),"--id",name]
        codes[name] = subprocess.run(command,check=False).returncode
    # All independent candidates run even if a numerical fold fails.
    report = compare(root)
    write_json(Path(root)/"batch_execution.json",dict(return_codes=codes,total_wall_seconds=time.perf_counter()-started,final_reference=report["final_reference"]))
    return report

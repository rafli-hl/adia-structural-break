"""Disk-backed Phase-1 execution; trusted replay/scoring remain authoritative."""
from collections import deque
from pathlib import Path
import json
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .backtest import e00, score_file
from .data import connect, iter_series, iter_frames, sha256
from .detectors import E01
from .experiments import load_spec, execution_order, PHASE1
from .folds import load_v2
from .metric import age_row, aggregate, bucket_summary
from .provenance import (ROOT, canonical_hash, write_json, source_manifest, versions,
                         quantiles, peak_memory_bytes, finalize_artifacts, verify_artifacts)
from .replay import replay


def check_cache(cache):
    cache = Path(cache)
    status = json.loads((cache / "profile_status.json").read_text())
    if status["label_alignment"] != "pass" or status.get("unresolved_repeated_block_pairs", 0):
        raise ValueError("Profile/label audit blocks Phase 1")
    if any(status["counts"].get(k, 0) for k in
           ("nonfinite_observations", "empty_online", "empty_historical", "nonmonotone_time_series")):
        raise ValueError("Invalid dataset for frozen Phase-1 contract")
    sources = json.loads((cache / "sources.json").read_text())
    if {k: v["sha256"] for k, v in sources.items()} != status["source_sha256"]:
        raise ValueError("Dataset source/profile mismatch")
    return sources


def context(cache, folds, spec_path):
    spec = load_spec(spec_path)
    manifest, settings = load_v2(cache, folds)
    source = source_manifest()
    result = dict(resolved_config_sha256=canonical_hash(spec),
                  fold_manifest_sha256=settings["fold_manifest_sha256"],
                  dataset_sha256={k: v["sha256"] for k, v in check_cache(cache).items()},
                  source_sha256=source["sha256"], dependency_versions=versions())
    return spec, manifest, settings, source, result


def validate_coverage(con, path, manifest):
    con.read_parquet(str(path)).create_view("coverage_predictions", replace=True)
    con.register("coverage_manifest", manifest[["id", "fold"]])
    con.execute("""CREATE OR REPLACE TEMP VIEW expected_online AS
        SELECT id,time,target,row_number() OVER(PARTITION BY id ORDER BY ordinal)-1 AS time_online
        FROM observations WHERE period=2""")
    count = int(con.execute("SELECT count(*) FROM coverage_predictions").fetchone()[0])
    if count != int(manifest.online_length.sum()):
        raise ValueError("Incomplete OOF row count")
    if con.execute("""SELECT count(*) FROM (SELECT id,time_online FROM coverage_predictions
        GROUP BY id,time_online HAVING count(*) != 1)""").fetchone()[0]:
        raise ValueError("Duplicate OOF keys")
    bad = con.execute("""SELECT count(*) FROM coverage_predictions p
        LEFT JOIN expected_online e USING(id,time_online)
        LEFT JOIN coverage_manifest m ON p.id=m.id
        WHERE e.id IS NULL OR p.time IS DISTINCT FROM e.time OR p.target IS DISTINCT FROM e.target
        OR p.fold IS DISTINCT FROM m.fold OR p.prediction IS NULL OR NOT isfinite(p.prediction)
        OR p.prediction < 0 OR p.prediction > 1""").fetchone()[0]
    if bad:
        raise ValueError("OOF key/label/fold/prediction coverage mismatch")
    return count


def summarize_ages(per_age):
    buckets = bucket_summary(per_age).to_dict("records")
    for row in buckets:
        if row["pair_weight"] == 0:
            row["ts_auc"] = None
        lo = int(row["age_range"].split("-")[0].rstrip("+"))
        hi = int(row["age_range"].split("-")[1]) if "-" in row["age_range"] else None
        g = per_age[(per_age.age >= lo) & ((per_age.age <= hi) if hi else True)]
        row["positive_rows"] = int(g.positive.sum())
        row["negative_rows"] = int(g.negative.sum())
    combined = {}
    for name, mask in (("le64", per_age.age <= 64), ("ge129", per_age.age >= 129),
                       ("ge257", per_age.age >= 257)):
        g = per_age[mask]
        weight = int(g.weight.sum())
        combined[name] = dict(ts_auc=float((g.auc.fillna(0) * g.weight).sum() / weight) if weight else None,
                              pair_weight=weight, eligible_ages=int((g.weight > 0).sum()),
                              positive_rows=int(g.positive.sum()), negative_rows=int(g.negative.sum()))
    return buckets, combined


def score_outputs(con, path, manifest, out):
    start = time.perf_counter()
    count = validate_coverage(con, path, manifest)
    score, ages, folds = score_file(con, path, count)
    ages.to_csv(out / "per_age.csv", index=False)
    buckets, combined = summarize_ages(ages)
    pd.DataFrame(buckets).to_csv(out / "age_buckets.csv", index=False)
    for fold, (_, per_age) in enumerate(folds):
        per_age.to_csv(out / f"per_age_fold{fold}.csv", index=False)
    return dict(pooled_oof_ts_auc=score, fold_ts_auc=[float(v[0]) for v in folds],
                age_bucket_ts_auc=buckets, combined_age_ts_auc=combined,
                scored_rows=count, series_count=len(manifest),
                scoring_seconds=time.perf_counter() - start)


def difference(candidate, parent):
    parent_buckets = {v["age_range"]: v for v in parent["age_bucket_ts_auc"]}
    def subtract(a, b):
        return a - b if a is not None and b is not None else None
    return dict(pooled=candidate["pooled_oof_ts_auc"] - parent["pooled_oof_ts_auc"],
                folds=[a - b for a, b in zip(candidate["fold_ts_auc"], parent["fold_ts_auc"])],
                age_buckets={v["age_range"]: subtract(v["ts_auc"], parent_buckets[v["age_range"]]["ts_auc"])
                             for v in candidate["age_bucket_ts_auc"]},
                combined={k: subtract(v["ts_auc"], parent["combined_age_ts_auc"][k]["ts_auc"])
                          for k, v in candidate["combined_age_ts_auc"].items()})


def load_result(directory, expected_context=None):
    directory = Path(directory)
    verify_artifacts(directory)
    result = json.loads((directory / "metrics.json").read_text())
    if result["status"] != "valid":
        raise ValueError("Cannot reuse an invalid/incomplete experiment")
    if expected_context:
        for key, value in expected_context.items():
            if result[key] != value:
                raise ValueError(f"Incompatible experiment provenance: {key}")
    return result


def rescore_e01(cache, folds, spec_path, original, out, memory="1GB"):
    spec, manifest, settings, source, ctx = context(cache, folds, spec_path)
    original, out = Path(original), Path(out)
    name = original.stem.removeprefix("oof_")
    if name not in E01:
        raise ValueError("Rescore requires a trusted E01 detector")
    original_hash = sha256(original)
    if out.exists():
        result = load_result(out, ctx)
        if result.get("original_prediction_artifact_sha256") != original_hash:
            raise ValueError("Original E01 artifact changed")
        return result
    out.mkdir(parents=True)
    started = time.perf_counter()
    con = connect(cache, memory)
    try:
        con.register("cv2", manifest[["id", "fold"]])
        con.read_parquet(str(original)).create_view("original_e01", replace=True)
        partial = out / "oof.partial.parquet"
        con.execute("""COPY (SELECT p.* EXCLUDE(fold), f.fold FROM original_e01 p
            JOIN cv2 f USING(id) ORDER BY id,time_online) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)""", [str(partial)])
        count = validate_coverage(con, partial, manifest)
        original_score = score_file(con, original, count)[0]
        measured = score_outputs(con, partial, manifest, out)
        if abs(measured["pooled_oof_ts_auc"] - original_score) > 1e-12:
            raise AssertionError("E01 pooled score changed under CV-v2")
        recorded_path = original.parent / "e01_results.json"
        recorded = next(v for v in json.loads(recorded_path.read_text()) if v["detector"] == name)
        if abs(original_score - recorded["pooled_oof_ts_auc"]) > 1e-12:
            raise AssertionError("Original E01 artifact does not reproduce recorded score")
        con.read_parquet(str(partial)).create_view("rescored_e01", replace=True)
        if con.execute("""SELECT count(*) FROM original_e01 a FULL JOIN rescored_e01 b USING(id,time_online)
            WHERE a.prediction IS DISTINCT FROM b.prediction OR a.id IS NULL OR b.id IS NULL""").fetchone()[0]:
            raise AssertionError("E01 prediction values changed")
        final = out / "oof.parquet"
        partial.replace(final)
        result = dict(**ctx, **measured, experiment_id=f"E01_{name}", parent_id=None,
                      dependency_ids=[], status="valid", original_pooled_ts_auc=original_score,
                      recorded_pooled_ts_auc=recorded["pooled_oof_ts_auc"],
                      pooled_invariance_absolute_delta=abs(measured["pooled_oof_ts_auc"] - original_score),
                      original_prediction_artifact_sha256=original_hash,
                      prediction_artifact_sha256=sha256(final),
                      total_wall_seconds=time.perf_counter() - started,
                      seed_manifest=dict(cv2=20260919), execution="rescoring unchanged E01 predictions")
        if sha256(original) != original_hash or source_manifest()["sha256"] != source["sha256"]:
            raise AssertionError("Source or original predictions changed during rescore")
        write_json(out / "metrics.json", result)
        write_json(out / "source_manifest.json", source)
        write_json(out / "resolved_config.json", spec)
        finalize_artifacts(out)
        print(f"{result['experiment_id']}: {result['pooled_oof_ts_auc']:.10f}; invariance PASS", flush=True)
        return result
    finally:
        con.close()


class Observer:
    def __init__(self):
        self.completed = deque()
        self.columns = None

    def __call__(self, event, state):
        if event == "initialize":
            self.current = dict(rows=[], initial_state_bytes=len(state.dumps()))
        elif event == "update":
            if self.columns is None:
                self.columns = list(state.last_blocks)
                if state.experiment == "E02":
                    self.columns += ["cumulative"]
            values = [state.last_blocks[k] for k in self.columns if k != "cumulative"]
            if "cumulative" in self.columns:
                values += [state.last_features[3]]
            self.current["rows"].append(values)
        else:
            self.current.update(serialized_state_bytes=max(self.current.pop("initial_state_bytes"), len(state.dumps())),
                                state_array_bytes=state.array_bytes(),
                                ar_mse_ratio=state.ar.mse_ratio if state.ar else None,
                                dependence_counts=state.dependence.counts if state.dependence else {},
                                historical_dependence_disabled=bool(state.dependence and state.dependence.reference_correlation is None))
            self.completed.append(self.current)


def run_one(cache, folds, spec_path, name, root_out, memory="1GB"):
    if name not in PHASE1:
        raise ValueError("Phase 1 stops before E08")
    spec, manifest, settings, source, ctx = context(cache, folds, spec_path)
    config = spec["experiments"][name]
    root_out = Path(root_out)
    dependencies = set(config["dependencies"] + [config["parent"]])
    dependency_results = {d: load_result(root_out / d, ctx) for d in sorted(dependencies)}
    out = root_out / name
    if out.exists():
        return load_result(out, ctx)
    out.mkdir(parents=True)
    start = time.perf_counter()
    con = connect(cache, memory)
    writer = None
    try:
        write_json(out / "resolved_config.json", spec)
        write_json(out / "source_manifest.json", source)
        write_json(out / "e00.json", e00())
        manifest.to_parquet(out / "fold_manifest.parquet", index=False)
        observer, seen, rows, series_records, buffers = Observer(), set(), 0, [], []
        lookup = dict(zip(manifest.id, manifest.fold))
        partial = out / "oof.partial.parquet"
        initialization = updates = 0.0
        timed_updates = ties = saturation = 0
        for s, predictions, timing in replay(iter_series(con), detector=name, state_observer=observer):
            if s.id not in lookup or s.id in seen or len(predictions) != len(s.target):
                raise AssertionError("Replay coverage mismatch")
            seen.add(s.id)
            detail = observer.completed.popleft()
            component_rows = np.asarray(detail.pop("rows"), dtype=np.float64)
            if len(component_rows) != len(predictions):
                raise AssertionError("Component/score coverage mismatch")
            n = len(predictions)
            rows += n
            initialization += timing["initialization_seconds"]
            updates += timing["update_seconds"]
            timed_updates += max(0, n - 1)
            ties += int(np.sum(predictions[1:] == predictions[:-1]))
            saturation += int(np.sum(predictions == 1))
            detail.update(id=s.id, online_rows=n, initialization_seconds=timing["initialization_seconds"],
                          update_seconds=timing["update_seconds"])
            series_records.append(detail)
            frame = pd.DataFrame(dict(id=s.id, time=s.time, time_online=np.arange(n, dtype=np.int64),
                                      target=s.target, prediction=predictions, fold=lookup[s.id]))
            for j, column in enumerate(observer.columns):
                frame[column] = component_rows[:, j]
            buffers.append(frame)
            if len(buffers) >= 32:
                table = pa.Table.from_pandas(pd.concat(buffers, ignore_index=True), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(partial, table.schema, compression="zstd")
                writer.write_table(table)
                buffers = []
            if len(seen) % 1000 == 0:
                print(f"{name}: replayed {len(seen)}/{len(manifest)} series", flush=True)
        if buffers:
            table = pa.Table.from_pandas(pd.concat(buffers, ignore_index=True), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(partial, table.schema, compression="zstd")
            writer.write_table(table)
        if writer is not None:
            writer.close()
            writer = None
        if seen != set(lookup) or rows != int(manifest.online_length.sum()) or observer.completed:
            raise AssertionError("Incomplete replay")
        replay_wall = time.perf_counter() - start
        measured = score_outputs(con, partial, manifest, out)
        regression = None
        if name == "E02":
            con.read_parquet(str(root_out / "E01_squared" / "oof.parquet")).create_view("e01_reference", replace=True)
            con.read_parquet(str(partial)).create_view("e02_comparison", replace=True)
            regression = float(con.execute("""SELECT max(abs(a.cumulative-b.prediction)) FROM e02_comparison a
                JOIN e01_reference b USING(id,time_online)""").fetchone()[0])
            if regression > 2e-14:
                raise AssertionError("E02 cumulative component differs from E01 squared")
        if source_manifest()["sha256"] != source["sha256"]:
            raise ValueError("Source changed during experiment; result invalid")
        final = out / "oof.parquet"
        partial.replace(final)
        support = {str(w): {k: sum(r["dependence_counts"].get(str(w), {}).get(k, 0) for r in series_records)
                   for k in ("supported", "insufficient_pairs", "online_degenerate", "historical_disabled")}
                   for w in (128, 256)} if name == "E07" else {}
        result = dict(**ctx, **measured, experiment_id=name, parent_id=config["parent"],
                      dependency_ids=config["dependencies"], status="valid",
                      dependency_prediction_sha256={d: r["prediction_artifact_sha256"] for d, r in dependency_results.items()},
                      primitive_feature_count=config["features"], model_input_count=0,
                      initialization_seconds_total=initialization,
                      initialization_ms_per_series=quantiles([1000*r["initialization_seconds"] for r in series_records]),
                      update_seconds_excluding_first_point=updates, timed_updates=timed_updates,
                      updates_per_second=timed_updates/updates, replay_wall_seconds=replay_wall,
                      serialized_state_bytes=quantiles([r["serialized_state_bytes"] for r in series_records]),
                      state_array_bytes=quantiles([r["state_array_bytes"] for r in series_records]),
                      shared_model_bytes=0, peak_process_memory_bytes_or_null=peak_memory_bytes(),
                      training_seconds_by_fold=[0.0]*5, training_seconds_total=0.0,
                      training_convergence_by_fold=None, bootstrap_seconds=0.0,
                      conditional_bootstrap_delta_interval=None, advancement_gate_results="pending comparisons",
                      seed_manifest=dict(cv2=20260919, bootstrap=20260920),
                      adjacent_score_ties=ties, scores_equal_one=saturation,
                      support_counts=support,
                      historical_dependence_disabled_series=sum(r["historical_dependence_disabled"] for r in series_records),
                      e02_cumulative_e01_max_absolute_difference=regression,
                      prediction_artifact_sha256=sha256(final),
                      timing_notes="Fresh process; updates include guarded replay, clock and component capture; first tick excluded. Serialization profiling is outside initialization/update timings.")
        result["delta_vs_parent"] = difference(result, dependency_results[config["parent"]])
        result["total_wall_seconds"] = time.perf_counter() - start
        write_json(out / "series_diagnostics.json", series_records)
        write_json(out / "metrics.json", result)
        finalize_artifacts(out)
        print(f"{name}: TS-AUC={result['pooled_oof_ts_auc']:.10f}; delta={result['delta_vs_parent']['pooled']:+.10f}", flush=True)
        return result
    except Exception as exc:
        write_json(out / "failure.json", dict(status="invalid", exception_type=type(exc).__name__,
                                             message=str(exc), source_sha256=ctx["source_sha256"]))
        raise
    finally:
        if writer is not None:
            writer.close()
        con.close()


def run_batch(cache, folds, spec_path, ids, out, memory="1GB"):
    spec = load_spec(spec_path)
    for name in execution_order(ids, spec):
        command = [sys.executable, str(ROOT / "run_research.py"), "_execute", "--id", name,
                   "--cache", str(cache), "--folds", str(folds), "--spec", str(spec_path),
                   "--out", str(out), "--memory", memory]
        subprocess.run(command, check=True)

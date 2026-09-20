"""Isolated Phase-2 execution and evaluation of the frozen E08 contract."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .data import connect, sha256
from .experiments import DEFAULT_SPEC, load_spec
from .logistic import INPUTS, SOLVER_OPTIONS, LogisticModel, InvalidFit, fit_fold
from .provenance import (ROOT, write_json, canonical_hash, source_manifest,
                         finalize_artifacts, verify_artifacts, peak_memory_bytes)
from .stage2 import context, score_outputs, validate_coverage, load_result, difference
from .comparison import gate_conditions, paired_bootstrap

CV2_SHA256 = "9c6b640a0b2c9a5b3fe026d5c28618346c20539d5750e31268802264cdb56434"
REFERENCES = ("E07", "E04", "E05", "E06", "E01_squared")


def verify_protected(baseline_path):
    baseline = json.loads(Path(baseline_path).read_text())
    files = baseline["protected_files"]
    if canonical_hash(files) != baseline["protected_sha256"]:
        raise ValueError("Invalid Phase-1 baseline inventory hash")
    for relative, expected in files.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Protected Phase-1 artifact changed: {relative}")
    # Only CLI routing and the validation report description may change in old code.
    for relative, expected in baseline["source"]["files"].items():
        if relative not in ("sbrt/cli.py", "sbrt/validation.py"):
            if sha256(ROOT / relative) != expected:
                raise ValueError(f"Protected Phase-1 source changed: {relative}")
    return dict(protected_file_count=len(files), protected_sha256=baseline["protected_sha256"],
                baseline_sha256=sha256(baseline_path), artifacts_unchanged=True,
                phase1_numerical_source_unchanged=True)


def admission(cache, folds, phase1, baseline, validation, spec_path):
    spec, manifest, settings, source, ctx = context(cache, folds, spec_path)
    if settings["fold_manifest_sha256"] != CV2_SHA256:
        raise ValueError("E08 requires the existing competition CV-v2 manifest")
    verify_artifacts(validation)
    test_result = json.loads((Path(validation) / "validation.json").read_text())
    if (not test_result["successful"] or test_result["source_sha256"] != source["sha256"]
            or test_result["failures"] or test_result["errors"]):
        raise ValueError("E08 requires passing tests of exactly the measured source")
    log = (Path(validation) / "unittest.log").read_text()
    if "test_e08" not in log:
        raise ValueError("Validation must include the E08-specific tests")
    integrity = verify_protected(baseline)
    stable = {k: ctx[k] for k in ("resolved_config_sha256", "fold_manifest_sha256", "dataset_sha256")}
    results = {name: load_result(Path(phase1) / name, stable)
               for name in ("E03", *REFERENCES)}
    verify_artifacts(Path(phase1) / "comparisons")
    phase1_report = json.loads((Path(phase1) / "comparisons/phase1_report.json").read_text())
    if phase1_report["final_reference"] != "E04":
        raise ValueError("Current qualifying reference is not the verified Phase-1 E04")
    ctx.update(validation_artifact_sha256=sha256(Path(validation) / "artifacts.sha256.json"),
               baseline_sha256=sha256(baseline),
               component_artifact_sha256=results["E07"]["prediction_artifact_sha256"])
    return spec, manifest, settings, source, ctx, results, integrity


def verify_component_artifacts(con, phase1, manifest):
    phase1 = Path(phase1)
    path = phase1 / "E07/oof.parquet"
    count = validate_coverage(con, path, manifest)
    con.read_parquet(str(path)).create_view("e08_blocks", replace=True)
    for experiment, column in (("E03", "R"), ("E04", "I"), ("E06", "P")):
        other = phase1 / experiment / "oof.parquet"
        validate_coverage(con, other, manifest)
        con.read_parquet(str(other)).create_view("e08_reference_block", replace=True)
        field = "P" if column == "P" else "prediction"
        bad = con.execute(f"""SELECT count(*) FROM e08_blocks a
            FULL JOIN e08_reference_block b USING(id,time_online)
            WHERE a.id IS NULL OR b.id IS NULL OR a.{column} IS DISTINCT FROM b.{field}""").fetchone()[0]
        if bad:
            raise ValueError(f"Frozen {column} block does not match {experiment}")
    invalid = " OR ".join(f"{k} IS NULL OR NOT isfinite({k}) OR {k}<0 OR {k}>1" for k in INPUTS)
    if con.execute(f"SELECT count(*) FROM e08_blocks WHERE {invalid}").fetchone()[0]:
        raise ValueError("Invalid frozen E08 component value")
    return dict(rows=count, R_equals_E03=True, I_equals_E04=True, P_equals_E06=True,
                D_source="unchanged E07 D", inputs=list(INPUTS))


def fit_oof(frame, destination):
    """Numerical core, also exercised end-to-end on labeled synthetic test data."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if set(frame.fold.unique()) != set(range(5)):
        raise ValueError("Exactly five CV-v2 folds required")
    if frame[["id", "time_online"]].duplicated().any() or frame.groupby("id").fold.nunique().max() != 1:
        raise ValueError("Duplicate keys or series crossing CV folds")
    models, diagnostics = [], []
    for fold in range(5):
        model, detail, weights = fit_fold(frame, fold)
        model_path = destination / f"fold{fold}.json"
        if model_path.exists():
            raise ValueError("Refusing to overwrite trained fold model")
        write_json(model_path, json.loads(model.dumps()))
        restored = LogisticModel.loads(model_path.read_bytes())
        if restored.dumps() != model.dumps():
            raise AssertionError("Model serialization changed fitted parameters")
        train_ids = sorted(frame.loc[frame.fold != fold, "id"].unique())
        valid_ids = sorted(frame.loc[frame.fold == fold, "id"].unique())
        if set(train_ids) & set(valid_ids):
            raise AssertionError("Series leakage")
        detail.update(training_series=len(train_ids), validation_series=len(valid_ids),
                      training_ids_sha256=canonical_hash(train_ids),
                      validation_ids_sha256=canonical_hash(valid_ids),
                      model_sha256=sha256(model_path), serialized_model_bytes=model_path.stat().st_size)
        write_json(destination / f"fold{fold}_training.json", detail)
        write_json(destination / f"fold{fold}_weights.json", weights)
        diagnostics.append(detail)
        models.append(restored)
        print(f"E08 fold {fold}: trained in {detail['training_seconds']:.3f}s; "
              f"iterations={detail['iterations']}; gradient_inf={detail['gradient_infinity_norm']:.3g}", flush=True)
    # Only after ALL folds are fitted do we produce/score held-out predictions.
    start = time.perf_counter()
    predictions = np.empty(len(frame), dtype=np.float64)
    seen = np.zeros(len(frame), dtype=bool)
    inference_seconds = 0.0
    for fold, model in enumerate(models):
        positions = np.flatnonzero(frame.fold.to_numpy() == fold)
        for begin in range(0, len(positions), 65536):
            indices = positions[begin:begin + 65536]
            x = frame.iloc[indices][list(INPUTS)].to_numpy(dtype=np.float64)
            tick = time.perf_counter()
            values = model.predict(x)
            inference_seconds += time.perf_counter() - tick
            predictions[indices] = values
            if seen[indices].any():
                raise AssertionError("Duplicate OOF prediction")
            seen[indices] = True
    if not seen.all() or not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 1)):
        raise AssertionError("Missing or invalid OOF predictions")
    return predictions, diagnostics, dict(replay_wall_seconds=time.perf_counter() - start,
                                          inference_seconds=inference_seconds,
                                          inference_rows_per_second=len(frame) / inference_seconds)


def run_e08(cache, folds, phase1, out, baseline, validation, spec_path=DEFAULT_SPEC, memory="1GB"):
    started = time.perf_counter()
    spec, manifest, settings, source, ctx, results, integrity = admission(
        cache, folds, phase1, baseline, validation, spec_path)
    out = Path(out)
    if out.exists():
        return load_result(out, ctx)
    out.mkdir(parents=True)
    con = connect(cache, memory)
    try:
        write_json(out / "resolved_config.json", spec)
        write_json(out / "execution.json", dict(inputs=list(INPUTS), solver_options=SOLVER_OPTIONS,
                   feature_replay="unchanged E07 OOF blocks; no feature regeneration",
                   coefficient_penalty="0.01/2 * ||beta||^2", intercept_penalty=0,
                   single_thread=True, no_tuning=True, no_age_or_context_inputs=True))
        write_json(out / "source_manifest.json", source)
        write_json(out / "admission.json", integrity)
        components = verify_component_artifacts(con, phase1, manifest)
        load_start = time.perf_counter()
        frame = pd.read_parquet(Path(phase1) / "E07/oof.parquet",
                                columns=["id", "time", "time_online", "target", "fold", *INPUTS])
        frame = frame.sort_values(["id", "time_online"], kind="stable").reset_index(drop=True)
        loading_seconds = time.perf_counter() - load_start
        predictions, diagnostics, timing = fit_oof(frame, out / "models")
        write_start = time.perf_counter()
        frame["prediction"] = predictions
        partial = out / "oof.partial.parquet"
        frame.to_parquet(partial, index=False, compression="zstd")
        writing_seconds = time.perf_counter() - write_start
        del frame, predictions
        measured = score_outputs(con, partial, manifest, out)
        if measured["series_count"] != 10000 or measured["scored_rows"] != 5036517:
            raise AssertionError("Unexpected full-population E08 coverage")
        final = out / "oof.parquet"
        partial.replace(final)
        integrity_after = verify_protected(baseline)
        if source_manifest()["sha256"] != source["sha256"]:
            raise AssertionError("Source changed after final tests or during E08")
        result = dict(**ctx, **measured, **timing, status="valid", experiment_id="E08", parent_id="E07",
                      dependency_ids=["E03", "E04", "E06", "E07"], model_inputs=list(INPUTS),
                      model_input_count=4, primitive_feature_count=18, new_features=0,
                      training_seconds_by_fold=[d["training_seconds"] for d in diagnostics],
                      training_seconds_total=sum(d["training_seconds"] for d in diagnostics),
                      training_convergence_by_fold=diagnostics, component_verification=components,
                      component_loading_seconds=loading_seconds, oof_writing_seconds=writing_seconds,
                      initialization_seconds_total=0.0,
                      initialization_note="Frozen components reused; zero feature initializations rerun. Model loading is included in training pipeline, not throughput.",
                      shared_model_bytes_by_fold=[d["serialized_model_bytes"] for d in diagnostics],
                      shared_model_bytes_total=sum(d["serialized_model_bytes"] for d in diagnostics),
                      peak_process_memory_bytes_or_null=peak_memory_bytes(),
                      prediction_artifact_sha256=sha256(final),
                      delta_vs_parent=difference(measured, results["E07"]),
                      protected_after=integrity_after, source_unchanged_since_validation=True,
                      total_wall_seconds=time.perf_counter() - started,
                      execution_config_sha256=sha256(out / "execution.json"),
                      bootstrap_seconds=0.0, advancement_gate_results="pending E08 comparisons",
                      timing_notes="Cached block replay only. Throughput is batched logistic inference; excludes existing E07 feature generation, indexing, and disk I/O.")
        write_json(out / "metrics.json", result)
        finalize_artifacts(out)
        print(f"E08 valid: pooled OOF TS-AUC={result['pooled_oof_ts_auc']:.10f}", flush=True)
        return result
    except Exception as exc:
        failure = dict(status="invalid", exception_type=type(exc).__name__, message=str(exc),
                       source_sha256=source["sha256"])
        if isinstance(exc, InvalidFit):
            failure["convergence"] = exc.diagnostics
        write_json(out / "failure.json", failure)
        raise
    finally:
        con.close()


def compare_e08(cache, folds, phase1, out, baseline, validation, spec_path=DEFAULT_SPEC, memory="1GB"):
    started = time.perf_counter()
    spec, manifest, settings, source, ctx, results, integrity = admission(
        cache, folds, phase1, baseline, validation, spec_path)
    out = Path(out)
    candidate = load_result(out, ctx)
    destination = out.parent / "comparisons"
    fingerprint = {k: v["prediction_artifact_sha256"] for k, v in results.items()}
    fingerprint["E08"] = candidate["prediction_artifact_sha256"]
    if destination.exists():
        verify_artifacts(destination)
        report = json.loads((destination / "phase2_report.json").read_text())
        if report["prediction_hashes"] != fingerprint or report["provenance"] != ctx:
            raise ValueError("Incompatible completed E08 comparisons")
        return report
    destination.mkdir()
    con = connect(cache, memory)
    comparisons = {}
    try:
        for name in REFERENCES:
            delta = difference(candidate, results[name])
            conditions = gate_conditions(delta, spec["gate"])
            # E01 is an additional descriptive delta, not another promotion gate.
            required = name != "E01_squared" and all(conditions.values())
            bootstrap = paired_bootstrap(con, out / "oof.parquet", Path(phase1) / name / "oof.parquet",
                                         manifest, spec["gate"]) if required else None
            comparisons[name] = dict(candidate="E08", comparator=name, delta=delta,
                    conditions_1_to_4=conditions, bootstrap_required=required, bootstrap=bootstrap,
                    passes=bool(required and bootstrap["interval_95"][0] > 0))
            write_json(destination / f"E08_vs_{name}.json", comparisons[name])
            print(f"E08 vs {name}: delta={delta['pooled']:+.10f}; gate={comparisons[name]['passes']}", flush=True)
        early = comparisons["E01_squared"]["delta"]["combined"]["le64"]
        early_ok = early is not None and early >= spec["gate"]["minimum_age_le64_vs_e01_delta"]
        # Parent and all mandatory comparisons, including strongest prior reference.
        promoted = all(comparisons[n]["passes"] for n in ("E07", "E04", "E05", "E06")) and early_ok
        boot_seconds = sum(v["bootstrap"]["seconds"] for v in comparisons.values() if v["bootstrap"])
        after = verify_protected(baseline)
        if source_manifest()["sha256"] != source["sha256"]:
            raise AssertionError("Source changed during E08 comparisons")
        beta = np.array([d["beta"] for d in candidate["training_convergence_by_fold"]])
        consistency = {name: dict(min=float(beta[:, j].min()), max=float(beta[:, j].max()),
                                  mean=float(beta[:, j].mean()), positive_folds=int((beta[:, j] > 0).sum()),
                                  negative_folds=int((beta[:, j] < 0).sum()))
                       for j, name in enumerate(INPUTS)}
        report = dict(provenance=ctx, prediction_hashes=fingerprint, cv2=settings, experiment=candidate,
                      comparisons=comparisons, coefficient_consistency=consistency,
                      advancement_gate=dict(parent_passes=comparisons["E07"]["passes"],
                          mandatory_passes={n: comparisons[n]["passes"] for n in ("E04", "E05", "E06")},
                          early_vs_e01_delta=early, early_vs_e01_passes=early_ok, promoted=promoted,
                          decision="advance" if promoted else "reject advancement; valid experiment"),
                      final_reference="E08" if promoted else "E04", bootstrap_seconds=boot_seconds,
                      comparison_wall_seconds=time.perf_counter() - started,
                      total_wall_seconds=candidate["total_wall_seconds"] + time.perf_counter() - started,
                      protected_after=after, source_unchanged_since_validation=True,
                      uncertainty="Conditional fixed-OOF whole-series bootstrap; no refitting or selection adjustment",
                      stopping_point="E08 completed; no Stage-3 implementation, execution or tuning")
        write_json(destination / "phase2_report.json", report)
        finalize_artifacts(destination)
        return report
    finally:
        con.close()


def add_commands(sub):
    for name in ("e08", "e08-compare"):
        parser = sub.add_parser(name, help="Execute/evaluate only the frozen Phase-2 logistic model")
        for argument in ("cache", "folds", "phase1", "out", "baseline", "validation"):
            parser.add_argument("--" + argument, required=True)
        parser.add_argument("--spec", default=str(DEFAULT_SPEC))
        parser.add_argument("--memory", default="1GB")


def dispatch(args):
    operation = run_e08 if args.command == "e08" else compare_e08
    return operation(args.cache, args.folds, args.phase1, args.out, args.baseline,
                     args.validation, args.spec, args.memory)

"""Local evaluator and packaging only; never included in the submission."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from . import runtime as rt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT/"production_results"
E08_HASH = "99fec6a4d33573f4e86ca86b54e8e277d288896d1db5b5c08bc3cd85e75c0010"
CV_HASH = "9c6b640a0b2c9a5b3fe026d5c28618346c20539d5750e31268802264cdb56434"


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()+b"\n")


def source():
    paths = [ROOT/p for p in rt.RUNTIME_FILES]
    paths += [ROOT/"production.py", ROOT/"production_official_test.py", *sorted((ROOT/"deployment").glob("*")),
              *sorted((ROOT/"production_e08").glob("*.py")),
              *sorted((ROOT/"deployment_tests").glob("*.py"))]
    files = {p.relative_to(ROOT).as_posix(): rt.file_hash(p) for p in sorted(set(paths)) if p.is_file()}
    return dict(files=files, sha256=rt.digest(rt.canonical(files)))


def verify_protected():
    from sbrt.provenance import source_manifest
    baseline = json.loads((RESULTS/"before.json").read_bytes())
    if rt.digest(rt.canonical(baseline["files"])) != baseline["sha256"]:
        raise ValueError("Protected baseline manifest changed")
    if source_manifest() != baseline["source"]:
        raise ValueError("Research source changed")
    for folder in baseline["protected_roots"]:
        actual = {p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob("*") if p.is_file()}
        if actual != {n for n in baseline["files"] if n.startswith(folder+"/")}:
            raise ValueError(f"Protected inventory changed: {folder}")
    for name, expected in baseline["files"].items():
        if rt.file_hash(ROOT/name) != expected:
            raise ValueError(f"Protected artifact changed: {name}")
    if rt.file_hash(ROOT/"stage2_phase2/E08/oof.parquet") != E08_HASH or rt.file_hash(ROOT/"local_cache/cv2/folds.parquet") != CV_HASH:
        raise ValueError("E08 or CV-v2 hash mismatch")
    return dict(unchanged=True, files=len(baseline["files"]), sha256=baseline["sha256"],
                E08_oof_sha256=E08_HASH, cv2_sha256=CV_HASH,
                research_source_sha256=baseline["source"]["sha256"])


def validate_tests():
    protected = verify_protected()
    frozen = source()
    logs, suites = [], {}
    for directory in ("tests", "deployment_tests"):
        start = time.perf_counter()
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", directory, "-v"],
                                cwd=ROOT, capture_output=True, text=True)
        output = result.stdout+result.stderr
        print(output, flush=True)
        logs.append(output)
        suites[directory] = dict(exit_code=result.returncode, seconds=time.perf_counter()-start)
        if result.returncode:
            raise RuntimeError(f"{directory} tests failed; no training admission")
    if source() != frozen:
        raise ValueError("Source changed during tests")
    out = RESULTS/"validation"
    out.mkdir(exist_ok=True)
    (out/"unittest.log").write_text("\n".join(logs), encoding="utf-8")
    record = dict(successful=True, suites=suites, source=frozen, environment=rt.environment(),
                  configuration=rt.CONFIG, config_sha256=rt.digest(rt.canonical(rt.CONFIG)), protected=protected,
                  log_sha256=rt.file_hash(out/"unittest.log"))
    write(out/"freeze.json", record)
    return record


def admitted():
    freeze = json.loads((RESULTS/"validation/freeze.json").read_bytes())
    if not freeze["successful"] or freeze["source"] != source() or freeze["configuration"] != rt.CONFIG:
        raise ValueError("Require passing tests and source/config freeze before execution")
    if rt.file_hash(RESULTS/"validation/unittest.log") != freeze["log_sha256"]:
        raise ValueError("Validation log changed")
    rt.require_dependencies()
    return freeze


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def training_datasets(data):
    """The official _load_train layout, retaining physical per-series order."""
    x = pd.read_parquet(Path(data)/"X_train.parquet")
    index = pd.read_parquet(Path(data)/"y_train_index.parquet")
    if not x.index.is_unique or not index.index.is_unique:
        raise ValueError("Duplicate training keys")
    for key, group in x.groupby(level="id"):
        tau = index.loc[key, "tau_index"]
        yield key, group.loc[group.period == 1, "value"].to_numpy(), group.loc[group.period == 2, "value"].to_numpy(), None if tau == -1 else tau


def run_train(output, data):
    freeze = admitted()
    protection = verify_protected()
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory; production refits never overwrite")
    output.mkdir(parents=True)
    original_data = {p.name: rt.file_hash(p) for p in sorted(Path(data).glob("*train*.parquet"))}
    def audit(frame, model, report):
        if report["data"]["series"] != 10000 or len(frame) != 5036517:
            raise ValueError("Production reference requires the full 10000-series population")
        trusted = pd.read_parquet(ROOT/"stage2_results/E07/oof.parquet", columns=["id", "time_online", "target", *rt.INPUTS])
        trusted = trusted.sort_values(["id", "time_online"], kind="stable").reset_index(drop=True)
        for key in ["id", "time_online", "target", *rt.INPUTS]:
            np.testing.assert_array_equal(frame[key].to_numpy(), trusted[key].to_numpy(), err_msg=key)
        report["frozen_E07_full_population_exact_match"] = True
        repeat, repeat_fit, repeat_weights = rt.fit_full(frame)
        if repeat.dumps() != model.dumps() or repeat_weights != report["weights"]:
            raise ValueError("Full-data refit is not exactly deterministic")
        report["determinism_repeat_fit"] = repeat_fit
        report["determinism_model_bytes_identical"] = True
        report["original_training_files_sha256"] = original_data
        report["source_freeze"] = freeze["source"]
        report["protected_before_training"] = protection
        frame.to_parquet(output/"training_blocks.parquet", index=False)
        report["training_blocks_sha256"] = rt.file_hash(output/"training_blocks.parquet")
    start = time.perf_counter()
    report = rt.train(training_datasets(data), output/"resources", audit_callback=audit)
    report["training_and_full_audits_seconds"] = time.perf_counter()-start
    report["source_unchanged_since_final_tests"] = source() == freeze["source"]
    if not report["source_unchanged_since_final_tests"]:
        raise ValueError("Source changed during full-data production training")
    report["resources_sha256"] = {p.name: rt.file_hash(p) for p in sorted((output/"resources").iterdir())}
    write(output/"training.json", report)
    print(json.dumps({k: report[k] for k in ("data", "feature_generation_seconds", "fit", "determinism_model_bytes_identical", "training_and_full_audits_seconds")}, indent=2))
    return report


def reduced_records(data):
    from sbrt.data import Series
    # Labels are not loaded here. Replay owns keys; inference sees only values.
    x = pd.read_parquet(Path(data)/"X_test.reduced.parquet")
    if not x.index.is_unique:
        raise ValueError("Duplicate reduced-test observations")
    records = []
    for key, group in x.groupby(level="id"):
        h = group.loc[group.period == 1, "value"].to_numpy()
        online = group.loc[group.period == 2, "value"]
        records.append(Series(str(key), h, online.to_numpy(), np.empty(0),
                              online.index.get_level_values("time").to_numpy(), -1))
    return records


def score_reduced(prediction, data):
    from sbrt.metric import ts_auc, bucket_summary
    labels = pd.read_parquet(Path(data)/"y_test.reduced.parquet")
    if not prediction.index.is_unique or not labels.index.is_unique or not prediction.index.equals(labels.index):
        raise ValueError("Reduced prediction/label keys must match exactly and in order")
    joined = prediction.join(labels, validate="one_to_one").reset_index()
    joined["time_online"] = joined.groupby("id").cumcount()
    indices = pd.read_parquet(Path(data)/"y_test_index.reduced.parquet")
    for key, group in joined.groupby("id"):
        tau = indices.loc[key, "tau_index"]
        expected = np.zeros(len(group), dtype=int) if tau == -1 else (np.arange(len(group)) >= tau)
        np.testing.assert_array_equal(group.target.to_numpy(), expected)
    score, ages = ts_auc(joined)
    return dict(ts_auc=score, age_buckets=bucket_summary(ages).to_dict("records"),
                purpose="100-series sanity check only; never model selection"), ages


def reduced(output, resources, data):
    from sbrt.replay import replay
    freeze = admitted()
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh reduced validation output directory")
    output.mkdir(parents=True)
    module = load_module(ROOT/"deployment/main.py", "deployment_submission")
    io_start = time.perf_counter()
    records = reduced_records(data)
    io_seconds = time.perf_counter()-io_start
    start = time.perf_counter()
    rows, reference = [], {}
    for record, values, _ in replay(records, infer_fn=lambda ds: module.infer(ds, str(resources))):
        reference[record.id] = values
        rows.append(pd.DataFrame(dict(id=int(record.id), time=record.time, prediction=values)))
    inference_seconds = time.perf_counter()-start
    prediction = pd.concat(rows, ignore_index=True).set_index(["id", "time"])
    prediction.to_parquet(output/"prediction.parquet")
    # Separate profiling/serialization pass; uses the same frozen state update.
    model = rt.load_bundle(resources)
    init, update, sizes, resumed = [], [], [], 0
    with rt.threadpool_limits(limits=1):
        for record in records:
            clock = time.perf_counter()
            state = rt.E08Detector(record.historical, model)
            init.append(time.perf_counter()-clock)
            initial = state.dumps()
            restart = rt.E08Detector.loads(initial)
            clock = time.perf_counter()
            values = np.array([state.update(float(v)) for v in record.online])
            update.append(time.perf_counter()-clock)
            np.testing.assert_array_equal(values, reference[record.id])
            cut = len(record.online)//2
            for value in record.online[:cut]:
                restart.update(float(value))
            restart = rt.E08Detector.loads(restart.dumps())
            rest = np.array([restart.update(float(v)) for v in record.online[cut:]])
            np.testing.assert_array_equal(rest, values[cut:])
            resumed += 1
            sizes.append(max(len(initial), len(state.dumps())))
    # Score only after the fixed production predictions are completely emitted.
    scoring_start = time.perf_counter()
    scoring, ages = score_reduced(prediction, data)
    scoring_seconds = time.perf_counter()-scoring_start
    ages.to_csv(output/"per_age.csv", index=False)
    count = len(prediction)
    report = dict(valid=True, series=len(records), predictions=count, score=scoring,
                  input_loading_seconds=io_seconds, end_to_end_inference_seconds=inference_seconds,
                  end_to_end_predictions_per_second=count/inference_seconds,
                  end_to_end_definition="main.infer + model load + AR/CDF initialization + rolling updates + logistic + guarded replay; excludes parquet loading and scoring",
                  separate_profile=dict(initialization_seconds=sum(init), update_and_logistic_seconds=sum(update),
                      serialized_state_bytes=dict(median=float(np.median(sizes)), max=max(sizes)), resumed_series_exact=resumed),
                  scoring_seconds=scoring_seconds, projected_10000_series_seconds=inference_seconds*10000/len(records),
                  prediction_sha256=rt.file_hash(output/"prediction.parquet"), model_sha256=rt.file_hash(Path(resources)/"e08.json"),
                  source_sha256=freeze["source"]["sha256"], environment=rt.environment(),
                  reduced_data_sha256={p.name: rt.file_hash(p) for p in sorted(Path(data).glob("*test.reduced.parquet"))})
    report["reduced_data_sha256"]["y_test_index.reduced.parquet"] = rt.file_hash(Path(data)/"y_test_index.reduced.parquet")
    if source() != freeze["source"]:
        raise ValueError("Source changed during reduced validation")
    write(output/"validation.json", report)
    print(json.dumps(report, indent=2))
    return report


def package(output, resources):
    admitted()
    return write_package(output, resources)


def write_package(output, resources):
    """Deterministic allowlist archive; also exercised on synthetic test bundles."""
    rt.load_bundle(resources)
    output = Path(output)
    if output.exists() or output.with_suffix(".zip").exists():
        raise FileExistsError("Refusing to overwrite submission package")
    files = {name: (ROOT/name).read_bytes() for name in rt.RUNTIME_FILES}
    files.update({"main.py": (ROOT/"deployment/main.py").read_bytes(),
                  "requirements.txt": (ROOT/"deployment/requirements.txt").read_bytes()})
    for name in ("e08.json", "manifest.json"):
        files["resources/"+name] = (Path(resources)/name).read_bytes()
    manifest = {name: rt.digest(value) for name, value in sorted(files.items())}
    files["package.sha256.json"] = rt.canonical(dict(files=manifest, sha256=rt.digest(rt.canonical(manifest))))
    output.mkdir(parents=True)
    with zipfile.ZipFile(output.with_suffix(".zip"), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, contents in sorted(files.items()):
            target = output/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 19, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, contents, compresslevel=9)
    report = dict(files=manifest, package_manifest_sha256=rt.digest(rt.canonical(manifest)),
                  zip_sha256=rt.file_hash(output.with_suffix(".zip")),
                  zip_bytes=output.with_suffix(".zip").stat().st_size,
                  model_bytes=(output/"resources/e08.json").stat().st_size)
    write(output.with_name(output.name+"_package_report.json"), report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="Frozen E08 production engineering; no research commands")
    parser.add_argument("command", choices=("test", "verify", "train", "reduced", "package"))
    parser.add_argument("--data", type=Path, default=ROOT/"data")
    parser.add_argument("--resources", type=Path, default=RESULTS/"full_train/resources")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "test":
        validate_tests()
    elif args.command == "verify":
        print(json.dumps(verify_protected(), indent=2))
    else:
        output = args.output or RESULTS/{"train": "full_train", "reduced": "reduced", "package": "submission"}[args.command]
        if args.command == "train":
            run_train(output, args.data)
        elif args.command == "reduced":
            reduced(output, args.resources, args.data)
        else:
            package(output, args.resources)

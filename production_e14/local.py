"""Local evaluator and packaging only; never included in the submission."""
import argparse
import importlib.util
import json
import os
import re
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
RESULTS = ROOT/"production_e14_results"
E14_HASH = rt.REFERENCES["E14_oof_sha256"]
CV_HASH = "9c6b640a0b2c9a5b3fe026d5c28618346c20539d5750e31268802264cdb56434"


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()+b"\n")


def source():
    paths = [ROOT/p for p in rt.RUNTIME_FILES]
    paths += [ROOT/"production_e14.py", ROOT/"production_e14_official_test.py", *sorted((ROOT/"deployment_e14").glob("*")),
              *sorted((ROOT/"production_e14").glob("*.py")),
              *sorted((ROOT/"deployment_e14_tests").glob("*.py"))]
    files = {p.relative_to(ROOT).as_posix(): rt.file_hash(p) for p in sorted(set(paths)) if p.is_file()}
    return dict(files=files, sha256=rt.digest(rt.canonical(files)))


def verify_protected():
    from stage4_research.provenance import protected, source as stage4_source, SPEC
    from stage4_research.contract import CONFIG as research_config
    result = protected()
    if stage4_source()["sha256"] != rt.REFERENCES["stage4_source_sha256"]:
        raise ValueError("Protected Stage-4 source changed")
    if rt.digest(rt.canonical(research_config)) != rt.REFERENCES["stage4_config_sha256"]:
        raise ValueError("Protected Stage-4 configuration changed")
    if rt.file_hash(SPEC) != rt.REFERENCES["stage4_spec_sha256"]:
        raise ValueError("Protected Stage-4 specification changed")
    if rt.file_hash(ROOT/"stage4_results/E14/oof.parquet") != E14_HASH:
        raise ValueError("Protected E14 OOF changed")
    path = ROOT/"stage4_results/delivery_manifest.json"
    delivery = json.loads(path.read_bytes())
    if rt.digest(rt.canonical(delivery["files"])) != delivery["sha256"]:
        raise ValueError("Stage-4 delivery inventory changed")
    for name, expected in delivery["files"].items():
        if rt.file_hash(ROOT/name) != expected:
            raise ValueError(f"Protected Stage-4 artifact changed: {name}")
    result.update(stage4_files=len(delivery["files"]), stage4_delivery_sha256=rt.file_hash(path),
                  E14_oof_sha256=E14_HASH, references=rt.REFERENCES)
    return result


def validate_tests():
    protected = verify_protected()
    frozen = source()
    previous = RESULTS/"validation/freeze.json"
    if previous.exists():
        admitted()  # Never admit changed source on top of measured artifacts.
    logs, suites = [], {}
    for directory in ("tests", "deployment_tests", "stage4_tests", "deployment_e14_tests"):
        start = time.perf_counter()
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", directory, "-v"],
                                cwd=ROOT, capture_output=True, text=True)
        output = result.stdout+result.stderr
        print(output, flush=True)
        logs.append(output)
        match = re.search(r"Ran (\d+) tests? in", output)
        suites[directory] = dict(exit_code=result.returncode, seconds=time.perf_counter()-start,
                                tests_run=int(match.group(1)) if match else None)
        if result.returncode:
            raise RuntimeError(f"{directory} tests failed; no training admission")
    if source() != frozen or verify_protected() != protected:
        raise ValueError("Source/protected artifacts changed during tests")
    out = RESULTS/"validation"
    out.mkdir(parents=True, exist_ok=True)
    if previous.exists():
        (out/"recheck.log").write_text("\n".join(logs), encoding="utf-8")
        write(out/"recheck.json",dict(successful=True,suites=suites,source=frozen))
        return json.loads(previous.read_bytes())
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
    if verify_protected() != freeze["protected"]:
        raise ValueError("Protected checkpoint drift")
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
        trusted = pd.read_parquet(ROOT/"stage4_results/features/raw.parquet", columns=["id", "time_online", "target", *rt.RAW])
        trusted = trusted.sort_values(["id", "time_online"], kind="stable").reset_index(drop=True)
        for key in ["id", "time_online", "target", *rt.RAW]:
            np.testing.assert_array_equal(frame[key].to_numpy(), trusted[key].to_numpy(), err_msg=key)
        report["frozen_stage4_full_population_exact_match"] = True
        repeat, repeat_fit, repeat_weights = rt.fit_full(frame)
        if repeat.dumps() != model.dumps() or repeat_weights != report["weights"]:
            raise ValueError("Full-data refit is not exactly deterministic")
        report["determinism_repeat_fit"] = repeat_fit
        report["determinism_model_bytes_identical"] = True
        report["data"]["original_training_files_sha256"] = original_data
        report["original_training_files_sha256"] = original_data
        report["source_freeze"] = freeze["source"]
        report["data"]["deployment_source_sha256"] = freeze["source"]["sha256"]
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
    module = load_module(ROOT/"deployment_e14/main.py", "deployment_submission")
    io_start = time.perf_counter()
    records = reduced_records(data)
    io_seconds = time.perf_counter()-io_start
    measurement = load_module(ROOT/"deployment_e14/measure.py", "e14_local_measure")
    start = time.perf_counter()
    rows, reference = [], {}
    with measurement.MemoryProbe() as memory:
        for record, values, _ in replay(records, infer_fn=lambda ds: module.infer(ds, str(resources))):
            reference[record.id] = values
            rows.append(pd.DataFrame(dict(id=int(record.id), time=record.time, prediction=values)))
    inference_seconds = time.perf_counter()-start
    prediction = pd.concat(rows, ignore_index=True).set_index(["id", "time"])
    if len(records)!=100 or len(prediction)!=50983:
        raise ValueError("Incomplete reduced population")
    for record, values, _ in replay(records[:len(records)//10], infer_fn=lambda ds: module.infer(ds,str(resources))):
        np.testing.assert_array_equal(values,reference[record.id])
    prediction.to_parquet(output/"prediction.parquet")
    # Separate profiling/serialization pass; uses the same frozen state update.
    model = rt.load_bundle(resources)
    init, update, sizes, resumed, support = [], [], [], 0, {}
    with rt.threadpool_limits(limits=1):
        for record in records:
            clock = time.perf_counter()
            state = rt.E14Detector(record.historical, model)
            init.append(time.perf_counter()-clock)
            initial = state.dumps()
            restart = rt.E14Detector.loads(initial, model)
            clock = time.perf_counter()
            values = np.array([state.update(float(v)) for v in record.online])
            update.append(time.perf_counter()-clock)
            np.testing.assert_array_equal(values, reference[record.id])
            cut = len(record.online)//2
            for value in record.online[:cut]:
                restart.update(float(value))
            restart = rt.E14Detector.loads(restart.dumps(), model)
            rest = np.array([restart.update(float(v)) for v in record.online[cut:]])
            np.testing.assert_array_equal(rest, values[cut:])
            resumed += 1
            sizes.append(max(len(initial), len(state.dumps())))
            for key,value in state.features.scales["slow"].counts.items():
                support[key] = support.get(key,0)+value
    # Score only after the fixed production predictions are completely emitted.
    scoring_start = time.perf_counter()
    scoring, ages = score_reduced(prediction, data)
    scoring_seconds = time.perf_counter()-scoring_start
    ages.to_csv(output/"per_age.csv", index=False)
    count = len(prediction)
    report = dict(valid=True, series=len(records), predictions=count, score=scoring,
                  memory=memory.report, deterministic_10percent_replay_exact=True,
                  support_counts=support,
                  input_loading_seconds=io_seconds, end_to_end_inference_seconds=inference_seconds,
                  end_to_end_predictions_per_second=count/inference_seconds,
                  end_to_end_definition="main.infer + model load + AR/conditional-scale historical replay/CDF initialization + rolling updates + logistic + guarded replay; excludes parquet loading and scoring",
                  separate_profile=dict(initialization_seconds=sum(init), update_and_logistic_seconds=sum(update),
                      serialized_state_bytes=dict(median=float(np.median(sizes)), max=max(sizes)), resumed_series_exact=resumed),
                  scoring_seconds=scoring_seconds, projected_10000_series_seconds=inference_seconds*10000/len(records),
                  prediction_sha256=rt.file_hash(output/"prediction.parquet"), model_sha256=rt.file_hash(Path(resources)/"e14.json"),
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
    files.update({"main.py": (ROOT/"deployment_e14/main.py").read_bytes(),
                  "requirements.txt": (ROOT/"deployment_e14/requirements.txt").read_bytes()})
    for name in ("e14.json", "manifest.json"):
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
                  model_bytes=(output/"resources/e14.json").stat().st_size)
    write(output.with_name(output.name+"_package_report.json"), report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="Frozen E14 production engineering; no research commands")
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

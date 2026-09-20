"""Run the unmodified official runner using already supplied local data.

Only the CLI's authenticated data-download step is replaced with a local path.
Official train/infer dispatch, socket protocol, guards and determinism check are
unchanged. No project credentials, uploads, invented competition metadata or
network access are needed. This helper is NOT submitted to the cloud.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package, data, output = args.package.resolve(), args.data.resolve(), args.output.resolve()
    root = Path(__file__).resolve().parent
    if output.exists():
        raise FileExistsError("Use a new official local-test output directory")
    output.mkdir(parents=True)
    sys.path.insert(0, str(package))
    import main as submitted
    from production_e14 import runtime as rt
    if not Path(rt.__file__).resolve().is_relative_to(package):
        raise RuntimeError("Official local test must import the packaged runtime")
    import pandas as pd
    import numpy as np
    from crunch.api import Competition
    from crunch.runner.local import LocalRunner
    from crunch.tester import install_logger
    from crunch.unstructured import RunnerModule, LocalCodeLoader
    import importlib.metadata

    evidence = root/"production_results/official_sources"
    expected = json.loads((evidence/"sources.json").read_bytes())
    for name, info in expected["files"].items():
        if rt.file_hash(evidence/name) != info["sha256"]:
            raise ValueError(f"Official source drift: {name}")
    if importlib.metadata.version("crunch-cli") != "12.0.3":
        raise RuntimeError("Require the same pinned Crunch CLI as validated E08")
    package_manifest = json.loads((package/"package.sha256.json").read_bytes())
    if rt.digest(rt.canonical(package_manifest["files"])) != package_manifest["sha256"]:
        raise ValueError("Package manifest changed")
    for name, checksum in package_manifest["files"].items():
        if rt.file_hash(package/name) != checksum:
            raise ValueError(f"Package changed: {name}")
    # Fail closed on remote networking; official loopback socket transport remains
    # unmodified. This is a local test guard, not a submitted runtime dependency.
    network = dict(loopback_connections=0, denied_connections=0)
    def audit_network(event, arguments):
        if event == "socket.connect":
            address = arguments[1]
            host = address[0] if isinstance(address,tuple) else None
            if host not in ("127.0.0.1","::1","localhost"):
                network["denied_connections"] += 1
                raise RuntimeError(f"Non-loopback networking forbidden in local validation: {address}")
            network["loopback_connections"] += 1
    sys.addaudithook(audit_network)
    competition = Competition(attrs=json.loads((evidence/"competition.json").read_bytes()))
    runner_module = RunnerModule.load(LocalCodeLoader(path=str(evidence/"runner.py")))

    class ProvidedLocalDataRunner(LocalRunner):
        def _download_data(self):
            # The only override: do not authenticate/download/overwrite the
            # user-provided immutable files. No guard/score behavior is changed.
            self.data_directory_path = str(data)

    runner = ProvidedLocalDataRunner(submitted, runner_module, str(package/"resources"),
                                    str(output), False, 1, None, competition, False,
                                    True, install_logger(), None)
    start = time.perf_counter()
    import importlib.util
    measurement_spec = importlib.util.spec_from_file_location("e14_official_measure",root/"deployment_e14/measure.py")
    measurement = importlib.util.module_from_spec(measurement_spec)
    measurement_spec.loader.exec_module(measurement)
    with measurement.MemoryProbe() as memory:
        runner.start()
    elapsed = time.perf_counter()-start
    if runner.deterministic is not True:
        raise ValueError("Official runner determinism failed")
    prediction = pd.read_parquet(output/"prediction.parquet")
    # The published quickstarter scores the supplied reduced labels exactly this
    # way. The frozen project scorer is imported as an evaluator, not inference.
    import importlib.util
    spec = importlib.util.spec_from_file_location("trusted_local_metric", root/"sbrt/metric.py")
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    labels = pd.read_parquet(data/"y_test.reduced.parquet")
    if len(prediction)!=50983 or prediction.index.get_level_values("id").nunique()!=100:
        raise ValueError("Incomplete reduced population")
    if not prediction.index.is_unique or not prediction.index.equals(labels.index):
        raise ValueError("Incomplete official OOF-independent test coverage")
    joined = prediction.join(labels, validate="one_to_one").reset_index()
    joined["time_online"] = joined.groupby("id").cumcount()
    score_start = time.perf_counter()
    score, ages = metric.ts_auc(joined)
    scoring_seconds = time.perf_counter()-score_start
    direct = pd.read_parquet(args.direct.resolve())
    if not prediction.index.equals(direct.index):
        raise ValueError("Direct/official keys differ")
    # Official wire protocol quantizes predictions to float32; model is unchanged.
    np.testing.assert_array_equal(prediction.prediction.to_numpy(), direct.prediction.to_numpy().astype(np.float32))
    report = dict(valid=True, memory=memory.report, network_audit=network,
                  model_sha256=rt.file_hash(package/"resources/e14.json"),
                  package_manifest_sha256=package_manifest["sha256"], mode="official LocalRunner with provided-data download override only",
                  official_runner_commit=expected["commit"], official_sources=expected,
                  crunch_cli=importlib.metadata.version("crunch-cli"), series=int(prediction.index.get_level_values("id").nunique()),
                  predictions=len(prediction), ts_auc=score, scoring_seconds=scoring_seconds,
                  runner_seconds_including_10percent_determinism=elapsed,
                  conservative_10000_series_seconds=elapsed*100,
                  effective_predictions_per_second=len(prediction)/elapsed,
                  official_determinism_passed=True, exact_direct_predictions_after_float32_transport=True,
                  prediction_sha256=rt.file_hash(output/"prediction.parquet"),
                  packaged_runtime_path=str(Path(rt.__file__).resolve()), environment=rt.environment())
    for name, checksum in package_manifest["files"].items():
        if rt.file_hash(package/name) != checksum:
            raise ValueError(f"Runner changed package: {name}")
    # All numerical code actually imported by the package must resolve inside it.
    for name in rt.RUNTIME_FILES:
        if not name.endswith(".py"):
            continue
        module_name = name[:-3].replace("/",".").removesuffix(".__init__")
        module = sys.modules.get(module_name)
        if module is None or not Path(module.__file__).resolve().is_relative_to(package):
            raise ValueError(f"Runtime dependency escaped submission: {module_name}")
    (output/"official_validation.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    ages.to_csv(output/"per_age.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Official train/infer adapter around the frozen E08 numerical implementation.

No data paths, evaluator labels, IDs, horizons or fold models enter inference.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

# Must precede numerical imports in a fresh official worker. The context manager
# also limits libraries already imported by the official host process.
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _name in THREADS:
    os.environ[_name] = "1"

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from sbrt.logistic import INPUTS, L2, SOLVER_OPTIONS, LogisticModel, E08Detector, fit_fold
from sbrt.stage2_detectors import Stage2Detector

CORE = ("sbrt/__init__.py", "sbrt/logistic.py", "sbrt/stage2_detectors.py",
        "sbrt/history.py", "sbrt/features.py")
RUNTIME_FILES = (*CORE, "production_e08/__init__.py", "production_e08/runtime.py")
DEPENDENCIES = {"numpy": "2.3.5", "scipy": "1.17.0", "pandas": "2.2.3",
                "threadpoolctl": "3.7.0", "pyarrow": "25.0.1"}
CONFIG = dict(version=1, reference="E08", fit="single full-population refit",
              inputs=list(INPUTS), block_detector="E07", l2=L2,
              penalty="0.01 / 2 * dot(beta, beta); intercept unpenalized",
              solver="L-BFGS-B", solver_options=SOLVER_OPTIONS, zero_initialization=True,
              gradient_infinity_norm_max=1e-6, dtype="float64", native_threads=1,
              row_order="lexicographic str(id), then zero-based time_online",
              weights="W_t/(2*Z*N_y(t)); exclude W_t=0; recompute on full population",
              standardization="frozen sbrt.logistic.standardize_fit; weighted population variance")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def runtime_source():
    root = Path(__file__).resolve().parent.parent
    files = {name: file_hash(root/name) for name in RUNTIME_FILES}
    return dict(files=files, sha256=digest(canonical(files)))


def environment():
    return dict(python=platform.python_version(), platform=platform.platform(),
                dependencies={k: importlib.metadata.version(k) for k in DEPENDENCIES},
                thread_environment={k: os.environ.get(k) for k in THREADS},
                native_threads=threadpool_info())


def require_dependencies():
    actual = {k: importlib.metadata.version(k) for k in DEPENDENCIES}
    if actual != DEPENDENCIES:
        raise RuntimeError(f"Frozen E08 dependency mismatch: {actual}; expected {DEPENDENCIES}")


def build_training_frame(datasets, progress=None):
    """Consume the official training tuples; labels are evaluator-side only.

    Training may materialize labeled rows. Inference cannot call this function.
    Canonical row ordering matches the trusted E08 fit and is independent of the
    order in which the official training loader supplies series.
    """
    import pandas as pd
    frames, records, seen = [], [], set()
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        for identifier, history, online, tau in datasets:
            key = str(identifier)
            if key in seen:
                raise ValueError("Duplicate normalized training ID")
            seen.add(key)
            h = np.asarray(history, dtype=np.float64)
            o = np.asarray(online, dtype=np.float64)
            if h.ndim != 1 or o.ndim != 1 or not len(o) or not np.isfinite(o).all():
                raise ValueError("Invalid training trajectory")
            if tau is not None and (not np.isfinite(tau) or int(tau) != tau or not 0 <= tau < len(o)):
                raise ValueError("tau must be None or a zero-based first-positive position")
            tau = None if tau is None else int(tau)
            state = Stage2Detector(h, "E07")
            x = np.empty((len(o), 4), dtype=np.float64)
            for t, value in enumerate(o):
                state.update(float(value))
                x[t] = [state.last_blocks[k] for k in INPUTS]
            age = np.arange(len(o), dtype=np.int64)
            frame = pd.DataFrame(x, columns=INPUTS)
            frame["id"], frame["time_online"] = key, age
            frame["target"] = np.zeros(len(o), dtype=np.int8) if tau is None else (age >= tau).astype(np.int8)
            frames.append(frame)
            records.append(dict(id=key, history=len(h), online=len(o), tau=tau,
                                history_sha256=digest(h.astype("<f8").tobytes()),
                                online_sha256=digest(o.astype("<f8").tobytes())))
            if progress is not None and len(seen) % 500 == 0:
                progress(len(seen))
    if not frames:
        raise ValueError("Empty training population")
    frame = pd.concat(frames, ignore_index=True).sort_values(["id", "time_online"], kind="stable").reset_index(drop=True)
    records.sort(key=lambda r: r["id"])
    data = dict(series=len(records), predictions=len(frame),
                historical_observations=sum(r["history"] for r in records),
                break_series=sum(r["tau"] is not None for r in records),
                canonical_training_sha256=digest(canonical(records)))
    return frame, data, time.perf_counter()-start


def fit_full(frame):
    """Reuse the entire trusted E08 fitting procedure, without a held-out set.

    All rows receive fold=0; heldout=-1 is an absent sentinel. No input fold
    assignment survives. This is NOT an average of OOF fold coefficients.
    """
    required = [*INPUTS, "time_online", "target"]
    if any(k not in frame for k in required):
        raise ValueError("Missing E08 training columns")
    model, diagnostics, audit = fit_fold(frame.loc[:, required].assign(fold=0), -1)
    diagnostics.pop("fold")
    diagnostics["scope"] = "all supplied training rows; no validation split"
    return model, diagnostics, audit


def save_bundle(directory, model, data):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = model.dumps()
    manifest = dict(version=1, model_file="e08.json", model_sha256=digest(payload),
                    configuration=CONFIG, configuration_sha256=digest(canonical(CONFIG)),
                    runtime_source=runtime_source(), training_data=data,
                    python=platform.python_version(), dependencies=DEPENDENCIES)
    # Manifest is committed last; an interrupted write cannot silently load a
    # mismatched pair. Canonical artifact bytes contain no wall clocks or paths.
    for name, contents in (("e08.json", payload), ("manifest.json", canonical(manifest))):
        temporary = directory/(name+".partial")
        temporary.write_bytes(contents)
        temporary.replace(directory/name)
    return manifest


def load_bundle(directory):
    require_dependencies()
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_bytes())
    if (manifest["version"] != 1 or manifest["model_file"] != "e08.json"
            or manifest["configuration"] != CONFIG
            or manifest["configuration_sha256"] != digest(canonical(CONFIG))
            or manifest["runtime_source"] != runtime_source()
            or manifest["dependencies"] != DEPENDENCIES):
        raise ValueError("Incompatible E08 bundle/source/configuration")
    payload = (directory/"e08.json").read_bytes()
    if digest(payload) != manifest["model_sha256"]:
        raise ValueError("E08 model artifact hash mismatch")
    return LogisticModel.loads(payload)


def train(datasets, model_directory_path, *, audit_callback=None):
    """Official full-data training path; optional callback is local auditing only."""
    require_dependencies()
    start = time.perf_counter()
    frame, data, feature_seconds = build_training_frame(datasets, progress=lambda n: print(f"E08 training features: {n} series", flush=True))
    model, diagnostics, weights = fit_full(frame)
    manifest = save_bundle(model_directory_path, model, data)
    report = dict(data=data, feature_generation_seconds=feature_seconds, fit=diagnostics,
                  weights=weights, manifest=manifest, total_seconds=time.perf_counter()-start)
    if audit_callback is not None:
        audit_callback(frame, model, report)
    return report


def infer(datasets, model_directory_path):
    """Exact official readiness handshake, fresh history-only state per series."""
    model = load_bundle(model_directory_path)
    with threadpool_limits(limits=1):
        yield
        for history, online in datasets:
            state = E08Detector(history, model)
            for value in online:
                yield state.update(value)

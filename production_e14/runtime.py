"""Single full-population E14; exact trusted numerical code, no research options."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _name in THREADS:
    os.environ[_name] = "1"

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from stage4_research.features import FeatureState, Detector as ResearchDetector
from stage4_research.models import Model, fit_fold
from stage4_research.contract import CONFIG as STAGE4_CONFIG, INPUTS as STAGE4_INPUTS

RAW = ("R", "I", "P", "D", "J_s")
INPUTS = tuple(STAGE4_INPUTS["E14"])
CORE = ("sbrt/__init__.py", "sbrt/logistic.py", "sbrt/stage2_detectors.py",
        "sbrt/history.py", "sbrt/features.py", "sbrt/stage3_features.py",
        "stage4_research/__init__.py", "stage4_research/contract.py",
        "stage4_research/scale.py", "stage4_research/features.py", "stage4_research/models.py")
RUNTIME_FILES = (*CORE, "production_e14/__init__.py", "production_e14/runtime.py")
DEPENDENCIES = {"numpy": "2.3.5", "scipy": "1.17.0", "pandas": "2.2.3",
                "threadpoolctl": "3.7.0", "pyarrow": "25.0.1"}
REFERENCES = dict(
    E14_oof_sha256="f617a259568cba2414f6beea1c7723747e357d00dbfee7ed92a9c8d5d1e70146",
    stage4_source_sha256="2152f8cf9ca1ac25b7e9c1a1a9b3270a1634161d895e0771c9fea8c604657f79",
    stage4_config_sha256="d0b955a9772b560a7b6d1a0b57c0839744826d2ed67404b77d0b19fd7d062bf6",
    stage4_spec_sha256="d760b839b2e10f38e698fc775e4f4ecfb51c6d36f5d3f4ad69d03ceb63f71a39")
CONFIG = dict(version=1, reference="E14", fit="single full-population refit",
    inputs=list(INPUTS), raw=list(RAW), references=REFERENCES,
    historical="unchanged frozen Stage-4 FeatureState(history, E14); ordinary slow alpha=0.01 only",
    evidence="unchanged R/I/P/D plus J_s; C_s=(J_s-I)/sqrt(2)",
    timing="AR residual -> old h -> J_s -> R/I/P/D -> score -> h_next -> yield -> next x",
    training=STAGE4_CONFIG["training"], row_order="lexicographic str(id), then time_online",
    weights="full-population W_t/(2*Z*N_y(t)); exclude W_t=0",
    preprocessing="exact separate E08 base and Stage-4 extension weighted population moments")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def runtime_source():
    root = Path(__file__).resolve().parent.parent
    files = {name: file_hash(root/name) for name in RUNTIME_FILES}
    return dict(files=files, sha256=digest(canonical(files)))


def environment():
    return dict(python=platform.python_version(), platform=platform.platform(),
        dependencies={k: importlib.metadata.version(k) for k in DEPENDENCIES},
        thread_environment={k: os.environ.get(k) for k in THREADS}, native_threads=threadpool_info())


def require_dependencies():
    actual = {k: importlib.metadata.version(k) for k in DEPENDENCIES}
    if actual != DEPENDENCIES:
        raise RuntimeError(f"Frozen E14 dependencies differ: {actual}")


def observe(state, value):
    """Reorder independent trusted updates; do not redefine any numerical routine.

    Research FeatureState advances the base blocks first. Here the cloned AR and
    J_s advance first to satisfy the explicit production timing contract. Neither
    reads the other's mutable state. Full-population equality is audited locally.
    h remains unchanged until the caller has computed the logistic score.
    """
    if state.experiment != "E14" or state.pending or tuple(state.scales) != ("slow",):
        raise ValueError("Require fresh/committed E14-only state")
    e, _ = state.ar.update(value)
    result = state.scales["slow"].observe(e)
    state.blocks.update(value)
    state.last = {k: state.blocks.last_blocks[k] for k in RAW[:4]}
    state.last["J_s"] = result["J"]
    state.diagnostic = dict(e_squared=e*e, v_s=result["variance"])
    state.pending = True
    return state.last


class E14Detector(ResearchDetector):
    def __init__(self, history, model):
        if model.pre.experiment != "E14" or len(model.beta) != 5:
            raise ValueError("Production only admits E14")
        super().__init__(history, model)

    def update(self, value):
        row = observe(self.features, value)
        score = float(self.model.predict(row))
        self.features.commit()
        return score

    @classmethod
    def loads(cls, payload, model):
        if model.pre.experiment != "E14":
            raise ValueError("Production only admits E14")
        return super().loads(payload, model)


def build_training_frame(datasets, progress=None):
    import pandas as pd
    frames, records, seen = [], [], set()
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        for identifier, history, online, tau in datasets:
            key = str(identifier)
            if key in seen:
                raise ValueError("Duplicate normalized training ID")
            seen.add(key)
            h, o = np.asarray(history, dtype=np.float64), np.asarray(online, dtype=np.float64)
            if h.ndim != 1 or o.ndim != 1 or not len(o) or not np.isfinite(o).all():
                raise ValueError("Invalid training trajectory")
            if tau is not None and (not np.isfinite(tau) or int(tau)!=tau or not 0<=tau<len(o)):
                raise ValueError("tau must be None or zero-based first-positive position")
            tau = None if tau is None else int(tau)
            state = FeatureState(h, "E14")
            x = np.empty((len(o), len(RAW)), dtype=np.float64)
            for t, value in enumerate(o):
                row = observe(state, float(value))
                x[t] = [row[k] for k in RAW]
                state.commit()
            age = np.arange(len(o), dtype=np.int64)
            frame = pd.DataFrame(x, columns=RAW)
            frame["id"], frame["time_online"] = key, age
            frame["target"] = np.zeros(len(o), dtype=np.int8) if tau is None else (age>=tau).astype(np.int8)
            frames.append(frame)
            records.append(dict(id=key, history=len(h), online=len(o), tau=tau,
                history_sha256=digest(h.astype("<f8").tobytes()), online_sha256=digest(o.astype("<f8").tobytes())))
            if progress is not None and len(seen)%500 == 0:
                progress(len(seen))
    if not frames:
        raise ValueError("Empty training population")
    frame = pd.concat(frames, ignore_index=True).sort_values(["id", "time_online"], kind="stable").reset_index(drop=True)
    records.sort(key=lambda r:r["id"])
    data = dict(series=len(records), predictions=len(frame), historical_observations=sum(r["history"] for r in records),
        break_series=sum(r["tau"] is not None for r in records), canonical_training_sha256=digest(canonical(records)))
    return frame, data, time.perf_counter()-start


def fit_full(frame):
    required = [*RAW, "time_online", "target"]
    model, detail, weights = fit_fold(frame.loc[:,required].assign(fold=0), -1, "E14")
    detail.pop("fold")
    detail["scope"] = "all supplied training rows; no validation split or fold averaging"
    return model, detail, weights


def save_bundle(directory, model, data):
    if model.pre.experiment != "E14":
        raise ValueError("Only E14 can be saved for production")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = model.dumps()
    manifest = dict(version=1, model_file="e14.json", model_sha256=digest(payload),
        configuration=CONFIG, configuration_sha256=digest(canonical(CONFIG)), runtime_source=runtime_source(),
        training_data=data, python=platform.python_version(), dependencies=DEPENDENCIES)
    for name, contents in (("e14.json",payload), ("manifest.json",canonical(manifest))):
        temporary = directory/(name+".partial")
        temporary.write_bytes(contents)
        temporary.replace(directory/name)
    return manifest


def load_bundle(directory):
    require_dependencies()
    directory = Path(directory)
    m = json.loads((directory/"manifest.json").read_bytes())
    if (m["version"]!=1 or m["model_file"]!="e14.json" or m["configuration"]!=CONFIG
        or m["configuration_sha256"]!=digest(canonical(CONFIG)) or m["runtime_source"]!=runtime_source()
        or m["dependencies"]!=DEPENDENCIES):
        raise ValueError("Incompatible E14 bundle/source/configuration")
    payload = (directory/"e14.json").read_bytes()
    if digest(payload)!=m["model_sha256"]:
        raise ValueError("E14 artifact hash mismatch")
    model = Model.loads(payload)
    if model.pre.experiment!="E14":
        raise ValueError("Only E14 production models admitted")
    return model


def train(datasets, model_directory_path, *, audit_callback=None):
    require_dependencies()
    start = time.perf_counter()
    frame, data, seconds = build_training_frame(datasets, progress=lambda n:print(f"E14 training features: {n} series",flush=True))
    model, detail, weights = fit_full(frame)
    report = dict(data=data, feature_generation_seconds=seconds, fit=detail, weights=weights)
    if audit_callback is not None:
        audit_callback(frame, model, report)
    report["manifest"] = save_bundle(model_directory_path, model, report["data"])
    report["total_seconds"] = time.perf_counter()-start
    return report


def infer(datasets, model_directory_path):
    model = load_bundle(model_directory_path)
    with threadpool_limits(limits=1):
        yield
        for history, online in datasets:
            state = E14Detector(history, model)
            for value in online:
                yield state.update(value)

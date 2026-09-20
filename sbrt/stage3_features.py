"""Exact Stage-3 column construction; no labels or evaluator imports."""
import hashlib
import json
import numpy as np
from .logistic import INPUTS, standardize_fit
from .stage2_detectors import Stage2Detector

RAW = (*INPUTS, "I32", "I128", "I256", "Icumulative", "P_location", "P_energy")
INPUT_LISTS = {
    "E09": ["z_R", "z_I", "z_P", "z_D", "S((q-qbar)*z_R)", "S((q-qbar)*z_I)"],
    "E10": ["z_R", "z_I", "z_P", "z_D", "S(C1)", "S(C2)", "S(C3)"],
    "E11": ["z_R", "z_I", "z_P", "z_D", "S(K)"],
    "E12": ["z_R", "z_I", "z_P", "z_D", "S((a-abar)*z_I)", "S((a-abar)*z_P)", "S((a-abar)*z_D)"],
    "E13": ["z_R", "z_I", "z_P", "z_D"],
}
CONTRASTS = np.array([[1, -1, 0, 0], [1, 1, -2, 0], [1, 1, 1, -3]], dtype=float) / np.sqrt([2, 6, 12])[:, None]


def context_q(ratio):
    if ratio is None:
        return 0.0
    if not np.isfinite(ratio) or ratio < 0:
        raise ValueError("Invalid historical AR MSE ratio")
    return float(np.clip(1.0 - ratio, 0.0, 1.0))


def raw_values(state):
    if state.experiment != "E07" or not state.age:
        raise ValueError("Updated authoritative E07 state required")
    result = dict(state.last_blocks)
    result.update(zip(RAW[4:8], state.last_features[4:8]))
    return {key: result[key] for key in RAW}


def maturity(time_online):
    t = np.asarray(time_online, dtype=np.float64)
    if not np.isfinite(t).all() or np.any(t < 0) or np.any(t != np.floor(t)):
        raise ValueError("Invalid causal age")
    return np.log1p(t + 1.0)


def weighted_center(u, omega):
    u = np.asarray(u, dtype=np.float64)
    # Exact constants have zero centered products, as in E08 standardization.
    return float(u[0]) if np.all(u == u[0]) else float(np.dot(omega, u) / omega.sum())


def standardize_extra(x, omega):
    if x.ndim != 2 or len(x) != len(omega) or not np.isfinite(x).all():
        raise ValueError("Invalid extension columns")
    mean = np.array([weighted_center(x[:, j], omega) for j in range(x.shape[1])])
    centered = x - mean
    std = np.sqrt([np.dot(omega, centered[:, j] ** 2) / omega.sum() for j in range(x.shape[1])])
    return mean, std, np.divide(centered, std, out=np.zeros_like(centered), where=std > 0)


def extension(experiment, frame, base, center):
    if experiment == "E09":
        q = np.asarray(frame["q"], dtype=float)
        if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
            raise ValueError("Invalid frozen historical context")
        return (q - center)[:, None] * base[:, :2]
    if experiment == "E10":
        a, b, c, d = (np.asarray(frame[key], dtype=float) for key in RAW[4:8])
        return np.column_stack(((a-b)/np.sqrt(2), (a+b-2*c)/np.sqrt(6), (a+b+c-3*d)/np.sqrt(12)))
    if experiment == "E11":
        return ((np.asarray(frame["P_energy"])-np.asarray(frame["P_location"]))/np.sqrt(2))[:, None]
    if experiment == "E12":
        return (maturity(frame["time_online"]) - center)[:, None] * base[:, 1:4]
    if experiment == "E13":
        return np.empty((len(base), 0))
    raise ValueError("Unknown frozen experiment")


class Preprocessing:
    def __init__(self, experiment, mean, std, extra_mean, extra_std, center):
        if experiment not in INPUT_LISTS:
            raise ValueError("Unknown experiment")
        self.experiment, self.center = experiment, float(center)
        for name, value, size in (("mean", mean, 4), ("std", std, 4),
                ("extra_mean", extra_mean, len(INPUT_LISTS[experiment])-4),
                ("extra_std", extra_std, len(INPUT_LISTS[experiment])-4)):
            array = np.array(value, dtype=np.float64, copy=True)
            if array.shape != (size,) or not np.isfinite(array).all():
                raise ValueError("Invalid preprocessing")
            if "std" in name and np.any(array < 0):
                raise ValueError("Negative standard deviation")
            array.setflags(write=False)
            setattr(self, name, array)
        if not np.isfinite(self.center):
            raise ValueError("Invalid centering constant")

    @classmethod
    def fit(cls, experiment, frame, omega):
        # Deliberately invoke E08's four-column routine, without extending it.
        # Canonical column-major layout matches the saved E08 parquet path and
        # prevents held-out DataFrame block fragmentation changing dot reductions.
        x = np.asfortranarray(frame.loc[:, list(INPUTS)].to_numpy(dtype=np.float64))
        mean, std, base = standardize_fit(x, omega)
        center = weighted_center(frame.q, omega) if experiment == "E09" else (
            weighted_center(maturity(frame.time_online), omega) if experiment == "E12" else 0.0)
        extra = extension(experiment, frame, base, center)
        em, es, scaled = standardize_extra(extra, omega)
        return cls(experiment, mean, std, em, es, center), np.column_stack((base, scaled))

    def transform(self, frame):
        x = frame.loc[:, list(INPUTS)].to_numpy(dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite base columns")
        base = np.divide(x-self.mean, self.std, out=np.zeros_like(x), where=self.std > 0)
        extra = extension(self.experiment, frame, base, self.center)
        if not np.isfinite(extra).all():
            raise ValueError("Nonfinite extension")
        scaled = np.divide(extra-self.extra_mean, self.extra_std,
                           out=np.zeros_like(extra), where=self.extra_std > 0)
        return np.column_stack((base, scaled))

    def as_dict(self):
        return dict(experiment=self.experiment, inputs=INPUT_LISTS[self.experiment], center=self.center,
                    **{k: getattr(self, k).tolist() for k in ("mean", "std", "extra_mean", "extra_std")})

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.pop("inputs") != INPUT_LISTS[value["experiment"]]:
            raise ValueError("Changed input allowlist")
        return cls(**value)


class Stage3Detector:
    """A shared fitted model over unchanged streaming Stage-2 state."""
    def __init__(self, history, model):
        self.blocks = Stage2Detector(history, "E07")
        self.model = model
        self.q = context_q(self.blocks.ar.mse_ratio) if model.pre.experiment == "E09" else None

    def update(self, value):
        import pandas as pd
        self.blocks.update(value)
        row = raw_values(self.blocks)
        row.update(q=self.q, time_online=self.blocks.age-1)
        return float(self.model.predict(pd.DataFrame([row]))[0])

    def dumps(self):
        # The shared model is external, hash-bound, not repeated in every series.
        return json.dumps(dict(version=1, model_sha256=hashlib.sha256(self.model.dumps()).hexdigest(),
                               q=self.q, blocks=json.loads(self.blocks.dumps())), sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode()

    @classmethod
    def loads(cls, payload, model):
        value = json.loads(payload)
        if set(value) != {"version", "model_sha256", "q", "blocks"} or value["version"] != 1:
            raise ValueError("Invalid Stage-3 state")
        if value["model_sha256"] != hashlib.sha256(model.dumps()).hexdigest():
            raise ValueError("Shared model mismatch")
        obj = cls.__new__(cls)
        obj.blocks = Stage2Detector.loads(json.dumps(value["blocks"]))
        obj.model, obj.q = model, value["q"]
        expected_q = context_q(obj.blocks.ar.mse_ratio) if model.pre.experiment == "E09" else None
        if obj.blocks.experiment != "E07" or obj.q != expected_q:
            raise ValueError("Historical context changed")
        return obj


def infer(datasets, model):
    yield
    for history, online in datasets:
        state = Stage3Detector(history, model)
        for value in online:
            yield state.update(value)

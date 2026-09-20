"""Frozen E02-E07 inference, without evaluator or profiler imports."""
import base64
import json
import math
import numpy as np
from .history import ARChannel, EmpiricalCDF, moments
from .features import MultiScaleEvidence, RollingMean, SignDependence, SignWindow


class Stage2Detector:
    def __init__(self, history, experiment):
        if experiment not in ("E02", "E03", "E04", "E05", "E06", "E07"):
            raise ValueError("Only frozen Phase-1 experiments E02-E07 are executable")
        h = np.asarray(history, dtype=np.float64)
        if not np.isfinite(h).all() or len(h) < 2:
            raise ValueError("Invalid history")
        self.experiment = experiment
        self.age = 0
        self.raw = self.ar = self.innovation = self.cdf = None
        self.rank_location = self.rank_energy = self.dependence = None
        self.raw_mu = self.raw_sd = None
        self.last_features = []
        self.last_blocks = {}
        if experiment != "E04":
            ref = h if experiment == "E02" else h[int(.7 * len(h)):]
            self.raw_mu, self.raw_sd = moments(ref)
            z = (ref - self.raw_mu) / self.raw_sd
            self.raw = MultiScaleEvidence(z * z)
        if experiment in ("E04", "E05", "E06", "E07"):
            self.ar = ARChannel(h)
            ref_e = self.ar.reference(h)
            ref_r = (ref_e - self.ar.ref_mu) / self.ar.ref_sd
            self.innovation = MultiScaleEvidence(ref_r * ref_r)
            if experiment in ("E06", "E07"):
                self.cdf = EmpiricalCDF(ref_r)
                ref_v = 2.0 * self.cdf.transform(ref_r) - 1.0
                self.rank_location = MultiScaleEvidence(ref_v)
                self.rank_energy = MultiScaleEvidence(ref_v * ref_v)
            if experiment == "E07":
                self.dependence = SignDependence(ref_e, self.ar.ref_median)

    def update(self, x):
        x = float(x)
        if not math.isfinite(x):
            raise ValueError("Nonfinite online observation")
        self.age += 1
        features, blocks = [], {}
        if self.raw is not None:
            z = (x - self.raw_mu) / self.raw_sd
            values = self.raw.update(z * z)
            features.extend(values)
            blocks["R"] = sum(values) / 4
        if self.ar is not None:
            e, r = self.ar.update(x)
            values = self.innovation.update(r * r)
            features.extend(values)
            blocks["I"] = sum(values) / 4
            if self.cdf is not None:
                v = 2.0 * float(self.cdf.transform(r)) - 1.0
                location = self.rank_location.update(v)
                energy = self.rank_energy.update(v * v)
                features.extend(location + energy)
                blocks["P_location"] = sum(location) / 4
                blocks["P_energy"] = sum(energy) / 4
                blocks["P"] = sum(location + energy) / 8
            if self.dependence is not None:
                values = self.dependence.update(e)
                features.extend(values)
                blocks["D"] = sum(values) / 2
        self.last_features, self.last_blocks = features, blocks
        selected = [blocks[k] for k in ("R", "I", "P", "D") if k in blocks]
        score = sum(selected) / len(selected)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Nonfinite or out-of-range Stage-2 prediction")
        return score

    def dumps(self):
        return json.dumps(dict(version=1, state=_encode(self)), sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")

    @classmethod
    def loads(cls, payload):
        obj = json.loads(payload)
        if obj.get("version") != 1 or set(obj) != {"version", "state"}:
            raise ValueError("Incompatible state schema")
        state = _decode(obj["state"])
        if not isinstance(state, cls):
            raise ValueError("Not a Stage-2 state")
        return state

    def array_bytes(self):
        return _array_bytes(self, set())


_CLASSES = {c.__name__: c for c in (Stage2Detector, ARChannel, EmpiricalCDF,
            MultiScaleEvidence, RollingMean, SignDependence, SignWindow)}


def _encode(value):
    if isinstance(value, np.ndarray):
        return dict(array=base64.b64encode(value.tobytes()).decode("ascii"),
                    dtype=value.dtype.str, shape=list(value.shape),
                    writeable=bool(value.flags.writeable))
    if type(value).__name__ in _CLASSES:
        return dict(type=type(value).__name__, fields=_encode(value.__dict__))
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(type(value))


def _decode(value):
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if isinstance(value, dict):
        if "array" in value:
            dtype = np.dtype(value["dtype"])
            if dtype.hasobject:
                raise ValueError("Object arrays are forbidden")
            a = np.frombuffer(base64.b64decode(value["array"], validate=True), dtype=dtype).copy()
            a = a.reshape(value["shape"])
            a.setflags(write=value["writeable"])
            return a
        if "type" in value:
            if value["type"] not in _CLASSES:
                raise ValueError("Unknown serialized state type")
            result = _CLASSES[value["type"]].__new__(_CLASSES[value["type"]])
            result.__dict__.update(_decode(value["fields"]))
            return result
        return {k: _decode(v) for k, v in value.items()}
    return value


def _array_bytes(value, seen):
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_array_bytes(v, seen) for v in value.values())
    if isinstance(value, list):
        return sum(_array_bytes(v, seen) for v in value)
    if hasattr(value, "__dict__"):
        return _array_bytes(value.__dict__, seen)
    return 0

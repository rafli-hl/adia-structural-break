"""Inference-only code: no labels, IDs, final horizons, or cross-series state."""
import math
import numpy as np


class OfficialEWMABaseline:
    ALPHA = 0.05
    KAPPA = 3.0

    def __init__(self, historical):
        h = np.asarray(historical, dtype=np.float64)
        self.mu_h = float(h.mean()) if len(h) else 0.0
        self.sd_h = float(h.std(ddof=1)) if len(h) > 1 else 1.0
        self.sd_h = max(self.sd_h, 1e-8)
        self.mu_ewma = self.mu_h
        self.n_eff = 0.0

    def update(self, x):
        x = float(x)
        a = self.ALPHA
        self.mu_ewma = (1.0 - a) * self.mu_ewma + a * x
        self.n_eff = (1.0 - a) * self.n_eff + 1.0
        se = self.sd_h / math.sqrt(max(self.n_eff, 1.0))
        z = (self.mu_ewma - self.mu_h) / max(se, 1e-8)
        return float(math.tanh(abs(z) / self.KAPPA))


class EvidenceDetector:
    """Fixed E01 configuration; Welford updates, frozen historical reference.

    Scale floors: 1e-8 for raw normalization and feature-level standard
    deviations; 1e-12 for normalized variance before taking a log.
    Page-Hinkley uses the inclusive online running mean and delta=0.05.
    """
    def __init__(self, historical, kind):
        h = np.asarray(historical, dtype=np.float64)
        self.mu = float(h.mean()) if len(h) else 0.0
        self.sd = max(float(h.std(ddof=1)) if len(h) > 1 else 1.0, 1e-8)
        self.kind = kind
        z = (h - self.mu) / self.sd
        self.reference_variance = float(z.var(ddof=1)) if len(z) > 1 else 1.0
        a = np.abs(z) if kind == "absolute" else z * z
        self.ref_mean = float(a.mean()) if len(a) else 0.0
        self.ref_sd = max(float(a.std(ddof=1)) if len(a) > 1 else 1.0, 1e-8)
        self.n = 0
        self.mean = self.m2 = self.transformed_mean = 0.0
        self.pos = self.neg = self.min_pos = self.min_neg = 0.0

    def update(self, x):
        z = (float(x) - self.mu) / self.sd
        self.n += 1
        delta = z - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (z - self.mean)
        if self.kind == "cusum":
            self.pos = max(0.0, self.pos + z - 0.5)
            self.neg = max(0.0, self.neg - z - 0.5)
            q = max(self.pos, self.neg)
        elif self.kind == "page_hinkley":
            centered = z - self.mean
            self.pos += centered - 0.05
            self.neg += -centered - 0.05
            self.min_pos = min(self.min_pos, self.pos)
            self.min_neg = min(self.min_neg, self.neg)
            q = max(self.pos - self.min_pos, self.neg - self.min_neg)
        elif self.kind == "running_mean":
            q = math.sqrt(self.n) * abs(self.mean)
        elif self.kind == "variance_ratio":
            q = abs(math.log(max(self.m2 / (self.n - 1), 1e-12)
                             / max(self.reference_variance, 1e-12))) if self.n > 1 else 0.0
        elif self.kind in ("absolute", "squared"):
            a = abs(z) if self.kind == "absolute" else z * z
            self.transformed_mean += (a - self.transformed_mean) / self.n
            q = math.sqrt(self.n) * abs(self.transformed_mean - self.ref_mean) / self.ref_sd
        else:
            raise ValueError(self.kind)
        if not math.isfinite(q):
            raise ValueError("Nonfinite evidence; inspect data quality")
        return q / (1.0 + q)


class Constant:
    def __init__(self, historical):
        pass

    def update(self, x):
        return 0.5


class AgeOnly:
    def __init__(self, historical):
        self.t = 0

    def update(self, x):
        self.t += 1
        return self.t / (self.t + 1.0)


E01 = ("official_ewma", "cusum", "page_hinkley", "running_mean",
       "variance_ratio", "absolute", "squared")


def make_detector(name, history):
    if name in ("E02", "E03", "E04", "E05", "E06", "E07"):
        from .stage2_detectors import Stage2Detector
        return Stage2Detector(history, name)
    if name == "official_ewma":
        return OfficialEWMABaseline(history)
    if name == "constant":
        return Constant(history)
    if name == "age_only":
        return AgeOnly(history)
    if name not in E01:
        raise ValueError(name)
    return EvidenceDetector(history, name)


def infer(datasets, detector="official_ewma"):
    """Actual generator handshake. Online streams are never materialized."""
    yield
    for historical, online in datasets:
        state = make_detector(detector, historical)
        for x in online:
            yield state.update(x)

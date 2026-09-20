"""Predict first, standardize, then advance a frozen historical scale recursion."""
import math
import numpy as np
from sbrt.history import moments, EmpiricalCDF
from sbrt.features import MultiScaleEvidence, correlation_from_sums
from sbrt.stage2_detectors import _encode, _decode


def lag_one(values):
    a, b = np.asarray(values[:-1]), np.asarray(values[1:])
    return correlation_from_sums(len(a), float(a.sum()), float(b.sum()),
        float(a@a), float(b@b), float(a@b))


class ScaleEvidence:
    def __init__(self, residuals, cut, kind, rank=False, *, audit_alpha=None):
        if kind not in ("slow", "fast", "robust"):
            raise ValueError("Unknown scale stream")
        e = np.asarray(residuals, dtype=np.float64)
        if cut < 7 or len(e)-cut < 2 or not np.isfinite(e[5:]).all():
            raise ValueError("Invalid historical innovation support")
        # audit_alpha=0 is used by arithmetic unit tests only, never a candidate.
        if audit_alpha not in (None, 0):
            raise ValueError("Only the nonselectable alpha-zero arithmetic audit")
        self.kind = kind
        self.alpha = (0.05 if kind == "fast" else 0.01) if audit_alpha is None else 0.0
        a2 = e[5:cut]**2
        self.b_raw = float(np.mean(a2))
        self.b = max(self.b_raw, 1e-16)
        self.v_min = max(1e-16, 1e-4*self.b)
        self.h = self.b
        self.cap = self.clipped_mean = self.factor = None
        self.robust_supported = False
        if kind == "robust":
            self.cap = float(np.quantile(a2, .99, method="linear"))
            self.clipped_mean = float(np.mean(np.minimum(a2, self.cap)))
            self.robust_supported = self.clipped_mean > 1e-16
            if self.robust_supported:
                self.factor = self.b/self.clipped_mean
        self.counts = dict(baseline_floor=int(self.b_raw < 1e-16), warmup_floor=0,
            reference_floor=0, online_floor=0, historical_clipping=0, online_clipping=0,
            reference_sd_floor=0, robust_fallback=int(kind == "robust" and not self.robust_supported))
        reference = []
        for j in range(5, len(e)):
            forecast = self.forecast()
            if self.h < self.v_min:
                self.counts["warmup_floor" if j < cut else "reference_floor"] += 1
            if j >= cut:
                reference.append(float(e[j])/math.sqrt(forecast))
            self.advance(float(e[j]), historical=True)
        reference = np.asarray(reference, dtype=np.float64)
        if not np.isfinite(reference).all():
            raise ValueError("Nonfinite standardized reference")
        self.mu, self.sd = moments(reference)
        self.counts["reference_sd_floor"] += int(float(reference.std(ddof=1)) < 1e-8)
        ref = (reference-self.mu)/self.sd
        self.energy = MultiScaleEvidence(ref*ref)
        self.counts["reference_sd_floor"] += int(float((ref*ref).std(ddof=1)) < 1e-8)
        self.reference_squared_acf = lag_one(ref*ref)
        self.cdf = self.location = self.rank_energy = None
        if rank:
            if kind != "slow":
                raise ValueError("Only slow rank stream is preregistered")
            self.cdf = EmpiricalCDF(ref)
            v = 2.0*self.cdf.transform(ref)-1.0
            self.location, self.rank_energy = MultiScaleEvidence(v), MultiScaleEvidence(v*v)
            self.counts["reference_sd_floor"] += int(float(v.std(ddof=1)) < 1e-8)+int(float((v*v).std(ddof=1)) < 1e-8)
        self.pending = None
        self.last = None

    def forecast(self):
        if not math.isfinite(self.h) or self.h < 0:
            raise ValueError("Nonfinite/negative conditional second moment")
        return max(self.h, self.v_min)

    def advance(self, e, historical=False):
        e2 = e*e
        if not math.isfinite(e2):
            raise ValueError("Nonfinite squared innovation")
        driver = e2
        if self.kind == "robust" and self.robust_supported:
            self.counts["historical_clipping" if historical else "online_clipping"] += int(e2 > self.cap)
            driver = self.factor*min(e2, self.cap)
        self.h = (1.0-self.alpha)*self.h+self.alpha*driver
        self.forecast()  # Fail closed immediately, including final observation.

    def observe(self, e):
        if self.pending is not None or not math.isfinite(e):
            raise ValueError("Uncommitted update or nonfinite innovation")
        variance = self.forecast()
        self.counts["online_floor"] += int(self.h < self.v_min)
        u = e/math.sqrt(variance)
        a = (u-self.mu)/self.sd
        values = self.energy.update(a*a)
        energy = sum(values)/4
        rank = None
        if self.cdf is not None:
            v = 2.0*float(self.cdf.transform(a))-1.0
            rank_values = self.location.update(v)+self.rank_energy.update(v*v)
            rank = sum(rank_values)/8
        if not all(math.isfinite(v) for v in (e,variance,u,a,energy)):
            raise ValueError("Nonfinite conditional evidence")
        self.pending = e
        self.last = dict(variance=variance, u=u, a=a, J=energy, P=rank, energy_components=values)
        return self.last

    def commit(self):
        if self.pending is None:
            raise ValueError("No pending innovation")
        self.advance(self.pending)
        self.pending = None

    def state(self):
        return _encode(self.__dict__)

    @classmethod
    def restore(cls, fields):
        obj = cls.__new__(cls)
        obj.__dict__.update(_decode(fields))
        obj.forecast()
        return obj

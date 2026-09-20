"""History-only numerical helpers; safe to import from streaming inference."""
import numpy as np


# Extracted verbatim from the validated profiler. AST regression tests pin both
# procedures to the pre-Stage-2 foundation, including operation order.
def fit_ar(h, p, ridge=1.0):
    if len(h) < max(32, 4 * p) or not np.isfinite(h).all():
        return None
    mu, sd = float(np.mean(h)), float(np.std(h, ddof=1))
    if sd <= 1e-8:
        return None
    z = (np.asarray(h) - mu) / sd
    windows = np.lib.stride_tricks.sliding_window_view(z, p + 1)
    X = np.column_stack([np.ones(len(windows)), windows[:, :-1][:, ::-1]])
    y = windows[:, -1]
    penalty = np.eye(p + 1) * ridge
    penalty[0, 0] = 0
    beta = np.linalg.solve(X.T @ X + penalty, X.T @ y)
    return mu, sd, beta


def residuals(values, model):
    mu, sd, beta = model
    p = len(beta) - 1
    z = (np.asarray(values, dtype=float) - mu) / sd
    out = np.full(len(z), np.nan)
    if len(z) > p:
        w = np.lib.stride_tricks.sliding_window_view(z, p + 1)
        out[p:] = w[:, -1] - beta[0] - w[:, :-1][:, ::-1] @ beta[1:]
    return out


def moments(values):
    a = np.asarray(values, dtype=np.float64)
    if len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("Invalid historical reference")
    return float(a.mean()), max(float(a.std(ddof=1)), 1e-8)


class ARChannel:
    def __init__(self, history):
        h = np.asarray(history, dtype=np.float64)
        self.cut = int(.7 * len(h))
        model = fit_ar(h[:self.cut], 5, ridge=1.0)
        if model is None:
            raise ValueError("Historical AR(5) fitting failed; no channel substitution")
        self.mu, self.sd, self.beta = model
        self.beta.setflags(write=False)
        ref = residuals(h, model)[self.cut:]
        if not np.isfinite(ref).all():
            raise ValueError("Nonfinite historical AR innovations")
        self.ref_mu, self.ref_sd = moments(ref)
        self.ref_median = float(np.median(ref))
        baseline = (h[self.cut:] - self.mu) / self.sd
        denom = float(np.mean(baseline * baseline))
        self.mse_ratio = float(np.mean(ref * ref)) / denom if denom > 0 else None
        self.lags = ((h[-5:][::-1] - self.mu) / self.sd).copy()

    def reference(self, history):
        return residuals(history, (self.mu, self.sd, self.beta))[self.cut:]

    def update(self, x):
        y = (float(x) - self.mu) / self.sd
        e = float(y - self.beta[0] - self.lags @ self.beta[1:])
        for j in range(4, 0, -1):
            self.lags[j] = self.lags[j - 1]
        self.lags[0] = y
        return e, (e - self.ref_mu) / self.ref_sd


class EmpiricalCDF:
    def __init__(self, reference):
        self.reference = np.sort(np.asarray(reference, dtype=np.float64)).copy()
        if not len(self.reference) or not np.isfinite(self.reference).all():
            raise ValueError("Invalid empirical CDF reference")
        self.reference.setflags(write=False)

    def transform(self, value):
        left = np.searchsorted(self.reference, value, side="left")
        right = np.searchsorted(self.reference, value, side="right")
        return (left + .5 * (right - left) + .5) / (len(self.reference) + 1)

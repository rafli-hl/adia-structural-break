"""Bounded causal state for the frozen Stage-2 statistics."""
import math
import numpy as np
from .history import moments

WINDOWS = (32, 128, 256)


def evidence(n, mean, ref_mean, ref_sd):
    q = math.sqrt(n) * abs(mean - ref_mean) / ref_sd
    if not math.isfinite(q):
        raise ValueError("Nonfinite evidence; experiment invalid")
    return q / (1.0 + q)


class RollingMean:
    def __init__(self, window):
        self.buffer = np.zeros(window, dtype=np.float64)
        self.window = window
        self.pointer = self.n = 0
        self.total = 0.0

    def update(self, value):
        if self.n == self.window:
            self.total -= float(self.buffer[self.pointer])
        else:
            self.n += 1
        self.total += value
        self.buffer[self.pointer] = value
        self.pointer = (self.pointer + 1) % self.window
        return self.total / self.n


class MultiScaleEvidence:
    def __init__(self, reference):
        self.ref_mean, self.ref_sd = moments(reference)
        self.windows = [RollingMean(w) for w in WINDOWS]
        self.n = 0
        self.mean = 0.0

    def update(self, value):
        self.n += 1
        # Same cumulative operation order as E01 squared.
        self.mean += (value - self.mean) / self.n
        values = [evidence(win.n, avg, self.ref_mean, self.ref_sd)
                  for win in self.windows for avg in [win.update(value)]]
        values.append(evidence(self.n, self.mean, self.ref_mean, self.ref_sd))
        return values


def correlation_from_sums(m, sa, sb, saa, sbb, sab):
    if m <= 0:
        return None
    va, vb = saa - sa * sa / m, sbb - sb * sb / m
    if va <= 1e-12 or vb <= 1e-12:
        return None
    return max(-1.0, min(1.0, (sab - sa * sb / m) / math.sqrt(va * vb)))


def historical_sign_correlation(signs):
    a, b = signs[:-1], signs[1:]
    return correlation_from_sums(len(a), float(a.sum()), float(b.sum()),
                                 float(a @ a), float(b @ b), float(a @ b))


class SignWindow:
    def __init__(self, window):
        self.buffer = np.zeros(window, dtype=np.int8)
        self.window = window
        self.pointer = self.n = 0
        self.sa = self.sb = self.saa = self.sbb = self.sab = 0.0

    def pair(self, a, b, direction):
        self.sa += direction * a
        self.sb += direction * b
        self.saa += direction * a * a
        self.sbb += direction * b * b
        self.sab += direction * a * b

    def update(self, value):
        if self.n == self.window:
            self.pair(int(self.buffer[self.pointer]),
                      int(self.buffer[(self.pointer + 1) % self.window]), -1)
        if self.n:
            self.pair(int(self.buffer[(self.pointer - 1) % self.window]), value, 1)
        self.buffer[self.pointer] = value
        self.pointer = (self.pointer + 1) % self.window
        self.n = min(self.n + 1, self.window)
        return correlation_from_sums(self.n - 1, self.sa, self.sb,
                                     self.saa, self.sbb, self.sab)


class SignDependence:
    def __init__(self, reference_innovations, median):
        self.median = median
        signs = np.sign(reference_innovations - median).astype(np.float64)
        # One full-C reference, deliberately shared by both online windows.
        self.reference_correlation = historical_sign_correlation(signs)
        self.windows = [SignWindow(128), SignWindow(256)]
        self.counts = {str(w): dict(supported=0, insufficient_pairs=0,
                                    online_degenerate=0, historical_disabled=0)
                       for w in (128, 256)}

    def update(self, innovation):
        value = int(innovation > self.median) - int(innovation < self.median)
        result = []
        for window in self.windows:
            rho = window.update(value)
            counts = self.counts[str(window.window)]
            m = window.n - 1
            if self.reference_correlation is None:
                counts["historical_disabled"] += 1
                result.append(0.0)
            elif m < 8:
                counts["insufficient_pairs"] += 1
                result.append(0.0)
            elif rho is None:
                counts["online_degenerate"] += 1
                result.append(0.0)
            else:
                counts["supported"] += 1
                q = math.sqrt(m) * abs(rho - self.reference_correlation)
                result.append(q / (1.0 + q))
        return result

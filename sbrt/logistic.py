"""Frozen E08: four blocks, training-only preprocessing, exact weighted objective.

No evaluator metadata reaches LogisticModel or E08Detector. The optimizer uses
SciPy's unbounded L-BFGS-B (L-BFGS), analytic gradients, and one native thread.
"""
import json
import time
import hashlib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits, threadpool_info
from .stage2_detectors import Stage2Detector

INPUTS = ("R", "I", "P", "D")
L2 = 0.01
SOLVER_OPTIONS = dict(maxiter=1000, gtol=1e-6, ftol=0.0, maxcor=10,
                      maxls=20, maxfun=20001)


def training_weights(age, target):
    """Vectorized exact Phase-1 weighting formula; inputs are training rows only."""
    age = np.asarray(age)
    target = np.asarray(target)
    if (age.ndim != 1 or target.shape != age.shape or not len(age)
            or not np.isfinite(age).all() or np.any(age < 0)
            or np.any(age != np.floor(age)) or not np.isin(target, [0, 1]).all()):
        raise ValueError("Invalid training ages or binary labels")
    age = age.astype(np.int64)
    target = target.astype(np.int64)
    neg = np.bincount(age[target == 0], minlength=int(age.max()) + 1)
    pos = np.bincount(age[target == 1], minlength=len(neg))
    pairs = pos * neg
    z = float(pairs.sum())
    if z <= 0:
        raise ValueError("No eligible training ages")
    eligible = pairs[age] > 0
    t, y = age[eligible], target[eligible]
    n_y = np.where(y == 1, pos[t], neg[t])
    omega = pairs[t] / (2.0 * z * n_y)
    expected = pairs / z
    mass = np.bincount(t, weights=omega, minlength=len(pairs))
    pos_mass = np.bincount(t[y == 1], weights=omega[y == 1], minlength=len(pairs))
    neg_mass = np.bincount(t[y == 0], weights=omega[y == 0], minlength=len(pairs))
    errors = dict(total=abs(float(omega.sum()) - 1.0),
                  age=float(np.max(np.abs(mass - expected))),
                  positive=float(np.max(np.abs(pos_mass - expected / 2))),
                  negative=float(np.max(np.abs(neg_mass - expected / 2))))
    if max(errors.values()) > 1e-12:
        raise ArithmeticError("Frozen weight identities failed")
    audit = dict(training_rows=len(age), eligible_rows=int(eligible.sum()),
                 excluded_rows=int((~eligible).sum()), eligible_ages=int((pairs > 0).sum()),
                 Z=z, total_weight=float(omega.sum()), maximum_absolute_errors=errors,
                 ages=[dict(time_online=int(k), age=int(k + 1), n_pos=int(pos[k]),
                            n_neg=int(neg[k]), pair_weight=int(pairs[k]),
                            expected_mass=float(expected[k]), actual_mass=float(mass[k]),
                            positive_mass=float(pos_mass[k]), negative_mass=float(neg_mass[k]))
                       for k in range(len(pairs))],
                 weights_sha256=hashlib.sha256(omega.tobytes()).hexdigest())
    return eligible, omega, audit


def standardize_fit(x, omega):
    x = np.asarray(x, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    if x.ndim != 2 or x.shape != (len(omega), 4) or not np.isfinite(x).all():
        raise ValueError("Exactly four finite training inputs required")
    total = float(omega.sum())
    mean = np.array([np.dot(omega, x[:, j]) / total for j in range(4)])
    # An exactly constant column has exactly zero variance, not rounding noise.
    for j in range(4):
        if np.all(x[:, j] == x[0, j]):
            mean[j] = x[0, j]
    centered = x - mean
    variance = np.array([np.dot(omega, centered[:, j] ** 2) / total for j in range(4)])
    std = np.sqrt(variance)
    scaled = np.divide(centered, std, out=np.zeros_like(centered), where=std > 0)
    return mean, std, scaled


def objective(theta, x, target, omega):
    """sum(omega * logistic_loss) + .01/2 * ||beta||^2; sum omega=1."""
    intercept, beta = theta[0], theta[1:]
    logits = intercept + x @ beta
    value = float(np.dot(omega, np.logaddexp(0.0, logits) - target * logits)
                  + L2 / 2 * np.dot(beta, beta))
    error = omega * (expit(logits) - target)
    gradient = np.r_[error.sum(), x.T @ error + L2 * beta]
    return value, gradient


class LogisticModel:
    def __init__(self, mean, std, intercept, beta):
        self.mean = np.array(mean, dtype=np.float64, copy=True)
        self.std = np.array(std, dtype=np.float64, copy=True)
        self.beta = np.array(beta, dtype=np.float64, copy=True)
        self.intercept = float(intercept)
        if (any(a.shape != (4,) or not np.isfinite(a).all()
                for a in (self.mean, self.std, self.beta))
                or np.any(self.std < 0) or not np.isfinite(self.intercept)):
            raise ValueError("Invalid frozen model parameters")
        for a in (self.mean, self.std, self.beta):
            a.setflags(write=False)

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim not in (1, 2) or x.shape[-1] != 4 or not np.isfinite(x).all():
            raise ValueError("Only four finite R/I/P/D inputs accepted")
        z = np.divide(x - self.mean, self.std, out=np.zeros_like(x), where=self.std > 0)
        # Fixed arithmetic order: scalar streaming and batched cached replay agree.
        logits = self.intercept + z[..., 0] * self.beta[0]
        for j in range(1, 4):
            logits = logits + z[..., j] * self.beta[j]
        return expit(logits)

    def dumps(self):
        return json.dumps(dict(version=1, inputs=list(INPUTS), mean=self.mean.tolist(),
                               std=self.std.tolist(), intercept=self.intercept,
                               beta=self.beta.tolist()), sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")

    @classmethod
    def loads(cls, payload):
        value = json.loads(payload)
        if (set(value) != {"version", "inputs", "mean", "std", "intercept", "beta"}
                or value["version"] != 1 or value["inputs"] != list(INPUTS)):
            raise ValueError("Invalid E08 model schema/input order")
        return cls(value["mean"], value["std"], value["intercept"], value["beta"])


class InvalidFit(RuntimeError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__("E08 invalid: optimization did not meet frozen convergence requirement")


def fit_fold(frame, heldout_fold):
    """Split first: held-out labels/features/ages never enter numerical fitting."""
    start = time.perf_counter()
    train = frame.loc[frame.fold != heldout_fold, [*INPUTS, "time_online", "target"]]
    with threadpool_limits(limits=1):
        active_threads = threadpool_info()
        eligible, omega, audit = training_weights(train.time_online.to_numpy(), train.target.to_numpy())
        x = train.loc[eligible, list(INPUTS)].to_numpy(dtype=np.float64)
        target = train.loc[eligible, "target"].to_numpy(dtype=np.float64)
        mean, std, x = standardize_fit(x, omega)
        result = minimize(objective, np.zeros(5, dtype=np.float64), args=(x, target, omega),
                          method="L-BFGS-B", jac=True, bounds=None, options=SOLVER_OPTIONS)
        final_objective, gradient = objective(result.x, x, target, omega)
    diagnostics = dict(fold=int(heldout_fold), iterations=int(result.nit),
                       function_evaluations=int(result.nfev), objective=final_objective,
                       gradient_infinity_norm=float(np.max(np.abs(gradient))),
                       final_gradient=gradient.tolist(), optimizer_success=bool(result.success),
                       optimizer_status=int(result.status), optimizer_message=str(result.message),
                       solver="scipy.optimize.minimize L-BFGS-B without bounds", options=SOLVER_OPTIONS,
                       native_threads=active_threads, weighted_mean=mean.tolist(),
                       weighted_std=std.tolist(), zero_variance=(std == 0).tolist(),
                       intercept=float(result.x[0]), beta=result.x[1:].tolist(),
                       training_seconds=time.perf_counter() - start)
    if (not result.success or not np.isfinite(result.x).all() or not np.isfinite(final_objective)
            or not np.isfinite(gradient).all() or diagnostics["gradient_infinity_norm"] > 1e-6
            or result.nit > 1000):
        raise InvalidFit(diagnostics)
    return LogisticModel(mean, std, result.x[0], result.x[1:]), diagnostics, audit


class E08Detector:
    """O(1) logistic layer on the unchanged E07 causal four-block state."""
    def __init__(self, history, model):
        self.blocks = Stage2Detector(history, "E07")
        self.model = model

    def update(self, value):
        self.blocks.update(value)
        return float(self.model.predict([self.blocks.last_blocks[k] for k in INPUTS]))

    def dumps(self):
        return json.dumps(dict(version=1, model=json.loads(self.model.dumps()),
                               blocks=json.loads(self.blocks.dumps())), sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")

    @classmethod
    def loads(cls, payload):
        value = json.loads(payload)
        if set(value) != {"version", "model", "blocks"} or value["version"] != 1:
            raise ValueError("Invalid E08 state schema")
        obj = cls.__new__(cls)
        obj.model = LogisticModel.loads(json.dumps(value["model"]))
        obj.blocks = Stage2Detector.loads(json.dumps(value["blocks"]))
        if obj.blocks.experiment != "E07":
            raise ValueError("E08 requires unchanged E07 block state")
        return obj


def infer(datasets, model):
    """Existing competition handshake; one already-trained shared model."""
    yield
    for history, online in datasets:
        state = E08Detector(history, model)
        for value in online:
            yield state.update(value)

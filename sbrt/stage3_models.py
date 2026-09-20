"""Frozen Stage-3 estimators; reuse E08 weights, objective and solver options."""
import hashlib
import json
import pickle
import time
import numpy as np
import sklearn
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits, threadpool_info
from .logistic import training_weights, objective, SOLVER_OPTIONS, InvalidFit
from .stage3_features import Preprocessing, INPUT_LISTS, CONTRASTS, maturity

TREE_PARAMS = dict(loss="log_loss", learning_rate=.05, max_iter=100, max_depth=2,
    max_leaf_nodes=4, min_samples_leaf=10000, max_bins=64, l2_regularization=.01,
    max_features=1., categorical_features=None, monotonic_cst=None, interaction_cst=None,
    class_weight=None, early_stopping=False, validation_fraction=None, warm_start=False,
    random_state=20260921, scoring="loss", n_iter_no_change=10, tol=1e-7, verbose=0)


class Model:
    def __init__(self, pre, intercept=None, beta=None, tree=None):
        self.pre, self.tree = pre, tree
        self.intercept = None if intercept is None else float(intercept)
        self.beta = None if beta is None else np.asarray(beta, dtype=np.float64).copy()
        if pre.experiment == "E13":
            if not isinstance(tree, HistGradientBoostingClassifier) or tree.n_iter_ != 100:
                raise ValueError("Invalid native boosting model")
        else:
            if (tree is not None or self.beta.shape != (len(INPUT_LISTS[pre.experiment]),)
                    or not np.isfinite(self.beta).all() or not np.isfinite(self.intercept)):
                raise ValueError("Invalid logistic coefficients")
            self.beta.setflags(write=False)

    def predict(self, frame):
        x = self.pre.transform(frame)
        with threadpool_limits(limits=1):
            if self.tree is not None:
                p = self.tree.predict_proba(x)[:, 1]
            else:
                logits = self.intercept + x[:, 0]*self.beta[0]
                for j in range(1, len(self.beta)):
                    logits = logits + x[:, j]*self.beta[j]
                p = expit(logits)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Invalid prediction")
        return p

    def dumps(self):
        if self.tree is not None:
            # Local trusted artifacts only; verify the file hash before loading pickle.
            return pickle.dumps(("stage3_native_1", self.pre.as_dict(), self.tree), protocol=5)
        return json.dumps(dict(version=1, pre=self.pre.as_dict(), intercept=self.intercept,
                               beta=self.beta.tolist()), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    @classmethod
    def loads(cls, payload, native=False):
        if native:
            version, pre, tree = pickle.loads(payload)
            if version != "stage3_native_1":
                raise ValueError("Unknown native model schema")
            return cls(Preprocessing.from_dict(pre), tree=tree)
        value = json.loads(payload)
        if set(value) != {"version", "pre", "intercept", "beta"} or value["version"] != 1:
            raise ValueError("Unknown logistic schema")
        return cls(Preprocessing.from_dict(value["pre"]), value["intercept"], value["beta"])


def coefficient_diagnostics(model):
    pre, beta = model.pre, model.beta
    if beta is None:
        return {}
    raw = np.divide(beta[:4], pre.std, out=np.zeros(4), where=pre.std > 0)
    extra = np.divide(beta[4:], pre.extra_std, out=np.zeros(len(beta)-4), where=pre.extra_std > 0)
    result = dict(intercept=model.intercept, beta=beta.tolist(), raw_base_slopes=raw.tolist())
    if pre.experiment == "E10":
        result["implied_innovation_scale_slopes"] = (raw[1]/4 + extra @ CONTRASTS).tolist()
    elif pre.experiment == "E11":
        result["implied_rank_slopes_location_energy"] = [float(raw[2]/2-extra[0]/np.sqrt(2)), float(raw[2]/2+extra[0]/np.sqrt(2))]
    elif pre.experiment == "E12":
        result["implied_raw_block_slopes_by_age"] = {str(t):
            np.divide(beta[1:4]+extra*(np.log1p(t)-pre.center), pre.std[1:4],
                      out=np.zeros(3), where=pre.std[1:4] > 0).tolist() for t in (32,128,256,512)}
    return result


def fit_fold(frame, heldout, experiment):
    start = time.perf_counter()
    # Split BEFORE any feature, context, class-weight, or bin computation.
    train = frame.loc[frame.fold != heldout]
    with threadpool_limits(limits=1):
        threads = threadpool_info()
        eligible, omega, audit = training_weights(train.time_online.to_numpy(), train.target.to_numpy())
        selected = train.loc[eligible]
        pre, x = Preprocessing.fit(experiment, selected, omega)
        y = selected.target.to_numpy(dtype=np.float64)
        audit["eligible_mask_sha256"] = hashlib.sha256(eligible.tobytes()).hexdigest()
        if experiment == "E13":
            if sklearn.__version__ != "1.8.0":
                raise ValueError("E13 requires pinned scikit-learn 1.8.0")
            tree = HistGradientBoostingClassifier(**TREE_PARAMS)
            tree.fit(x, y, sample_weight=omega)
            model = Model(pre, tree=tree)
            prob = tree.predict_proba(x)[:, 1]
            eps = np.finfo(float).eps
            loss = float(np.dot(omega, -(y*np.log(np.clip(prob,eps,1-eps))+(1-y)*np.log(np.clip(1-prob,eps,1-eps)))))
            detail = dict(n_iter=int(tree.n_iter_), parameters=tree.get_params(),
                tree_depths=[int(t[0].nodes["depth"].max()) for t in tree._predictors],
                leaf_counts=[int(t[0].nodes["is_leaf"].sum()) for t in tree._predictors],
                weighted_training_log_loss=loss, sample_weight_sum=float(omega.sum()),
                sample_weight_sha256=hashlib.sha256(omega.tobytes()).hexdigest(),
                internal_precision="pinned native float32 gradient/Hessian arrays; float64 supplied X/weights")
            if max(detail["tree_depths"]) > 2 or max(detail["leaf_counts"]) > 4:
                raise ValueError("Native model violates frozen tree bounds")
        else:
            result = minimize(objective, np.zeros(x.shape[1]+1), args=(x,y,omega),
                              method="L-BFGS-B", jac=True, bounds=None, options=SOLVER_OPTIONS)
            value, gradient = objective(result.x, x, y, omega)
            detail = dict(iterations=int(result.nit), function_evaluations=int(result.nfev),
                objective=value, gradient_infinity_norm=float(np.max(np.abs(gradient))),
                final_gradient=gradient.tolist(), optimizer_success=bool(result.success),
                optimizer_message=str(result.message), options=SOLVER_OPTIONS)
            if (not result.success or not np.isfinite(result.x).all() or not np.isfinite(value)
                    or not np.isfinite(gradient).all() or detail["gradient_infinity_norm"] > 1e-6):
                raise InvalidFit(detail)
            model = Model(pre, result.x[0], result.x[1:])
            detail.update(coefficient_diagnostics(model))
    detail.update(fold=int(heldout), preprocessing=pre.as_dict(), native_threads=threads,
        zero_variance_base=(pre.std==0).tolist(), zero_variance_extension=(pre.extra_std==0).tolist(),
        training_seconds=time.perf_counter()-start, serialized_model_bytes=len(model.dumps()))
    return model, detail, audit

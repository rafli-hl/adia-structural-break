"""Official pair-weighted metric; labels are evaluator-only."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def age_row(t, labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("Invalid label/score shape")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Nonbinary label")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Score must be finite and in [0,1]")
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    weight = n_pos * n_neg
    auc = float(roc_auc_score(labels, scores)) if weight else np.nan
    return dict(time_online=int(t), age=int(t) + 1, active=len(labels),
                positive=n_pos, negative=n_neg, weight=weight, auc=auc)


def aggregate(rows):
    # Same operation order as the supplied official reference.
    weighted_auc_sum = 0.0
    total_weight = 0.0
    for r in rows:
        if r["weight"]:
            weighted_auc_sum += r["weight"] * r["auc"]
            total_weight += r["weight"]
    for r in rows:
        r["weight_fraction"] = r["weight"] / total_weight if total_weight else 0.0
    return weighted_auc_sum / total_weight if total_weight else 0.5, pd.DataFrame(rows)


def ts_auc(df, expected_keys=None):
    required = {"id", "time_online", "target", "prediction"}
    if not required <= set(df.columns):
        raise ValueError("Missing scorer fields")
    keys = df[["id", "time_online"]]
    if keys.isna().any().any() or keys.duplicated().any():
        raise ValueError("Missing or duplicate prediction key")
    ages = df.time_online.to_numpy(dtype=np.float64)
    if not np.isfinite(ages).all() or (ages < 0).any() or (ages != np.floor(ages)).any():
        raise ValueError("time_online must be a nonnegative integer")
    if expected_keys is not None:
        expected = pd.MultiIndex.from_frame(expected_keys[["id", "time_online"]])
        actual = pd.MultiIndex.from_frame(keys)
        if not expected.is_unique or len(actual) != len(expected) or not actual.isin(expected).all():
            raise ValueError("Prediction coverage mismatch")
    return aggregate([age_row(t, g.target, g.prediction)
                      for t, g in df.groupby("time_online", sort=True)])


BUCKETS = [(1, 8), (9, 16), (17, 32), (33, 64), (65, 128),
           (129, 256), (257, 512), (513, None)]


def bucket_summary(per_age):
    total = float(per_age.weight.sum()) if len(per_age) else 0.0
    rows = []
    for lo, hi in BUCKETS:
        g = per_age[(per_age.age >= lo) & ((per_age.age <= hi) if hi else True)]
        weight = float(g.weight.sum())
        score = float((g.auc.fillna(0) * g.weight).sum() / weight) if weight else None
        rows.append(dict(age_range=f"{lo}-{hi}" if hi else f"{lo}+",
                         pair_weight=weight, weight_fraction=weight / total if total else 0,
                         ts_auc=score, eligible_ages=int((g.weight > 0).sum())))
    return pd.DataFrame(rows)

"""Only the preregistered weighting contract; no E08 model or training run."""
import numpy as np


def metric_training_weights(frame, heldout_fold):
    train = frame[frame.fold != heldout_fold]
    counts = train.groupby(["time_online", "target"]).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
    pairs = counts[0] * counts[1]
    z = float(pairs.sum())
    if z <= 0:
        raise ValueError("No eligible training ages")
    eligible = train[train.time_online.isin(pairs[pairs > 0].index)]
    weights = np.asarray([float(pairs.loc[t]) / (2*z*int(counts.loc[t, y]))
                          for t, y in zip(eligible.time_online, eligible.target)], dtype=np.float64)
    return eligible.index.to_numpy(), weights

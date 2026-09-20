"""Immutable, separately versioned CV manifests; evaluator-only metadata."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from .data import sha256
from .profile import Union, write_json


def fixed_folds(cache, seed=20260918, groups_path=None):
    cache = Path(cache)
    frame = pd.read_parquet(cache / "series_manifest.parquet").sort_values("id").reset_index(drop=True)
    if groups_path:
        external = pd.read_csv(groups_path, dtype=str)
        if not {"id", "group"} <= set(external) or external.id.duplicated().any() or external[["id", "group"]].isna().any().any():
            raise ValueError("Group file must contain unique id,group rows")
        if set(external.id) != set(frame.id):
            raise ValueError("External group file must cover all IDs exactly")
        # Union external groups with verified duplicate groups, never split an
        # existing component by supplying a new group file.
        u = Union()
        for table in (frame[["id", "group"]], external):
            for _, g in table.groupby("group", sort=True):
                first = g.id.iloc[0]
                for sid in g.id:
                    u.join(first, sid)
        frame["group"] = [u.root(sid) for sid in frame.id]
    if frame.group.nunique() < 5 or frame.has_break.value_counts().min() < 5 or frame.has_break.nunique() != 2:
        raise ValueError("Need >=5 groups and >=5 series of each class for initial five-fold evaluation")
    grouped = bool(frame.group.duplicated().any())
    splitter = (StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed) if grouped
                else StratifiedKFold(n_splits=5, shuffle=True, random_state=seed))
    frame["fold"] = -1
    splits = splitter.split(frame.id, frame.has_break, frame.group) if grouped else splitter.split(frame.id, frame.has_break)
    for fold, (train, valid) in enumerate(splits):
        if set(frame.id.iloc[train]) & set(frame.id.iloc[valid]):
            raise AssertionError("Series leakage")
        if set(frame.group.iloc[train]) & set(frame.group.iloc[valid]):
            raise AssertionError("Group leakage")
        frame.loc[valid, "fold"] = fold
    if (frame.fold < 0).any():
        raise AssertionError("Incomplete folds")
    path = cache / "folds.parquet"
    settings = dict(seed=seed, n_splits=5, grouped=grouped,
                    groups_sha256=sha256(groups_path) if groups_path else None,
                    sklearn=sklearn.__version__)
    if path.exists():
        old = pd.read_parquet(path)
        if not old.equals(frame) or json.loads((cache / "fold_settings.json").read_text()) != settings:
            raise ValueError("Existing fixed folds differ; use a new cache for a new split experiment")
    else:
        frame.to_parquet(path, index=False)
        write_json(cache / "fold_settings.json", settings)
    return frame, settings


STRATA = ("no_break", "break_1_64", "break_65_128", "break_129_256",
          "break_257_512", "break_513_plus")
CV2_SEED = 20260919


def stratify(frame):
    tau = frame.tau_index.to_numpy()
    if not np.isfinite(tau).all() or (tau < -1).any() or (tau != np.floor(tau)).any():
        raise ValueError("Invalid tau_index")
    if not np.array_equal(frame.has_break.to_numpy(), (tau >= 0).astype(int)):
        raise ValueError("Inconsistent series class")
    if ((tau >= 0) & (tau >= frame.online_length.to_numpy())).any():
        raise ValueError("Break outside online stream")
    age = tau + 1
    return np.select([tau == -1, age <= 64, age <= 128, age <= 256, age <= 512],
                     list(STRATA[:-1]), default=STRATA[-1])


def expected_v2(cache, seed=CV2_SEED):
    if seed != CV2_SEED:
        raise ValueError("CV-v2 seed is frozen at 20260919")
    cache = Path(cache)
    status = json.loads((cache / "profile_status.json").read_text())
    if status["label_alignment"] != "pass" or status.get("unresolved_repeated_block_pairs", 0):
        raise ValueError("Profile audits must pass before ungrouped CV-v2")
    frame = pd.read_parquet(cache / "series_manifest.parquet").sort_values("id").reset_index(drop=True)
    if frame.id.isna().any() or frame.id.duplicated().any() or not frame.id.map(lambda v: isinstance(v, str)).all():
        raise ValueError("Manifest IDs must already be unique normalized strings")
    if frame.group.duplicated().any():
        raise ValueError("CV-v2 ungrouped specification incompatible with verified groups")
    frame["stratum"] = stratify(frame)
    counts = frame.stratum.value_counts()
    if any(counts.get(s, 0) < 5 for s in STRATA):
        raise ValueError("Need at least five series in every frozen CV-v2 stratum")
    frame["fold"] = -1
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train, valid) in enumerate(splitter.split(frame.id, frame.stratum)):
        if set(frame.id.iloc[train]) & set(frame.id.iloc[valid]):
            raise AssertionError("Series leakage")
        frame.loc[valid, "fold"] = fold
    table = pd.crosstab(frame.stratum, frame.fold).reindex(STRATA)
    if not (table.max(axis=1) - table.min(axis=1) <= 1).all():
        raise AssertionError("Unbalanced stratum allocation")
    metadata = dict(version="cv2", seed=seed, n_splits=5, grouped=False,
                    sklearn=sklearn.__version__, splitter="StratifiedKFold",
                    id_order="lexicographic normalized strings", break_age="tau_index + 1",
                    strata=list(STRATA), series_manifest_sha256=sha256(cache / "series_manifest.parquet"),
                    sources_sha256=sha256(cache / "sources.json"),
                    profile_status_sha256=sha256(cache / "profile_status.json"),
                    stratum_counts={s: int(counts[s]) for s in STRATA},
                    stratum_counts_by_fold={s: [int(v) for v in table.loc[s]] for s in STRATA})
    return frame, metadata


def create_v2(cache, out, seed=CV2_SEED):
    frame, metadata = expected_v2(cache, seed)
    out = Path(out)
    path, settings = out / "folds.parquet", out / "fold_settings.json"
    if path.exists() or settings.exists():
        return load_v2(cache, path)
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "folds.partial.parquet"
    if partial.exists():
        raise ValueError("Incomplete CV-v2 manifest exists; inspect before retry")
    frame.to_parquet(partial, index=False)
    partial.replace(path)
    metadata["fold_manifest_sha256"] = sha256(path)
    write_json(settings, metadata)
    return frame, metadata


def load_v2(cache, path):
    path = Path(path)
    expected, settings = expected_v2(cache)
    if not path.exists() or not path.with_name("fold_settings.json").exists():
        raise ValueError("Missing immutable CV-v2 manifest/settings")
    metadata = json.loads(path.with_name("fold_settings.json").read_text())
    settings["fold_manifest_sha256"] = sha256(path)
    actual = pd.read_parquet(path)
    if metadata != settings or not actual.equals(expected):
        raise ValueError("Incompatible or modified CV-v2 manifest/settings; refusing overwrite")
    return actual, metadata

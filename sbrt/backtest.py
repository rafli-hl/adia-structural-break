"""Five fixed folds, guarded online replay, complete disk-backed OOF outputs."""
from pathlib import Path
import json
import platform
try:
    import resource
except ImportError:  # Windows
    resource = None
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from .data import iter_series, iter_frames, sha256
from .detectors import E01, make_detector
from .metric import age_row, aggregate, bucket_summary, ts_auc
from .profile import write_json, Union
from .replay import replay


def process_peak_rss():
    """Return process peak RSS where the stdlib resource module exists.

    Windows does not provide ``resource``. Peak-memory reporting is diagnostic
    only, so return None there instead of blocking the backtest.
    """
    if resource is None:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss



def e00():
    # Multiple unequal-age pair counts and ineligible ages.
    labels = [[0, 0, 1, 1], [0, 1, 1], [0, 0], [1, 1, 1, 1], [0]]
    base = pd.DataFrame([dict(id=str(i), time_online=t, target=y)
                         for i, ys in enumerate(labels) for t, y in enumerate(ys)])
    results = {}
    for name, values in (("constant", np.full(len(base), .5)),
                         ("age_only", (base.time_online + 1) / (base.time_online + 2)),
                         ("perfect_oracle", base.target),
                         ("reversed_oracle", 1 - base.target)):
        results[name] = ts_auc(base.assign(prediction=values))[0]
    expected = dict(constant=.5, age_only=.5, perfect_oracle=1., reversed_oracle=0.)
    for key in expected:
        if results[key] != expected[key]:
            raise AssertionError((key, results[key]))
    return dict(scope="synthetic metric controls only", scores=results,
                no_eligible_age_fallback=ts_auc(base.assign(target=0, prediction=.5))[0])


from .folds import fixed_folds


def score_file(con, path, expected_rows):
    con.read_parquet(str(path)).create_view("scored", replace=True)
    if con.execute("SELECT count(*) FROM scored").fetchone()[0] != expected_rows:
        raise ValueError("OOF row count mismatch")
    if con.execute("SELECT count(*) FROM (SELECT id,time_online FROM scored GROUP BY id,time_online HAVING count(*)>1)").fetchone()[0]:
        raise ValueError("Duplicate OOF keys")
    rows, fold_rows = [], [[] for _ in range(5)]
    for g in iter_frames(con, "SELECT * FROM scored ORDER BY time_online,id", key="time_online"):
        t = int(g.time_online.iloc[0])
        rows.append(age_row(t, g.target, g.prediction))
        for fold, f in g.groupby("fold", sort=True):
            fold_rows[int(fold)].append(age_row(t, f.target, f.prediction))
    score, per_age = aggregate(rows)
    fold_results = [aggregate(r) for r in fold_rows]
    return score, per_age, fold_results


def run_backtest(con, cache, out, detectors=E01, seed=20260918, groups_path=None):
    out, cache = Path(out), Path(cache)
    out.mkdir(parents=True, exist_ok=True)
    status = json.loads((cache / "profile_status.json").read_text())
    if status["label_alignment"] != "pass":
        raise ValueError("Label audit requires review before E01")
    if status["counts"].get("nonfinite_observations", 0) or status["counts"].get("empty_online", 0):
        raise ValueError("Nonfinite observations or empty online segments need an explicit data policy")
    if status["unresolved_repeated_block_pairs"] and groups_path is None:
        raise ValueError("Repeated segments cross groups. Review local cache/related_candidates.json and supply a complete --groups CSV")
    manifest, fold_settings = fixed_folds(cache, seed, groups_path)
    if status["unresolved_repeated_block_pairs"]:
        groups = dict(zip(manifest.id, manifest.group))
        candidates = json.loads((cache / "related_candidates.json").read_text())
        if any(groups[a] != groups[b] for a, b in candidates["repeated_block_pairs"]):
            raise ValueError("Supplied groups leave repeated-segment candidates across folds/components")
    fold_lookup = dict(zip(manifest.id, manifest.fold))
    manifest.to_parquet(out / "fold_manifest.parquet", index=False)
    write_json(out / "e00.json", e00())
    records = []
    expected_rows = int(manifest.online_length.sum())
    for name in detectors:
        if name not in E01 and name not in ("constant", "age_only"):
            raise ValueError(f"Unknown detector: {name}")
        output_file = out / f"oof_{name}.parquet"
        if output_file.exists():
            raise ValueError(f"Refusing to overwrite completed OOF artifact: {output_file}")
        temporary = out / f"oof_{name}.partial.parquet"
        start = time.perf_counter()
        initialization = updates = 0.0
        seen, prediction_count, timed_updates, ties, saturated = set(), 0, 0, 0, 0
        buffers, writer = [], None
        try:
            for s, scores, timing in replay(iter_series(con), detector=name):
                if s.id in seen or s.id not in fold_lookup:
                    raise AssertionError("Unexpected replay series")
                seen.add(s.id)
                if len(scores) != len(s.target):
                    raise AssertionError("Incomplete score coverage")
                n = len(scores)
                initialization += timing["initialization_seconds"]
                updates += timing["update_seconds"]
                timed_updates += max(0, n - 1)
                prediction_count += n
                ties += int(np.sum(scores[1:] == scores[:-1]))
                saturated += int(np.sum(scores == 1))
                buffers.append(pd.DataFrame(dict(id=s.id, time=s.time,
                    time_online=np.arange(n, dtype=np.int64), target=s.target,
                    prediction=scores, fold=int(fold_lookup[s.id]))))
                if len(buffers) >= 32:
                    table = pa.Table.from_pandas(pd.concat(buffers, ignore_index=True), preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                    writer.write_table(table)
                    buffers = []
            if buffers:
                table = pa.Table.from_pandas(pd.concat(buffers, ignore_index=True), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        if seen != set(fold_lookup) or prediction_count != expected_rows:
            raise AssertionError("Replay omitted series or observations")
        temporary.replace(output_file)
        replay_wall = time.perf_counter() - start
        score, per_age, fold_results = score_file(con, output_file, expected_rows)
        per_age.to_csv(out / f"per_age_{name}.csv", index=False)
        bucket_summary(per_age).to_csv(out / f"age_buckets_{name}.csv", index=False)
        for fold, (_, age) in enumerate(fold_results):
            age.to_csv(out / f"per_age_{name}_fold{fold}.csv", index=False)
        state = make_detector(name, np.arange(100, dtype=float))
        state_bytes = sys.getsizeof(state) + sys.getsizeof(state.__dict__) + sum(sys.getsizeof(v) for v in state.__dict__.values())
        row = dict(experiment="E01" if name in E01 else "E00", detector=name,
                   pooled_oof_ts_auc=score, fold_scores=[float(v[0]) for v in fold_results],
                   fold_eligible_ages=[int((v[1].weight > 0).sum()) for v in fold_results],
                   scored_points=prediction_count, initialization_seconds=initialization,
                   update_seconds_excluding_first_point=updates,
                   timed_updates=timed_updates,
                   updates_per_second=timed_updates / updates if updates else None,
                   replay_wall_seconds=replay_wall, total_wall_seconds=time.perf_counter() - start,
                   state_bytes_shallow=state_bytes,
                   process_peak_rss_platform_units=process_peak_rss(),
                   adjacent_score_ties=ties, scores_equal_one=saturated,
                   update_complexity="O(1)", state_complexity="O(1)", initialization_complexity="O(H)",
                   notes="No global fit; fold-held-out trajectories use each series' history only. Update timing includes generator/guard/clock overhead and excludes the first point per series.")
        records.append(row)
        write_json(out / "e01_results.json", records)
        flat = [{**r, "fold_scores": json.dumps(r["fold_scores"]),
                 "fold_eligible_ages": json.dumps(r["fold_eligible_ages"])} for r in records]
        pd.DataFrame(flat).to_csv(out / "experiment_log.csv", index=False)
        print(f"{name}: TS-AUC={score:.8f}; replay={replay_wall:.3f}s", flush=True)
    write_json(out / "run_metadata.json", dict(
        fold_settings=fold_settings, profile_status=status,
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, pyarrow=pa.__version__),
        official_ewma=dict(alpha=.05, kappa=3., scale_floor=1e-8, ddof=1),
        statistical_detectors=dict(cusum_k=.5, page_hinkley_delta=.05, normalization_scale_floor=1e-8,
                                   variance_log_floor=1e-12, evidence_mapping="q/(1+q)"),
        limitations=["No uncertainty interval yet; fold scores are diagnostics, not five independent repetitions.",
                     "Peak RSS is cumulative for this process; run detectors separately for isolated peaks.",
                     "Labels and final lengths appear only in evaluator outputs, never detector inputs."] ))
    return records

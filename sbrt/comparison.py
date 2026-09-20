"""Preregistered paired comparisons; no score-dependent model changes."""
from pathlib import Path
import time
import numpy as np
from .data import connect, iter_frames
from .folds import STRATA
from .provenance import write_json, finalize_artifacts, verify_artifacts, canonical_hash
from .stage2 import context, difference, load_result
from .metric import age_row, aggregate


def gate_conditions(delta, gate):
    return dict(pooled_delta=delta["pooled"] >= gate["minimum_pooled_delta"],
                positive_folds=sum(v > 0 for v in delta["folds"]) >= gate["minimum_positive_folds"],
                age_ge129=delta["combined"]["ge129"] is not None and
                          delta["combined"]["ge129"] >= gate["minimum_age_ge129_delta"],
                age_le64=delta["combined"]["le64"] is not None and
                         delta["combined"]["le64"] >= gate["minimum_age_le64_delta"])


def sorted_age(series, labels, scores):
    order = np.argsort(scores, kind="stable")
    values = np.asarray(scores)[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    return np.asarray(series, dtype=np.int32)[order], np.asarray(labels, dtype=np.int8)[order], starts


def weighted_age_terms(prepared, multiplicity):
    """Exact weighted tie groups; columns are bootstrap replicates."""
    series, labels, starts = prepared
    w = np.asarray(multiplicity[series], dtype=np.float64)
    if w.ndim == 1:
        w = w[:, None]
    positives = np.add.reduceat(w * labels[:, None], starts, axis=0)
    negatives = np.add.reduceat(w * (1 - labels[:, None]), starts, axis=0)
    before = np.cumsum(negatives, axis=0) - negatives
    numerator = np.sum(positives * (before + .5 * negatives), axis=0)
    denominator = positives.sum(axis=0) * negatives.sum(axis=0)
    return numerator, denominator


def bootstrap_multiplicities(manifest, replicates, seed):
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(manifest.stratum.to_numpy() == s) for s in STRATA]
    values = np.zeros((len(manifest), replicates), dtype=np.int32)
    for b in range(replicates):
        for indices in strata:
            sampled = rng.choice(indices, size=len(indices), replace=True)
            values[:, b] += np.bincount(sampled, minlength=len(manifest)).astype(np.int32)
    return values


def prepare_pair(con, candidate_path, parent_path, manifest):
    con.read_parquet(str(candidate_path)).create_view("boot_candidate", replace=True)
    con.read_parquet(str(parent_path)).create_view("boot_parent", replace=True)
    numbered = manifest[["id"]].copy()
    numbered["series_number"] = np.arange(len(numbered), dtype=np.int32)
    con.register("boot_series", numbered)
    # iter_frames uses a separate DuckDB cursor. Registered Python relations
    # are connection-local, so materialize this evaluator-only lookup first.
    con.execute("CREATE OR REPLACE TABLE stage2_boot_series AS SELECT * FROM boot_series")
    prepared = []
    query = """SELECT a.time_online, a.target, m.series_number,
        a.prediction AS candidate, b.prediction AS parent FROM boot_candidate a
        JOIN boot_parent b USING(id,time_online) JOIN stage2_boot_series m ON a.id=m.id
        ORDER BY a.time_online,a.id"""
    for g in iter_frames(con, query, key="time_online"):
        prepared.append((sorted_age(g.series_number, g.target, g.candidate),
                         sorted_age(g.series_number, g.target, g.parent)))
    con.execute("DROP TABLE stage2_boot_series")
    return prepared


def paired_bootstrap(con, candidate_path, parent_path, manifest, gate):
    start = time.perf_counter()
    prepared = prepare_pair(con, candidate_path, parent_path, manifest)
    multiplicity = bootstrap_multiplicities(manifest, gate["bootstrap_replicates"], gate["bootstrap_seed"])
    deltas = []
    # Batching bounds temporary allocations without changing resampling.
    for begin in range(0, multiplicity.shape[1], 16):
        block = multiplicity[:, begin:begin + 16]
        a_total = np.zeros(block.shape[1])
        b_total = np.zeros(block.shape[1])
        weights = np.zeros(block.shape[1])
        for a, b in prepared:
            av, aw = weighted_age_terms(a, block)
            bv, bw = weighted_age_terms(b, block)
            if not np.array_equal(aw, bw):
                raise AssertionError("Bootstrap pair weights differ")
            a_total += av
            b_total += bv
            weights += aw
        delta = np.divide(a_total - b_total, weights, out=np.zeros_like(weights), where=weights > 0)
        deltas.extend(delta.tolist())
        if begin % 160 == 0:
            print(f"bootstrap {candidate_path.parent.name} vs {parent_path.parent.name}: {min(begin+16, multiplicity.shape[1])}/{multiplicity.shape[1]}", flush=True)
    interval = np.quantile(deltas, gate["bootstrap_interval"]).tolist()
    return dict(interval_95=interval, seed=gate["bootstrap_seed"], replicates=len(deltas),
                delta_replicates=deltas, seconds=time.perf_counter() - start,
                multiplicities_sha256=__import__("hashlib").sha256(multiplicity.tobytes()).hexdigest(),
                scope=gate["bootstrap_scope"], refitting=False,
                selection_adjusted=False)


def side_diagnostics(con, root_out, destination):
    import json
    import pandas as pd
    diagnostics = json.loads((root_out / "E04" / "series_diagnostics.json").read_text())
    ratios = pd.DataFrame([dict(id=r["id"], mse_ratio=r["ar_mse_ratio"]) for r in diagnostics])
    con.register("historical_ratios", ratios)
    con.execute("CREATE OR REPLACE TABLE stage2_historical_ratios AS SELECT * FROM historical_ratios")
    con.read_parquet(str(root_out / "E03" / "oof.parquet")).create_view("diagnostic_raw", replace=True)
    con.read_parquet(str(root_out / "E04" / "oof.parquet")).create_view("diagnostic_ar", replace=True)
    output = {}
    for name, condition in (("lt_0.9", "m.mse_ratio < 0.9"),
                            ("0.9_to_1.0", "m.mse_ratio >= 0.9 AND m.mse_ratio <= 1.0"),
                            ("gt_1.0", "m.mse_ratio > 1.0")):
        rows = {"E03": [], "E04": []}
        query = f"""SELECT a.time_online,a.target,a.prediction AS raw,b.prediction AS innovation
            FROM diagnostic_raw a JOIN diagnostic_ar b USING(id,time_online)
            JOIN stage2_historical_ratios m ON a.id=m.id WHERE {condition} ORDER BY a.time_online,a.id"""
        for g in iter_frames(con, query, key="time_online"):
            rows["E03"].append(age_row(int(g.time_online.iloc[0]),g.target,g.raw))
            rows["E04"].append(age_row(int(g.time_online.iloc[0]),g.target,g.innovation))
        entry = {}
        for experiment, values in rows.items():
            score, ages = aggregate(values)
            ages.to_csv(destination / f"ar_mse_{name}_{experiment}_per_age.csv",index=False)
            entry[experiment] = dict(ts_auc=score,eligible_ages=int((ages.weight>0).sum()) if len(ages) else 0,
                                     pair_weight=int(ages.weight.sum()) if len(ages) else 0)
        entry["series"] = int(con.execute(f"SELECT count(*) FROM stage2_historical_ratios m WHERE {condition}").fetchone()[0])
        entry["delta_E04_minus_E03"] = entry["E04"]["ts_auc"]-entry["E03"]["ts_auc"]
        output[name] = entry
    con.execute("DROP TABLE stage2_historical_ratios")
    con.read_parquet(str(root_out / "E06" / "oof.parquet")).create_view("rank_diagnostics",replace=True)
    rows = {name: [] for name in ("P", "P_location", "P_energy")}
    for g in iter_frames(con,"SELECT time_online,target,P,P_location,P_energy FROM rank_diagnostics ORDER BY time_online,id",key="time_online"):
        for name in rows:
            rows[name].append(age_row(int(g.time_online.iloc[0]),g.target,g[name]))
    rank = {}
    from .stage2 import summarize_ages
    for name, values in rows.items():
        score, ages = aggregate(values)
        buckets, combined = summarize_ages(ages)
        rank[name] = dict(pooled_ts_auc=score,age_buckets=buckets,combined=combined)
        ages.to_csv(destination/f"{name}_per_age.csv",index=False)
    result = dict(historical_ar_mse_subsets=output,rank_subblocks=rank,
                  note="Prespecified descriptive diagnostics only; subsets do not decompose pooled TS-AUC and do not select channels")
    write_json(destination/"side_diagnostics.json",result)
    return result


def run_comparisons(cache, folds, spec_path, root_out, memory="1GB"):
    import json
    spec, manifest, settings, source, ctx = context(cache, folds, spec_path)
    root_out = Path(root_out)
    destination = root_out / "comparisons"
    ids = ["E01_squared"] + spec["phase1_ids"]
    results = {name: load_result(root_out / name, ctx) for name in ids}
    fingerprint = {name: r["prediction_artifact_sha256"] for name, r in results.items()}
    if destination.exists():
        verify_artifacts(destination)
        summary = json.loads((destination / "phase1_report.json").read_text())
        if summary["prediction_hashes"] != fingerprint or summary["provenance"] != ctx:
            raise ValueError("Incompatible completed comparisons")
        return summary
    destination.mkdir()
    con = connect(cache, memory)
    comparisons, experiments = {}, {}
    gate = spec["gate"]

    def compare(a, b):
        key = f"{a}_vs_{b}"
        if key not in comparisons:
            delta = difference(results[a], results[b])
            conditions = gate_conditions(delta, gate)
            boot = None
            if all(conditions.values()):
                boot = paired_bootstrap(con, root_out / a / "oof.parquet", root_out / b / "oof.parquet", manifest, gate)
            comparisons[key] = dict(candidate=a, comparator=b, delta=delta, conditions_1_to_4=conditions,
                                    bootstrap=boot, bootstrap_required=all(conditions.values()),
                                    bootstrap_lower_positive=bool(boot and boot["interval_95"][0] > 0),
                                    passes=bool(all(conditions.values()) and boot and boot["interval_95"][0] > 0))
            write_json(destination / f"{key}.json", comparisons[key])
        return comparisons[key]

    champion = "E01_squared"
    try:
        for name in spec["phase1_ids"]:
            config = spec["experiments"][name]
            primary = compare(name, config["parent"])
            mandatory = {p: compare(name, p) for p in config.get("mandatory_comparisons", [])}
            previous_champion = champion
            global_comparison = compare(name, previous_champion)
            early = difference(results[name], results["E01_squared"])["combined"]["le64"]
            early_ok = early is not None and early >= gate["minimum_age_le64_vs_e01_delta"]
            local_keep = primary["passes"] and all(v["passes"] for v in mandatory.values())
            promote = local_keep and global_comparison["passes"] and early_ok
            if promote:
                champion = name
            extras = {p: difference(results[name], results[p]) for p in config.get("additional_comparisons", [])}
            pair_keys = {f"{name}_vs_{p}" for p in [config["parent"], previous_champion] + config.get("mandatory_comparisons", [])}
            bootstrap_seconds = sum(comparisons[k]["bootstrap"]["seconds"] for k in pair_keys if comparisons[k]["bootstrap"])
            entry = dict(**results[name])
            entry.update(advancement_gate_results=dict(parent=primary, mandatory=mandatory,
                         previous_reference=previous_champion, reference_comparison=global_comparison,
                         early_vs_e01_delta=early, early_vs_e01_passes=early_ok,
                         local_keep=local_keep, promoted=promote,
                         control_artifact_retained=bool(config.get("control_artifact")),
                         decision="advance" if promote else "control retained; no advancement" if name == "E03" else "reject advancement; valid experiment"),
                         additional_comparisons=extras,
                         conditional_bootstrap_delta_interval=primary["bootstrap"]["interval_95"] if primary["bootstrap"] else None,
                         bootstrap_seconds=bootstrap_seconds,
                         total_wall_seconds=results[name]["total_wall_seconds"]+bootstrap_seconds)
            experiments[name] = entry
            write_json(destination / f"{name}.json", entry)
            print(f"{name}: valid; parent gate={primary['passes']}; promote={promote}; reference={champion}", flush=True)
        side = side_diagnostics(con,root_out,destination)
        summary = dict(provenance=ctx, prediction_hashes=fingerprint, cv2=settings,
                       experiments=experiments, final_reference=champion,
                       side_diagnostics=side,
                       E08_status="not implemented or executed; frozen specification unchanged",
                       uncertainty="Conditional fixed-OOF series bootstrap; no refitting or selection adjustment")
        write_json(destination / "phase1_report.json", summary)
        finalize_artifacts(destination)
        return summary
    finally:
        con.close()

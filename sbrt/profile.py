"""Offline, aggregate-only diagnostics. Never import this module in inference."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from .data import iter_series, get_history, get_values
from .metric import BUCKETS, bucket_summary

QUANTILES = [0, .01, .05, .25, .5, .75, .95, .99, 1]
QNAMES = ["min", "p1", "p5", "p25", "median", "p75", "p95", "p99", "max"]
LAGS = (1, 2, 5, 10, 20)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + "\n")


def summarize(values):
    v = np.asarray(values, dtype=float)
    x = v[np.isfinite(v)]
    result = dict(count=len(v), valid=len(x), nonfinite=int(len(v) - len(x)))
    result.update(dict(zip(QNAMES, np.quantile(x, QUANTILES).tolist())) if len(x)
                  else {k: None for k in QNAMES})
    result["mean"] = float(x.mean()) if len(x) else None
    return result


def distribution_table(records, expected_count=None):
    keys = sorted({k for r in records for k in r})
    rows = []
    for k in keys:
        values = [r.get(k, np.nan) for r in records]
        row = dict(statistic=k, **summarize(values))
        row["eligible_population"] = expected_count if expected_count is not None else len(records)
        rows.append(row)
    return pd.DataFrame(rows)


def corr(x, lag):
    if len(x) <= lag:
        return np.nan
    a, b = x[:-lag], x[lag:]
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 4:
        return np.nan
    a, b = a[mask], b[mask]
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else np.nan


def stats(x):
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    f = x[finite]
    result = dict(length=len(x), nonfinite=int((~finite).sum()))
    if not len(f):
        return result
    mu, med = float(f.mean()), float(np.median(f))
    sd = float(f.std(ddof=1)) if len(f) > 1 else np.nan
    mad = float(np.median(np.abs(f - med)))
    result.update(mean=mu, std=sd, median=med, mad=mad,
                  constant=float(np.ptp(f) == 0), zero_mad=float(mad == 0))
    for q, value in zip((1, 5, 25, 50, 75, 95, 99), np.quantile(f, [.01, .05, .25, .5, .75, .95, .99])):
        result[f"q{q:02d}"] = float(value)
    pop_sd = float(f.std())
    if pop_sd > 0:
        z = (f - mu) / pop_sd
        if len(f) >= 8:
            result.update(skewness=float(np.mean(z ** 3)), excess_kurtosis=float(np.mean(z ** 4) - 3))
    if mad > 0:
        result["extreme_fraction_robust_5sigma"] = float(np.mean(np.abs(f - med) > 5 * 1.4826 * mad))
    if len(f) >= 2:
        positions = np.flatnonzero(finite).astype(float)
        positions -= positions.mean()
        slope = float(positions @ (f - mu) / (positions @ positions))
        result["slope_per_tick"] = slope
        if sd > 0:
            result["slope_times_length_over_sd"] = slope * len(x) / sd
        d = np.diff(x)
        d = d[np.isfinite(d)]
        result["difference_variance"] = float(d.var(ddof=1)) if len(d) > 1 else np.nan
    for lag in LAGS:
        for name, v in (("acf", x), ("abs_acf", np.abs(x)), ("squared_acf", x * x),
                        ("centered_abs_acf", np.abs(x - med)), ("centered_squared_acf", (x - mu) ** 2)):
            result[f"{name}_{lag}"] = corr(v, lag)
    return result


from .history import fit_ar, residuals


def whitening(h):
    result = {}
    cut = int(.7 * len(h))
    for p in (1, 5):
        model = fit_ar(h[:cut], p)
        if model is None or len(h) - cut < 16:
            continue
        mu, sd, beta = model
        e = residuals(h, model)[cut:]
        baseline = (h[cut:] - mu) / sd
        if not np.isfinite(e).all() or np.var(baseline) <= 0:
            continue
        prefix = f"ar{p}_"
        result[prefix + "heldout_variance_ratio"] = float(np.var(e) / np.var(baseline))
        result[prefix + "heldout_mse_ratio"] = float(np.mean(e * e) / np.mean(baseline * baseline))
        for k, v in stats(e).items():
            if k in ("std", "skewness", "excess_kurtosis") or k.startswith("acf_"):
                result[prefix + "heldout_residual_" + k] = v
        for k, v in stats(baseline).items():
            if k in ("std", "skewness", "excess_kurtosis"):
                result[prefix + "heldout_raw_" + k] = v
        for j, b in enumerate(beta):
            result[prefix + f"coefficient_{j}"] = float(b)
        half = len(h) // 2
        m1, m2 = fit_ar(h[:half], p), fit_ar(h[half:], p)
        if m1 is not None and m2 is not None:
            result[prefix + "half_coefficient_l2_change"] = float(np.linalg.norm(m1[2][1:] - m2[2][1:]))
    return result


def signature(reference, post, thresholds, residual=False):
    a, b = stats(reference), stats(post)
    out = {"reference_length": len(reference), "post_length": len(post)}
    sd, mad = a.get("std", 0), a.get("mad", 0)
    if sd > 0 and "mean" in b:
        out["mean_change_over_reference_sd"] = (b["mean"] - a["mean"]) / sd
        for q in (1, 5, 25, 50, 75, 95, 99):
            out[f"q{q:02d}_change_over_reference_sd"] = (b[f"q{q:02d}"] - a[f"q{q:02d}"]) / sd
    if sd > 0 and b.get("std", 0) > 0:
        out["log_variance_ratio"] = 2 * np.log(b["std"] / sd)
    if mad > 0 and b.get("mad", 0) > 0:
        out["log_mad_ratio"] = np.log(b["mad"] / mad)
    for key in ("skewness", "excess_kurtosis") + tuple(f"acf_{lag}" for lag in LAGS):
        if key in a and key in b:
            out[("residual_" if residual else "") + key + "_change"] = b[key] - a[key]
    if thresholds is not None:
        lo, hi = thresholds
        fa, fb = reference[np.isfinite(reference)], post[np.isfinite(post)]
        if len(fa) and len(fb):
            out["tail_frequency_change"] = float(np.mean((fb < lo) | (fb > hi)) - np.mean((fa < lo) | (fa > hi)))
    return out


def canonical(x):
    a = np.array(x, dtype="<f8", copy=True)
    a[a == 0] = 0
    a[np.isnan(a)] = np.nan
    return a.tobytes()


class Union:
    def __init__(self):
        self.parent = {}

    def root(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.root(self.parent[x])
        return self.parent[x]

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        self.parent[max(a, b)] = min(a, b)


def run_profile(con, source, cache, out, near_cap=2000):
    start = time.perf_counter()
    out, cache = Path(out), Path(cache)
    out.mkdir(parents=True, exist_ok=True)
    historical, white, lengths, timing, manifests = [], [], [], [], []
    signatures = defaultdict(list)
    alignment, counts = Counter(), Counter()
    match_counts = Counter()
    active_diff = np.zeros(2, dtype=np.int64)
    positive = np.zeros(2, dtype=np.int64)
    hist_hashes, full_hashes, block_hashes = {}, {}, {}
    fingerprints, u = [], Union()
    block_matches = 0
    # Repeated segments are candidates, not automatically grouped. Store
    # positions only; confirmation reads the candidate history from cache.
    block_pairs = set()
    for s in iter_series(con):
        h, o, y = s.historical, s.online, s.target
        u.root(s.id)
        n = len(o)
        counts["series"] += 1
        counts["total_observations"] += len(h) + n
        counts["historical_observations"] += len(h)
        counts["online_observations"] += n
        counts["nonfinite_observations"] += int((~np.isfinite(h)).sum() + (~np.isfinite(o)).sum())
        counts["nonmonotone_time_series"] += int(not s.time_monotone)
        counts["empty_historical"] += int(len(h) == 0)
        counts["empty_online"] += int(n == 0)
        counts["no_break"] += int(s.tau_index == -1)
        lengths.append(dict(historical=len(h), online=n, total=len(h) + n))
        historical.append(stats(h))
        white.append(whitening(h))
        if n + 2 > len(active_diff):
            extra = n + 2 - len(active_diff)
            active_diff = np.pad(active_diff, (0, extra))
            positive = np.pad(positive, (0, extra))
        if n:
            active_diff[1] += 1
            active_diff[n + 1] -= 1
            positive[1:n + 1] += y
        cumulative = bool(np.all(np.diff(y.astype(int)) >= 0))
        alignment["noncumulative_series"] += int(not cumulative)
        first = int(np.flatnonzero(y)[0]) if y.any() else None
        if s.tau_index == -1:
            alignment["no_break_nonzero_labels"] += int(y.any())
        else:
            shifts = [shift for shift in (-1, 0, 1)
                      if np.array_equal(y, (np.arange(n) >= s.tau_index + shift).astype(np.int8))]
            alignment["break_without_positive"] += int(first is None)
            alignment["unmatched_break_series"] += int(not shifts)
            alignment["ambiguous_shift_matches"] += int(len(shifts) > 1)
            for shift in shifts:
                match_counts[str(shift)] += 1
            timing.append(dict(tau_index=s.tau_index,
                               tau_index_over_online_length=s.tau_index / n if n else np.nan,
                               first_positive_position=first if first is not None else np.nan,
                               first_positive_age=first + 1 if first is not None else np.nan,
                               position_minus_tau=first - s.tau_index if first is not None else np.nan))
            if first is not None:
                counts[("early", "middle", "late")[min(2, int(3 * (first + 1 - 1e-12) / n))] ] += 1
                for lo, hi in BUCKETS:
                    if first + 1 >= lo and (hi is None or first + 1 <= hi):
                        counts[f"break_age_{lo}_{hi}"] += 1
        if first is not None and cumulative and s.tau_index != -1:
            post = o[first:]
            pre = np.r_[h, o[:first]]
            fh = h[np.isfinite(h)]
            thresholds = tuple(np.quantile(fh, [.01, .99])) if len(fh) else None
            signatures["historical_vs_post"].append(signature(h, post, thresholds))
            window = min(128, len(pre), len(post))
            if window >= 8:
                signatures["matched_pre_vs_post"].append(signature(pre[-window:], post[:window], thresholds))
            model = fit_ar(h[:int(.7 * len(h))], 5)
            if model is not None:
                r = residuals(np.r_[h, o], model)
                ref = r[int(.7 * len(h)):len(h)]
                after = r[len(h) + first:]
                fr = ref[np.isfinite(ref)]
                rt = tuple(np.quantile(fr, [.01, .99])) if len(fr) else None
                signatures["historical_holdout_residual_vs_post"].append(signature(ref, after, rt, residual=True))
        # Exact history checks include degenerate series in counts; only
        # informative histories automatically group, to avoid connecting all
        # independently constant histories. Degenerate matches are reported.
        hh = hashlib.sha256(canonical(h)).hexdigest()
        informative = len(h) >= 16 and np.isfinite(h).all() and np.unique(h).size >= 8
        if hh in hist_hashes:
            other = hist_hashes[hh]
            if np.array_equal(h, get_history(con, other), equal_nan=True):
                counts["duplicate_histories"] += 1
                if informative:
                    u.join(s.id, other)
                else:
                    counts["degenerate_duplicate_histories"] += 1
        else:
            hist_hashes[hh] = s.id
        # Complete training series means chronological values AND periods.
        values = {"value": s.complete_values, "period": s.periods}
        payload = canonical(values["value"]) + np.asarray(values["period"], dtype="<i8").tobytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest in full_hashes:
            other = get_values(con, full_hashes[digest])
            if (np.array_equal(values["value"], other["value"], equal_nan=True)
                    and np.array_equal(values["period"], other["period"])):
                counts["duplicate_complete_series"] += 1
                u.join(s.id, full_hashes[digest])
        else:
            full_hashes[digest] = s.id
        if informative:
            z = (h - h.mean()) / h.std()
            fp = np.array([v.mean() for v in np.array_split(z, 16)])
            fingerprints.append((s.id, len(h), fp))
            # At most 16 nonoverlapping blocks per history; deterministic
            # coverage from first to last eligible aligned block.
            starts = np.arange(0, max(0, len(h) - 127), 128)
            if len(starts) > 16:
                starts = starts[np.linspace(0, len(starts) - 1, 16).astype(int)]
            for pos in starts:
                block = h[pos:pos + 128]
                if np.unique(block).size < 16:
                    continue
                counts["repeated_block_positions_checked"] += 1
                digest = hashlib.sha256(canonical(block)).hexdigest()
                if digest in block_hashes:
                    other_id, other_pos = block_hashes[digest]
                    if other_id != s.id and np.array_equal(block, get_history(con, other_id)[other_pos:other_pos + 128]):
                        block_matches += 1
                        block_pairs.add(tuple(sorted((s.id, other_id))))
                else:
                    block_hashes[digest] = (s.id, int(pos))
        manifests.append(dict(id=s.id, tau_index=s.tau_index, online_length=n,
                              historical_length=len(h), has_break=int(s.tau_index != -1)))
    # Near-duplicate search is historical only; same-length shapes, four nearest
    # compressed fingerprints, bounded full-resolution comparisons.
    by_length = defaultdict(list)
    for entry in fingerprints:
        by_length[entry[1]].append(entry)
    candidates = set()
    for entries in by_length.values():
        if len(entries) < 2:
            continue
        tree = cKDTree(np.stack([e[2] for e in entries]))
        distances, neighbors = tree.query(np.stack([e[2] for e in entries]), k=min(5, len(entries)))
        for i, js in enumerate(neighbors):
            for j in np.atleast_1d(js):
                if i != j:
                    candidates.add(tuple(sorted((entries[i][0], entries[int(j)][0]))))
    near_pairs = []
    checked = 0
    for a, b in sorted(candidates):
        if checked >= near_cap:
            break
        if u.root(a) == u.root(b):
            continue
        x, y = get_history(con, a), get_history(con, b)
        checked += 1
        zx, zy = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
        rms = float(np.sqrt(np.mean((zx - zy) ** 2)))
        if rms <= .01:
            near_pairs.append((a, b, rms))
            # Conservative grouping, not a claim of shared provenance.
            u.join(a, b)
    # Repeated blocks left across different components need explicit review.
    unresolved_blocks = sum(u.root(a) != u.root(b) for a, b in block_pairs)
    manifest = pd.DataFrame(manifests)
    manifest["group"] = [u.root(sid) for sid in manifest.id]
    manifest.to_parquet(cache / "series_manifest.parquet", index=False)
    # IDs remain in internal cache, never in aggregate diagnostics.
    write_json(cache / "related_candidates.json", {"near_pairs": near_pairs,
               "repeated_block_pairs": sorted(block_pairs)})
    active = np.cumsum(active_diff)[1:-1]
    pos = positive[1:-1]
    per_age = pd.DataFrame(dict(time_online=np.arange(len(active)), age=np.arange(1, len(active) + 1),
                               active=active, positive=pos, negative=active - pos,
                               weight=pos * (active - pos), auc=np.nan))
    total_weight = int(per_age.weight.sum())
    per_age["weight_fraction"] = per_age.weight / total_weight if total_weight else 0.0
    per_age.drop(columns="auc").to_csv(out / "age_support.csv", index=False)
    bucket_summary(per_age).drop(columns="ts_auc").to_csv(out / "age_weight_buckets.csv", index=False)
    pd.DataFrame([dict(age_range=f"{lo}-{hi}" if hi else f"{lo}+",
                       break_count=counts[f"break_age_{lo}_{hi}"])
                  for lo, hi in BUCKETS]).to_csv(out / "break_age_counts.csv", index=False)
    for name, records in (("lengths", lengths), ("historical_diagnostics", historical),
                          ("whitening_diagnostics", white), ("break_timing", timing)):
        distribution_table(records).to_csv(out / f"{name}.csv", index=False)
    tables = []
    for name, records in signatures.items():
        table = distribution_table(records, expected_count=counts["series"] - counts["no_break"])
        table.insert(0, "comparison", name)
        tables.append(table)
    pd.concat(tables, ignore_index=True).to_csv(out / "break_signatures.csv", index=False) if tables else pd.DataFrame().to_csv(out / "break_signatures.csv", index=False)
    bad_alignment = sum(alignment[k] for k in ("noncumulative_series", "no_break_nonzero_labels",
                                               "break_without_positive", "unmatched_break_series"))
    offsets = [r["position_minus_tau"] for r in timing if np.isfinite(r["position_minus_tau"])]
    offset_counts = dict(Counter(str(int(v)) for v in offsets))
    consistent = len(offset_counts) <= 1
    alignment_report = dict(counts=dict(alignment), shifted_identity_matches=dict(match_counts),
                            first_positive_minus_tau_counts=offset_counts,
                            consistent_offset=consistent, status="pass" if not bad_alignment and consistent else "review_required")
    write_json(out / "label_alignment.json", alignment_report)
    n = counts["series"]
    breaks = n - counts["no_break"]
    counts["break"] = breaks
    milestones = {}
    if total_weight:
        cumulative = per_age.weight.cumsum().to_numpy() / total_weight
        milestones = {str(q): int(np.searchsorted(cumulative, q) + 1) for q in (.25, .5, .75, .9, .95)}
    summary = dict(counts=dict(counts), no_break_fraction=counts["no_break"] / n if n else None,
                   early_middle_late_break_proportions={k: counts[k] / breaks if breaks else None for k in ("early", "middle", "late")},
                   total_pair_weight=total_weight, cumulative_metric_weight_age_milestones=milestones,
                   sources=source, wall_seconds=time.perf_counter() - start,
                   versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                   definitions={"age": "one-based online position", "std": "ddof=1", "MAD": "unscaled median absolute deviation",
                                "skew_kurtosis": "moment estimates; excess kurtosis; >=8 finite points",
                                "early_middle_late": "thirds of first-positive one-based age / observed online length",
                                "tail_thresholds": "reference historical 1% and 99% quantiles",
                                "matched_window": "min(128, available pre, available post), >=8",
                                "AR": "ridge=1, unpenalized intercept, training-only standardization, historical 70/30 holdout"})
    write_json(out / "summary.json", summary)
    group_sizes = manifest.groupby("group").size().to_numpy()
    audit = dict(duplicate_histories=counts["duplicate_histories"],
                 degenerate_duplicate_histories=counts["degenerate_duplicate_histories"],
                 duplicate_complete_series=counts["duplicate_complete_series"],
                 grouping="informative exact histories, exact complete series, verified same-length normalized RMS <=0.01 candidates",
                 group_size_distribution=summarize(group_sizes), group_count=len(group_sizes),
                 near_candidates=len(candidates), near_comparisons=checked, near_cap=near_cap,
                 verified_near_pairs=len(near_pairs), repeated_block_matches=block_matches,
                 repeated_block_pairs=len(block_pairs), unresolved_repeated_block_pairs=unresolved_blocks,
                 limitations=["Approximate search is not exhaustive; same-length histories only.",
                              "At most 16 aligned 128-point blocks per history; shifted overlaps may be missed.",
                              "Near grouping is conservative shape similarity, not established provenance.",
                              "Degenerate identical histories alone are not grouped; full-series duplicates are."])
    write_json(out / "related_series_audit.json", audit)
    readiness = dict(label_alignment=alignment_report["status"], counts=dict(counts),
                     unresolved_repeated_block_pairs=unresolved_blocks,
                     source_sha256={k: v["sha256"] for k, v in source.items()})
    write_json(cache / "profile_status.json", readiness)
    return summary

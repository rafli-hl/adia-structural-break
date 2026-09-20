# Structural Break Real-Time: E00/E01 foundation

## Frozen Astra Stage 2, Phase 1

The original foundation notes below describe the initial delivery. Real training
data have subsequently been profiled and E01 evaluated on 10,000 series; those
recorded artifacts are in `diagnostics/` and `e01_results/`.

Phase 1 implements the immutable contract in `research_specs/stage2_v1.json`:
E02 multi-scale raw energy; E03 matched final-30% historical reference; E04
historical 70/30 frozen ridge AR(5) innovations; E05 equal raw/innovation blend;
E06 frozen empirical ranks (location and bounded energy); E07 residual sign
dependence. Windows are 32, 128, 256 and cumulative; dependence uses 128/256.
AR numerical helpers are extracted verbatim from the validated profiler.
E08 remains specified but is not implemented or executable in this phase.

Run from the repository root with the Python 3.12 environment:

```powershell
.\.venv\Scripts\python.exe -m sbrt.validation --out stage2_results/validation
.\.venv\Scripts\python.exe run_research.py folds --cache local_cache --version cv2 --seed 20260919 --out local_cache/cv2
.\.venv\Scripts\python.exe run_research.py rescore --cache local_cache --folds local_cache/cv2/folds.parquet --oof e01_results/oof_squared.parquet --out stage2_results/E01_squared
.\.venv\Scripts\python.exe run_research.py batch --ids E02 E03 E04 E05 E06 E07 --cache local_cache --folds local_cache/cv2/folds.parquet --out stage2_results
.\.venv\Scripts\python.exe run_research.py compare --cache local_cache --folds local_cache/cv2/folds.parquet --out stage2_results
```

Rescore each other E01 file in the same way, using `E01_<detector>` output
directories. `experiment --id E04` resolves its declared dependencies. Batch
execution starts each missing experiment in a fresh process, verifies existing
artifacts before reuse, and never changes parameters based on scores. Completed
directories are immutable. `_execute` is the batch worker entry point.

CV-v2 preserves CV-v1 and uses six first-positive-age strata with seed 20260919.
Source, configuration, original dataset, fold and prediction hashes bind every
result. The initial repository inspection found only commit `b0014f6`, no tags,
and untracked foundation files; `research_specs/foundation_baseline.json` records
the actual pre-edit source and CV-v1 hashes instead of asserting a Git checkpoint.

Each experiment has complete OOF predictions, component scores, exact serialized
state sizes, array storage, series support diagnostics, age/fold metrics and an
artifact manifest. `comparisons/phase1_report.json` is the finalized report with
parent/fold/age deltas, conditional bootstrap intervals where required, advancement
decisions, and historical AR-MSE/rank-subblock diagnostics. Per-experiment metrics
remain immutable; their initial pending gate fields are superseded by the merged
comparison report. Statistical initialization and supervised training are reported
separately (supervised training time is zero in Phase 1).

Advancement requires delta >=0.002, >=4 positive fold deltas, nonnegative delta at
ages >=129, loss <=0.005 at ages <=64, and a positive lower bound from 999 paired
whole-series bootstrap replicates within the six strata (seed 20260920). Bootstrap
is run only if the first four conditions pass. Promotion additionally checks the
prior eligible reference and the early-age E01 guard. This is conditional
fixed-prediction uncertainty, not refit uncertainty or a selection-adjusted claim.

The unit suite preserves the original 19 tests and adds direct offline arithmetic,
prefix/truncation/series-order invariance, E01 AST and scorer byte regressions,
AR/CDF freezing, exact serialization, coverage, immutable hashes, metric-training
weight identities and tie-aware series-bootstrap checks. Synthetic fixtures are
used only to verify execution and mathematics. E08 trained-model tests remain
deferred with E08; no learned-model success is claimed.

State serialization uses versioned JSON and base64 numeric arrays, not pickle.
Reported update throughput includes replay guards, clocks and component capture,
and excludes the first point of every series as in the foundation. Serialized
state size is the larger initialized/completed snapshot per series; NumPy array
storage is reported separately. The maximum buffers are allocated at initialization.

This is a local research toolkit, not a trained final submission. It implements
the long-form parquet layout and executable EWMA specification supplied in this
conversation. No competition dataset is included or has been analyzed here.

## Run locally

Use Python 3.12 in a fresh environment. Keep the `sbrt/` directory beside the
entry-point scripts. The old list-column profiler is superseded by this bundle.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

python profile_structural_break.py \
  --x /path/to/X_train.parquet \
  --y /path/to/y_train.parquet \
  --index /path/to/y_train_index.parquet \
  --cache ./local_cache \
  --out ./diagnostics

python run_research.py backtest \
  --cache ./local_cache \
  --out ./e01_results
```

Alternatively, use `run_research.py profile --data-dir /path/to/files --cache
./local_cache --out ./diagnostics`. No list-valued loader is present.

Windows: activate `.venv\Scripts\activate` instead. Linux is the tested target;
the resource-based peak-memory report is Unix-specific.

Start E01 with a single detector if desired:

```bash
python run_research.py backtest --cache ./local_cache \
  --out ./official_ewma_results --detectors official_ewma
```

Use a new output directory for another run. Completed OOF files are never
silently overwritten. Fixed folds are reused across detector runs. A changed
dataset or changed fold configuration requires a new cache directory.

Set `--memory 1GB` (the default) on either main command to control DuckDB's
memory budget. This is not a hard limit on total Python-process memory. The
cache and external-sort spill directory need free disk space: allow several
times the uncompressed input size for preparation, plus OOF files. Profile
preparation hashes inputs, validates keys, joins labels, and stores a local
ordered cache. It can take substantially longer than model inference.

## E00 without any data

```bash
python run_research.py e00 --out ./e00.json
```

Expected synthetic scores: constant 0.5; age-only 0.5; perfect oracle 1.0;
reversed oracle 0.0. All-ineligible ages produce 0.5. Oracle predictions exist
only in evaluator tests, never in a detector.

Optional end-to-end synthetic smoke test:

```bash
python run_research.py make-synthetic --out ./synthetic_data
python run_research.py profile --data-dir ./synthetic_data \
  --cache ./synthetic_cache --out ./synthetic_diagnostics
python run_research.py backtest --cache ./synthetic_cache \
  --out ./synthetic_results
```

Synthetic results establish execution and invariants only. They are not evidence
about competition performance and must not guide hyperparameter selection.

## Data and label contract

- `X`: unique `(id,time)`, `value`, `period`; period 1 is history, period 2 online.
- `y`: unique `(id,time)`, binary `target`; online coverage must be complete.
  Historical labels may be present or absent. Extra keys outside X are rejected.
- `y_train_index`: unique `id`, integer `tau_index`, and `tau`.
  `tau_index == -1` means no break. IDs are string-normalized in local outputs.
- Physical parquet order is retained within each series, as in the supplied
  runner filter. The `time` key joins labels but is never silently used to
  reorder observations. Nonmonotone time is reported for inspection.
- The scorer uses original target labels. Profile tests shifts -1, 0, +1 against
  the entire online label vector, reports the first-positive offset, checks
  cumulative labels, and reports ambiguous boundary matches.
- Nonfinite values are profiled, but E01 fails closed instead of dropping rows.
  Empty online segments and unresolved label inconsistencies also block E01.

## Metric

At each zero-based `time_online`, ordinary sklearn ROC AUC is weighted by
`n_pos*n_neg`. One-class ages are skipped. No eligible ages returns 0.5.
`age` in reports is one-based. Tie behavior is sklearn's ordinary ROC AUC.

The implementation preserves the supplied reference accumulation order.
Pooled OOF TS-AUC is primary; fold TS-AUCs are separate diagnostics and their
unweighted average is not a replacement for the pooled score. Prediction
coverage is checked by guarded replay and complete local OOF keys/counts.

## Baselines and immutable configuration

`official_ewma` follows the supplied executable quickstarter semantics:

```
ALPHA = 0.05; KAPPA = 3.0
mu_h = historical mean, or 0 for empty history
sd_h = historical sample standard deviation, or 1 for length <= 1
sd_h = max(sd_h, 1e-8)
mu_ewma = mu_h; n_eff = 0
mu_ewma = 0.95 * mu_ewma + 0.05 * x
n_eff = 0.95 * n_eff + 1
se = sd_h / sqrt(max(n_eff, 1))
z = (mu_ewma - mu_h) / max(se, 1e-8)
score = tanh(abs(z) / 3)
```

There is no substitution of a textbook EWMA standard error. `n_eff` converges
to 20; that recurrence is reproduced, not reinterpreted as an IID-valid sample
size. Float saturation can create ties; it is measured and not silently fixed.

Other fixed E01 detectors: two-sided CUSUM (k=0.5), two-sided Page-Hinkley
(inclusive online running mean, delta=0.05), historical-normalized running mean,
variance ratio, absolute-deviation evidence, and squared-deviation evidence.
They use historical arithmetic mean/sample scale and `q/(1+q)` for nonnegative
evidence. No learned combination is fitted. Each has O(1) update/state cost and
O(H) initialization. The score is evidence, not calibrated break probability.
The variance ratio compares online and historical normalized sample variance,
flooring both at 1e-12 before the ratio. This gives neutral evidence for an
unchanged constant series. It is neutral before two online observations.

All seven methods fit only the current series' history. Hence five-fold OOF
does not involve fitting a global model yet. The fold infrastructure is in
place for later supervised experiments; no such model is included.

## Causality boundary

SAFE: frozen historical normalization; historical EWMA initialization; causal
running means/variances; CUSUM/PH recursion; current age.

EVALUATOR-ONLY: final lengths, targets, tau fields, full observations, retrospective
break signatures, group/ID/fold metadata, OOF scoring and timing.

UNSAFE and absent from inference: whole-online normalization, centered windows,
hindsight segmentation features, horizon features, row-level splits, or shared
mutable cross-series detector state.

`sbrt/detectors.py` is the inference-only boundary. `infer()` yields its initial
handshake, then exactly one score per new observation. It does not inspect the
online iterable's length or materialize it. Profile and data-cache modules must
never be imported into a submission's inference path.

The evaluator guards the online and dataset iterators. Tests exercise prefix,
truncation, repeatability, series order, lookahead, array/list conversion, missing
and extra scores, skipped datasets, and handshake violations. The guard protects
against accidental misuse; it is not a security sandbox for malicious code.

## Profiling and grouping interpretation

Historical moments, quantiles, raw/absolute/squared and centered autocorrelations,
trend, difference variance, and extreme counts are summarized across series.
Quantiles are distributions of per-series statistics, not pooled observations.
Skewness/kurtosis require at least eight finite observations; insufficient or
degenerate statistics are reported as unavailable with support counts.

Retrospective break signatures compare history with observed post-break data,
matched pre/post windows up to 128 points, and AR residuals against a historical
holdout reference. They are descriptive diagnostics, not inferred ground-truth
break-type labels. Small-sample and dependence effects remain relevant.

AR(1)/ridge AR(5) use the first 70% of history for fitting and the last 30% for
one-step evaluation. Ridge penalty is 1 with an unpenalized intercept. Fitting
separate historical halves diagnoses slope-coefficient stability. No AR model
is used by E01. These diagnostics do not establish that whitening is beneficial
for TS-AUC.

Exact informative duplicate histories and duplicate full series are grouped.
Near-duplicate search compares at most `--near-cap 2000` candidate pairs from
same-length 16-bin historical fingerprints. Normalized full-resolution RMS
<=0.01 causes conservative grouping. This is similarity, not proof of shared
provenance. Exact degenerate histories alone are counted but not grouped.

At most 16 nonoverlapping 128-point historical blocks are checked per series;
nondegenerate exact matches are verified. Unresolved repeated segments across
components block E01 pending local group review. Candidate IDs remain in
`local_cache/related_candidates.json`, not aggregate reports. To combine proposed
groups with existing duplicate groups, supply a complete `id,group` CSV:

```bash
python run_research.py backtest --cache ./local_cache --out ./e01_results \
  --groups ./reviewed_groups.csv
```

Approximate searches are not exhaustive. Shifted or different-length overlap
may be missed. The initial fold design is five fixed stratified series folds,
or stratified group folds where detected components require it. There must be
enough groups and both series classes to support five folds. Targets after tau
remain the scorer's labels; `has_break` is used only for splitting.

## Results and what to share

Share the small aggregate files in `diagnostics/`, plus `e00.json`,
`e01_results.json`, `experiment_log.csv`, `run_metadata.json`, and optionally
the per-age/age-bucket result files. Keep raw data, the cache, fold manifests,
candidate IDs, and `oof_*.parquet` local.

OOF files contain every online `(id,time,time_online,target,prediction,fold)`.
Initialization time is separate from guarded update time. The first point of
each series is excluded from update throughput because its generator call also
loads the evaluator record. Update timings include clock/guard/generator
overhead. Replay wall time includes data loading and OOF writes. Full wall time
also includes scoring. Peak RSS is process-cumulative, in platform units (KiB
on Linux), not an isolated per-detector allocation. Run one detector per fresh
process for isolated peak comparisons. State-size reporting is a shallow
Python-object estimate, not process memory.

Fold scores are not five independent repetitions and no bootstrap confidence
interval is claimed. Statistical uncertainty and break-regime performance
comparisons will follow after valid real-data baseline results exist.

The pinned dependencies, input content hashes, fixed seeds, single-thread CLI,
fold files, configurations, and per-run metadata support reproducibility.

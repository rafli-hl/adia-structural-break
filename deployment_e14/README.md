# Frozen E14 production handoff

This is one full-training refit of selected E14, not a new experiment. Existing
E08 deployment and all Stage-1/2/3/4 sources and artifacts remain protected.
Only z_R/z_I/z_P/z_D/S((J_s-I)/sqrt(2)) enter the logistic model. Reused source
modules contain other research definitions, but this adapter admits only E14;
no Stage-3 feature, fast/robust scale, conditional-rank block or ensemble is used.

The exact Stage-4 FeatureState initializes AR(5), ordinary alpha=.01 scale and
C references. The adapter reorders two independent updates to execute AR/scale
observation before the unchanged base blocks. The score is computed before
scale commit and yielded before consuming the next observation. Full-population
feature equality against the frozen cache and exact state/prediction equivalence
against the research detector are required.

The full-population fitter reuses Stage-4 fit_fold with all rows assigned to
fold 0 and absent held-out sentinel -1. It recalculates metric weights and
separate weighted base/contrast standardization. Objective is sum(weighted
logistic loss)+.01/2*dot(beta,beta), intercept unpenalized, frozen float64
zero-init L-BFGS-B options, one native thread, gradient infinity norm <=1e-6.
There is no fold averaging or hyperparameter search.

## Reproduce locally (PowerShell)

Start at D:\adia-structural-break. The commands refuse overwriting completed
training, reduced-test or package directories. For a repeat use new output paths
and the corresponding --resources, --package and --direct paths.

```powershell
Set-Location D:\adia-structural-break
.\.venv\Scripts\python.exe production_e14.py test
.\.venv\Scripts\python.exe production_e14.py verify
.\.venv\Scripts\python.exe production_e14.py train --output production_e14_results\full_train
.\.venv\Scripts\python.exe production_e14.py reduced --resources production_e14_results\full_train\resources --output production_e14_results\reduced
.\.venv\Scripts\python.exe production_e14.py package --resources production_e14_results\full_train\resources --output production_e14_results\submission
.\.venv\Scripts\python.exe production_e14.py package --resources production_e14_results\full_train\resources --output production_e14_results\submission_repeat
.\.venv-crunch\Scripts\python.exe production_e14_official_test.py --package production_e14_results\submission --data data --direct production_e14_results\reduced\prediction.parquet --output production_e14_results\official_local
```

`test` runs all four suites and freezes source/configuration/dependencies/protected
hashes. Rechecks preserve the initial freeze. Full training regenerates all five
raw quantities causally, checks the entire frozen Stage-4 cache, and requires
identical coefficients/preprocessing/weights on a second full numerical fit.
Model and manifest include content hashes; wall times are kept in separate reports.

The reduced 100-series score is a sanity check only. Labels are loaded after
inference. Primary direct timing includes model loading, AR/CDF/conditional-scale
historical initialization, rolling updates and logistic scoring. Input loading
and scoring are separate. A profiling pass measures initialization/updates/state
size and verifies exact resume for every reduced series. Memory is the measured
whole Windows host process, not isolated model memory; sampled maxima may miss
very brief peaks, and native lifetime peak counters are also recorded.

Official validation uses unchanged competition runner commit
5a24c2413122942eae97b6bad376ef0ec143ce22 and crunch-cli==12.0.3 from the successful
E08 environment. Only authenticated download is overridden with provided data.
Official guards, socket transport and deterministic 10% replay remain unchanged.
Predictions must exactly match direct predictions after float32 transport.
The helper rejects non-loopback networking and verifies all numerical imports
resolve inside the submission. No account credentials are required for this test.

## Submission package

Use only `production_e14_results/submission/`, NOT the repository root and NOT
`production_results/submission/` (the untouched E08 fallback).

The package contains main.py, requirements.txt, allowlisted unchanged numerical
helpers plus production_e14/runtime.py, resources/e14.json, resources/manifest.json
and package.sha256.json. No data, evaluator/scorer, CLI, tests or credentials.
The sibling ZIP is a reproducible archive, not a claim about ZIP upload support.
Repeat packaging must produce exactly the same ZIP hash. Do not rewrite line
endings or edit files: the bundle verifies its exact runtime source on load.

## User-only official CLI / cloud actions

Use your account-specific `crunch setup` command from the competition page to
create a NEW E14 workspace. Do not overwrite the E08 workspace. Copy the submission
directory contents into that configured workspace, preserving all relative paths.
Do not copy research data or a Python environment. The exact pinned CLI commands,
executed FROM THAT E14 ACCOUNT-CONFIGURED WORKSPACE, are:

```powershell
D:\adia-structural-break\.venv-crunch\Scripts\crunch.exe test --no-force-first-train --main-file main.py --model-directory resources
D:\adia-structural-break\.venv-crunch\Scripts\crunch.exe push --no-pip-freeze --main-file main.py --model-directory resources --message "Frozen E14 full-training refit"
```

No account token/model name is invented here. `--no-force-first-train` validates
the supplied full-data artifact; remove it only if you deliberately want a new
official full retraining. `--no-pip-freeze` preserves the explicit numerical pins.
For the exact pretrained artifact, select the cloud no-training option. Upload
and cloud execution are user actions only; this workflow never pushes/runs remotely.

## Dependencies and runtime limits

Use the same five numerical pins as validated E08: NumPy 2.3.5, SciPy 1.17.0,
pandas 2.2.3, PyArrow 25.0.1, threadpoolctl 3.7.0. No new package is needed.
Crunch CLI/psutil/DuckDB/scikit-learn are not submission dependencies. The module
named sbrt/stage3_features.py is included unchanged solely because trusted
Stage-4 preprocessing imports its standardize_extra helper.

Local Python is 3.12; the previous E08 Linux wheel check requires x86-64 and
glibc >=2.28 overall. Reuse the known successful E08 cloud runtime configuration;
do not silently alter dependency versions. Actual cloud allocation, remaining
quota and runtime are account-specific and are not established by local tests.
The local 100-to-10,000-series extrapolation is an estimate, not a cloud guarantee.
The earlier E08 cloud score/runtime are external references, never tuning targets.

Stop after local validation and packaging. Do not push, combine candidates or
start Stage 5 without a separate request.

# Production E08: local handoff and submission checklist

This is deployment engineering, not a new experiment. The model is one refit on
all 10,000 training series. It reuses `sbrt.logistic.fit_fold` with every row in
training (fold 0, absent held-out sentinel -1), the four E07 block outputs R/I/P/D,
the full-population metric weights, and the exact frozen E08 objective. The
intercept is unpenalized; the objective adds `0.01 / 2 * dot(beta, beta)`.
No fold coefficients are averaged. No age/context/Stage-3 inputs are included.

All commands below are PowerShell, starting in `D:\adia-structural-break`.
The original `.venv`, `requirements.txt`, `sbrt/`, and `tests/` are preserved.
New tests live in `deployment_tests/` to preserve Stage-3 source integrity.

## 1. Test, protect the checkpoint, and freeze deployment source

```powershell
Set-Location D:\adia-structural-break
.\.venv\Scripts\python.exe production.py test
.\.venv\Scripts\python.exe production.py verify
```

`test` runs the complete original suite and deployment suite, records their logs,
hashes deployment source/configuration, and checks every protected data/research
artifact against `production_results/before.json`. Training refuses source/config
drift after this test freeze. No research files are regenerated or overwritten.

## 2. Train the full-population artifact

```powershell
.\.venv\Scripts\python.exe production.py train --output production_results\full_train
```

This calls the production training implementation on the real training tuples,
causally regenerates the four blocks, verifies every block/key/label against E07,
fits the full population, and repeats the numerical fit to require byte-identical
model serialization and identical training weights. It records all 999 age-count
and class-mass identities. This is not a new OOF measurement.

Output paths must be new. For reproduction use `--output production_results\full_train_repeat`
and compare `resources\e08.json` and `resources\manifest.json` hashes. Training time
and platform diagnostics are intentionally not part of canonical model bytes.
`training.json` preserves original parquet SHA-256 hashes, source/configuration,
fit diagnostics, and the full training weight audit. The model manifest also
includes a canonical digest of all per-series history/online/label content.

## 3. Validate the actual production generator on the reduced test

```powershell
.\.venv\Scripts\python.exe production.py reduced --resources production_results\full_train\resources --output production_results\reduced
```

The official quickstarter explicitly permits scoring the supplied 100-series
reduced labels. Labels are loaded only after all production predictions are
emitted. The check uses `deployment/main.py:infer` through the trusted streaming
guard. Timings include model loading, historical AR/CDF initialization, rolling
updates, logistic scoring and replay overhead, with input I/O reported separately.
An additional profiling/resume pass must reproduce every prediction exactly.
Do not change the model in response to this sanity-check score.

## 4. Package, then run the pinned official local runner without credentials

```powershell
.\.venv\Scripts\python.exe production.py package --resources production_results\full_train\resources --output production_results\submission
# Create this isolated environment if it does not already exist:
# .\.venv\Scripts\python.exe -m venv .venv-crunch
# .\.venv-crunch\Scripts\python.exe -m pip install -r deployment\requirements-cli.txt
.\.venv-crunch\Scripts\python.exe production_official_test.py --package production_results\submission --data data --output production_results\official_local
```

The helper uses the official `crunch-cli==12.0.3` LocalRunner and the unmodified
competition runner at commit `5a24c2413122942eae97b6bad376ef0ec143ce22`.
Only its authenticated download step is overridden to use the already supplied
local data. Official train/infer dispatch, sockets, double-protection guards, and
the 10% determinism replay are retained. Public competition metadata is a saved
real API response, not invented configuration. No workspace token is needed.
This verifies the packaged runtime imports rather than the repository's runtime.
The official transport casts predictions to float32; its output must equal the
direct production predictions after that transport cast. Report both scores if
rounding changes ties. This is a local check, not proof of cloud execution.

Fresh output paths are required. `production_official_test.py` compares to the
canonical direct run at `production_results/reduced/prediction.parquet`.

## 5. Submission files and deterministic packaging

`production_results/submission/` is the submission root. A deterministic ZIP
snapshot is written alongside it; the ZIP is a transport/archive convenience,
not a claim that Crunch accepts ZIP upload. Use the contents through the Files
tab or CLI. The allowlist is:

- `main.py`, `requirements.txt`;
- `production_e08/__init__.py`, `production_e08/runtime.py`;
- unchanged `sbrt/__init__.py`, `logistic.py`, `stage2_detectors.py`, `history.py`, `features.py`;
- `resources/e08.json`, `resources/manifest.json`;
- `package.sha256.json` (file checksums).

No data, labels, research outputs, credentials, caches, evaluator/scorer, notebook,
CLI, Stage-3 features, or virtual environment is in this package. Repeating the
package command with a different new output directory must produce the same ZIP
SHA-256. Do not edit packaged source or rewrite line endings: runtime source
digests are validated when the model loads.

## 6. Upload and cloud run: USER ACTION ONLY

No upload, account setup, submission, or cloud run is executed by the engineering
workflow. Two options:

1. In the competition's **Submit / Files** interface, add the entire contents of
   `production_results/submission/`, preserving directory paths and including
   `requirements.txt` and `resources/`. Do not select the repository root.
2. For CLI submission, copy the account-specific `crunch setup` command from the
   competition **Submit via CLI** page. It supplies your real model name and
   short-lived token; neither is stored or guessed here. Set up a NEW sibling
   workspace, never the research repository. Copy only the package contents into
   that configured workspace, then execute:

```powershell
# Set-Location to YOUR account-configured Crunch workspace first.
D:\adia-structural-break\.venv-crunch\Scripts\crunch.exe test --no-force-first-train
D:\adia-structural-break\.venv-crunch\Scripts\crunch.exe push --no-pip-freeze --message "Frozen E08 full-training refit"
```

The exact setup syntax is `crunch setup <competition> <model> --token <token> [directory]`.
Use the real command from your account. `--no-force-first-train` exercises the
provided artifact; remove that flag to test full retraining. `--no-pip-freeze`
preserves the explicit submitted pins rather than freezing unrelated CLI packages.

After upload select the submission and **Run in the Cloud**. The competition
allows the pretrained artifact with its reproducible training code; choose
**No training** to use this exact artifact, or training to exercise the same
full-data refit. Inspect logs/coverage/determinism and obtain a successful cloud
run before considering the submission cloud-validated.

## Runtime and dependency caveats

Inference uses one native numerical thread and `INFER_PARALLELISM=1`. No GPU is
needed. Estimate 10,000-series inference using both the real 100-series generator
timing and the slower official socket-runner timing (which includes its 10%
determinism replay). These are extrapolations, not guaranteed cloud benchmarks.
The published competition quota is 15 hours per week; actual remaining quota,
CPU, memory and timeout must be checked in your account's run dialog.

Submit the unchanged pins: NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, PyArrow 25.0.1,
threadpoolctl 3.7.0. The production code needs no scikit-learn, DuckDB, or Crunch CLI.
Local scoring/tooling does. The official runner requires Crunch CLI >=11.7.0;
this local adapter is pinned/tested with 12.0.3.

The five dependency names were checked against the public Crunch whitelist.
CPython 3.12 Linux x86-64 wheels are checked separately. PyArrow's wheel requires
glibc >=2.28; NumPy/SciPy require >=2.27. Cloud Python/architecture/image details
are not guaranteed by the public competition page. Select/confirm compatible
Python (3.12 matches local validation), Linux x86-64 and glibc >=2.28 in the run
environment. Do not silently downgrade dependencies to fit an older image.
If the cloud offers an incompatible image, request an appropriate runtime from
Crunch support and repeat deployment validation before use. Inference performs
no networking or filesystem reads outside submitted code/model resources.

Official sources:

- https://docs.crunchdao.com/competitions/competitions/structural-break-real-time
- https://docs.crunchdao.com/competitions/participate
- https://docs.crunchdao.com/competitions/participate/resources-limit
- https://github.com/crunchdao/competitions/tree/5a24c2413122942eae97b6bad376ef0ec143ce22/competitions/structural-break-real-time

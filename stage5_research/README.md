# Frozen Stage 5

Authoritative contract: `stage5_research/specs/stage5_v1.md` (pinned SHA-256 in contract.py).
Run from `D:\adia-structural-break` using the existing research environment. No dependency upgrades.

```powershell
.\.venv\Scripts\python.exe run_stage5.py partition
.\.venv\Scripts\python.exe run_stage5.py validate
.\.venv\Scripts\python.exe run_stage5.py freeze
.\.venv\Scripts\python.exe run_stage5.py batch
```

The CLI sets the four numerical thread environment variables to 1 before importing numerical libraries.
Validation runs all five suites (research, E08 deployment, Stage 4, E14 deployment, Stage 5).
The partition is immutable and must precede export. Freeze requires final passing tests, unchanged source,
configuration, specification, partition, dependencies, thread environment and protected artifacts.

Equivalent separated execution after freeze:

```powershell
.\.venv\Scripts\python.exe run_stage5.py export
.\.venv\Scripts\python.exe run_stage5.py run --experiment E18
.\.venv\Scripts\python.exe run_stage5.py run --experiment E19
.\.venv\Scripts\python.exe run_stage5.py run --experiment E20
.\.venv\Scripts\python.exe run_stage5.py run --experiment E21
.\.venv\Scripts\python.exe run_stage5.py compare
.\.venv\Scripts\python.exe run_stage5.py report
```

Completed hash-verified outputs are reused without refitting. Incomplete or incompatible outputs fail closed.
Do not delete results and rerun after inspecting scores. Any necessary engineering correction requires a
documented audit and fresh validation, never a modeling-setting adjustment.

Each candidate's runtime includes full-population fresh causal inference and an exact prediction audit against
its cached OOF computation. Cached throughput is not production throughput. The historical null replay uses
the same C that fitted the inclusive CDF and references: it is explicitly in-sample, not held-out calibration.
The 5000/5000 A/B slices reuse previously researched series and overlapping CV training populations.

Outputs: stage5_results/{partition,validation,freeze,features,E18,E19,E20,E21,comparisons}, REPORT.md and
delivery_manifest.json. Scratch DuckDB/spill files are evaluator-only and excluded from the delivery inventory.
No Stage-1–4/production files are modified. Stop at the G5 report; no production refit or cloud submission.

# Frozen Stage-3 execution

The authoritative contract is `research_specs/stage3_v1.md`; its SHA-256 is
`35b21bc6e1f3c09f45a7ab49a31dd216872a4669c516ca34a40efc0ea2c1601e`.
`stage3_v1.json` resolves the same contract for execution. E09 through E13 are
independent E08 children, not an accumulating feature sequence.

Use the pinned local environment. `run_research.py` sets native numerical thread
environment variables to 1 before numerical imports. No hyperparameter CLI
overrides exist. Do not overwrite completed outputs or reuse failed/incompatible
artifacts. The pre-edit protection inventory is in
`stage3_results/before/baseline.json`.

## Admission and execution

Run the complete final tests into a new directory, then freeze that exact source:

```powershell
.\.venv\Scripts\python.exe -m sbrt.validation --out stage3_results\validation_final
.\.venv\Scripts\python.exe run_research.py stage3-freeze --out stage3_results --cache local_cache --folds local_cache\cv2\folds.parquet --validation stage3_results\validation_final
.\.venv\Scripts\python.exe run_research.py stage3-batch --out stage3_results
```

The batch performs the one permitted causal feature export, then E09, E10, E11,
E12 and E13 in separate processes, followed by comparisons. Each candidate trains
five outer-fold models exactly once. Already completed compatible artifacts are
verified and reused, not refitted. An invalid numerical candidate does not cancel
later candidates. Source, contract, dependencies, threads, folds and protected
artifacts are checked at admission and after each run.

Equivalent individual commands (do not refit an already measured candidate):

```powershell
.\.venv\Scripts\python.exe run_research.py stage3-export --out stage3_results
.\.venv\Scripts\python.exe run_research.py stage3-run --out stage3_results --id E09
.\.venv\Scripts\python.exe run_research.py stage3-run --out stage3_results --id E10
.\.venv\Scripts\python.exe run_research.py stage3-run --out stage3_results --id E11
.\.venv\Scripts\python.exe run_research.py stage3-run --out stage3_results --id E12
.\.venv\Scripts\python.exe run_research.py stage3-run --out stage3_results --id E13
.\.venv\Scripts\python.exe run_research.py stage3-compare --out stage3_results
```

## Artifact semantics

- `freeze/`: resolved configuration, tested source, versions, environment and hashes.
- `features/`: raw causal quantities only; separate history-only context; export
  audit, support counts, per-series state sizes and shared export timings.
- `E09/` through `E13/`: fold models, preprocessing, normalized-weight audits,
  training diagnostics, one complete OOF artifact, official metrics and age files.
- `comparisons/`: complete per-candidate reports, conditional bootstrap draws for
  serious candidates, G3 decisions, historical subgroups and final winner.
- `batch_execution.json`: process exit statuses and measured batch wall time.

E13 uses native scikit-learn serialization through pickle. Load only local,
hash-verified artifacts created by this program; pickle is not an interchange
format for untrusted models. Logistic parameters and preprocessing use JSON.
Streaming state is separately serialized and bound to the shared model hash.

Cached-model throughput is not end-to-end streaming throughput. The shared
guarded feature-export update rate is separately labeled. Shared feature-export
cost must be counted once when computing whole-batch cost.

The final decision retains E08 if nothing qualifies. Multiple qualifiers are
ordered by pooled score, then fewer inputs, then lower experiment ID. The five-way
one-sided bootstrap safeguard is approximate and conditional on fixed OOF models;
it does not remove prior research-selection risk. Stop after this batch: no
combined extensions, new objectives, feature selection, tuning, or Stage 4.

## Completed run and evaluator-only recovery

All five measured candidates were trained using source hash
`756e756a46f975b0bd8c4f1885083bd080d3968992c562cf13061fec8320170b`,
after 73 passing tests. The resolved configuration hash is
`ccbea8925e5ffdc48043a704530c85198ada11edceec4b49963fb0aca651658d`.

The initial batch driver encountered a transient DuckDB open-file lock before
E13 began fitting. Its empty output directory was archived, and E13 then ran once
under the same original frozen source. No candidate was refitted.

The E09 subgroup query also exposed a case-insensitive SQL collision between
historical `r` and raw block `R`. An evaluator-only qualification correction was
made after all measured models finished. The incomplete comparison output was
empty and was archived. The correction was retested (75 passing tests) and
separately frozen at source hash
`f910be6b34a09b8bbfd481c412a1ba2bf27234381cb65418068a87c162c16063`.
Feature/model/gate/bootstrap code and all completed candidate artifacts were
unchanged. The original source snapshot, reversible diff, incident record, and
evaluation freeze are under `stage3_results/recovery/`.

The current corrected checkout intentionally permits only audited comparison
reuse, not additional model execution against the old measurement freeze:

```powershell
.\.venv\Scripts\python.exe run_research.py stage3-compare --out stage3_results
```

The earlier full-batch/individual commands document the original frozen
measurement path. Its exact changed-file source snapshots and hashes are retained
for independent reproduction; do not overwrite this completed run.

Final decision: all E09-E13 are valid, none passes G3, and E08 remains reference.
No conditional bootstrap was required because no pooled delta reached +0.002.
Full precision results, fold/age diagnostics, parameters, model hashes and gates
are in `stage3_results/comparisons/stage3_report.json`.

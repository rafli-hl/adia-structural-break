"""Persist the complete unittest run and immutable-foundation checks."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from pathlib import Path
import json
import sys
import time
import unittest
from .data import sha256
from .provenance import ROOT, write_json, source_manifest, versions, finalize_artifacts


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, value):
        self.stream.write(value)
        self.file.write(value)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise ValueError("Refusing to overwrite a validation run")
    out.mkdir(parents=True)
    source = source_manifest()
    baseline = json.loads((ROOT/"research_specs/foundation_baseline.json").read_text())
    before = {path: sha256(ROOT/path) for path in ("local_cache/folds.parquet","local_cache/fold_settings.json")}
    if any(before[p] != baseline["files"][p] for p in before):
        raise AssertionError("CV-v1 differs from the initial foundation snapshot")
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"))
    with (out/"unittest.log").open("w",encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=Tee(sys.stderr,log),verbosity=2).run(suite)
    unchanged = all(sha256(ROOT/p)==h for p,h in before.items())
    stable = source_manifest()["sha256"] == source["sha256"]
    report = dict(command="python -m sbrt.validation --out " + str(out),
                  discovery_equivalent="python -m unittest discover -s tests -v",
                  tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                  skipped=len(result.skipped),seconds=time.perf_counter()-start,
                  successful=result.wasSuccessful() and unchanged and stable,
                  cv1_unchanged=unchanged,source_unchanged=stable,source_sha256=source["sha256"],
                  versions=versions(), foundation_before=baseline["tests_before"],
                  deferred_tests="None in E08 scope; Phase-1 tests are preserved and E08 numerical, isolation, determinism and serialization tests are included.")
    write_json(out/"validation.json",report)
    write_json(out/"source_manifest.json",source)
    finalize_artifacts(out)
    if not report["successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

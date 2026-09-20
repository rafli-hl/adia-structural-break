"""Independent Stage-4 admission; old artifact roots remain read-only."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import duckdb
from threadpoolctl import threadpool_info
from sbrt.data import sha256
from sbrt.folds import load_v2
from sbrt.provenance import ROOT, canonical_hash, write_json, source_manifest, versions, finalize_artifacts, verify_artifacts
from production_e08.local import verify_protected, source as production_source
from .contract import CONFIG, THREADS, E08_HASH, CV_HASH

DEFAULT = ROOT/"stage4_results"
SPEC = ROOT/"stage4_research/specs/stage4_v1.md"


def source():
    paths = [*sorted((ROOT/"stage4_research").rglob("*.py")),*sorted((ROOT/"stage4_research/specs").glob("*")),
             *sorted((ROOT/"stage4_tests").glob("*.py")),ROOT/"run_stage4.py"]
    files = {p.relative_to(ROOT).as_posix():sha256(p) for p in paths if p.is_file()}
    files.update(source_manifest()["files"])
    files.update(production_source()["files"])
    return dict(files=files,sha256=canonical_hash(files))


def protected():
    result = verify_protected()
    delivery = json.loads((ROOT/"production_results/delivery.json").read_text())
    for name,digest in delivery["files"].items():
        if sha256(ROOT/"production_results"/name)!=digest:
            raise ValueError(f"Protected production artifact changed: {name}")
    result.update(production_delivery_sha256=sha256(ROOT/"production_results/delivery.json"),
        production_source=production_source(),production_delivery_files=len(delivery["files"]))
    return result


def evaluator(root):
    # Never open the protected database with write access. Scratch tables/views
    # and bootstrap materializations belong exclusively to Stage 4.
    out = Path(root)/"scratch"
    out.mkdir(parents=True,exist_ok=True)
    con = duckdb.connect(str(out/"evaluation.duckdb"))
    con.execute("SET threads=1")
    con.execute("SET memory_limit='2GB'")
    con.execute("SET temp_directory=?",[str(out/"spill")])
    literal = str(ROOT/"local_cache/data.duckdb").replace("'","''")
    con.execute(f"ATTACH '{literal}' AS trusted (READ_ONLY)")
    for name in ("observations","series_index"):
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM trusted.{name}")
    return con


def validate(root):
    root = Path(root)
    if (root/"freeze").exists():
        raise ValueError("Already frozen; validation cannot replace measured admission")
    start = time.perf_counter()
    before, frozen_source = protected(), source()
    out = root/"validation"
    out.mkdir(parents=True,exist_ok=True)
    suites = {}
    for directory in ("tests","deployment_tests","stage4_tests"):
        tick = time.perf_counter()
        process = subprocess.run([sys.executable,"-m","unittest","discover","-s",directory,"-v"],
            cwd=ROOT,capture_output=True,text=True)
        log = process.stdout+process.stderr
        (out/(directory+".log")).write_text(log,encoding="utf-8")
        match = re.search(r"Ran (\d+) tests? in",log)
        suites[directory] = dict(exit_code=process.returncode,tests_run=int(match.group(1)) if match else None,
            seconds=time.perf_counter()-tick)
        print(f"{directory}: {suites[directory]}",flush=True)
        if process.returncode:
            print(log,flush=True)
            raise RuntimeError(f"{directory} failed; no full-data admission")
    if source()!=frozen_source or protected()!=before:
        raise ValueError("Source/artifacts changed during tests")
    value = dict(successful=True,suites=suites,source=frozen_source,protected=before,
        config_sha256=canonical_hash(CONFIG),spec_sha256=sha256(SPEC),versions=versions(),seconds=time.perf_counter()-start)
    write_json(out/"validation.json",value)
    finalize_artifacts(out)
    return value


def freeze(root):
    root = Path(root)
    if (root/"freeze").exists():
        raise FileExistsError("Immutable freeze already exists")
    verify_artifacts(root/"validation")
    passed = json.loads((root/"validation/validation.json").read_text())
    current = source()
    if not passed["successful"] or passed["source"]!=current or passed["config_sha256"]!=canonical_hash(CONFIG) or passed["spec_sha256"]!=sha256(SPEC):
        raise ValueError("Require passing tests on final source/config/spec")
    if any(os.environ.get(k)!="1" for k in THREADS) or any(p["num_threads"]!=1 for p in threadpool_info()):
        raise ValueError("Native numerical threads must be one")
    manifest,metadata = load_v2(ROOT/"local_cache",ROOT/"local_cache/cv2/folds.parquet")
    if metadata["fold_manifest_sha256"]!=CV_HASH or sha256(ROOT/"stage2_phase2/E08/oof.parquet")!=E08_HASH:
        raise ValueError("Immutable reference hash mismatch")
    if len(manifest)!=10000 or int(manifest.online_length.sum())!=5036517:
        raise ValueError("Population coverage mismatch")
    protection = protected()
    if protection!=passed["protected"]:
        raise ValueError("Protected checkpoint drift")
    value = dict(config=CONFIG,config_sha256=canonical_hash(CONFIG),source=current,spec_sha256=sha256(SPEC),
        versions=versions(),native_threads=threadpool_info(),protected=protection,cv2=metadata,
        data=json.loads((ROOT/"local_cache/sources.json").read_text()),
        validation_sha256=sha256(root/"validation/artifacts.sha256.json"))
    write_json(root/"freeze/freeze.json",value)
    write_json(root/"freeze/resolved_config.json",CONFIG)
    finalize_artifacts(root/"freeze")
    return value


def admission(root):
    root = Path(root)
    verify_artifacts(root/"freeze")
    f = json.loads((root/"freeze/freeze.json").read_text())
    if f["source"]!=source() or f["config"]!=CONFIG or f["spec_sha256"]!=sha256(SPEC):
        raise ValueError("Source/config/spec changed after freeze")
    if f["versions"]!=versions() or any(os.environ.get(k)!="1" for k in THREADS) or any(p["num_threads"]!=1 for p in threadpool_info()):
        raise ValueError("Dependency/thread drift")
    if protected()!=f["protected"] or sha256(root/"validation/artifacts.sha256.json")!=f["validation_sha256"]:
        raise ValueError("Protected/validation artifact drift")
    verify_artifacts(root/"validation")
    manifest,metadata = load_v2(ROOT/"local_cache",ROOT/"local_cache/cv2/folds.parquet")
    if metadata!=f["cv2"]:
        raise ValueError("CV drift")
    return f,manifest


def provenance(frozen):
    return dict(source_sha256=frozen["source"]["sha256"],config_sha256=frozen["config_sha256"],
        spec_sha256=frozen["spec_sha256"],fold_manifest_sha256=CV_HASH,E08_oof_sha256=E08_HASH,
        data_sha256={k:v["sha256"] for k,v in frozen["data"].items()})

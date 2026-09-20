"""Stage-5 admission and source/configuration freeze; old roots read-only."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from threadpoolctl import threadpool_info
from sbrt.data import sha256
from sbrt.folds import load_v2
from sbrt.provenance import ROOT, canonical_hash, write_json, versions, finalize_artifacts, verify_artifacts
from production_e14.local import verify_protected, source as e14_source
from stage4_research.provenance import source as stage4_source, evaluator
from .contract import CONFIG, THREADS, E14_HASH, CV_HASH, SPEC_HASH
from .partition import create

DEFAULT=ROOT/"stage5_results"
SPEC=ROOT/"stage5_research/specs/stage5_v1.md"


def source():
    files={**stage4_source()["files"],**e14_source()["files"]}
    paths=[*sorted((ROOT/"stage5_research").rglob("*.py")),*sorted((ROOT/"stage5_tests").glob("*.py")),
           SPEC,ROOT/"run_stage5.py",ROOT/"stage5_research/README.md"]
    files.update({p.relative_to(ROOT).as_posix():sha256(p) for p in paths if p.is_file()})
    return dict(files=files,sha256=canonical_hash(files))


def protected():
    result=verify_protected()
    path=ROOT/"production_e14_results/delivery_manifest.json"
    delivery=json.loads(path.read_bytes())
    if canonical_hash(delivery["files"])!=delivery["sha256"]: raise ValueError("E14 delivery inventory changed")
    for name,expected in delivery["files"].items():
        if sha256(ROOT/name)!=expected: raise ValueError(f"Protected E14 production changed: {name}")
    if sha256(ROOT/"stage4_results/E14/oof.parquet")!=E14_HASH or sha256(ROOT/"local_cache/cv2/folds.parquet")!=CV_HASH:
        raise ValueError("Protected reference mismatch")
    result.update(E14_production_files=len(delivery["files"]),E14_delivery_sha256=sha256(path))
    return result


def partition(root):
    if sha256(SPEC)!=SPEC_HASH: raise ValueError("Authoritative Stage-5 spec changed")
    manifest,meta=load_v2(ROOT/"local_cache",ROOT/"local_cache/cv2/folds.parquet")
    if meta["fold_manifest_sha256"]!=CV_HASH: raise ValueError("CV-v2 changed")
    if not (Path(root)/"partition").exists() and any((Path(root)/p).exists() for p in ("features",*CONFIG["execution_order"])):
        raise ValueError("Partition must precede export and measurements")
    return create(root,manifest,CV_HASH,SPEC_HASH)


def validate(root):
    root=Path(root)
    if (root/"freeze").exists(): raise ValueError("Already frozen; do not replace admission")
    before,current=protected(),source()
    _,parts=partition(root)
    out=root/"validation"; out.mkdir(parents=True,exist_ok=True)
    suites={}; start=time.perf_counter()
    for directory in ("tests","deployment_tests","stage4_tests","deployment_e14_tests","stage5_tests"):
        tick=time.perf_counter()
        run=subprocess.run([sys.executable,"-m","unittest","discover","-s",directory,"-v"],cwd=ROOT,capture_output=True,text=True)
        log=run.stdout+run.stderr
        (out/(directory+".log")).write_text(log,encoding="utf-8")
        match=re.search(r"Ran (\d+) tests? in",log)
        suites[directory]=dict(exit_code=run.returncode,tests_run=int(match.group(1)) if match else None,seconds=time.perf_counter()-tick)
        print(f"{directory}: {suites[directory]}",flush=True)
        if run.returncode: print(log,flush=True); raise RuntimeError(f"{directory} failed")
    if source()!=current or protected()!=before: raise ValueError("Source/protection drift during tests")
    record=dict(successful=True,suites=suites,source=current,protected=before,partition=parts,
        config_sha256=canonical_hash(CONFIG),spec_sha256=sha256(SPEC),versions=versions(),seconds=time.perf_counter()-start)
    write_json(out/"validation.json",record); finalize_artifacts(out)
    return record


def threads():
    if any(os.environ.get(k)!="1" for k in THREADS) or any(p["num_threads"]!=1 for p in threadpool_info()):
        raise ValueError("Require single native numerical thread")


def freeze(root):
    root=Path(root)
    if (root/"freeze").exists(): raise FileExistsError("Immutable freeze exists")
    verify_artifacts(root/"validation")
    passed=json.loads((root/"validation/validation.json").read_text())
    current=source(); protection=protected(); _,parts=partition(root)
    if (not passed["successful"] or passed["source"]!=current or passed["protected"]!=protection or
            passed["partition"]!=parts or passed["config_sha256"]!=canonical_hash(CONFIG) or
            passed["spec_sha256"]!=sha256(SPEC) or passed["versions"]!=versions()):
        raise ValueError("Require passing tests on unchanged final configuration")
    threads()
    manifest,meta=load_v2(ROOT/"local_cache",ROOT/"local_cache/cv2/folds.parquet")
    if len(manifest)!=10000 or int(manifest.online_length.sum())!=5036517: raise ValueError("Coverage mismatch")
    value=dict(config=CONFIG,config_sha256=canonical_hash(CONFIG),spec_sha256=sha256(SPEC),source=current,
        partition=parts,protected=protection,versions=versions(),native_threads=threadpool_info(),cv2=meta,
        data=json.loads((ROOT/"local_cache/sources.json").read_text()),
        validation_sha256=sha256(root/"validation/artifacts.sha256.json"))
    write_json(root/"freeze/freeze.json",value); write_json(root/"freeze/resolved_config.json",CONFIG)
    finalize_artifacts(root/"freeze")
    return value


def admission(root):
    root=Path(root); verify_artifacts(root/"freeze")
    f=json.loads((root/"freeze/freeze.json").read_text())
    if f["source"]!=source() or f["config"]!=CONFIG or f["spec_sha256"]!=sha256(SPEC): raise ValueError("Frozen source/config/spec drift")
    if f["versions"]!=versions(): raise ValueError("Dependency drift")
    threads()
    verify_artifacts(root/"validation")
    if sha256(root/"validation/artifacts.sha256.json")!=f["validation_sha256"] or protected()!=f["protected"]: raise ValueError("Protection drift")
    _,parts=partition(root)
    manifest,meta=load_v2(ROOT/"local_cache",ROOT/"local_cache/cv2/folds.parquet")
    if parts!=f["partition"] or meta!=f["cv2"]: raise ValueError("Partition/CV drift")
    return f,manifest


def provenance(f):
    return dict(source_sha256=f["source"]["sha256"],config_sha256=f["config_sha256"],spec_sha256=f["spec_sha256"],
        fold_manifest_sha256=CV_HASH,partition_sha256=f["partition"]["manifest_sha256"],E14_oof_sha256=E14_HASH,
        data_sha256={k:v["sha256"] for k,v in f["data"].items()})

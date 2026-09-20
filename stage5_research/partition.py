"""Evaluator-only immutable stability slices; never training inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sbrt.folds import STRATA
from sbrt.data import sha256
from sbrt.provenance import write_json, finalize_artifacts, verify_artifacts, canonical_hash
from .contract import CONFIG


def expected(manifest):
    if manifest.id.isna().any() or manifest.id.duplicated().any() or not manifest.id.map(lambda x:isinstance(x,str)).all():
        raise ValueError("Require existing unique normalized string IDs")
    if not set(manifest.stratum)<=set(STRATA) or not set(manifest.fold)<=set(range(5)):
        raise ValueError("Unknown CV-v2 cell")
    cells = []
    for stratum in STRATA:
        for fold in range(5):
            group = manifest.loc[(manifest.stratum==stratum)&(manifest.fold==fold),["id","stratum","fold"]].copy()
            group["digest"] = group.id.map(lambda sid:hashlib.sha256(("stage5_v1|20260924|"+sid).encode("utf-8")).digest())
            cells.append(group.sort_values(["digest","id"]).drop(columns="digest"))
    result = pd.concat(cells,ignore_index=True)
    result["partition"] = np.where(np.arange(len(result))%2==0,"A","B")
    for keys in (["stratum","fold"],["stratum"]):
        counts = result.groupby(keys+["partition"]).size().unstack(fill_value=0).reindex(columns=["A","B"],fill_value=0)
        if ((counts.A-counts.B).abs()>1).any(): raise AssertionError("Partition imbalance")
    return result


def create(root, manifest, cv_hash, spec_hash):
    out = Path(root)/"partition"
    frame = expected(manifest)
    metadata = dict(algorithm=CONFIG["partition"],cv2_sha256=cv_hash,spec_sha256=spec_hash,
        assignments_sha256=canonical_hash(frame.to_dict("records")),
        counts={str(k):int(v) for k,v in frame.partition.value_counts().items()},
        cells=frame.groupby(["stratum","fold","partition"]).size().rename("series").reset_index().to_dict("records"),
        limitation="Stability slices, not untouched holdouts or independent replication")
    if out.exists():
        verify_artifacts(out)
        prior=json.loads((out/"metadata.json").read_text())
        if prior!={**metadata,"manifest_sha256":sha256(out/"manifest.parquet")} or not pd.read_parquet(out/"manifest.parquet").equals(frame):
            raise ValueError("Incompatible immutable A/B partition; refusing overwrite")
        return frame,prior
    out.mkdir(parents=True)
    frame.to_parquet(out/"manifest.parquet",index=False,compression="zstd")
    metadata["manifest_sha256"] = sha256(out/"manifest.parquet")
    write_json(out/"metadata.json",metadata)
    finalize_artifacts(out)
    return frame,metadata

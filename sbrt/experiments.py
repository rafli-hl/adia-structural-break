"""Frozen registry. Phase 1 deliberately cannot execute E08."""
import json
from pathlib import Path
from .provenance import ROOT, canonical_hash

FROZEN_SPEC_SHA256 = "d003f4de1789aeca32d687bf443239fb984438aab9870435d4a9f26cb2c4758a"
PHASE1 = ("E02", "E03", "E04", "E05", "E06", "E07")
DEFAULT_SPEC = ROOT / "research_specs" / "stage2_v1.json"


def load_spec(path=DEFAULT_SPEC):
    spec = json.loads(Path(path).read_text())
    if canonical_hash(spec) != FROZEN_SPEC_SHA256:
        raise ValueError("Specification differs from frozen Astra Stage-2 contract")
    return spec


def execution_order(ids, spec):
    result = []
    def add(name):
        if name not in PHASE1:
            raise ValueError("Phase 1 stops before E08")
        for dependency in spec["experiments"][name]["dependencies"]:
            add(dependency)
        if name not in result:
            result.append(name)
    for name in ids:
        add(name)
    return result

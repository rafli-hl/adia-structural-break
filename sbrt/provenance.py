"""Reproducibility metadata and validated artifact reuse (evaluator only)."""
import hashlib
import json
import os
import platform
from pathlib import Path
import importlib.metadata
import numpy as np
from .data import sha256

ROOT = Path(__file__).resolve().parent.parent


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)


def source_manifest():
    paths = sorted(list((ROOT / "sbrt").glob("*.py")) + list((ROOT / "tests").glob("*.py"))
                   + [ROOT / "run_research.py", ROOT / "requirements.txt"]
                   + list((ROOT / "research_specs").glob("*.json")))
    files = {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}
    return dict(files=files, sha256=canonical_hash(files))


def versions():
    return dict(python=platform.python_version(), platform=platform.platform(),
                thread_environment={name: os.environ.get(name) for name in
                                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                **{name: importlib.metadata.version(name) for name in
                   ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "duckdb")})


def quantiles(values):
    a = np.asarray(values, dtype=float)
    return dict(p50=float(np.quantile(a, .5)), p95=float(np.quantile(a, .95)), max=float(a.max())) if len(a) else None


def peak_memory_bytes():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return None
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if platform.system() == "Darwin" else value * 1024)
    except (ImportError, OSError):
        return None


def finalize_artifacts(directory):
    directory = Path(directory)
    files = {p.relative_to(directory).as_posix(): sha256(p) for p in sorted(directory.rglob("*"))
             if p.is_file() and p.name != "artifacts.sha256.json" and ".partial" not in p.name}
    write_json(directory / "artifacts.sha256.json", dict(files=files, sha256=canonical_hash(files)))
    return files


def verify_artifacts(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "artifacts.sha256.json").read_text())
    if canonical_hash(manifest["files"]) != manifest["sha256"]:
        raise ValueError("Artifact manifest hash mismatch")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
              if p.is_file() and p.name != "artifacts.sha256.json"}
    if actual != set(manifest["files"]):
        raise ValueError("Completed artifact inventory changed or contains partial files")
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    return manifest

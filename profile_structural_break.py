#!/usr/bin/env python3
"""Replacement long-form profiler; keep this script with the sbrt/ package."""
import os
import sys
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
from sbrt.cli import main
if __name__ == "__main__":
    main(["profile", *sys.argv[1:]])

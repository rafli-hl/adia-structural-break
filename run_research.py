#!/usr/bin/env python3
# Set deterministic native-library threading before importing NumPy.
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
from sbrt.cli import main
if __name__ == "__main__":
    main()

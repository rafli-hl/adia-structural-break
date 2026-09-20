"""Crunch Structural Break: single full-training frozen E14, independent of E08."""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from production_e14 import runtime

# @crunch/keep:on
INFER_PARALLELISM = 1


def train(datasets, model_directory_path):
    runtime.train(datasets, model_directory_path)


def infer(datasets, model_directory_path):
    yield from runtime.infer(datasets, model_directory_path)

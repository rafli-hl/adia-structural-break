"""CrunchDAO Structural Break Real-Time: single full-training frozen E08."""
import os

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

from production_e08 import runtime

# @crunch/keep:on
INFER_PARALLELISM = 1


def train(datasets, model_directory_path):
    runtime.train(datasets, model_directory_path)


def infer(datasets, model_directory_path):
    yield from runtime.infer(datasets, model_directory_path)

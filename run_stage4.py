"""Stage-4 CLI: establish single-thread execution before numerical imports."""
import os
for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

if __name__ == "__main__":
    from stage4_research.cli import main
    main()

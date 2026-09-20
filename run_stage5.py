"""Set single-thread native execution before importing numerical libraries."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"

if __name__=="__main__":
    from stage5_research.cli import main
    main()

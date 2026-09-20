import argparse
import json
from pathlib import Path
from .contract import IDS
from .provenance import DEFAULT, validate, freeze, admission
from .pipeline import export_features, run_one, batch, compare


def main():
    parser = argparse.ArgumentParser(description="Frozen Stage-4 E14-E17; no tuning or submission")
    parser.add_argument("command",choices=("validate","freeze","export","run","batch","compare","report"))
    parser.add_argument("--out",type=Path,default=DEFAULT)
    parser.add_argument("--experiment",choices=IDS)
    args = parser.parse_args()
    if args.command=="run":
        if not args.experiment: parser.error("run requires --experiment")
        result = run_one(args.out,args.experiment)
    else:
        result = {"validate":validate,"freeze":freeze,"export":export_features,"batch":batch,
                  "compare":compare,"report":compare}[args.command](args.out)
    # Full machine-readable reports stay on disk; keep terminal output small.
    summary = {k:result[k] for k in ("status","experiment_id","pooled_oof_ts_auc","final_reference","source_unchanged","config_sha256","spec_sha256","successful","suites") if k in result}
    print(json.dumps(summary,indent=2),flush=True)

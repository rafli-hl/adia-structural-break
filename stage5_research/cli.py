import argparse
import json
from pathlib import Path
from .provenance import DEFAULT, partition, validate, freeze
from .contract import IDS


def main():
    parser=argparse.ArgumentParser(description="Frozen Stage-5 E18-E21; no modeling options")
    parser.add_argument("command",choices=("partition","validate","freeze","export","run","batch","compare","report"))
    parser.add_argument("--out",type=Path,default=DEFAULT)
    parser.add_argument("--experiment",choices=IDS)
    args=parser.parse_args()
    if args.command=="partition": result=partition(args.out)[1]
    elif args.command=="validate": result=validate(args.out)
    elif args.command=="freeze": result=freeze(args.out)
    else:
        from .pipeline import export_features,run_one,batch,compare,report
        if args.command=="export": result=export_features(args.out)
        elif args.command=="run":
            if args.experiment is None: parser.error("run requires --experiment")
            result=run_one(args.out,args.experiment)
        elif args.command=="batch": result=batch(args.out)
        elif args.command=="compare": result=compare(args.out)
        else: result=report(args.out)
    print(json.dumps({k:result[k] for k in ("status","final_reference","source_sha256","manifest_sha256","pooled_oof_ts_auc") if k in result},indent=2),flush=True)

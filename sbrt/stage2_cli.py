"""Phase-1 CLI dispatch; every newly executed experiment gets a fresh process."""
from pathlib import Path
import json
from .experiments import DEFAULT_SPEC, PHASE1

COMMANDS = ("folds", "rescore", "experiment", "batch", "compare", "_execute")


def add_commands(sub):
    folds = sub.add_parser("folds", help="Create or verify immutable CV-v2")
    folds.add_argument("--cache", required=True)
    folds.add_argument("--out", required=True)
    folds.add_argument("--version", choices=["cv2"], default="cv2")
    folds.add_argument("--seed", type=int, default=20260919)
    for name in COMMANDS[1:]:
        parser = sub.add_parser(name)
        parser.add_argument("--cache", required=True)
        parser.add_argument("--folds", required=True)
        parser.add_argument("--out", required=True)
        parser.add_argument("--spec", default=str(DEFAULT_SPEC))
        parser.add_argument("--memory", default="1GB")
        if name in ("experiment", "_execute"):
            parser.add_argument("--id", required=True, choices=list(PHASE1))
        elif name == "batch":
            parser.add_argument("--ids", nargs="+", required=True, choices=list(PHASE1))
        elif name == "rescore":
            parser.add_argument("--oof", required=True)


def dispatch(args):
    if args.command == "folds":
        from .folds import create_v2
        _, metadata = create_v2(args.cache, args.out, args.seed)
        print(json.dumps(metadata, indent=2))
        return
    from .stage2 import rescore_e01, run_batch, run_one
    if args.command == "rescore":
        rescore_e01(args.cache, args.folds, args.spec, args.oof, args.out, args.memory)
    elif args.command == "_execute":
        run_one(args.cache, args.folds, args.spec, args.id, args.out, args.memory)
    elif args.command in ("experiment", "batch"):
        ids = [args.id] if args.command == "experiment" else args.ids
        run_batch(args.cache, args.folds, args.spec, ids, args.out, args.memory)
    elif args.command == "compare":
        from .comparison import run_comparisons
        run_comparisons(args.cache, args.folds, args.spec, args.out, args.memory)

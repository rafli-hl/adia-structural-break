"""Reproducible frozen Stage-3 CLI; no modeling options are exposed."""
from . import stage3

COMMANDS = ("stage3-freeze", "stage3-export", "stage3-run", "stage3-batch", "stage3-compare")


def add_commands(sub):
    for name in COMMANDS:
        parser = sub.add_parser(name, help=__doc__)
        parser.add_argument("--out", required=True)
        if name == "stage3-freeze":
            for key in ("cache", "folds", "validation"):
                parser.add_argument("--"+key, required=True)
        if name == "stage3-run":
            parser.add_argument("--id", choices=stage3.IDS, required=True)


def dispatch(args):
    if args.command == "stage3-freeze":
        return stage3.freeze(args.out, args.cache, args.folds, args.validation)
    if args.command == "stage3-export":
        return stage3.export_features(args.out)
    if args.command == "stage3-run":
        return stage3.run_one(args.out, args.id)
    if args.command == "stage3-batch":
        return stage3.batch(args.out)
    if args.command == "stage3-compare":
        return stage3.compare(args.out)
    raise ValueError("Unknown Stage-3 command")

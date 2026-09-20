"""Local entry points; data never leave the machine."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import prepare, connect
from .profile import run_profile, write_json
from .backtest import run_backtest, e00
from .detectors import E01


def make_synthetic(out, n_series=30):
    """Integration fixture, not a model-selection dataset."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260918)
    rows, targets, metadata = [], [], []
    shared = rng.normal(size=128).astype(np.float32)
    for i in range(n_series):
        sid = f"synthetic_{i:03d}"
        h = shared.copy() if i < 2 else rng.normal(size=96 + i % 64).astype(np.float32)
        n = 24 + i % 24
        tau = (i * 7) % n if i % 2 else -1
        online = rng.normal(size=n).astype(np.float32)
        if tau >= 0:
            if i % 3 == 0:
                online[tau:] *= 3
            else:
                online[tau:] += 2
        for t, value in enumerate(h):
            rows.append((sid, t, value, 1))
        for t, value in enumerate(online):
            rows.append((sid, len(h) + t, value, 2))
            targets.append((sid, len(h) + t, int(tau >= 0 and t >= tau)))
        metadata.append((sid, tau, len(h) + tau if tau >= 0 else -1))
    X = pd.DataFrame(rows, columns=["id", "time", "value", "period"])
    X["value"] = X.value.astype(np.float32)
    X["period"] = X.period.astype(np.int64)
    # Physical interleaving while preserving each series' observation order.
    X.sort_values(["time", "id"], kind="stable").set_index(["id", "time"]).to_parquet(out / "X_train.parquet")
    pd.DataFrame(targets, columns=["id", "time", "target"]).sample(frac=1, random_state=7).set_index(["id", "time"]).to_parquet(out / "y_train.parquet")
    pd.DataFrame(metadata, columns=["id", "tau_index", "tau"]).set_index("id").to_parquet(out / "y_train_index.parquet")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    from .stage2_cli import add_commands, dispatch, COMMANDS
    add_commands(sub)
    from .e08 import add_commands as add_e08_commands, dispatch as dispatch_e08
    add_e08_commands(sub)
    from .stage3_cli import add_commands as add_stage3, dispatch as dispatch_stage3, COMMANDS as STAGE3
    add_stage3(sub)
    profile = sub.add_parser("profile")
    profile.add_argument("--data-dir", default=".")
    profile.add_argument("--x")
    profile.add_argument("--y")
    profile.add_argument("--index")
    profile.add_argument("--cache", required=True)
    profile.add_argument("--out", required=True)
    profile.add_argument("--memory", default="1GB")
    profile.add_argument("--near-cap", type=int, default=2000)
    backtest = sub.add_parser("backtest")
    backtest.add_argument("--cache", required=True)
    backtest.add_argument("--out", required=True)
    backtest.add_argument("--memory", default="1GB")
    backtest.add_argument("--seed", type=int, default=20260918)
    backtest.add_argument("--groups", help="Local complete id,group CSV, combined with detected groups")
    backtest.add_argument("--detectors", nargs="+", default=list(E01), choices=list(E01) + ["constant", "age_only"])
    controls = sub.add_parser("e00")
    controls.add_argument("--out", required=True)
    synth = sub.add_parser("make-synthetic")
    synth.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.command in STAGE3:
        return dispatch_stage3(args)
    if args.command in ("e08", "e08-compare"):
        return dispatch_e08(args)
    if args.command in COMMANDS:
        return dispatch(args)
    if args.command == "profile":
        root = Path(args.data_dir)
        con, source = prepare(args.x or root / "X_train.parquet", args.y or root / "y_train.parquet",
                              args.index or root / "y_train_index.parquet", args.cache, args.memory)
        try:
            result = run_profile(con, source, args.cache, args.out, near_cap=args.near_cap)
            print(json.dumps({"series": result["counts"]["series"], "diagnostics": args.out}))
        finally:
            con.close()
    elif args.command == "backtest":
        con = connect(args.cache, args.memory)
        try:
            run_backtest(con, args.cache, args.out, args.detectors, args.seed, args.groups)
        finally:
            con.close()
    elif args.command == "e00":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        result = e00()
        write_json(args.out, result)
        print(json.dumps(result))
    elif args.command == "make-synthetic":
        make_synthetic(args.out)
        print("Wrote synthetic integration fixtures; not competition data.")


if __name__ == "__main__":
    main()

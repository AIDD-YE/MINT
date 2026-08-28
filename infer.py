import argparse
import json
import os

from mint_engine.cli import build_parser
from mint_engine.engine import run_pipeline


EXPERIMENTS = [
    ("pocketnet_distill_anneal_v2", "pocketnet", "MINT D+A"),
    ("pocketnet_distill_supervised_v2_woa", "pocketnet", "MINT w/o A"),
    ("pocketnet_supervised_v2_wod", "pocketnet", "MINT w/o D"),
    ("mlp_distill_anneal_v2", "mlp", "MLP D+A"),
    ("mlp_distill_supervised_v2_woa", "mlp", "MLP w/o A"),
    ("mlp_supervised_v2_baseline", "mlp", "MLP w/o D"),
]


def run_single(args):
    required = ["exp", "kind", "ckpt_dir", "manifest", "prot_dir", "lig_dir", "out_dir"]
    missing = [name for name in required if not getattr(args, name, None)]
    if missing:
        raise SystemExit("missing required arguments: %s" % ", ".join("--" + name.replace("_", "-") for name in missing))
    return run_pipeline(args)


def run_all(args):
    fields = ["ckpt_dir", "manifest", "prot_dir", "lig_dir", "out_dir",
              "batch_size", "device", "verify_arch", "write_integrity", "workers", "quiet", "folds"]
    summary = []
    for exp, kind, label in EXPERIMENTS:
        print("=" * 70)
        print("  %s  (%s)" % (exp, label))
        print("=" * 70)
        child = argparse.Namespace()
        for name in fields:
            setattr(child, name, getattr(args, name, None))
        child.exp = exp
        child.kind = kind
        child.ckpt_dir = os.path.join(args.ckpt_dir, exp) if getattr(args, "ckpt_dir", None) else None
        code = run_pipeline(child)
        summary.append({"exp": exp, "kind": kind, "exit": code})
    out_dir = getattr(args, "out_dir", None) or "."
    with open(os.path.join(out_dir, "run_all_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    return 0


def main(argv=None):
    parser = build_parser()
    parser.add_argument("--run-all", action="store_true", help="run all experiments sequentially")
    args = parser.parse_args(argv)
    if getattr(args, "run_all", False):
        return run_all(args)
    return run_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
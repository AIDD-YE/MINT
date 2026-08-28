import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RunOptions:
    exp: str
    kind: str
    ckpt_dir: str
    manifest: str
    prot_dir: str
    lig_dir: str
    out_dir: str
    batch_size: int
    device: Optional[str]
    verify_arch: bool
    write_integrity: bool
    folds: Tuple[int, ...]
    workers: int
    quiet: bool

    def validate(self) -> None:
        if self.kind not in ("pocketnet", "mlp"):
            raise ValueError("kind must be one of: pocketnet, mlp")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.workers < 0:
            raise ValueError("workers must be non-negative")
        if len(self.folds) == 0:
            raise ValueError("at least one fold must be selected")
        if len(set(self.folds)) != len(self.folds):
            raise ValueError("fold list must not contain duplicates")
        for path in (self.ckpt_dir, self.manifest, self.prot_dir, self.lig_dir, self.out_dir):
            if not path:
                raise ValueError("all path arguments must be provided")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infer.py",
        description="minimal inference over pre-encoded features with trained checkpoints",
    )
    parser.add_argument("--exp", required=False, default=None, help="experiment label used for output naming")
    parser.add_argument("--kind", choices=["pocketnet", "mlp"], required=False, default=None)
    parser.add_argument("--ckpt-dir", required=False, default=None)
    parser.add_argument("--manifest", required=False, default=None)
    parser.add_argument("--prot-dir", required=False, default=None)
    parser.add_argument("--lig-dir", required=False, default=None)
    parser.add_argument("--out-dir", required=False, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--verify-arch", type=int, default=1)
    parser.add_argument("--write-integrity", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    return parser


def options_from_args(args: argparse.Namespace) -> RunOptions:
    opts = RunOptions(
        exp=args.exp,
        kind=args.kind,
        ckpt_dir=args.ckpt_dir,
        manifest=args.manifest,
        prot_dir=args.prot_dir,
        lig_dir=args.lig_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        device=args.device,
        verify_arch=bool(getattr(args, "verify_arch", 1)),
        write_integrity=bool(getattr(args, "write_integrity", 1)),
        folds=tuple(getattr(args, "folds", (1, 2, 3, 4, 5))),
        workers=int(getattr(args, "workers", 0)),
        quiet=bool(getattr(args, "quiet", False)),
    )
    opts.validate()
    return opts
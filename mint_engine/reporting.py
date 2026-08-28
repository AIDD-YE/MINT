from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np

from mint_engine.io_utils import list_pkl_files, path_exists, sha256_file, write_json_atomic


@dataclass
class FoldResult:
    fold: int
    checkpoint: str
    samples: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    agreement: Dict[str, bool] = field(default_factory=dict)
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0))
    probabilities: np.ndarray = field(default_factory=lambda: np.zeros(0))
    seconds: float = 0.0


def print_table(rows: Sequence[Dict[str, Any]], headers: Sequence[str]) -> None:
    rendered = []
    for row in rows:
        rendered.append({header: str(row.get(header, "")) for header in headers})
    widths = {header: max(len(header), max((len(item[header]) for item in rendered), default=0)) for header in headers}
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for item in rendered:
        print("  ".join(item[header].ljust(widths[header]) for header in headers))


def compose_summary(exp: str, kind: str, device: str, results: Sequence[FoldResult], arch: Dict[str, Any], features: Dict[str, Any], manifest: Dict[str, Any], environment: Dict[str, Any]) -> Dict[str, Any]:
    aucs = [result.metrics["auc"] for result in results if result.metrics.get("auc") == result.metrics.get("auc")]
    summary = {
        "exp": exp,
        "kind": kind,
        "device": device,
        "environment": environment,
        "manifest": manifest,
        "feature_stats": features,
        "architecture": arch,
        "folds": [
            {
                "fold": result.fold,
                "samples": result.samples,
                "checkpoint": result.checkpoint,
                "auc": result.metrics.get("auc"),
                "manual_auc": result.metrics.get("manual_auc"),
                "aupr": result.metrics.get("aupr"),
                "accuracy": result.metrics.get("accuracy"),
                "agreement": result.agreement,
                "seconds": round(result.seconds, 3),
            }
            for result in results
        ],
    }
    if aucs:
        summary["mean_auc"] = float(np.mean(aucs))
        summary["std_auc"] = float(np.std(aucs))
    return summary


def collect_integrity(ckpt_dir: str, prot_dir: str, lig_dir: str, folds: Sequence[int]) -> Dict[str, Any]:
    integrity = {"checkpoints": {}, "features": {"protein": {}, "ligand": {}}}
    for fold in folds:
        checkpoint = "%s/fold%d/stage2_best.pt" % (ckpt_dir, fold)
        if path_exists(checkpoint):
            integrity["checkpoints"]["fold%d" % fold] = sha256_file(checkpoint)
        else:
            integrity["checkpoints"]["fold%d" % fold] = None
    for name in list_pkl_files(prot_dir):
        integrity["features"]["protein"][name] = sha256_file("%s/%s" % (prot_dir, name))
    for name in list_pkl_files(lig_dir):
        integrity["features"]["ligand"][name] = sha256_file("%s/%s" % (lig_dir, name))
    return integrity


def dump_summary(path: str, payload: Dict[str, Any]) -> None:
    write_json_atomic(path, payload)


def dump_integrity(path: str, payload: Dict[str, Any]) -> None:
    write_json_atomic(path, payload)
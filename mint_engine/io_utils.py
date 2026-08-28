import csv
import hashlib
import json
import os
import pickle
import tempfile
from typing import Any, Dict, Iterable, Optional, Sequence


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def path_exists(path: str) -> bool:
    return os.path.exists(path)


def regular_file_size(path: str) -> Optional[int]:
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def sha256_file(path: str, block_size: int = 65536) -> Optional[str]:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while True:
                block = handle.read(block_size)
                if not block:
                    break
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def read_pickle(path: str) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def write_json_atomic(path: str, payload: Any) -> None:
    directory = os.path.dirname(path) or "."
    ensure_dir(directory)
    fd, temporary = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, default=_json_default)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def _json_default(value: Any) -> Any:
    import numpy as np
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("cannot serialize %r" % type(value))


def stream_write_csv(path: str, header: Sequence[str], rows: Iterable[Sequence[Any]], precision: int = 10) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            rendered = []
            for value in row:
                if isinstance(value, float):
                    rendered.append("%.*f" % (precision, value))
                else:
                    rendered.append(value)
            writer.writerow(rendered)


def list_pkl_files(directory: str) -> list:
    if not os.path.isdir(directory):
        return []
    return sorted(name for name in os.listdir(directory) if name.endswith(".pkl"))
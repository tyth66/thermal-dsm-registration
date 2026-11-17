
"""Cache exact footprint samples shared by panel-model and geometry stages."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _index_path(path: Path) -> Path:
    return Path(path).with_suffix(".json")


def cache_available(path: Path) -> bool:
    """True when both the NPZ payload and its panel index exist."""
    path = Path(path)
    return path.exists() and _index_path(path).exists()


def write_footprint_cache(path: Path, samples: dict, *, compressed: bool = False) -> None:
    """Persist per-panel visible/thermal (E,N,Z) samples in an NPZ cache.

    Uncompressed is the default: compression spends several seconds of CPU for
    a cache that is read once by the geometry stage.  Set compressed=True only
    when disk footprint matters more than pipeline latency.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    index = {}
    arrays = {}
    for number, (panel_id, entry) in enumerate(sorted(samples.items())):
        index[panel_id] = number
        for kind, prefix in (("visible", "v"), ("thermal", "t")):
            values = entry.get(kind)
            if values is None:
                continue
            east, north, z = values
            arrays[f"{prefix}{number}_e"] = np.asarray(east, dtype=float)
            arrays[f"{prefix}{number}_n"] = np.asarray(north, dtype=float)
            arrays[f"{prefix}{number}_z"] = np.asarray(z, dtype=float)
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)
    _index_path(path).write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")


def read_footprint_cache(path: Path) -> dict:
    path = Path(path)
    index = json.loads(_index_path(path).read_text(encoding="utf-8"))
    result = {}
    with np.load(path) as data:
        for panel_id, number in index.items():
            entry = {}
            for kind, prefix in (("visible", "v"), ("thermal", "t")):
                key = f"{prefix}{number}_e"
                if key not in data:
                    continue
                entry[kind] = (data[f"{prefix}{number}_e"],
                               data[f"{prefix}{number}_n"],
                               data[f"{prefix}{number}_z"])
            result[panel_id] = entry
    return result

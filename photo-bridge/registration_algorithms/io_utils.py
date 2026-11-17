"""Small shared IO and statistics helpers."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stats(values) -> dict:
    array = np.asarray([value for value in values if value is not None], float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"n": 0}
    return {"n": int(len(array)), "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)), "max": float(np.max(array))}


def abs_stats(values) -> dict:
    array = np.abs(np.asarray([value for value in values if value is not None], float))
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"n": 0}
    return {"n": int(len(array)), "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)), "max": float(np.max(array))}

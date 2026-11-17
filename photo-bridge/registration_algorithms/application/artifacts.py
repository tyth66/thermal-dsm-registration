
"""Artifact boundary helpers.

Application code returns domain objects; these helpers are the only place that
knows how to serialize them to JSON/CSV files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def write_json(path: Path, payload: Any, *, indent: int = 1,
               compact: bool = False, allow_nan: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if compact:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                             allow_nan=allow_nan)
    else:
        encoded = json.dumps(payload, ensure_ascii=False, indent=indent,
                             allow_nan=allow_nan)
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)
    return path


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return path
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path

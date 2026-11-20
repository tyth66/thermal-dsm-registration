"""Build world_model.json from standard panels, visible ortho and visible DSM."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from registration_algorithms.application import write_json
from registration_algorithms.visible_world_model import build_visible_world_model

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "analysis/photo-bridge/outputs/visible_world_model"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build visible world model")
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--visible-ortho", type=Path, required=True)
    parser.add_argument("--visible-dsm", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dsm-erode-m", type=float, default=0.35)
    parser.add_argument("--minimum-dsm-pixels", type=int, default=20)
    args = parser.parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_visible_world_model(
        args.panels.resolve(), args.visible_ortho.resolve(),
        args.visible_dsm.resolve(), args.dsm_erode_m, args.minimum_dsm_pixels)
    path = out_dir / "world_model.json"
    text = json.dumps(model, ensure_ascii=False, indent=2, allow_nan=False)
    write_json(path, model, indent=2, allow_nan=False)
    summary = {
        "model": str(path),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "panels": len(model["panels"]),
        "surfaces": len(model["surfaces"]),
        "diagnostics": len(model["diagnostics"]),
        "input_panels": model["input_panel_count"],
        "crs": model["crs"],
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

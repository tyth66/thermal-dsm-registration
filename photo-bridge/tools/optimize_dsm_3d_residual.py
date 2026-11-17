"""Optimize DSM registration and fit the planimetric field.

Vertical component medians are produced by the panel geometry stage from its
shared thermal footprint samples; this stage emits planimetric shifts only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from registration_algorithms.application import (
    PlanimetricInputs, estimate_planimetric, write_csv, write_json,
)
from registration_algorithms.io_utils import load_json, stats

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT = (REPO_ROOT / "analysis/photo-bridge/output/raw_registration"
                  / "run277_gnss_sg5_20261114_verified")
DEFAULT_VISIBLE_DSM = (REPO_ROOT / "analysis/ortho-data"
                       / "visible_test_run_277_station_141/map/dsm.tif")
DEFAULT_THERMAL_DSM = (REPO_ROOT / "analysis/ortho-data"
                       / "thermal_test_run_277_station_141/map/dsm.tif")
DEFAULT_OUT = (REPO_ROOT / "analysis/photo-bridge/output/visualizations"
               / "run277_gnss_sg5_20261114_verified" / "thermal_world_model"
               / "dsm_3d_residual_optimized")
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Optimized DSM 3D residual fit")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--visible-dsm", type=Path, default=DEFAULT_VISIBLE_DSM)
    parser.add_argument("--thermal-dsm", type=Path, default=DEFAULT_THERMAL_DSM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resolution-m", type=float, default=0.1)
    parser.add_argument("--tile-m", type=float, default=40.0,
                        help="requested maximum tile size; adaptive tiling may shrink it")
    parser.add_argument("--step-m", type=float, default=20.0,
                        help="requested maximum step; capped at tile/2")
    parser.add_argument("--highpass-sigma-m", type=float, default=1.5)
    parser.add_argument("--min-cc", type=float, default=0.5)
    parser.add_argument("--max-shift-m", type=float, default=1.5)
    parser.add_argument("--fixed-tiling", action="store_true",
                        help="use the requested tile/step exactly (disable adaptive tiling)")
    parser.add_argument("--erode-m", type=float, default=0.35)
    parser.add_argument("--minimum-pixels", type=int, default=20)
    parser.add_argument("--field-max-distance-m", type=float, default=30.0)
    parser.add_argument("--regularized-kernel-m", type=float, default=None)
    parser.add_argument("--regularized-ridge", type=float, default=0.1)
    parser.add_argument("--fixed-field-support", action="store_true",
                        help="use the requested IDW radius/support distance exactly")
    parser.add_argument("--warp-threads", type=int,
                        default=min(8, os.cpu_count() or 1),
                        help="GDAL warp threads for DSM reprojection (default: 8)")
    parser.add_argument("--ecc-workers", type=int,
                        default=min(8, os.cpu_count() or 1),
                        help="parallel ECC tile workers (default: 8)")
    args = parser.parse_args(argv)

    resolution = args.resolution_m
    result = args.result_dir.resolve()
    visible_model = load_json(result / "world_model.json")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    estimation = estimate_planimetric(PlanimetricInputs(
        visible_dsm=args.visible_dsm.resolve(),
        thermal_dsm=args.thermal_dsm.resolve(),
        visible_world=visible_model,
        resolution_m=resolution,
        tile_m=args.tile_m,
        step_m=args.step_m,
        highpass_sigma_m=args.highpass_sigma_m,
        min_cc=args.min_cc,
        max_shift_m=args.max_shift_m,
        field_max_distance_m=args.field_max_distance_m,
        warp_threads=args.warp_threads,
        ecc_workers=args.ecc_workers,
        adaptive_tiling=not args.fixed_tiling,
        adaptive_field_support=not args.fixed_field_support,
        regularized_kernel_m=args.regularized_kernel_m,
        regularized_ridge=args.regularized_ridge,
    ))
    field = estimation.field
    records = estimation.records
    accepted = estimation.accepted
    outliers = estimation.outliers
    components = estimation.components
    cv_checker = estimation.cv_checker
    cv_quadrant = estimation.cv_quadrant
    cv_block = estimation.cv_block
    joint = estimation.joint_shape
    field_path = out_dir / "planimetric_field.json"
    write_json(field_path, field.to_model())
    write_csv(out_dir / "tiles.csv", [
        {key: value for key, value in record.items() if key != "candidates"}
        for record in records])
    write_json(out_dir / "tile_candidates.json", [{
        "row": record["row"], "column": record["column"],
        "east": record["east"], "north": record["north"],
        "accepted": record["accepted"], "outlier": record["outlier"],
        "candidate_count": record.get("candidate_count", 0),
        "second_margin": record.get("second_margin", 0.0),
        "periodic_ambiguity": record.get("periodic_ambiguity", False),
        "candidates": record.get("candidates", []),
    } for record in records])
    write_csv(out_dir / "component_residuals_3d.csv", components)
    summary = {
        "method": "multi-scale ECC high-pass DSM registration + similarity_regularized_rbf"
                  " trend + IDW residual field; vertical resampled at shifted footprints",
        "grid_shape": list(joint), "resolution_m": resolution,
        "warp_threads": args.warp_threads, "ecc_workers": args.ecc_workers,
        "tile_m": estimation.tiling.tile_m,
        "step_m": estimation.tiling.step_m,
        "tile_px": estimation.tiling.tile_px,
        "step_px": estimation.tiling.step_px,
        "requested_tile_m": args.tile_m,
        "requested_step_m": args.step_m,
        "tiling_attempts": list(estimation.attempts),
        "overlap_extent_m": list(estimation.overlap_extent_m),
        "field_support": {
            "adaptive": estimation.field_support.adaptive,
            "short_overlap_m": estimation.field_support.short_overlap_m,
            "field_max_distance_m": estimation.field_support.max_distance_m,
            "requested_field_max_distance_m": estimation.field_support.requested_max_distance_m,
        },
        "field_max_distance_m": estimation.field_support.max_distance_m,
        "highpass_sigma_m": args.highpass_sigma_m,
        "tiles": len(records), "accepted_tiles": len(accepted) + outliers,
        "outlier_tiles": outliers,
        "used_tiles": len(accepted),
        "tile_candidates": {
            "tiles": len(records),
            "accepted": len([record for record in records if record["accepted"]]),
            "periodic_ambiguity": len([record for record in records
                                        if record.get("periodic_ambiguity")]),
            "unique": len([record for record in records
                           if record.get("unique", True)]),
            "initial_consistent": len([record for record in records
                                       if record.get("initial_consistent", True)]),
            "candidate_count": stats([record.get("candidate_count", 0) for record in records]),
            "second_margin": stats([record.get("second_margin", 0.0) for record in records]),
        },
        "registration_cc": stats([record["cc"] for record in accepted]),
        "ncc_improvement": stats([record["ncc_after"] - record["ncc_before"]
                                  for record in accepted]),
        "shift_thermal_vs_visible_m": {
            "dx": stats([record["dx_m"] for record in accepted]),
            "dy": stats([record["dy_m"] for record in accepted]),
            "magnitude": stats([record["magnitude_m"] for record in accepted]),
        },
        "planimetric_field_model": "similarity_regularized",
        "polynomial_coefficients": {
            "dx": field.cx.tolist(),
            "dy": field.cy.tolist(),
            "normalization": field.to_model()["normalization"],
            "basis": ["1", "e", "n", "e2", "en", "n2"],
        },
        "regularized_residual": {
            "kernel_length_m": field.kernel_length_m,
            "ridge": field.ridge,
            "centers": len(field.centers),
        },
        "planimetric_field": str(field_path),
        "cross_validation": {
            "checkerboard": cv_checker, "quadrant": cv_quadrant,
            "block": cv_block,
        },
        "vertical_sampling": "deferred_to_panel_thermal_model",
        "caveats": [
            "registration is still product-level; validate with independent thermal edges",
            "regularized RBF residual decays outside the sampled tile support",
        ],
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

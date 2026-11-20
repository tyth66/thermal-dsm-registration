"""Current production pipeline: visible world -> planimetric field -> position mapping."""
from __future__ import annotations

import argparse
import importlib.util
import io
import contextlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from registration_algorithms.application import (
    FinalPipelineInputs, PlanimetricInputs, run_final_pipeline,
)
from registration_algorithms.io_utils import load_json

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS = REPO_ROOT / "analysis/photo-bridge/tools"
DEFAULT_VISIBLE_DSM = (REPO_ROOT / "analysis/ortho-data"
                       / "visible_test_run_277_station_141/map/dsm.tif")
DEFAULT_THERMAL_DSM = (REPO_ROOT / "analysis/ortho-data"
                       / "thermal_test_run_277_station_141/map/dsm.tif")
DEFAULT_VISIBLE_ORTHO = (REPO_ROOT / "analysis/ortho-data"
                         / "visible_test_run_277_station_141/map/result.tif")
DEFAULT_THERMAL_ORTHO = (REPO_ROOT / "analysis/ortho-data"
                         / "thermal_test_run_277_station_141/map/result.tif")
PLANIMETRIC_RESOLUTION_M = 0.1
PLANIMETRIC_TILE_M = 40.0
PLANIMETRIC_STEP_M = 20.0
PLANIMETRIC_HIGHPASS_SIGMA_M = 1.5
PLANIMETRIC_FIELD_MAX_DISTANCE_M = 30.0
PLANIMETRIC_WORKERS = min(8, os.cpu_count() or 1)


@dataclass(frozen=True)
class PipelineConfig:
    result_dir: Path
    mapping_root: Path
    visible_dsm: Path
    thermal_dsm: Path
    visible_ortho: Path
    thermal_ortho: Path
    out_dir: Path
    panels_json: Path | None = None
    grid_step: int = 16
    planimetric_resolution_m: float = PLANIMETRIC_RESOLUTION_M
    planimetric_tile_m: float = PLANIMETRIC_TILE_M
    planimetric_step_m: float = PLANIMETRIC_STEP_M
    planimetric_highpass_sigma_m: float = PLANIMETRIC_HIGHPASS_SIGMA_M
    planimetric_field_max_distance_m: float = PLANIMETRIC_FIELD_MAX_DISTANCE_M
    planimetric_workers: int = PLANIMETRIC_WORKERS
    adaptive_planimetric_tiling: bool = True
    adaptive_planimetric_field: bool = True
    planimetric_regularized_kernel_m: float | None = None
    planimetric_regularized_ridge: float = 0.1
    panel_mapping: str = "rigid"

    def __post_init__(self):
        if self.grid_step < 1:
            raise ValueError("grid_step must be positive")
        if self.planimetric_workers < 1:
            raise ValueError("planimetric_workers must be positive")
        if not self.planimetric_regularized_ridge > 0.0:
            raise ValueError("planimetric_regularized_ridge must be positive")
        # Same contract as PlanimetricInputs: every scale parameter must be a
        # strictly positive number (NaN included).
        for name in ("planimetric_resolution_m", "planimetric_tile_m",
                     "planimetric_step_m", "planimetric_highpass_sigma_m",
                     "planimetric_field_max_distance_m"):
            if not getattr(self, name) > 0:
                raise ValueError("%s must be positive" % name)


def _build_world_model(config: PipelineConfig) -> Path:
    world_dir = config.out_dir / "visible_world_model"
    world_path = world_dir / "world_model.json"
    if config.panels_json is None:
        candidate = config.result_dir / "world_model.json"
        if candidate.exists():
            return candidate
        candidate = config.result_dir / "visible_world_model" / "world_model.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError("world_model.json not found under %s" % config.result_dir)
    spec = importlib.util.spec_from_file_location(
        "pipeline_stage_build_visible_world_model",
        TOOLS / "build_visible_world_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = module.main([
            "--panels", str(config.panels_json),
            "--visible-ortho", str(config.visible_ortho),
            "--visible-dsm", str(config.visible_dsm),
            "--out-dir", str(world_dir),
        ])
    if code not in (0, None):
        raise RuntimeError("visible_world_model failed with code %s" % code)
    return world_path


def run_pipeline(config: PipelineConfig) -> dict:
    world_path = _build_world_model(config)
    world = load_json(world_path)
    mapping_out = config.out_dir / "visible_to_thermal_mapping"
    result = run_final_pipeline(FinalPipelineInputs(
        world=world,
        planimetric=PlanimetricInputs(
            visible_dsm=config.visible_dsm,
            thermal_dsm=config.thermal_dsm,
            visible_world=world,
            resolution_m=config.planimetric_resolution_m,
            tile_m=config.planimetric_tile_m,
            step_m=config.planimetric_step_m,
            highpass_sigma_m=config.planimetric_highpass_sigma_m,
            field_max_distance_m=config.planimetric_field_max_distance_m,
            warp_threads=config.planimetric_workers,
            ecc_workers=config.planimetric_workers,
            adaptive_tiling=config.adaptive_planimetric_tiling,
            adaptive_field_support=config.adaptive_planimetric_field,
            regularized_kernel_m=config.planimetric_regularized_kernel_m,
            regularized_ridge=config.planimetric_regularized_ridge,
        ),
        visible_ortho=config.visible_ortho,
        thermal_ortho=config.thermal_ortho,
        output_json=mapping_out / "visible_to_thermal_mapping_result.json",
        grid_step=config.grid_step,
        panel_mapping=config.panel_mapping,
    ))
    return {
        "pipeline": "visible_world_planimetric_position_mapping",
        "execution": "in_memory",
        "result_dir": str(config.result_dir),
        "mapping_root": str(config.mapping_root),
        "mapping_out": str(mapping_out),
        "final_json": str(result.final_json),
        "stages": list(result.stages),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Visible world planimetric position mapping pipeline")
    parser.add_argument("--result-dir", type=Path, required=True,
                        help="directory containing world_model.json")
    parser.add_argument("--mapping-root", type=Path, default=None)
    parser.add_argument("--panels", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--visible-dsm", type=Path, default=DEFAULT_VISIBLE_DSM)
    parser.add_argument("--thermal-dsm", type=Path, default=DEFAULT_THERMAL_DSM)
    parser.add_argument("--visible-ortho", type=Path, default=DEFAULT_VISIBLE_ORTHO)
    parser.add_argument("--thermal-ortho", type=Path, default=DEFAULT_THERMAL_ORTHO)
    parser.add_argument("--grid-step", type=int, default=16)
    parser.add_argument("--panel-mapping", choices=("pointwise", "rigid", "similarity"),
                        default="rigid")
    parser.add_argument("--planimetric-resolution-m", type=float, default=PLANIMETRIC_RESOLUTION_M)
    parser.add_argument("--planimetric-tile-m", type=float, default=PLANIMETRIC_TILE_M)
    parser.add_argument("--planimetric-step-m", type=float, default=PLANIMETRIC_STEP_M)
    parser.add_argument("--planimetric-highpass-sigma-m", type=float, default=PLANIMETRIC_HIGHPASS_SIGMA_M)
    parser.add_argument("--planimetric-field-max-distance-m", type=float, default=PLANIMETRIC_FIELD_MAX_DISTANCE_M)
    parser.add_argument("--planimetric-workers", type=int, default=PLANIMETRIC_WORKERS)
    parser.add_argument("--fixed-tiling", action="store_false", dest="adaptive_planimetric_tiling")
    parser.add_argument("--fixed-field-support", action="store_false", dest="adaptive_planimetric_field")
    parser.add_argument("--regularized-kernel-m", type=float, default=None)
    parser.add_argument("--regularized-ridge", type=float, default=0.1)
    args = parser.parse_args(argv)
    mapping_root = (args.mapping_root or args.out_dir or Path.cwd() / "mapping").resolve()
    out_dir = (args.out_dir or mapping_root / "pipeline").resolve()
    config = PipelineConfig(
        result_dir=args.result_dir.resolve(), mapping_root=mapping_root,
        visible_dsm=args.visible_dsm.resolve(), thermal_dsm=args.thermal_dsm.resolve(),
        visible_ortho=args.visible_ortho.resolve(), thermal_ortho=args.thermal_ortho.resolve(),
        panels_json=(args.panels.resolve() if args.panels else None),
        out_dir=out_dir, grid_step=args.grid_step,
        planimetric_resolution_m=args.planimetric_resolution_m,
        planimetric_tile_m=args.planimetric_tile_m,
        planimetric_step_m=args.planimetric_step_m,
        planimetric_highpass_sigma_m=args.planimetric_highpass_sigma_m,
        planimetric_field_max_distance_m=args.planimetric_field_max_distance_m,
        planimetric_workers=args.planimetric_workers,
        adaptive_planimetric_tiling=args.adaptive_planimetric_tiling,
        adaptive_planimetric_field=args.adaptive_planimetric_field,
        planimetric_regularized_kernel_m=args.regularized_kernel_m,
        planimetric_regularized_ridge=args.regularized_ridge,
        panel_mapping=args.panel_mapping,
    )
    summary = run_pipeline(config)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

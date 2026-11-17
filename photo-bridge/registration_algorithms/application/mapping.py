
"""Mapping use case: canonical residual model -> thermal panel/grid samples.

This module computes results only; CSV/PNG/JSON serialization stays in tools.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rasterio

from ..domain import CanonicalResidualModel
from ..forward_mapping import grid_mapping, panel_points
from ..io_utils import stats


@dataclass(frozen=True)
class MappingInputs:
    visible_world: Mapping[str, Any]
    canonical: CanonicalResidualModel
    visible_ortho: Path
    visible_dsm: Path
    thermal_ortho: Path
    residual_model_path: Path
    grid_step: int = 16
    max_grid_rows: int = 50000
    field_max_distance_m: float | None = None
    panel_mapping: str = "rigid"


@dataclass(frozen=True)
class MappingComputation:
    panel_rows: tuple[dict, ...]
    grid: Mapping[str, np.ndarray]
    summary: dict
    thermal_resolution: float


def compute_mapping(inputs: MappingInputs) -> MappingComputation:
    field = inputs.canonical.planimetric
    panels = inputs.canonical.mapping_panels()
    with (rasterio.open(inputs.visible_ortho) as visible_ortho,
          rasterio.open(inputs.visible_dsm) as visible_dsm,
          rasterio.open(inputs.thermal_ortho) as thermal_ortho):
        alpha_stride = 1
        thermal_alpha = thermal_ortho.read(4)
        panel_rows = panel_points(
            inputs.visible_world, panels, field, visible_ortho.transform,
            thermal_ortho.transform, thermal_ortho.width, thermal_ortho.height,
            thermal_alpha, alpha_stride, inputs.field_max_distance_m,
            panel_mapping=inputs.panel_mapping)
        grid_height = visible_ortho.height // inputs.grid_step
        grid_width = visible_ortho.width // inputs.grid_step
        visible_z_grid = visible_dsm.read(
            1, out_shape=(grid_height, grid_width),
            resampling=rasterio.enums.Resampling.average).astype(float).ravel()
        grid = grid_mapping(
            inputs.visible_world, panels, field, visible_ortho.transform,
            visible_ortho.width, visible_ortho.height, visible_z_grid,
            thermal_ortho.transform, thermal_ortho.width, thermal_ortho.height,
            thermal_alpha, alpha_stride, inputs.grid_step,
            inputs.field_max_distance_m)
        thermal_resolution = abs(thermal_ortho.transform.a)
    panel_grouped = {}
    for row in panel_rows:
        panel_grouped.setdefault(row["panel_id"], []).append(row)
    valid_panel = [row for row in panel_rows if row["valid"]]
    fully_valid_panel_ids = {panel_id for panel_id, rows in panel_grouped.items()
                             if len(rows) == 5 and all(row["valid"] for row in rows)}
    all_original_nodata_panel_ids = {
        panel_id for panel_id, rows in panel_grouped.items()
        if len(rows) == 5 and not any(row["naive_valid"] for row in rows)
    }
    valid_grid = grid["valid"]
    panel_mask = grid["vertical_source"] != "ground"
    grid_size = len(grid["east"])
    summary = {
        "method": "visible ortho -> visible world surface -> 3D residual -> thermal world -> thermal ortho",
        "visible_ortho": str(inputs.visible_ortho),
        "thermal_ortho": str(inputs.thermal_ortho),
        "residual_model": str(inputs.residual_model_path),
        "grid_step_px": inputs.grid_step,
        "grid_points": int(grid_size),
        "panel_points": len(panel_rows),
        "panels_total": len(inputs.visible_world["panels"]),
        "panel_ids_total": len(inputs.visible_world["panels"]),
        "panels_with_vertical_model": sum(
            bool(model.get("visible_plane") is not None or model.get("thermal_plane") is not None)
            for model in panels.values()),
        "panel_points_valid": len(valid_panel),
        "panel_points_corrected_valid": sum(row["corrected_valid"] for row in panel_rows),
        "panel_points_unsupported": sum(not row["field_valid"] for row in panel_rows),
        "panel_points_target_nodata": sum(row["target_nodata"] for row in panel_rows),
        "panel_target_nodata_count": sum(row["target_nodata"] for row in panel_rows),
        "panel_ids_valid": len(fully_valid_panel_ids),
        "panel_ids_all_original_nodata": len(all_original_nodata_panel_ids),
        "panel_ids_incomplete": sum(len(rows) != 5 for rows in panel_grouped.values()),
        "field_max_distance_m": (field.max_distance_m if inputs.field_max_distance_m is None
                                 else inputs.field_max_distance_m),
        "mapping_policy": (
            "map all five points when any of the five original positions has thermal alpha > 0; "
            "exclude only components whose five original positions are all nodata"),
        "diagnostics_only": ["field_valid", "corrected_valid", "target_nodata"],
        "grid_naive_valid_fraction": float(grid["naive_valid"].mean()),
        "grid_corrected_valid_fraction": float(grid["corrected_valid"].mean()),
        "grid_valid_fraction": float(valid_grid.mean()),
        "grid_mapped_fraction": float(valid_grid.mean()),
        "grid_unsupported_fraction": float((~grid["field_valid"]).mean()),
        "grid_target_nodata_fraction": float(grid["target_nodata"].mean()),
        "grid_valid_points": int(valid_grid.sum()),
        "grid_unsupported_points": int((~grid["field_valid"]).sum()),
        "grid_target_nodata_points": int(grid["target_nodata"].sum()),
        "grid_field_fraction": float(grid["field_valid"].mean()),
        "grid_panel_fraction": float(panel_mask.mean()),
        "panel_field_shift_px": stats([row["field_shift_px"] for row in valid_panel]),
        "panel_residual_applied_count": sum(row["residual_applied"] for row in panel_rows),
        "delta_e_m": stats(grid["delta_e"]),
        "delta_n_m": stats(grid["delta_n"]),
        "delta_z_m": stats(grid["delta_z"]),
        "grid_field_shift_px": stats(
            np.hypot(grid["delta_e"][valid_grid], grid["delta_n"][valid_grid])
            / thermal_resolution),
    }
    return MappingComputation(panel_rows=tuple(panel_rows), grid=grid,
                              summary=summary, thermal_resolution=thermal_resolution)

def nested_panel_points(panel_rows) -> list[dict]:
    """Group point rows into one center/corners object per panel."""
    entries = {}
    order = []
    for row in panel_rows:
        panel_id = row["panel_id"]
        entry = entries.get(panel_id)
        if entry is None:
            entry = {"panel_id": panel_id, "center": None, "corners": {}}
            entries[panel_id] = entry
            order.append(panel_id)
        point = {key: value for key, value in row.items()
                 if key not in ("panel_id", "point")}
        point_name = row["point"]
        if point_name == "center":
            entry["center"] = point
        else:
            entry["corners"][point_name] = point
    result = []
    for panel_id in order:
        entry = entries[panel_id]
        corner_names = sorted(entry["corners"], key=lambda name: int(name.split("_")[-1]))
        result.append({"panel_id": panel_id, "center": entry["center"],
                       "corners": [entry["corners"][name] for name in corner_names]})
    return result


def mapping_result_payload(computation: MappingComputation,
                           schema: str = "visible_to_thermal_mapping_result/v2") -> dict:
    panels = nested_panel_points(computation.panel_rows)
    return {
        "schema": schema,
        "method": computation.summary["method"],
        "summary": computation.summary,
        "panel_count": len(panels),
        "panels": panels,
    }


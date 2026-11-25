
"""Mapping use case: planimetric field -> thermal panel/grid positions.

This module computes results only; CSV/PNG/JSON serialization stays in tools.
The mapping is positions-only: no vertical surface model participates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rasterio

from ..forward_mapping import grid_mapping, panel_points
from ..io_utils import stats
from ..residual_model import PlanimetricField


@dataclass(frozen=True)
class MappingInputs:
    visible_world: Mapping[str, Any]
    field: PlanimetricField
    visible_ortho: Path
    thermal_ortho: Path
    residual_model_path: Path
    grid_step: int = 16
    field_max_distance_m: float | None = None
    field_chunk_workers: int = 2


@dataclass(frozen=True)
class MappingComputation:
    panel_rows: tuple[dict, ...]
    grid: Mapping[str, np.ndarray]
    summary: dict
    thermal_resolution: float


def compute_mapping(inputs: MappingInputs) -> MappingComputation:
    field = inputs.field
    with (rasterio.open(inputs.visible_ortho) as visible_ortho,
          rasterio.open(inputs.thermal_ortho) as thermal_ortho):
        thermal_alpha = thermal_ortho.read(4)
        thermal_transform = thermal_ortho.transform
        thermal_bounds = (
            thermal_transform.c,
            thermal_transform.c + thermal_transform.a * thermal_ortho.width,
            thermal_transform.f + thermal_transform.e * thermal_ortho.height,
            thermal_transform.f,
        )
        panel_rows = panel_points(
            inputs.visible_world, field, visible_ortho.transform,
            thermal_transform, thermal_ortho.width, thermal_ortho.height,
            thermal_alpha, inputs.field_max_distance_m,
            boundary=thermal_bounds,
            field_chunk_workers=inputs.field_chunk_workers)
        grid = grid_mapping(
            inputs.visible_world, field, visible_ortho.transform,
            visible_ortho.width, visible_ortho.height,
            thermal_transform, thermal_ortho.width, thermal_ortho.height,
            thermal_alpha, inputs.grid_step,
            inputs.field_max_distance_m, boundary=thermal_bounds,
            field_chunk_workers=inputs.field_chunk_workers)
        thermal_resolution = abs(thermal_ortho.transform.a)
    row_count = len(panel_rows)
    panel_order = [panel["panel_id"] for panel in inputs.visible_world["panels"]]
    panel_index = {panel_id: index for index, panel_id in enumerate(panel_order)}

    def column(key, dtype):
        return np.fromiter((row[key] for row in panel_rows), dtype=dtype,
                           count=row_count)

    codes = np.fromiter((panel_index[row["panel_id"]] for row in panel_rows),
                        dtype=np.int64, count=row_count)
    valid = column("valid", bool)
    naive_valid = column("naive_valid", bool)
    corrected_valid = column("corrected_valid", bool)
    field_valid = column("field_valid", bool)
    conservative_valid = np.fromiter(
        (row.get("conservative_valid", 0) for row in panel_rows),
        dtype=bool, count=row_count)
    component_conservative = np.fromiter(
        (row.get("component_conservative_valid", 0) for row in panel_rows),
        dtype=bool, count=row_count)
    edge_distance = column("edge_distance_m", float)
    edge_tail = column("edge_tail_applied", bool)
    panel_count = len(panel_order)
    counts = np.bincount(codes, minlength=panel_count)
    valid_counts = np.bincount(codes, weights=valid, minlength=panel_count)
    conservative_counts = np.bincount(
        codes, weights=component_conservative, minlength=panel_count)
    naive_counts = np.bincount(codes, weights=naive_valid, minlength=panel_count)
    fully_valid_panel_ids = np.flatnonzero(
        (counts == 5) & (valid_counts == 5))
    conservative_panel_ids = np.flatnonzero(
        (counts == 5) & (conservative_counts > 0))
    all_original_nodata_panel_ids = np.flatnonzero(
        (counts == 5) & (naive_counts == 0))
    edge_mask = edge_distance < 40.0
    valid_grid = grid["valid"]
    panel_mask = grid["panel_mask"]
    grid_size = len(grid["east"])
    summary = {
        "method": "visible ortho -> planimetric field -> thermal ortho",
        "visible_ortho": str(inputs.visible_ortho),
        "thermal_ortho": str(inputs.thermal_ortho),
        "residual_model": str(inputs.residual_model_path),
        "grid_step_px": inputs.grid_step,
        "grid_points": int(grid_size),
        "panel_points": len(panel_rows),
        "panels_total": len(inputs.visible_world["panels"]),
        "panel_points_valid": int(valid.sum()),
        "panel_points_conservative_valid": int(conservative_valid.sum()),
        "panel_points_corrected_valid": int(corrected_valid.sum()),
        "panel_points_unsupported": int((~field_valid).sum()),
        "panel_points_target_nodata": int((~corrected_valid).sum()),
        "panel_ids_valid": len(fully_valid_panel_ids),
        "panel_ids_conservative_valid": len(conservative_panel_ids),
        "panel_ids_all_original_nodata": len(all_original_nodata_panel_ids),
        "panel_ids_incomplete": int(np.count_nonzero(counts != 5)),
        "edge_band_m": 2.0 * float(getattr(field, "tile_m", 20.0)),
        "edge_tail_points": int(edge_tail[edge_mask].sum()),
        "grid_edge_tail_points": int(np.count_nonzero(grid.get("edge_tail_applied", []))),
        "field_max_distance_m": (field.max_distance_m if inputs.field_max_distance_m is None
                                 else inputs.field_max_distance_m),
        "mapping_policy": MAPPING_POLICY,
        "diagnostics_only": ["field_valid", "corrected_valid", "target_nodata"],
        "vertical_model_removed": True,
        "grid_naive_valid_fraction": float(grid["naive_valid"].mean()),
        "grid_corrected_valid_fraction": float(grid["corrected_valid"].mean()),
        "grid_valid_fraction": float(valid_grid.mean()),
        "grid_unsupported_fraction": float((~grid["field_valid"]).mean()),
        "grid_target_nodata_fraction": float(grid["target_nodata"].mean()),
        "grid_valid_points": int(valid_grid.sum()),
        "grid_unsupported_points": int((~grid["field_valid"]).sum()),
        "grid_target_nodata_points": int(grid["target_nodata"].sum()),
        "grid_field_fraction": float(grid["field_valid"].mean()),
        "grid_panel_fraction": float(panel_mask.mean()),
        "panel_field_shift_px": stats(column("field_shift_px", float)[valid]),
        "panel_residual_applied_count": int(column("residual_applied", int).sum()),
        "delta_e_m": stats(grid["delta_e"]),
        "delta_n_m": stats(grid["delta_n"]),
        "grid_field_shift_px": stats(
            np.hypot(grid["delta_e"][valid_grid], grid["delta_n"][valid_grid])
            / thermal_resolution),
    }
    return MappingComputation(panel_rows=tuple(panel_rows), grid=grid,
                              summary=summary, thermal_resolution=thermal_resolution)

MAPPING_POLICY = (
    "only components with thermal alpha > 0 at any of their five original "
    "positions participate in the mapping; components whose five original "
    "positions are all nodata are excluded from the delivered payload")


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
                           schema: str = "visible_to_thermal_mapping_result/v3") -> dict:
    panels = nested_panel_points(computation.panel_rows)
    # Policy: components whose five original (uncorrected) thermal positions
    # are all nodata do not participate in the mapping.  They are dropped from
    # the delivered payload entirely; input-side counts stay in the summary so
    # the exclusion remains auditable downstream.
    included = [panel for panel in panels
                if panel["center"] is not None and panel["center"].get("valid", 0) == 1]
    excluded = len(panels) - len(included)
    # Stage provenance assembled by the caller (e.g. the planimetric block from
    # run_final_pipeline) passes through untouched; only payload-level counts
    # and the mapping policy are added here.
    summary = dict(computation.summary)
    summary["panel_ids_excluded_original_nodata"] = excluded
    summary["mapping_policy"] = MAPPING_POLICY
    return {
        "schema": schema,
        "method": computation.summary["method"],
        "summary": summary,
        "panel_count": len(included),
        "panels": included,
    }

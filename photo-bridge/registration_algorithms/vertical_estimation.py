"""Component-level vertical residual sampling and per-array plane fitting."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from .residual_model import PlanimetricField
from .dsm_sampling import sample_footprint
from .io_utils import stats
from .local_plane import LocalPlane, fit_local_plane


def sample_component_vertical(visible_panels, field: PlanimetricField,
                              thermal_dsm_path: Path, erode_m: float,
                              minimum_pixels: int):
    rows = []
    if not visible_panels:
        return rows
    centers = np.asarray([panel["center_xyz"][:2] for panel in visible_panels], float)
    corners = np.asarray([np.asarray(panel["corners_xyz"], float)[:, :2]
                          for panel in visible_panels], float)
    dx, dy, weights, _ = field.evaluate(centers[:, 0], centers[:, 1])
    support_points = np.concatenate([centers[:, None, :], corners], axis=1).reshape(-1, 2)
    support_valid = field.supports_points(support_points).reshape(len(visible_panels), 5)
    mapped_corners = field.map_points(corners.reshape(-1, 2)).reshape(len(visible_panels), 4, 2)
    with rasterio.open(thermal_dsm_path) as thermal_dsm:
        for index, panel in enumerate(visible_panels):
            east, north = (float(value) for value in panel["center_xyz"][:2])
            shift_x, shift_y = float(dx[index]), float(dy[index])
            weight = float(weights[index])
            horizontal_valid = bool(support_valid[index].all())
            footprint = sample_footprint(thermal_dsm, mapped_corners[index], erode_m)
            measured = None
            if (footprint["median_z"] is not None
                    and footprint["valid_pixels"] >= minimum_pixels):
                measured = float(footprint["median_z"]) - float(panel["center_xyz"][2])
            rows.append({
                "panel_id": panel["panel_id"], "array_id": str(panel["array_id"]),
                "east": east, "north": north,
                "delta_e_thermal_vs_visible_m": shift_x,
                "delta_n_thermal_vs_visible_m": shift_y,
                "delta_z_measured_m": measured,
                "dsm_mad_m": footprint["mad_z"],
                "dsm_valid_pixels": footprint["valid_pixels"],
                "horizontal_confidence": weight,
                "horizontal_valid": int(horizontal_valid),
            })
    return rows



def component_shifts(visible_panels, field: PlanimetricField):
    """Planimetric shifts/support for every component, without thermal DSM reads.

    Vertical medians are produced later from the panel-stage thermal samples, so
    the planimetric stage does not sample the thermal DSM a second time.
    """
    rows = []
    if not visible_panels:
        return rows
    centers = np.asarray([panel["center_xyz"][:2] for panel in visible_panels], float)
    corners = np.asarray([np.asarray(panel["corners_xyz"], float)[:, :2]
                          for panel in visible_panels], float)
    dx, dy, weights, _ = field.evaluate(centers[:, 0], centers[:, 1])
    support_points = np.concatenate([centers[:, None, :], corners], axis=1).reshape(-1, 2)
    support_valid = field.supports_points(support_points).reshape(len(visible_panels), 5)
    for index, panel in enumerate(visible_panels):
        east, north = (float(value) for value in panel["center_xyz"][:2])
        rows.append({
            "panel_id": panel["panel_id"], "array_id": str(panel["array_id"]),
            "east": east, "north": north,
            "delta_e_thermal_vs_visible_m": float(dx[index]),
            "delta_n_thermal_vs_visible_m": float(dy[index]),
            "delta_z_measured_m": None,
            "dsm_mad_m": None,
            "dsm_valid_pixels": 0,
            "horizontal_confidence": float(weights[index]),
            "horizontal_valid": int(bool(support_valid[index].all())),
        })
    return rows

def fit_vertical(rows: list[dict]):
    valid = [row for row in rows if row["delta_z_measured_m"] is not None]
    by_array = {}
    for row in valid:
        by_array.setdefault(row["array_id"], []).append(row)
    for array_id, items in by_array.items():
        east = np.asarray([item["east"] for item in items])
        north = np.asarray([item["north"] for item in items])
        values = np.asarray([item["delta_z_measured_m"] for item in items])
        design = np.column_stack([east - east.mean(), north - north.mean(),
                                  np.ones(len(items))])
        if len(items) >= 8 and np.linalg.matrix_rank(design) == 3:
            plane = fit_local_plane(east, north, values)
        else:
            plane = LocalPlane(float(np.mean(east)), float(np.mean(north)),
                               float(np.median(values)), 0.0, 0.0)
        for item in items:
            item["delta_z_fitted_m"] = float(plane.evaluate(item["east"], item["north"]))
            item["delta_z_correction_m"] = -item["delta_z_fitted_m"]
            item["vertical_support"] = len(items)
    for row in rows:
        row.setdefault("delta_z_fitted_m", None)
        row.setdefault("delta_z_correction_m", None)
        row.setdefault("vertical_support", 0)
    measured = [row["delta_z_measured_m"] for row in valid]
    residual = [row["delta_z_measured_m"] - row["delta_z_fitted_m"]
                for row in valid if row["delta_z_fitted_m"] is not None]
    return stats(measured), stats(residual)

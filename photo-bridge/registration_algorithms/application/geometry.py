"""Application use case: fit panel-level visible/thermal geometry."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rasterio

from ..dsm_sampling import sample_footprint_points as _samples
from ..local_plane import LocalPlane, fit_local_plane
from ..residual_model import PlanimetricField


def _stats(values) -> dict:
    array = np.asarray([value for value in values if value is not None], float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"n": 0}
    return {"n": int(len(array)), "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)), "max": float(np.max(array))}


def _fit(east, north, values):
    plane = fit_local_plane(east, north, values)
    residual = np.asarray(values, float) - plane.evaluate(east, north)
    return plane, float(np.sqrt(np.mean(residual ** 2)))


def panel_layer(visible_dsm_path: Path, thermal_dsm_path: Path, visible_world: dict,
                component_rows: dict[str, dict], erode_m: float,
                field: PlanimetricField, samples_cache: dict | None = None,
                plane_cache: dict | None = None):
    panels = visible_world["panels"]
    surfaces = visible_world.get("surfaces", {})
    visible_samples = {}
    thermal_samples = {}
    thermal_medians = {}
    if panels:
        mapped_stack = field.map_points(np.asarray(
            [np.asarray(panel["corners_xyz"], float)[:, :2] for panel in panels]
        ).reshape(-1, 2)).reshape(-1, 4, 2)
    else:
        mapped_stack = np.empty((0, 4, 2))
    with (rasterio.open(visible_dsm_path) as visible_dsm,
          rasterio.open(thermal_dsm_path) as thermal_dsm):
        for panel_index, panel in enumerate(panels):
            panel_id = panel["panel_id"]
            corners = np.asarray(panel["corners_xyz"], float)[:, :2]
            cached = (samples_cache or {}).get(panel_id, {})
            samples = cached.get("visible")
            if samples is None:
                samples = _samples(visible_dsm, corners, erode_m)
            if samples is not None:
                visible_samples[panel_id] = samples
            row = component_rows.get(panel_id)
            if row is None:
                continue
            samples = cached.get("thermal")
            if samples is None:
                samples = _samples(thermal_dsm, mapped_stack[panel_index], erode_m)
            if samples is None:
                continue
            thermal_samples[panel_id] = samples
            east, north, values = samples
            thermal_medians.setdefault(str(panel["array_id"]), []).append(
                (float(east.mean()), float(north.mean()), float(np.median(values))))
    thermal_arrays = {}
    for array_id, items in thermal_medians.items():
        east = np.asarray([item[0] for item in items])
        north = np.asarray([item[1] for item in items])
        design = np.column_stack([east - east.mean(), north - north.mean(),
                                  np.ones(len(items))])
        if len(items) >= 5 and np.linalg.matrix_rank(design) == 3:
            values = np.asarray([item[2] for item in items])
            thermal_arrays[array_id] = fit_local_plane(east, north, values)
        else:
            thermal_arrays[array_id] = LocalPlane(
                float(np.mean(east)), float(np.mean(north)),
                float(np.median([item[2] for item in items])), 0.0, 0.0)
    rows = []
    for panel in panels:
        panel_id = panel["panel_id"]
        if panel_id not in visible_samples or panel_id not in thermal_samples:
            continue
        component = component_rows[panel_id]
        array_id = str(panel["array_id"])
        shift = np.array([float(component["delta_e_thermal_vs_visible_m"]),
                          float(component["delta_n_thermal_vs_visible_m"])])
        center_visible = np.asarray(panel["center_xyz"], float)[:2]
        center_thermal = center_visible + shift
        visible_east, visible_north, visible_z = visible_samples[panel_id]
        thermal_east, thermal_north, thermal_z = thermal_samples[panel_id]
        cached_planes = (plane_cache or {}).get(panel_id, {})
        cached_visible = cached_planes.get("visible")
        if cached_visible is not None:
            visible_plane = LocalPlane.from_model(cached_visible["plane"])
            visible_rms = float(cached_visible["rms_m"])
        else:
            visible_plane, visible_rms = _fit(visible_east, visible_north, visible_z)
        cached_thermal = cached_planes.get("thermal")
        if cached_thermal is not None:
            thermal_plane = LocalPlane.from_model(cached_thermal["plane"])
            thermal_rms = float(cached_thermal["rms_m"])
        else:
            thermal_plane, thermal_rms = _fit(thermal_east, thermal_north, thermal_z)
        surface = surfaces.get(array_id, {})
        if surface.get("plane") is not None:
            visible_array = LocalPlane.from_model(surface["plane"])
        elif all(surface.get(key) is not None for key in ("plane_a", "plane_b", "plane_c")):
            visible_array = LocalPlane.from_model(surface)
        else:
            visible_array = visible_plane
        thermal_array = thermal_arrays[array_id]
        visible_center_z = float(visible_plane.evaluate(*center_visible))
        thermal_center_z = float(thermal_plane.evaluate(*center_thermal))
        visible_array_z = float(visible_array.evaluate(*center_visible))
        thermal_array_z = float(thermal_array.evaluate(*center_thermal))
        rows.append({
            "panel_id": panel_id, "array_id": array_id,
            "visible_plane_e0": float(visible_plane.e0),
            "visible_plane_n0": float(visible_plane.n0),
            "visible_plane_z0": float(visible_plane.z0),
            "visible_plane_a": float(visible_plane.a),
            "visible_plane_b": float(visible_plane.b),
            "visible_plane_rms_m": visible_rms,
            "thermal_plane_e0": float(thermal_plane.e0),
            "thermal_plane_n0": float(thermal_plane.n0),
            "thermal_plane_z0": float(thermal_plane.z0),
            "thermal_plane_a": float(thermal_plane.a),
            "thermal_plane_b": float(thermal_plane.b),
            "thermal_plane_rms_m": thermal_rms,
            "visible_offset_m": float(visible_center_z - visible_array_z),
            "thermal_offset_m": float(thermal_center_z - thermal_array_z),
            "delta_offset_m": float((thermal_center_z - thermal_array_z)
                                    - (visible_center_z - visible_array_z)),
            "visible_tilt_a": float(visible_plane.a - visible_array.a),
            "visible_tilt_b": float(visible_plane.b - visible_array.b),
            "thermal_tilt_a": float(thermal_plane.a - thermal_array.a),
            "thermal_tilt_b": float(thermal_plane.b - thermal_array.b),
            "delta_tilt_a": float((thermal_plane.a - thermal_array.a)
                                  - (visible_plane.a - visible_array.a)),
            "delta_tilt_b": float((thermal_plane.b - thermal_array.b)
                                  - (visible_plane.b - visible_array.b)),
            "visible_samples": int(len(visible_z)),
            "thermal_samples": int(len(thermal_z)),
        })
    summary = {
        "panels": len(rows),
        "visible_plane_rms_m": _stats([row["visible_plane_rms_m"] for row in rows]),
        "thermal_plane_rms_m": _stats([row["thermal_plane_rms_m"] for row in rows]),
        "delta_offset_m": _stats([row["delta_offset_m"] for row in rows]),
        "delta_tilt_a": _stats([row["delta_tilt_a"] for row in rows]),
        "delta_tilt_b": _stats([row["delta_tilt_b"] for row in rows]),
    }
    return rows, summary

@dataclass(frozen=True)
class GeometryInputs:
    visible_dsm_path: Path
    thermal_dsm_path: Path
    visible_world: Mapping[str, Any]
    component_rows: Mapping[str, Mapping[str, Any]]
    erode_m: float
    field: PlanimetricField
    samples_cache: Mapping[str, Any] | None = None
    plane_cache: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class GeometryResult:
    rows: tuple[dict, ...]
    summary: dict


def build_geometry(inputs: GeometryInputs) -> GeometryResult:
    rows, summary = panel_layer(
        inputs.visible_dsm_path, inputs.thermal_dsm_path, inputs.visible_world,
        inputs.component_rows, inputs.erode_m, inputs.field,
        samples_cache=inputs.samples_cache, plane_cache=inputs.plane_cache)
    return GeometryResult(rows=tuple(rows), summary=summary)

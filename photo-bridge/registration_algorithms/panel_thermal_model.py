"""Panel-level thermal model: sample shifted footprints and fit panel planes."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio

from .io_utils import stats
from .dsm_sampling import sample_footprint_points as sample_polygon_pixels
from .local_plane import LocalPlane, fit_local_plane
from .residual_model import PlanimetricField


def fit_plane_samples(east, north, values):
    plane = fit_local_plane(east, north, values)
    residual = np.asarray(values, float) - plane.evaluate(east, north)
    return plane, float(np.sqrt(np.mean(residual ** 2)))



def _sample_and_fit_panel(panel, row, mapped_corners, visible_dsm, thermal_dsm,
                          erode_m: float, minimum_pixels: int,
                          cache_samples: bool):
    panel_id = panel["panel_id"]
    corners = np.asarray(panel["corners_xyz"], float)[:, :2]
    visible = sample_polygon_pixels(visible_dsm, corners, erode_m, minimum_pixels)
    result = {"visible": visible if cache_samples else None,
              "visible_plane": None, "thermal": None,
              "thermal_plane": None, "thermal_median": None}
    if visible is not None:
        result["visible_plane"] = fit_plane_samples(*visible)
    if row is None:
        return result
    thermal = sample_polygon_pixels(thermal_dsm, mapped_corners,
                                    erode_m, minimum_pixels)
    if thermal is None:
        return result
    result["thermal"] = thermal if cache_samples else None
    plane, rms = fit_plane_samples(*thermal)
    shift = np.array([float(row["delta_e_thermal_vs_visible_m"]),
                      float(row["delta_n_thermal_vs_visible_m"])])
    result["thermal_plane"] = (plane, rms, len(thermal[2]), shift)
    thermal_median = float(np.median(thermal[2]))
    result["thermal_median"] = (float(thermal[0].mean()),
                                float(thermal[1].mean()), thermal_median)
    return result


def _process_panel_chunk(task):
    chunk, visible_path, thermal_path, erode_m, minimum_pixels, cache_samples = task
    with (rasterio.open(visible_path) as visible_dsm,
          rasterio.open(thermal_path) as thermal_dsm):
        return [_sample_and_fit_panel(panel, row, mapped_corners,
                                      visible_dsm, thermal_dsm,
                                      erode_m, minimum_pixels, cache_samples)
                for panel, row, mapped_corners in chunk]

def build_panel_thermal_model(visible_world: dict, component_rows: dict[str, dict],
                              visible_dsm_path: Path, thermal_dsm_path: Path,
                              erode_m: float, minimum_pixels: int,
                              maximum_rms: float, field: PlanimetricField,
                              return_samples: bool = False, workers: int = 1):
    panels = visible_world["panels"]
    visible_planes = {}
    thermal_planes = {}
    thermal_medians = {}
    samples_cache = {}
    plane_cache = {}
    if panels:
        corner_stack = np.asarray(
            [np.asarray(panel["corners_xyz"], float)[:, :2] for panel in panels], float)
        mapped_stack = field.map_points(corner_stack.reshape(-1, 2)).reshape(-1, 4, 2)
    else:
        mapped_stack = np.empty((0, 4, 2))
    tasks = [(panel, component_rows.get(panel["panel_id"]), mapped_stack[index])
             for index, panel in enumerate(panels)]
    workers = max(1, int(workers))
    if workers > 1 and len(tasks) > 1:
        chunk_size = max(1, (len(tasks) + workers * 4 - 1) // (workers * 4))
        chunks = [tasks[start:start + chunk_size]
                  for start in range(0, len(tasks), chunk_size)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            chunk_results = list(pool.map(
                _process_panel_chunk,
                [(chunk, str(visible_dsm_path), str(thermal_dsm_path),
                  erode_m, minimum_pixels, return_samples) for chunk in chunks]))
        results = [result for chunk in chunk_results for result in chunk]
    else:
        with (rasterio.open(visible_dsm_path) as visible_dsm,
              rasterio.open(thermal_dsm_path) as thermal_dsm):
            results = [_sample_and_fit_panel(
                panel, row, mapped_corners, visible_dsm, thermal_dsm,
                erode_m, minimum_pixels, return_samples)
                for panel, row, mapped_corners in tasks]
    for (panel, row, _), result in zip(tasks, results):
        panel_id = panel["panel_id"]
        visible_plane = result["visible_plane"]
        if visible_plane is not None:
            visible_planes[panel_id] = visible_plane
            plane_cache.setdefault(panel_id, {})["visible"] = {
                "plane": visible_plane[0].to_dict(), "rms_m": visible_plane[1]}
        if result["visible"] is not None:
            samples_cache.setdefault(panel_id, {})["visible"] = result["visible"]
        thermal_plane = result["thermal_plane"]
        if thermal_plane is None:
            continue
        thermal_planes[panel_id] = thermal_plane
        plane_cache.setdefault(panel_id, {})["thermal"] = {
            "plane": thermal_plane[0].to_dict(), "rms_m": thermal_plane[1],
            "support": thermal_plane[2]}
        if result["thermal"] is not None:
            samples_cache.setdefault(panel_id, {})["thermal"] = result["thermal"]
        if result["thermal_median"] is not None:
            thermal_medians.setdefault(str(panel["array_id"]), []).append(
                result["thermal_median"])
    arrays = {}
    for array_id, items in thermal_medians.items():
        east = np.asarray([item[0] for item in items])
        north = np.asarray([item[1] for item in items])
        values = np.asarray([item[2] for item in items])
        design = np.column_stack([east - east.mean(), north - north.mean(),
                                  np.ones(len(items))])
        if len(items) >= 5 and np.linalg.matrix_rank(design) == 3:
            plane = fit_local_plane(east, north, values)
            quality = "surface_valid"
        else:
            plane = LocalPlane(float(east.mean()), float(north.mean()),
                               float(np.median(values)), 0.0, 0.0)
            quality = "surface_unavailable"
        arrays[array_id] = {"plane": plane.to_dict(),
                            "quality_status": quality,
                            "support": len(items)}
    visible_arrays = {str(panel["array_id"]) for panel in panels}
    for array_id in sorted(visible_arrays - set(arrays)):
        arrays[array_id] = {"plane": None,
                            "quality_status": "surface_unavailable", "support": 0}
    model_panels = {}
    counts = {}
    for panel in panels:
        panel_id = panel["panel_id"]
        array_id = str(panel["array_id"])
        row = component_rows.get(panel_id)
        shift = np.zeros(2)
        if row is not None:
            shift = np.array([float(row["delta_e_thermal_vs_visible_m"]),
                              float(row["delta_n_thermal_vs_visible_m"])])
        center_en = np.asarray(panel["center_xyz"], float)[:2] + shift
        corners_en = field.map_points(np.asarray(panel["corners_xyz"], float)[:, :2])
        array = arrays[array_id]
        array_plane = (LocalPlane.from_model(array["plane"])
                       if array.get("plane") is not None else None)
        array_available = array_plane is not None
        array_z = float(array_plane.evaluate(*center_en)) if array_available else None
        panel_plane = None
        z_source = "unavailable"
        center_z = None
        corners_z = None
        if panel_id in thermal_planes:
            plane, rms, support, _ = thermal_planes[panel_id]
            if rms <= maximum_rms:
                panel_plane = {**plane.to_dict(), "rms_m": rms, "support": support}
                center_z = float(plane.evaluate(*center_en))
                corners_z = [float(plane.evaluate(e, n)) for e, n in corners_en]
                z_source = "thermal_panel_plane"
        if center_z is None and array_available:
            center_z = float(array_z)
            corners_z = [float(array_plane.evaluate(e, n)) for e, n in corners_en]
            z_source = "thermal_m0"
        counts[z_source] = counts.get(z_source, 0) + 1
        if center_z is None or corners_z is None:
            model_panels[panel_id] = {
                "array_id": array_id, "center_xyz": None, "corners_xyz": None,
                "z_source": "unavailable", "panel_plane": None}
        else:
            model_panels[panel_id] = {
                "array_id": array_id,
                "center_xyz": [float(center_en[0]), float(center_en[1]), float(center_z)],
                "corners_xyz": [[float(e), float(n), float(z)]
                                for (e, n), z in zip(corners_en, corners_z)],
                "z_source": z_source,
                "panel_plane": panel_plane,
            }
    new_residual = []
    array_residual = []
    for panel in panels:
        panel_id = panel["panel_id"]
        if panel_id not in visible_planes or panel_id not in model_panels:
            continue
        visible_plane = visible_planes[panel_id][0]
        visible_center = np.asarray(panel["center_xyz"], float)[:2]
        visible_z = float(visible_plane.evaluate(*visible_center))
        model = model_panels[panel_id]
        if model.get("center_xyz") is None:
            continue
        new_residual.append(abs(model["center_xyz"][2] - visible_z))
        row = component_rows.get(panel_id)
        shift = np.zeros(2) if row is None else np.array([
            float(row["delta_e_thermal_vs_visible_m"]),
            float(row["delta_n_thermal_vs_visible_m"])])
        center = np.asarray(panel["center_xyz"], float)[:2] + shift
        array = arrays[str(panel["array_id"])]
        array_plane = LocalPlane.from_model(array["plane"]) if array.get("plane") is not None else None
        if array_plane is None:
            continue
        array_z = array_plane.evaluate(*center)
        array_residual.append(abs(float(array_z) - visible_z))
    validation = {
        "panel_vs_visible_panel_residual_m": stats(new_residual),
        "array_vs_visible_panel_residual_m": stats(array_residual),
        "visible_panel_planes": len(visible_planes),
        "thermal_panel_planes": len(thermal_planes),
    }
    if return_samples:
        return arrays, model_panels, counts, validation, samples_cache, plane_cache
    return arrays, model_panels, counts, validation

"""Shrunk per-array height residual planes in corrected thermal EN coordinates."""
from __future__ import annotations

import copy
from collections import defaultdict

import numpy as np

from .io_utils import abs_stats
from .local_plane import LocalPlane, fit_local_plane, plane_from_row

DEFAULT_MINIMUM_SUPPORT = 20
DEFAULT_PLANE_SHRINKAGE = 0.5
ARRAY_HEIGHT_CORRECTION_VERSION = "array_residual_plane/v1"


def _validate_settings(minimum_support, plane_shrinkage):
    if not isinstance(minimum_support, int) or minimum_support < 1:
        raise ValueError("minimum_support must be a positive integer")
    if not np.isfinite(plane_shrinkage) or not 0 <= plane_shrinkage <= 1:
        raise ValueError("plane_shrinkage must be finite and between 0 and 1")


def _fit_correction(east, north, residual, plane_shrinkage):
    """Fit median + lambda * (robust plane - median), including its intercept."""
    median = float(np.median(residual))
    constant = LocalPlane(float(np.mean(east)), float(np.mean(north)), median, 0., 0.)
    if plane_shrinkage == 0:
        return constant, "per_array_median", ""
    try:
        plane = fit_local_plane(east, north, residual)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return constant, "per_array_median_fallback", str(exc)
    correction = LocalPlane(plane.e0, plane.n0,
                            median + plane_shrinkage * (plane.z0 - median),
                            plane_shrinkage * plane.a, plane_shrinkage * plane.b)
    return correction, "per_array_residual_plane", ""


def build_bias_rows(model: dict, geometry: dict[str, dict],
                    components: dict[str, dict]):
    rows = []
    for panel_id, panel in model["panels"].items():
        if panel.get("center_xyz") is None:
            continue
        if panel_id not in geometry or panel_id not in components:
            continue
        component = components[panel_id]
        shift_e = float(component["delta_e_thermal_vs_visible_m"])
        shift_n = float(component["delta_n_thermal_vs_visible_m"])
        east, north, z = (float(value) for value in panel["center_xyz"])
        visible_east = east - shift_e
        visible_north = north - shift_n
        row = geometry[panel_id]
        plane = plane_from_row(row, "visible_plane")
        if plane is None:
            continue
        visible_z = float(plane.evaluate(visible_east, visible_north))
        rows.append({
            "panel_id": panel_id,
            "array_id": str(panel["array_id"]),
            "east": east, "north": north, "z": z,
            "visible_z": visible_z,
            "residual": z - visible_z,
            "z_source": panel["z_source"],
        })
    return rows


def _spatial_folds(east, north):
    coordinates = np.column_stack([east, north])
    centered = coordinates - coordinates.mean(axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    axis = vectors[0].copy()
    if axis[0] < 0 or (abs(axis[0]) < 1e-12 and axis[1] < 0):
        axis *= -1
    projection = centered @ axis
    order = np.argsort(projection, kind="stable")
    for slab, test in enumerate(np.array_split(order, 5)):
        if not len(test):
            continue
        low, high = projection[test].min(), projection[test].max()
        train = np.flatnonzero((projection < low - 1.5) | (projection > high + 1.5))
        yield slab, train, test


def _residual_stats(values):
    result = abs_stats(values)
    if result["n"]:
        result["rmse"] = float(np.sqrt(np.mean(np.square(values))))
    return result


def validate_bias(rows: list[dict], minimum_support: int,
                  plane_shrinkage: float = DEFAULT_PLANE_SHRINKAGE) -> dict:
    """Spatial OOF validation matching the frozen five-slab research protocol."""
    _validate_settings(minimum_support, plane_shrinkage)
    by_array = defaultdict(list)
    for row in rows:
        by_array[row["array_id"]].append(row)
    errors = {"before": [], "constant": [], "residual_plane": []}
    ends = {"constant": [], "residual_plane": []}
    fallback_folds = 0
    for items in by_array.values():
        if len(items) < minimum_support:
            continue
        east = np.asarray([item["east"] for item in items], float)
        north = np.asarray([item["north"] for item in items], float)
        residual = np.asarray([item["residual"] for item in items], float)
        for slab, train, test in _spatial_folds(east, north):
            if len(train) < 8:
                continue
            correction, method, _ = _fit_correction(east[train], north[train], residual[train], plane_shrinkage)
            fallback_folds += int(method == "per_array_median_fallback")
            errors["before"].extend(residual[test])
            fold_errors = {
                "constant": residual[test] - np.median(residual[train]),
                "residual_plane": residual[test] - correction.evaluate(east[test], north[test]),
            }
            for name, values in fold_errors.items():
                errors[name].extend(values)
                if slab in (0, 4):
                    ends[name].extend(values)
    evaluated = len(errors["constant"])
    return {
        "method": "five_contiguous_pca_slabs", "buffer_m": 1.5,
        "minimum_train_support": 8, "plane_shrinkage": plane_shrinkage,
        "input_components": len(rows), "evaluated_components": evaluated,
        "unavailable_components": len(rows) - evaluated,
        "coverage": evaluated / len(rows) if rows else 0.,
        **{name: _residual_stats(values) for name, values in errors.items()},
        "end_slabs": {name: _residual_stats(values) for name, values in ends.items()},
        "plane_fallback_folds": fallback_folds,
        "limits": "Spatial holdout on existing DSM observations, not independent physical accuracy.",
    }


def fit_array_bias(model: dict, rows: list[dict], minimum_support: int,
                   plane_shrinkage: float = DEFAULT_PLANE_SHRINKAGE):
    _validate_settings(minimum_support, plane_shrinkage)
    by_array = defaultdict(list)
    for row in rows:
        by_array[str(row["array_id"])].append(row)
    all_residual = np.asarray([row["residual"] for row in rows], float)
    global_bias = float(np.median(all_residual)) if len(all_residual) else 0.0
    biases = {}
    corrections = {}
    for array_id, items in by_array.items():
        values = np.asarray([item["residual"] for item in items], float)
        east = np.asarray([item["east"] for item in items], float)
        north = np.asarray([item["north"] for item in items], float)
        if len(items) >= minimum_support:
            bias = float(np.median(values))
            correction, method, error = _fit_correction(east, north, values, plane_shrinkage)
        else:
            bias = global_bias
            method, error = "global_median_fallback", ""
            correction = LocalPlane(float(east.mean()), float(north.mean()), bias, 0., 0.)
        corrections[array_id] = correction
        biases[array_id] = {
            "array_id": array_id, "support": len(items),
            "bias_m": bias, "method": method,
            "plane_shrinkage": plane_shrinkage, "fit_error": error,
            **{f"correction_{key}": value for key, value in correction.to_dict().items() if key != "type"},
            "mad_m": float(1.4826 * np.median(np.abs(values - np.median(values)))),
            "before_p95_abs_m": float(np.percentile(np.abs(values), 95)),
        }
    corrected = copy.deepcopy(model)
    for panel_id, panel in corrected["panels"].items():
        array_id = str(panel["array_id"])
        correction = corrections.get(array_id, LocalPlane(0., 0., global_bias, 0., 0.))
        method = biases.get(array_id, {"method": "global_median_fallback"})["method"]
        if panel.get("center_xyz") is not None:
            bias = float(correction.evaluate(*panel["center_xyz"][:2]))
            panel["center_xyz"][2] -= bias
            panel["corners_xyz"] = [[corner[0], corner[1], corner[2] - float(correction.evaluate(*corner[:2]))]
                                    for corner in panel["corners_xyz"]]
            if isinstance(panel.get("panel_plane"), dict):
                original = panel["panel_plane"]
                panel_plane = LocalPlane.from_model(original).subtract(correction).to_dict()
                for key in ("rms_m", "support"):
                    if key in original:
                        panel_plane[key] = original[key]
                panel["panel_plane"] = panel_plane
            panel["vertical_bias_m"] = bias
            panel["vertical_correction_method"] = method
            panel["z_source"] += ("+array_residual_plane" if method == "per_array_residual_plane" else "+array_bias")
    corrected["array_height_corrections"] = {key: value.to_dict() for key, value in corrections.items()}
    corrected["provenance"] = {
        **corrected.get("provenance", {}),
        "kind": "panel_level_with_array_height_correction",
        "minimum_support": minimum_support, "global_bias_m": global_bias,
        "array_height_correction": {
            "version": ARRAY_HEIGHT_CORRECTION_VERSION, "coordinate_frame": "thermal_en",
            "minimum_support": minimum_support, "plane_shrinkage": plane_shrinkage,
        },
    }
    component_rows = []
    for row in rows:
        array_id = str(row["array_id"])
        bias = float(corrections[array_id].evaluate(row["east"], row["north"]))
        component_rows.append({
            "panel_id": row["panel_id"], "array_id": row["array_id"],
            "east": row["east"], "north": row["north"],
            "residual_before_m": row["residual"],
            "array_bias_m": bias, "array_constant_bias_m": biases[array_id]["bias_m"],
            "correction_method": biases[array_id]["method"],
            "residual_after_m": row["residual"] - bias,
            "before_abs_m": abs(row["residual"]),
            "after_abs_m": abs(row["residual"] - bias),
        })
    return biases, corrected, component_rows

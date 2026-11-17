"""Core visible-ortho to thermal-ortho forward mapping.

A component maps when any of its five original thermal positions has image
data. Coverage at corrected positions and planimetric support are diagnostics.
"""
from __future__ import annotations

import cv2
import numpy as np

from .residual_model import PlanimetricField, field_support_mask
from .local_plane import LocalPlane


def _plane_z(plane, east, north):
    if plane is None:
        return None
    return float(LocalPlane.from_model(plane).evaluate(east, north))


def _pixels(inverse, east, north):
    east = np.asarray(east, float)
    north = np.asarray(north, float)
    with np.errstate(invalid="ignore", over="ignore"):
        col, row = inverse @ (east, north)
    return col, row


def _alpha_at(u, v, width, height, alpha, stride):
    if not (np.isfinite(u) and np.isfinite(v)
            and 0 <= u < width and 0 <= v < height):
        return None
    row = min(max(int(v) // stride, 0), alpha.shape[0] - 1)
    col = min(max(int(u) // stride, 0), alpha.shape[1] - 1)
    return int(alpha[row, col])


def _fit_similarity_2d(source: np.ndarray, target: np.ndarray):
    """Least-squares similarity mapping source -> target (rotation+scale+t)."""
    source = np.asarray(source, float)
    target = np.asarray(target, float)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered = source - source_center
    desired = target - target_center
    denom = float(np.sum(centered * centered))
    if not np.isfinite(denom) or denom <= 1e-12:
        return None
    a = float(np.sum(centered[:, 0] * desired[:, 0]
                     + centered[:, 1] * desired[:, 1]) / denom)
    b = float(np.sum(centered[:, 0] * desired[:, 1]
                     - centered[:, 1] * desired[:, 0]) / denom)
    return a, b, source_center, target_center


def _apply_similarity(source: np.ndarray, transform):
    a, b, source_center, target_center = transform
    centered = np.asarray(source, float) - source_center
    x = centered[:, 0]
    y = centered[:, 1]
    return np.column_stack((target_center[0] + a * x - b * y,
                            target_center[1] + b * x + a * y))


def panel_points(visible_world: dict, panels: dict, field: PlanimetricField,
                 visible_transform, thermal_transform,
                 thermal_width: int, thermal_height: int,
                 thermal_alpha: np.ndarray, alpha_stride: int,
                 field_max_distance: float | None = None,
                 panel_mapping: str = "pointwise") -> list[dict]:
    visible_inverse = ~visible_transform
    thermal_inverse = ~thermal_transform
    entries = []
    groups = []
    for panel in visible_world["panels"]:
        panel_id = panel["panel_id"]
        model = panels.get(panel_id, {})
        start = len(entries)
        for point_name, point in _panel_points(panel):
            entries.append({
                "panel_id": panel_id, "point_name": point_name,
                "east": float(point[0]), "north": float(point[1]),
                "z_visible": float(point[2]),
            })
        groups.append((panel_id, model, start, len(entries)))
    if not entries:
        return []
    east = np.asarray([entry["east"] for entry in entries], float)
    north = np.asarray([entry["north"] for entry in entries], float)
    delta_e, delta_n, weight, nearest = field.evaluate(east, north)
    _require_finite_shift(delta_e, delta_n)
    field_valid = _field_valid_mask(delta_e, delta_n, weight, nearest, field,
                                    field_max_distance)
    naive_u, naive_v = _pixels(thermal_inverse, east, north)
    visible_u, visible_v = _pixels(visible_inverse, east, north)
    east_t = east + delta_e
    north_t = north + delta_n
    applied_delta_e = delta_e.copy()
    applied_delta_n = delta_n.copy()
    if panel_mapping not in ("pointwise", "translation", "rigid",
                            "similarity"):
        raise ValueError("unknown panel_mapping: %s" % panel_mapping)
    if panel_mapping != "pointwise":
        for _, _, start, end in groups:
            source = np.column_stack((east[start:end], north[start:end]))
            target = np.column_stack((east_t[start:end], north_t[start:end]))
            if panel_mapping in ("translation", "rigid"):
                shift_e = float(np.mean(delta_e[start:end]))
                shift_n = float(np.mean(delta_n[start:end]))
                mapped = source + np.array([shift_e, shift_n])
            else:
                transform = _fit_similarity_2d(source, target)
                if transform is None:
                    shift_e = float(np.mean(delta_e[start:end]))
                    shift_n = float(np.mean(delta_n[start:end]))
                    mapped = source + np.array([shift_e, shift_n])
                else:
                    mapped = _apply_similarity(source, transform)
            east_t[start:end] = mapped[:, 0]
            north_t[start:end] = mapped[:, 1]
            applied_delta_e[start:end] = mapped[:, 0] - source[:, 0]
            applied_delta_n[start:end] = mapped[:, 1] - source[:, 1]
    corrected_u, corrected_v = _pixels(thermal_inverse, east_t, north_t)
    naive_alpha = np.asarray([
        _alpha_at(naive_u[i], naive_v[i], thermal_width, thermal_height,
                  thermal_alpha, alpha_stride) or 0 for i in range(len(entries))],
        dtype=np.int16)
    corrected_alpha = np.asarray([
        _alpha_at(corrected_u[i], corrected_v[i], thermal_width, thermal_height,
                  thermal_alpha, alpha_stride) or 0 for i in range(len(entries))],
        dtype=np.int16)
    naive_valid = naive_alpha > 0
    corrected_valid = corrected_alpha > 0
    rows = []
    for panel_id, model, start, end in groups:
        visible_plane = model.get("visible_plane")
        thermal_plane = model.get("thermal_plane")
        visible_plane = (LocalPlane.from_model(visible_plane)
                         if visible_plane is not None else None)
        thermal_plane = (LocalPlane.from_model(thermal_plane)
                         if thermal_plane is not None else None)
        panel_rows = []
        for i in range(start, end):
            east_i = east[i]
            north_i = north[i]
            if visible_plane is not None:
                z_visible = float(visible_plane.evaluate(east_i, north_i))
            else:
                z_visible = entries[i]["z_visible"]
            visible_plane_z = z_visible if visible_plane is not None else None
            thermal_z = (float(thermal_plane.evaluate(east_t[i], north_t[i]))
                         if thermal_plane is not None else None)
            if thermal_z is not None and visible_plane_z is not None:
                delta_z = thermal_z - visible_plane_z
                vertical_map_source = "thermal_panel_plane_minus_visible"
            else:
                delta_z = 0.0
                vertical_map_source = "no_vertical_surface_model"
            z_thermal = z_visible + delta_z
            panel_rows.append({
                "panel_id": panel_id, "point": entries[i]["point_name"],
                "visible_u": float(visible_u[i]), "visible_v": float(visible_v[i]),
                "east_visible": float(east_i), "north_visible": float(north_i),
                "z_visible": z_visible,
                "delta_e": float(applied_delta_e[i]),
                "delta_n": float(applied_delta_n[i]),
                "delta_z": delta_z,
                "east_thermal": float(east_t[i]), "north_thermal": float(north_t[i]),
                "z_thermal": z_thermal,
                "thermal_u": float(corrected_u[i]), "thermal_v": float(corrected_v[i]),
                "field_shift_px": float(np.hypot(corrected_u[i] - naive_u[i],
                                                 corrected_v[i] - naive_v[i])),
                "valid": 0,
                "naive_alpha": int(naive_alpha[i]),
                "naive_valid": int(naive_valid[i]),
                "thermal_alpha": int(corrected_alpha[i]),
                "corrected_valid": int(corrected_valid[i]),
                "field_valid": int(bool(field_valid[i])),
                "residual_applied": 1,
                "target_nodata": int(not corrected_valid[i]),
                "vertical_source": ("visible_panel_plane" if visible_plane is not None
                                    else "visible_world_center_z") + "+" + vertical_map_source,
                "field_weight": float(weight[i]), "nearest_tile_m": float(nearest[i]),
            })
        component_mapped = bool(np.any(naive_valid[start:end]))
        for row in panel_rows:
            row["valid"] = int(component_mapped)
        rows.extend(panel_rows)
    return rows


def grid_mapping(visible_world: dict, panels: dict, field: PlanimetricField,
                 visible_transform, visible_width: int, visible_height: int,
                 visible_z_grid: np.ndarray,
                 thermal_transform, thermal_width: int, thermal_height: int,
                 thermal_alpha: np.ndarray, alpha_stride: int, step: int,
                 field_max_distance: float | None = None) -> dict:
    grid_height = visible_height // step
    grid_width = visible_width // step
    columns = (np.arange(grid_width) + 0.5) * step
    rows = (np.arange(grid_height) + 0.5) * step
    column_grid, row_grid = np.meshgrid(columns, rows)
    inverse = ~visible_transform
    east, north = visible_transform @ (column_grid.ravel(), row_grid.ravel())
    visible_z = np.asarray(visible_z_grid, float).ravel()
    if visible_z.shape[0] != grid_height * grid_width:
        raise ValueError("visible_z_grid shape does not match grid size")
    panel_ids = [panel["panel_id"] for panel in visible_world["panels"]]
    panel_index = {panel_id: index + 1 for index, panel_id in enumerate(panel_ids)}
    mask = np.zeros((grid_height, grid_width), np.uint16)
    for panel in visible_world["panels"]:
        panel_id = panel["panel_id"]
        corners = np.asarray(panel["corners_xyz"], float)[:, :2]
        col, row = inverse @ (corners[:, 0], corners[:, 1])
        polygon = np.column_stack([col / step, row / step]).astype(np.int32)
        cv2.fillPoly(mask, [polygon], panel_index[panel_id])
    panel_index_grid = mask.ravel()
    delta_e, delta_n, weight, nearest = field.evaluate(east, north)
    east_t_global = east + delta_e
    north_t_global = north + delta_n
    thermal_inverse = ~thermal_transform
    with np.errstate(invalid="ignore", over="ignore"):
        naive_u, naive_v = thermal_inverse @ (east, north)

    def alpha_sample(u, v):
        valid = (np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < thermal_width)
                 & (v >= 0) & (v < thermal_height))
        alpha = np.zeros(len(u), np.uint8)
        rows_idx = np.clip((v[valid] // alpha_stride).astype(int), 0, thermal_alpha.shape[0] - 1)
        cols_idx = np.clip((u[valid] // alpha_stride).astype(int), 0, thermal_alpha.shape[1] - 1)
        alpha[valid] = thermal_alpha[rows_idx, cols_idx]
        return valid, alpha

    naive_inside, naive_alpha = alpha_sample(naive_u, naive_v)
    naive_valid = naive_inside & (naive_alpha > 0)
    field_valid = _field_valid_mask(delta_e, delta_n, weight, nearest, field,
                                    field_max_distance)
    use_corrected = np.isfinite(delta_e) & np.isfinite(delta_n)
    if not np.all(use_corrected):
        raise ValueError("non-finite planimetric prediction")

    index = panel_index_grid
    panel_mask = index > 0
    safe = np.where(panel_mask, index, 0)
    count = len(panel_ids)
    vis_e0 = np.zeros(count + 1); vis_n0 = np.zeros(count + 1)
    vis_z0 = np.zeros(count + 1); vis_a = np.zeros(count + 1)
    vis_b = np.zeros(count + 1); has_vis = np.zeros(count + 1, bool)
    th_e0 = np.zeros(count + 1); th_n0 = np.zeros(count + 1)
    th_z0 = np.zeros(count + 1); th_a = np.zeros(count + 1)
    th_b = np.zeros(count + 1); has_th = np.zeros(count + 1, bool)
    method_code = np.zeros(count + 1, np.int8)
    for number, panel_id in enumerate(panel_ids):
        model = panels.get(panel_id, {})
        plane = model.get("visible_plane")
        if plane is not None:
            plane = LocalPlane.from_model(plane)
            slot = number + 1
            vis_e0[slot], vis_n0[slot], vis_z0[slot] = plane.e0, plane.n0, plane.z0
            vis_a[slot], vis_b[slot], has_vis[slot] = plane.a, plane.b, True
        plane = model.get("thermal_plane")
        if plane is not None:
            plane = LocalPlane.from_model(plane)
            slot = number + 1
            th_e0[slot], th_n0[slot], th_z0[slot] = plane.e0, plane.n0, plane.z0
            th_a[slot], th_b[slot], has_th[slot] = plane.a, plane.b, True
        if model.get("thermal_correction_method") == "per_array_residual_plane":
            method_code[number + 1] = 2
    z_visible = visible_z.copy()
    panel_visible_z = np.full(len(east), np.nan)
    panel_thermal_z = np.full(len(east), np.nan)
    visible_mask = panel_mask & has_vis[safe]
    if visible_mask.any():
        e0 = vis_e0[safe][visible_mask]
        n0 = vis_n0[safe][visible_mask]
        value = (vis_a[safe][visible_mask] * (east[visible_mask] - e0)
                 + vis_b[safe][visible_mask] * (north[visible_mask] - n0)
                 + vis_z0[safe][visible_mask])
        panel_visible_z[visible_mask] = value
        z_visible[visible_mask] = value
    thermal_mask = panel_mask & has_th[safe]
    if thermal_mask.any():
        e0 = th_e0[safe][thermal_mask]
        n0 = th_n0[safe][thermal_mask]
        panel_thermal_z[thermal_mask] = (
            th_a[safe][thermal_mask] * (east_t_global[thermal_mask] - e0)
            + th_b[safe][thermal_mask] * (north_t_global[thermal_mask] - n0)
            + th_z0[safe][thermal_mask])
    both = visible_mask & thermal_mask
    delta_z = np.zeros(len(east), float)
    delta_z[both] = panel_thermal_z[both] - panel_visible_z[both]
    source_code = np.zeros(len(east), np.int8)
    source_code[panel_mask] = 3
    source_code[both] = 1
    source_code[both & (method_code[safe] == 2)] = 2
    vertical_source = np.asarray([
        "ground", "panel_plane+array_bias",
        "panel_plane+array_residual_plane",
        "no_vertical_surface_model",
    ])[source_code]

    with np.errstate(invalid="ignore", over="ignore"):
        corrected_u, corrected_v = thermal_inverse @ (east_t_global, north_t_global)
    corrected_inside, corrected_alpha = alpha_sample(corrected_u, corrected_v)
    corrected_valid = corrected_inside & (corrected_alpha > 0)
    target_nodata = ~corrected_valid
    valid = naive_valid.copy()
    coverage_points = [_panel_points(panel) for panel in visible_world["panels"]]
    if coverage_points and all(len(points) == 5 for points in coverage_points):
        point_east = np.asarray([point[1][0] for points in coverage_points for point in points], float)
        point_north = np.asarray([point[1][1] for points in coverage_points for point in points], float)
        with np.errstate(invalid="ignore", over="ignore"):
            point_u, point_v = thermal_inverse @ (point_east, point_north)
        point_inside, point_alpha = alpha_sample(point_u, point_v)
        covered = (point_inside & (point_alpha > 0)).reshape(-1, 5).any(axis=1)
        valid[index > 0] = covered[index[index > 0] - 1]
    else:
        for panel in visible_world["panels"]:
            selected = np.flatnonzero(index == panel_index[panel["panel_id"]])
            if len(selected):
                valid[selected] = _original_coverage(
                    panel, thermal_inverse, thermal_width, thermal_height,
                    thermal_alpha, alpha_stride)

    z_thermal = z_visible + delta_z
    return {
        "grid_height": grid_height, "grid_width": grid_width,
        "visible_u": column_grid.ravel(), "visible_v": row_grid.ravel(),
        "east": east, "north": north, "z_visible": z_visible,
        "delta_e": delta_e, "delta_n": delta_n, "delta_z": delta_z,
        "east_t": east_t_global, "north_t": north_t_global, "z_thermal": z_thermal,
        "thermal_u": corrected_u, "thermal_v": corrected_v,
        "naive_thermal_u": naive_u, "naive_thermal_v": naive_v,
        "valid": valid, "naive_valid": naive_valid,
        "corrected_valid": corrected_valid,
        "use_corrected": use_corrected, "target_nodata": target_nodata,
        "panel_id": np.asarray([""] + panel_ids, dtype=object)[panel_index_grid],
        "vertical_source": vertical_source,
        "field_weight": weight, "nearest_tile_m": nearest,
        "field_valid": field_valid,
    }


def _field_valid_mask(delta_e, delta_n, weight, nearest, field,
                      max_distance: float | None):
    threshold = field.max_distance_m if max_distance is None else max_distance
    return field_support_mask(delta_e, delta_n, weight, nearest, threshold)


def _field_is_valid(delta_e, delta_n, weight, nearest, field,
                    max_distance: float | None) -> bool:
    return bool(_field_valid_mask(np.asarray([delta_e]), np.asarray([delta_n]),
                                  np.asarray([weight]), np.asarray([nearest]),
                                  field, max_distance)[0])


def _require_finite_shift(delta_e, delta_n):
    if not (np.all(np.isfinite(delta_e)) and np.all(np.isfinite(delta_n))):
        raise ValueError("non-finite planimetric prediction")


def _panel_points(panel):
    points = [("center", np.asarray(panel["center_xyz"], float))]
    points.extend(("corner_%d" % index, np.asarray(corner, float))
                  for index, corner in enumerate(panel["corners_xyz"]))
    return points


def _original_coverage(panel, thermal_inverse, width, height, alpha, stride):
    for _, point in _panel_points(panel):
        u, v = _pixels(thermal_inverse, point[0], point[1])
        sample = _alpha_at(float(u), float(v), width, height, alpha, stride)
        if sample is not None and sample > 0:
            return True
    return False

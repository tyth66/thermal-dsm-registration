"""Core visible-ortho to thermal-ortho forward position mapping.

A component maps when any of its five original thermal positions has image
data. Coverage at corrected positions and planimetric support are diagnostics.
The mapping is positions-only: the vertical (z) surface model was removed
because it never influenced the thermal pixel position.
"""
from __future__ import annotations

import cv2
import numpy as np

from .boundary_field import evaluate_boundary_field
from .residual_model import PlanimetricField, field_support_mask


def _evaluate_with_support(field, east, north, workers: int = 1):
    if hasattr(field, "evaluate_with_support"):
        return field.evaluate_with_support(east, north, workers=workers)
    delta_e, delta_n, weight, nearest = field.evaluate(east, north)
    support = (field.support_confidence(east, north)
               if hasattr(field, "support_confidence")
               else np.ones(len(east), float))
    return delta_e, delta_n, weight, nearest, support


def _pixels(inverse, east, north):
    east = np.asarray(east, float)
    north = np.asarray(north, float)
    with np.errstate(invalid="ignore", over="ignore"):
        col, row = inverse @ (east, north)
    return col, row


def _alpha_at(u, v, width, height, alpha):
    if not (np.isfinite(u) and np.isfinite(v)
            and 0 <= u < width and 0 <= v < height):
        return None
    # Alpha is sampled at full thermal resolution; clamp to the raster bounds so
    # a coordinate near the far edge still reads the last pixel.
    row = min(max(int(v), 0), alpha.shape[0] - 1)
    col = min(max(int(u), 0), alpha.shape[1] - 1)
    return int(alpha[row, col])


def _component_conservative(panel_rows):
    """A panel is delivery-conservative only with a common valid core."""
    center = next((row for row in panel_rows if row["point"] == "center"), None)
    corners = [row for row in panel_rows if row["point"] != "center"]
    if center is None or len(corners) != 4:
        return False
    return bool(center["conservative_valid"]
                and sum(row["conservative_valid"] for row in corners) >= 2)


def panel_points(visible_world: dict, field: PlanimetricField,
                 visible_transform, thermal_transform,
                 thermal_width: int, thermal_height: int,
                 thermal_alpha: np.ndarray,
                 field_max_distance: float | None = None,
                 boundary=None, field_chunk_workers: int = 1) -> list[dict]:
    visible_inverse = ~visible_transform
    thermal_inverse = ~thermal_transform
    entries = []
    groups = []
    for panel in visible_world["panels"]:
        panel_id = panel["panel_id"]
        start = len(entries)
        for point_name, point in _panel_points(panel):
            entries.append({
                "panel_id": panel_id, "point_name": point_name,
                "east": float(point[0]), "north": float(point[1]),
            })
        groups.append((panel_id, start, len(entries)))
    if not entries:
        return []
    east = np.asarray([entry["east"] for entry in entries], float)
    north = np.asarray([entry["north"] for entry in entries], float)
    delta_e, delta_n, weight, nearest, support_confidence = (
        _evaluate_with_support(field, east, north, field_chunk_workers))
    edge_distance = np.full(len(east), np.inf, float)
    edge_tail_applied = np.zeros(len(east), bool)
    if boundary is not None:
        delta_e, delta_n, edge_distance, edge_tail_applied = (
            evaluate_boundary_field(field, east, north, boundary,
                                    base_e=delta_e, base_n=delta_n))
    _require_finite_shift(delta_e, delta_n)
    field_valid = _field_valid_mask(delta_e, delta_n, weight, nearest, field,
                                    field_max_distance)
    naive_u, naive_v = _pixels(thermal_inverse, east, north)
    visible_u, visible_v = _pixels(visible_inverse, east, north)
    east_t = east + delta_e
    north_t = north + delta_n
    applied_delta_e = delta_e.copy()
    applied_delta_n = delta_n.copy()
    for _, start, end in groups:
        shift_e = float(np.mean(delta_e[start:end]))
        shift_n = float(np.mean(delta_n[start:end]))
        east_t[start:end] = east[start:end] + shift_e
        north_t[start:end] = north[start:end] + shift_n
        applied_delta_e[start:end] = shift_e
        applied_delta_n[start:end] = shift_n
    corrected_u, corrected_v = _pixels(thermal_inverse, east_t, north_t)
    naive_alpha = np.asarray([
        _alpha_at(naive_u[i], naive_v[i], thermal_width, thermal_height,
                  thermal_alpha) or 0 for i in range(len(entries))],
        dtype=np.int16)
    corrected_alpha = np.asarray([
        _alpha_at(corrected_u[i], corrected_v[i], thermal_width, thermal_height,
                  thermal_alpha) or 0 for i in range(len(entries))],
        dtype=np.int16)
    naive_valid = naive_alpha > 0
    corrected_valid = corrected_alpha > 0
    rows = []
    for panel_id, start, end in groups:
        panel_rows = []
        for i in range(start, end):
            panel_rows.append({
                "panel_id": panel_id, "point": entries[i]["point_name"],
                "visible_u": float(visible_u[i]), "visible_v": float(visible_v[i]),
                "east_visible": float(east[i]), "north_visible": float(north[i]),
                "delta_e": float(applied_delta_e[i]),
                "delta_n": float(applied_delta_n[i]),
                "east_thermal": float(east_t[i]), "north_thermal": float(north_t[i]),
                "thermal_u": float(corrected_u[i]), "thermal_v": float(corrected_v[i]),
                "field_shift_px": float(np.hypot(corrected_u[i] - naive_u[i],
                                                 corrected_v[i] - naive_v[i])),
                "valid": 0,
                "naive_alpha": int(naive_alpha[i]),
                "naive_valid": int(naive_valid[i]),
                "thermal_alpha": int(corrected_alpha[i]),
                "corrected_valid": int(corrected_valid[i]),
                "field_valid": int(bool(field_valid[i])),
                "source_valid": int(naive_valid[i]),
                "field_supported": int(bool(field_valid[i])),
                "target_valid": int(corrected_valid[i]),
                "residual_applied": 1,
                "field_weight": float(weight[i]), "nearest_tile_m": float(nearest[i]),
                "support_confidence": float(support_confidence[i]),
                "edge_distance_m": float(edge_distance[i]),
                "edge_tail_applied": int(bool(edge_tail_applied[i])),
            })
        component_mapped = bool(np.any(naive_valid[start:end]))
        center_row = next((row for row in panel_rows if row["point"] == "center"),
                          panel_rows[0])
        corner_supported = sum(row["field_supported"] for row in panel_rows
                               if row["point"] != "center")
        component_field_supported = bool(center_row["field_supported"]
                                         and corner_supported >= 3)
        component_target_valid = bool(sum(row["target_valid"] for row in panel_rows) >= 3)
        for row in panel_rows:
            row["conservative_valid"] = int(
                row["source_valid"] and row["field_supported"]
                and row["target_valid"])
        component_conservative_valid = _component_conservative(panel_rows)
        for row in panel_rows:
            row["valid"] = int(component_mapped)
            row["component_source_valid"] = int(component_mapped)
            row["component_field_supported"] = int(component_field_supported)
            row["component_target_valid"] = int(component_target_valid)
            row["low_confidence"] = int(not (component_field_supported
                                             and component_target_valid))
            row["component_conservative_valid"] = int(component_conservative_valid)
        rows.extend(panel_rows)
    return rows


def _grid_summary(visible_world: dict, field: PlanimetricField,
                  visible_transform, visible_width: int, visible_height: int,
                  thermal_transform, thermal_width: int, thermal_height: int,
                  thermal_alpha: np.ndarray, step: int,
                  field_max_distance: float | None,
                  boundary, field_chunk_workers: int) -> dict:
    grid_height = visible_height // step
    grid_width = visible_width // step
    if grid_height == 0 or grid_width == 0:
        raise ValueError("grid step exceeds visible raster dimensions")
    columns = (np.arange(grid_width) + 0.5) * step
    inverse = ~visible_transform
    thermal_inverse = ~thermal_transform
    panel_ids = [panel["panel_id"] for panel in visible_world["panels"]]
    panel_index = {panel_id: index + 1 for index, panel_id in enumerate(panel_ids)}
    mask = np.zeros((grid_height, grid_width), np.uint16)
    for panel in visible_world["panels"]:
        panel_id = panel["panel_id"]
        corners = np.asarray(panel["corners_xyz"], float)[:, :2]
        col, row = inverse @ (corners[:, 0], corners[:, 1])
        polygon = np.column_stack([col / step, row / step]).astype(np.int32)
        cv2.fillPoly(mask, [polygon], panel_index[panel_id])
    coverage_points = [_panel_points(panel) for panel in visible_world["panels"]]
    point_counts = {len(points) for points in coverage_points}
    if point_counts - {5}:
        raise ValueError("panel point count is not five: %s" % sorted(point_counts))
    point_east = np.asarray([point[1][0] for points in coverage_points for point in points], float)
    point_north = np.asarray([point[1][1] for points in coverage_points for point in points], float)
    with np.errstate(invalid="ignore", over="ignore"):
        point_u, point_v = thermal_inverse @ (point_east, point_north)

    def alpha_sample(u, v):
        inside = (np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < thermal_width)
                  & (v >= 0) & (v < thermal_height))
        alpha = np.zeros(len(u), np.uint8)
        rows_idx = np.clip(v[inside].astype(int), 0, thermal_alpha.shape[0] - 1)
        cols_idx = np.clip(u[inside].astype(int), 0, thermal_alpha.shape[1] - 1)
        alpha[inside] = thermal_alpha[rows_idx, cols_idx]
        return inside, alpha

    point_inside, point_alpha = alpha_sample(point_u, point_v)
    covered = (point_inside & (point_alpha > 0)).reshape(-1, 5).any(axis=1)
    chunk_rows = max(1, 100_000 // grid_width)
    fields = {name: [] for name in (
        "delta_e", "delta_n", "valid", "naive_valid", "corrected_valid",
        "target_nodata", "field_valid", "panel_mask", "edge_tail_applied")}
    for start in range(0, grid_height, chunk_rows):
        stop = min(start + chunk_rows, grid_height)
        row_grid, column_grid = np.meshgrid(
            (np.arange(start, stop) + 0.5) * step, columns, indexing="ij")
        east, north = visible_transform @ (column_grid.ravel(), row_grid.ravel())
        delta_e, delta_n, weight, nearest, _ = _evaluate_with_support(
            field, east, north, field_chunk_workers)
        edge_tail_applied = np.zeros(len(east), bool)
        if boundary is not None:
            delta_e, delta_n, _, edge_tail_applied = evaluate_boundary_field(
                field, east, north, boundary, base_e=delta_e, base_n=delta_n)
        east_t = east + delta_e
        north_t = north + delta_n
        with np.errstate(invalid="ignore", over="ignore"):
            naive_u, naive_v = thermal_inverse @ (east, north)
            corrected_u, corrected_v = thermal_inverse @ (east_t, north_t)
        naive_inside, naive_alpha = alpha_sample(naive_u, naive_v)
        corrected_inside, corrected_alpha = alpha_sample(corrected_u, corrected_v)
        naive_valid = naive_inside & (naive_alpha > 0)
        corrected_valid = corrected_inside & (corrected_alpha > 0)
        field_valid = _field_valid_mask(delta_e, delta_n, weight, nearest, field,
                                        field_max_distance)
        if not np.all(np.isfinite(delta_e) & np.isfinite(delta_n)):
            raise ValueError("non-finite planimetric prediction")
        index = mask[start:stop].ravel()
        panel_mask = index > 0
        valid = naive_valid.copy()
        valid[index > 0] = covered[index[index > 0] - 1]
        fields["delta_e"].append(delta_e)
        fields["delta_n"].append(delta_n)
        fields["valid"].append(valid)
        fields["naive_valid"].append(naive_valid)
        fields["corrected_valid"].append(corrected_valid)
        fields["target_nodata"].append(~corrected_valid)
        fields["field_valid"].append(field_valid)
        fields["panel_mask"].append(panel_mask)
        fields["edge_tail_applied"].append(edge_tail_applied)
    result = {name: np.concatenate(values) for name, values in fields.items()}
    result.update({"grid_height": grid_height, "grid_width": grid_width,
                   "grid_points": grid_height * grid_width})
    return result


def grid_mapping(visible_world: dict, field: PlanimetricField,
                 visible_transform, visible_width: int, visible_height: int,
                 thermal_transform, thermal_width: int, thermal_height: int,
                 thermal_alpha: np.ndarray, step: int,
                 field_max_distance: float | None = None,
                 boundary=None, field_chunk_workers: int = 1,
                 grid_mode: str = "full") -> dict:
    if grid_mode not in ("full", "compact", "summary"):
        raise ValueError("unsupported grid mode: %s" % grid_mode)
    if grid_mode == "summary":
        return _grid_summary(visible_world, field, visible_transform,
                             visible_width, visible_height, thermal_transform,
                             thermal_width, thermal_height, thermal_alpha, step,
                             field_max_distance, boundary, field_chunk_workers)
    grid_height = visible_height // step
    grid_width = visible_width // step
    if grid_height == 0 or grid_width == 0:
        raise ValueError("grid step exceeds visible raster dimensions")
    columns = (np.arange(grid_width) + 0.5) * step
    rows = (np.arange(grid_height) + 0.5) * step
    column_grid, row_grid = np.meshgrid(columns, rows)
    inverse = ~visible_transform
    east, north = visible_transform @ (column_grid.ravel(), row_grid.ravel())
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
    delta_e, delta_n, weight, nearest, support_confidence = (
        _evaluate_with_support(field, east, north, field_chunk_workers))
    edge_distance = np.full(len(east), np.inf, float)
    edge_tail_applied = np.zeros(len(east), bool)
    if boundary is not None:
        delta_e, delta_n, edge_distance, edge_tail_applied = (
            evaluate_boundary_field(field, east, north, boundary,
                                    base_e=delta_e, base_n=delta_n))
    east_t_global = east + delta_e
    north_t_global = north + delta_n
    thermal_inverse = ~thermal_transform
    with np.errstate(invalid="ignore", over="ignore"):
        naive_u, naive_v = thermal_inverse @ (east, north)

    def alpha_sample(u, v):
        valid = (np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < thermal_width)
                 & (v >= 0) & (v < thermal_height))
        alpha = np.zeros(len(u), np.uint8)
        rows_idx = np.clip(v[valid].astype(int), 0, thermal_alpha.shape[0] - 1)
        cols_idx = np.clip(u[valid].astype(int), 0, thermal_alpha.shape[1] - 1)
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

    with np.errstate(invalid="ignore", over="ignore"):
        corrected_u, corrected_v = thermal_inverse @ (east_t_global, north_t_global)
    corrected_inside, corrected_alpha = alpha_sample(corrected_u, corrected_v)
    corrected_valid = corrected_inside & (corrected_alpha > 0)
    target_nodata = ~corrected_valid
    valid = naive_valid.copy()
    coverage_points = [_panel_points(panel) for panel in visible_world["panels"]]
    point_counts = {len(points) for points in coverage_points}
    if point_counts - {5}:
        # The world model emits center + four corners by construction; any other
        # count means the component coverage rule cannot be applied to the panel.
        raise ValueError("panel point count is not five: %s" % sorted(point_counts))
    point_east = np.asarray([point[1][0] for points in coverage_points for point in points], float)
    point_north = np.asarray([point[1][1] for points in coverage_points for point in points], float)
    with np.errstate(invalid="ignore", over="ignore"):
        point_u, point_v = thermal_inverse @ (point_east, point_north)
    point_inside, point_alpha = alpha_sample(point_u, point_v)
    covered = (point_inside & (point_alpha > 0)).reshape(-1, 5).any(axis=1)
    valid[index > 0] = covered[index[index > 0] - 1]

    return {
        "grid_height": grid_height, "grid_width": grid_width,
        "grid_points": len(east),
        "visible_u": column_grid.ravel(), "visible_v": row_grid.ravel(),
        "east": east, "north": north,
        "delta_e": delta_e, "delta_n": delta_n,
        "east_t": east_t_global, "north_t": north_t_global,
        "thermal_u": corrected_u, "thermal_v": corrected_v,
        "naive_thermal_u": naive_u, "naive_thermal_v": naive_v,
        "valid": valid, "naive_valid": naive_valid,
        "corrected_valid": corrected_valid,
        "use_corrected": use_corrected, "target_nodata": target_nodata,
        "panel_id": np.asarray([""] + panel_ids, dtype=object)[panel_index_grid],
        "panel_mask": panel_mask,
        "field_weight": weight, "nearest_tile_m": nearest,
        "support_confidence": support_confidence,
        "field_valid": field_valid,
        "edge_distance_m": edge_distance,
        "edge_tail_applied": edge_tail_applied,
    }


def _field_valid_mask(delta_e, delta_n, weight, nearest, field,
                      max_distance: float | None):
    fallback = field.max_distance_m if max_distance is None else max_distance
    guard = 2.0 * float(getattr(field, "tile_m", fallback / 2.0))
    return field_support_mask(delta_e, delta_n, weight, nearest,
                              max(guard, 1e-6))


def _require_finite_shift(delta_e, delta_n):
    if not (np.all(np.isfinite(delta_e)) and np.all(np.isfinite(delta_n))):
        raise ValueError("non-finite planimetric prediction")


def _panel_points(panel):
    points = [("center", np.asarray(panel["center_xyz"], float)[:2])]
    points.extend(("corner_%d" % index, np.asarray(corner, float)[:2])
                  for index, corner in enumerate(panel["corners_xyz"]))
    return points

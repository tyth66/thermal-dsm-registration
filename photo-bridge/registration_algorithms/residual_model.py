"""Canonical 3D residual model: planimetric field + vertical panel planes."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

from .local_plane import LocalPlane, plane_from_row


def field_support_mask(delta_e, delta_n, weight, nearest, max_distance_m):
    """Shared support contract for height sampling and horizontal mapping."""
    return (np.isfinite(delta_e) & np.isfinite(delta_n) & np.isfinite(weight)
            & np.isfinite(nearest) & (weight > 0) & (nearest <= max_distance_m))


class PlanimetricField:
    """Common planimetric field interface."""

    max_distance_m = 30.0

    @classmethod
    def from_model(cls, model: dict):
        if model.get("type") != REGULARIZED_FIELD_TYPE:
            raise ValueError("unsupported planimetric field type: %s" % model.get("type"))
        return RegularizedPlanimetricField.from_model(model)

    @classmethod
    def from_tiles(cls, rows, radius: float = 90.0, power: float = 2.0,
                   max_distance_m: float = 30.0):
        rows = list(rows)
        if not rows:
            return RegularizedPlanimetricField.empty(max_distance_m)
        return fit_regularized_field(rows, tile_m=10.0,
                                     max_distance_m=max_distance_m)

    def evaluate(self, east, north):
        raise NotImplementedError

    def map_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, float)
        dx, dy, _, _ = self.evaluate(points[:, 0], points[:, 1])
        return points + np.column_stack([dx, dy])

    def supports_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, float)
        return field_support_mask(*self.evaluate(points[:, 0], points[:, 1]),
                                  self.max_distance_m)


REGULARIZED_FIELD_TYPE = "similarity_regularized_rbf/v1"


def _weighted_similarity(east: np.ndarray, north: np.ndarray,
                         target_east: np.ndarray, target_north: np.ndarray,
                         weights: np.ndarray):
    """Robust weighted similarity: target = A * source + b."""
    source = np.column_stack([east, north])
    target = np.column_stack([target_east, target_north])
    weights = np.asarray(weights, float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    mask = np.ones(len(source), bool)
    matrix = np.eye(2)
    translation = np.zeros(2)
    for _ in range(4):
        if mask.sum() < 3:
            break
        source_center = np.average(source[mask], axis=0, weights=weights[mask])
        target_center = np.average(target[mask], axis=0, weights=weights[mask])
        centered = source[mask] - source_center
        desired = target[mask] - target_center
        w = weights[mask]
        design = np.zeros((2 * len(centered), 2))
        rhs = desired.ravel()
        design[0::2, 0] = centered[:, 0]
        design[0::2, 1] = -centered[:, 1]
        design[1::2, 0] = centered[:, 1]
        design[1::2, 1] = centered[:, 0]
        root_weight = np.repeat(np.sqrt(w), 2)
        a, b = np.linalg.lstsq(design * root_weight[:, None],
                               rhs * root_weight, rcond=None)[0]
        matrix = np.array([[a, -b], [b, a]])
        translation = target_center - matrix @ source_center
        residual = target - (source @ matrix.T + translation)
        magnitude = np.hypot(residual[:, 0], residual[:, 1])
        median = float(np.median(magnitude[mask]))
        mad = 1.4826 * float(np.median(np.abs(magnitude[mask] - median)))
        threshold = max(3.0 * mad, 0.50)
        next_mask = magnitude <= threshold
        if next_mask.sum() < max(3, len(mask) // 2):
            next_mask = np.ones(len(mask), bool)
        if np.array_equal(next_mask, mask):
            break
        mask = next_mask
    return matrix, translation, mask


def fit_regularized_field(rows: Iterable[dict], tile_m: float,
                          max_distance_m: float = 30.0,
                          kernel_length_m: float | None = None,
                          ridge: float = 0.1,
                          support_weighting: bool = True):
    """Fit global similarity + regularized RBF residual.

    The residual model decays to zero outside the tile support, so weak-support
    areas fall back to the global similarity instead of extrapolating IDW noise.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("cannot fit a planimetric field without accepted tiles")
    east = np.asarray([float(row["east"]) for row in rows], float)
    north = np.asarray([float(row["north"]) for row in rows], float)
    target_east = east + np.asarray([float(row["dx_m"]) for row in rows], float)
    target_north = north + np.asarray([float(row["dy_m"]) for row in rows], float)
    cc = np.asarray([float(row["cc"]) for row in rows], float)
    cc = np.where(np.isfinite(cc) & (cc > 0), cc, 1.0)
    quality = np.asarray([float(row.get("quality_weight", 1.0)) for row in rows], float)
    quality = np.where(np.isfinite(quality) & (quality > 0), quality, 1.0)
    cc = cc * quality
    tree = cKDTree(np.column_stack([east, north]))
    if len(rows) > 1:
        spacing = tree.query(np.column_stack([east, north]), k=2)[0][:, 1]
        spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
        spacing_median = float(np.median(spacing)) if len(spacing) else float(tile_m)
    else:
        spacing_median = float(tile_m)
    if kernel_length_m is None:
        kernel_length_m = max(2.5 * float(tile_m), 4.0 * spacing_median)
    kernel_length_m = max(float(kernel_length_m), 1e-6)
    if support_weighting and len(rows) > 2:
        support = np.exp(-0.5 * ((tree.query(
            np.column_stack([east, north]), k=2)[0][:, 1]
            / max(spacing_median, 1e-6)) - 1.0) ** 2)
        weights = cc * np.where(np.isfinite(support), support, 1.0)
    else:
        weights = cc
    matrix, translation, inliers = _weighted_similarity(
        east, north, target_east, target_north, weights)
    target = np.column_stack([target_east, target_north])
    aligned = np.column_stack([east, north]) @ matrix.T + translation
    residual = target - aligned
    centers = np.column_stack([east, north])
    fit_weights = weights * np.where(inliers, 1.0, 0.2)
    fit_residual = residual
    distances = np.hypot(centers[:, None, 0] - centers[None, :, 0],
                         centers[:, None, 1] - centers[None, :, 1])
    kernel = np.exp(-0.5 * (distances / kernel_length_m) ** 2)
    weighted_kernel = kernel * fit_weights[:, None]
    normal = kernel.T @ weighted_kernel + float(ridge) * np.eye(len(centers))
    coefficients = np.linalg.solve(normal, weighted_kernel.T @ fit_residual)
    tiles = [{
        "east": float(east[i]), "north": float(north[i]), "cc": float(cc[i]),
        "residual_e": float(residual[i, 0]), "residual_n": float(residual[i, 1]),
    } for i in range(len(rows))]
    return RegularizedPlanimetricField(
        tiles=tiles, centers=centers, weights=fit_weights,
        coefficients=coefficients, matrix=matrix, translation=translation,
        kernel_length_m=kernel_length_m, ridge=float(ridge),
        max_distance_m=float(max_distance_m))


class RegularizedPlanimetricField(PlanimetricField):
    """Global similarity plus a smooth RBF residual that decays off-support."""

    EVALUATION_CHUNK = 250_000

    def __init__(self, tiles, centers, weights, coefficients, matrix,
                 translation, kernel_length_m, ridge, max_distance_m):
        self.tiles = list(tiles)
        self.east = np.asarray([tile["east"] for tile in self.tiles], float)
        self.north = np.asarray([tile["north"] for tile in self.tiles], float)
        self.cc = np.asarray([tile["cc"] for tile in self.tiles], float)
        self.residual_e = np.asarray([tile["residual_e"] for tile in self.tiles], float)
        self.residual_n = np.asarray([tile["residual_n"] for tile in self.tiles], float)
        self.centers = np.asarray(centers, float)
        self.weights = np.asarray(weights, float)
        self.coefficients = np.asarray(coefficients, float)
        self.matrix = np.asarray(matrix, float)
        self.translation = np.asarray(translation, float)
        self.kernel_length_m = float(kernel_length_m)
        self.ridge = float(ridge)
        self.max_distance_m = float(max_distance_m)
        self.tree = cKDTree(np.column_stack([self.east, self.north])) if len(self.east) else None

    @classmethod
    def empty(cls, max_distance_m: float = 30.0):
        return cls(tiles=[], centers=np.empty((0, 2)), weights=np.empty(0),
                   coefficients=np.empty((0, 2)), matrix=np.eye(2),
                   translation=np.zeros(2), kernel_length_m=1.0,
                   ridge=0.1, max_distance_m=max_distance_m)

    @classmethod
    def from_model(cls, model: dict):
        residual = model["residual"]
        similarity = model["similarity"]
        return cls(
            tiles=residual["tiles"],
            centers=np.asarray(residual["centers"], float),
            weights=np.asarray(residual["weights"], float),
            coefficients=np.column_stack([
                np.asarray(residual["coefficients_e"], float),
                np.asarray(residual["coefficients_n"], float)]),
            matrix=np.asarray(similarity["matrix"], float),
            translation=np.asarray(similarity["translation"], float),
            kernel_length_m=float(residual["kernel_length_m"]),
            ridge=float(residual["ridge"]),
            max_distance_m=float(model.get("max_distance_m", 30.0)))

    def to_model(self) -> dict:
        rotation = float(np.degrees(np.arctan2(self.matrix[1, 0], self.matrix[0, 0])))
        scale = float(np.hypot(self.matrix[0, 0], self.matrix[1, 0]))
        return {
            "type": REGULARIZED_FIELD_TYPE,
            "similarity": {
                "matrix": self.matrix.tolist(),
                "translation": self.translation.tolist(),
                "rotation_deg": rotation,
                "scale": scale,
            },
            "residual": {
                "kernel_length_m": self.kernel_length_m,
                "ridge": self.ridge,
                "centers": self.centers.tolist(),
                "weights": self.weights.tolist(),
                "coefficients_e": self.coefficients[:, 0].tolist(),
                "coefficients_n": self.coefficients[:, 1].tolist(),
                "tiles": self.tiles,
            },
            "max_distance_m": self.max_distance_m,
        }

    def evaluate(self, east, north):
        east = np.asarray(east, float)
        north = np.asarray(north, float)
        if len(east) <= self.EVALUATION_CHUNK:
            dx, dy, weight_sum = self._evaluate_chunk(east, north)
        else:
            dx = np.empty(len(east), float)
            dy = np.empty(len(east), float)
            weight_sum = np.empty(len(east), float)
            for start in range(0, len(east), self.EVALUATION_CHUNK):
                stop = start + self.EVALUATION_CHUNK
                dx[start:stop], dy[start:stop], weight_sum[start:stop] = (
                    self._evaluate_chunk(east[start:stop], north[start:stop]))
        if self.tree is None:
            nearest = np.full(len(east), np.inf)
        else:
            nearest, _ = self.tree.query(np.column_stack([east, north]), k=1)
        return dx, dy, weight_sum, nearest

    def _evaluate_chunk(self, east, north):
        points = np.column_stack([np.asarray(east, float), np.asarray(north, float)])
        aligned = points @ self.matrix.T + self.translation
        residual = np.zeros_like(points)
        density = np.zeros(len(points), float)
        if len(self.centers):
            squared = ((points[:, None, 0] - self.centers[None, :, 0]) ** 2
                       + (points[:, None, 1] - self.centers[None, :, 1]) ** 2)
            kernel = np.exp(-0.5 * squared / (self.kernel_length_m ** 2))
            residual = kernel @ self.coefficients
            density = kernel.sum(axis=1) / max(len(self.centers), 1)
        delta = aligned - points + residual
        return delta[:, 0], delta[:, 1], density


def vertical_model(panels: Iterable[dict], geometry: dict[str, dict],
                   thermal_panels: dict[str, dict]) -> dict:
    model = {}
    for panel in panels:
        panel_id = panel["panel_id"]
        visible = plane_from_row(geometry.get(panel_id, {}), "visible_plane")
        visible_plane = None if visible is None else visible.to_dict()
        thermal_panel = thermal_panels.get(panel_id, {})
        thermal = thermal_panel.get("panel_plane")
        thermal_plane = None if thermal is None else LocalPlane.from_model(thermal).to_dict()
        model[panel_id] = {
            "array_id": str(panel["array_id"]),
            "visible_plane": visible_plane,
            "thermal_plane": thermal_plane,
            "thermal_correction_method": thermal_panel.get("vertical_correction_method", "array_bias"),
        }
    return model


def panels_from_model(vertical: dict) -> dict[str, dict]:
    panels = {}
    for panel_id, row in vertical.items():
        visible_plane = (None if row.get("visible_plane") is None
                         else LocalPlane.from_model(row["visible_plane"]))
        thermal_plane = (None if row.get("thermal_plane") is None
                         else LocalPlane.from_model(row["thermal_plane"]))
        panels[panel_id] = {"visible_plane": visible_plane,
                            "thermal_plane": thermal_plane,
                            "thermal_correction_method": row.get("thermal_correction_method", "array_bias"),
                            "array_id": row.get("array_id")}
    return panels


def validate_residual_model(model: dict) -> None:
    """Validate the canonical residual model structure."""
    if not isinstance(model, dict):
        raise ValueError("residual model must be an object")
    planimetric = model.get("planimetric")
    vertical = model.get("vertical")
    if not isinstance(planimetric, dict) or not isinstance(vertical, dict):
        raise ValueError("residual model requires planimetric and vertical objects")
    panels = vertical.get("panels")
    if not isinstance(panels, dict):
        raise ValueError("vertical.panels must be an object")
    for panel_id, row in panels.items():
        if not isinstance(row, dict):
            raise ValueError("panel %s must be an object" % panel_id)
        for key in ("visible_plane", "thermal_plane"):
            plane = row.get(key)
            if plane is not None:
                try:
                    LocalPlane.from_model(plane)
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError("panel %s %s has invalid plane: %s" % (panel_id, key, exc)) from exc
    if planimetric.get("type") != REGULARIZED_FIELD_TYPE:
        raise ValueError("unsupported planimetric type: %s" % planimetric.get("type"))
    similarity = planimetric.get("similarity")
    residual = planimetric.get("residual")
    if not isinstance(similarity, dict) or not isinstance(residual, dict):
        raise ValueError("regularized planimetric model requires similarity and residual")
    if np.asarray(similarity.get("matrix")).shape != (2, 2):
        raise ValueError("similarity.matrix must be 2x2")
    if np.asarray(similarity.get("translation")).shape != (2,):
        raise ValueError("similarity.translation must have two values")
    if len(residual.get("centers", [])) != len(residual.get("weights", [])):
        raise ValueError("residual centers/weights length mismatch")
    if len(residual.get("coefficients_e", [])) != len(residual.get("centers", [])):
        raise ValueError("residual coefficients_e length mismatch")
    if len(residual.get("coefficients_n", [])) != len(residual.get("centers", [])):
        raise ValueError("residual coefficients_n length mismatch")
    if not isinstance(residual.get("tiles"), list):
        raise ValueError("residual.tiles must be a list")

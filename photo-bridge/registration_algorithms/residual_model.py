"""Planimetric field: global similarity + regularized RBF residual."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


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
    def from_tiles(cls, rows, max_distance_m: float = 30.0):
        """Fit a field from tile rows by delegating to fit_regularized_field.

        Empty rows produce the empty field: identity similarity with no
        residual support.
        """
        rows = list(rows)
        if not rows:
            return RegularizedPlanimetricField.empty(max_distance_m)
        return fit_regularized_field(rows, tile_m=10.0,
                                     max_distance_m=max_distance_m)

    def evaluate(self, east, north):
        raise NotImplementedError

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
                          ridge: float = 0.1):
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
    # A non-finite or non-positive correlation is the least trustworthy
    # observation, not the most: floor it at the minimum weight instead of
    # replacing it with the maximum (1.0).  clip also bounds finite values.
    cc = np.clip(np.where(np.isfinite(cc), cc, 1e-3), 1e-3, 1.0)
    quality = np.asarray([float(row.get("quality_weight", 1.0)) for row in rows], float)
    quality = np.clip(np.where(np.isfinite(quality), quality, 1e-3), 1e-3, 1.0)
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
    support = np.exp(-0.5 * ((tree.query(
        np.column_stack([east, north]), k=2)[0][:, 1]
        / max(spacing_median, 1e-6)) - 1.0) ** 2)
    weights = cc * np.where(np.isfinite(support), support, 1.0)
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
    rhs = weighted_kernel.T @ fit_residual
    coefficients = np.linalg.solve(normal, rhs)
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

    def _kernel(self, east, north):
        points = np.column_stack([np.asarray(east, float), np.asarray(north, float)])
        squared = ((points[:, None, 0] - self.centers[None, :, 0]) ** 2
                   + (points[:, None, 1] - self.centers[None, :, 1]) ** 2)
        return np.exp(-0.5 * squared / (self.kernel_length_m ** 2))

    def rbf_component(self, east, north):
        """Return only the RBF residual displacement g(p)."""
        return self._kernel(east, north) @ self.coefficients

    def jacobian(self, east, north):
        """Return J_f(p), det(J), sigma_min, sigma_max for each query point."""
        points = np.column_stack([np.asarray(east, float), np.asarray(north, float)])
        kernel = self._kernel(east, north)
        offsets = points[:, None, :] - self.centers[None, :, :]
        correction = (1.0 / (self.kernel_length_m ** 2)) * np.einsum(
            "nm,ma,nmb->nab", kernel, self.coefficients, offsets)
        jacobian = self.matrix[None, :, :] - correction
        determinant = np.linalg.det(jacobian)
        singular = np.linalg.svd(jacobian, compute_uv=False)
        return {
            "jacobian": jacobian,
            "det": determinant,
            "sigma_min": singular[:, -1],
            "sigma_max": singular[:, 0],
        }

    def support_metrics(self, east, north):
        """Local effective count and weighted covariance of observation centres."""
        points = np.column_stack([np.asarray(east, float), np.asarray(north, float)])
        kernel = self._kernel(east, north)
        a = self.weights[None, :] * kernel
        total = np.sum(a, axis=1)
        total_sq = np.sum(a * a, axis=1)
        effective = np.divide(total * total, total_sq,
                              out=np.zeros_like(total), where=total_sq > 0.0)
        mean = np.divide(a @ self.centers, total[:, None],
                         out=np.zeros((len(points), 2)),
                         where=total[:, None] > 0.0)
        centered = self.centers[None, :, :] - mean[:, None, :]
        covariance = np.einsum("nm,nmb,nmc->nbc", a, centered, centered)
        covariance = np.divide(covariance, total[:, None, None],
                               out=np.zeros_like(covariance),
                               where=total[:, None, None] > 0.0)
        eigenvalues = np.linalg.eigvalsh(covariance)
        nearest = np.full(len(points), np.inf)
        if self.tree is not None and len(points):
            nearest, _ = self.tree.query(points, k=1)
        return {
            "weight_sum": total,
            "effective_count": effective,
            "covariance": covariance,
            "lambda_min": eigenvalues[:, 0],
            "lambda_max": eigenvalues[:, -1],
            "nearest_m": nearest,
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




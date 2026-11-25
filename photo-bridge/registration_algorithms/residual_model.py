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

        Every row must carry sigma_e, sigma_n and sigma_en.  Empty rows produce
        the empty field: identity similarity with no residual support.
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



def _coverage_weight_scale(centers: np.ndarray, tile_m: float) -> np.ndarray:
    """Mean tile-footprint coverage count for each observation centre."""
    centers = np.asarray(centers, float)
    if len(centers) < 2:
        return np.ones(len(centers), float)
    cell = max(float(tile_m) / 8.0, 1e-3)
    half = float(tile_m) / 2.0
    low = centers.min(axis=0) - half
    high = centers.max(axis=0) + half
    shape = np.maximum(1, np.ceil((high - low) / cell).astype(int) + 1)
    counts = np.zeros(tuple(shape), np.float32)
    for east, north in centers:
        left = int(np.floor((east - half - low[0]) / cell))
        right = int(np.ceil((east + half - low[0]) / cell))
        bottom = int(np.floor((north - half - low[1]) / cell))
        top = int(np.ceil((north + half - low[1]) / cell))
        left, right = max(0, left), min(shape[0], right)
        bottom, top = max(0, bottom), min(shape[1], top)
        if right > left and top > bottom:
            counts[left:right, bottom:top] += 1.0
    scales = np.ones(len(centers), float)
    for index, (east, north) in enumerate(centers):
        left = int(np.floor((east - half - low[0]) / cell))
        right = int(np.ceil((east + half - low[0]) / cell))
        bottom = int(np.floor((north - half - low[1]) / cell))
        top = int(np.ceil((north + half - low[1]) / cell))
        left, right = max(0, left), min(shape[0], right)
        bottom, top = max(0, bottom), min(shape[1], top)
        if right > left and top > bottom:
            scales[index] = max(1.0, float(np.mean(
                counts[left:right, bottom:top])))
    return scales

def _weighted_displacement_similarity(east: np.ndarray, north: np.ndarray,
                                      delta_e: np.ndarray, delta_n: np.ndarray,
                                      weights: np.ndarray, center: np.ndarray):
    """Robust weighted similarity in centered displacement coordinates.

    The model is d = B (p - center) + t0, with B = A - I constrained to
    [[u, -v], [v, u]].  Centering keeps the large projected coordinates out of
    the rotation/translation coupling.
    """
    source = np.column_stack([east, north]) - np.asarray(center, float)
    target = np.column_stack([delta_e, delta_n])
    weights = np.asarray(weights, float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    mask = np.ones(len(source), bool)
    gradient = np.zeros((2, 2))
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
        u, v = np.linalg.lstsq(design * root_weight[:, None],
                               rhs * root_weight, rcond=None)[0]
        gradient = np.array([[u, -v], [v, u]])
        translation = target_center - gradient @ source_center
        residual = target - (source @ gradient.T + translation)
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
    return gradient, translation, mask


def _observation_precision(rows):
    """Return normalized inverse ECC covariances as (a, b, d).

    Each row must carry sigma_e, sigma_n and sigma_en.  For every row,
    Sigma^-1 = [[a, b], [b, d]].  The precision is normalized by the median
    average precision so the ridge scale remains meaningful across stations;
    a common scaling of all covariances is absorbed by that normalization.
    """
    try:
        sigma_e = np.asarray([float(row["sigma_e"]) for row in rows], float)
        sigma_n = np.asarray([float(row["sigma_n"]) for row in rows], float)
        sigma_en = np.asarray([float(row["sigma_en"]) for row in rows], float)
    except KeyError as exc:
        raise ValueError(
            "covariance RBF weighting requires sigma_e/sigma_n/sigma_en "
            "on every accepted row") from exc
    variance_e = sigma_e ** 2
    variance_n = sigma_n ** 2
    determinant = variance_e * variance_n - sigma_en ** 2
    if not np.all(np.isfinite(determinant)) or np.any(determinant <= 0.0):
        raise ValueError("observation covariance must be positive definite")
    a = variance_n / determinant
    b = -sigma_en / determinant
    d = variance_e / determinant
    scale = float(np.median(0.5 * (a + d)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("observation covariance precision scale must be positive")
    return a / scale, b / scale, d / scale


def fit_regularized_field(rows: Iterable[dict], tile_m: float,
                          max_distance_m: float = 30.0,
                          kernel_length_m: float | None = None,
                          ridge: float = 0.1,
                          normalize_coverage: bool = True):
    """Fit global similarity + covariance-weighted RBF residual.

    The RBF residual is fitted by generalized least squares using each tile's
    2x2 ECC covariance, so east/north precision and their correlation are
    respected.  Every accepted row must carry sigma_e, sigma_n and sigma_en;
    there is no scalar-weight fallback.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("cannot fit a planimetric field without accepted tiles")
    east = np.asarray([float(row["east"]) for row in rows], float)
    north = np.asarray([float(row["north"]) for row in rows], float)
    delta_e = np.asarray([float(row["dx_m"]) for row in rows], float)
    delta_n = np.asarray([float(row["dy_m"]) for row in rows], float)
    center = np.array([float(np.mean(east)), float(np.mean(north))])
    cc = np.asarray([float(row["cc"]) for row in rows], float)
    # A non-finite or non-positive correlation is the least trustworthy
    # observation, not the most: floor it at the minimum weight instead of
    # replacing it with the maximum (1.0).  clip also bounds finite values.
    cc = np.clip(np.where(np.isfinite(cc), cc, 1e-3), 1e-3, 1.0)
    cc_raw = cc.copy()
    quality = np.asarray([float(row.get("quality_weight", 1.0)) for row in rows], float)
    quality = np.clip(np.where(np.isfinite(quality), quality, 1e-3), 1e-3, 1.0)
    cc = cc * quality
    if normalize_coverage:
        cc = cc / _coverage_weight_scale(
            np.column_stack([east, north]), tile_m)
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
    gradient, translation, inliers = _weighted_displacement_similarity(
        east, north, delta_e, delta_n, weights, center)
    centers = np.column_stack([east, north])
    centered = centers - center
    residual = np.column_stack([delta_e, delta_n]) - (
        centered @ gradient.T + translation)
    fit_weights = weights * np.where(inliers, 1.0, 0.2)
    fit_residual = residual
    distances = np.hypot(centers[:, None, 0] - centers[None, :, 0],
                         centers[:, None, 1] - centers[None, :, 1])
    kernel = np.exp(-0.5 * (distances / kernel_length_m) ** 2)
    a, b, d = _observation_precision(rows)
    a = a * fit_weights
    b = b * fit_weights
    d = d * fit_weights
    identity = np.eye(len(centers))
    m_ee = kernel.T @ (a[:, None] * kernel) + float(ridge) * identity
    m_en = kernel.T @ (b[:, None] * kernel)
    m_nn = kernel.T @ (d[:, None] * kernel) + float(ridge) * identity
    rhs_e = kernel.T @ (a * fit_residual[:, 0] + b * fit_residual[:, 1])
    rhs_n = kernel.T @ (b * fit_residual[:, 0] + d * fit_residual[:, 1])
    solution = np.linalg.solve(
        np.block([[m_ee, m_en], [m_en, m_nn]]),
        np.concatenate([rhs_e, rhs_n]))
    coefficients = np.column_stack([solution[:len(centers)],
                                    solution[len(centers):]])
    tiles = [{
        "east": float(east[i]), "north": float(north[i]), "cc": float(cc_raw[i]),
        "residual_e": float(residual[i, 0]), "residual_n": float(residual[i, 1]),
    } for i in range(len(rows))]
    return RegularizedPlanimetricField(
        tiles=tiles, centers=centers, weights=fit_weights,
        coefficients=coefficients, displacement_gradient=gradient,
        translation=translation, center=center,
        kernel_length_m=kernel_length_m, ridge=float(ridge),
        max_distance_m=float(max_distance_m), tile_m=float(tile_m))


class RegularizedPlanimetricField(PlanimetricField):
    """Global similarity plus a smooth RBF residual that decays off-support."""

    EVALUATION_CHUNK = 250_000

    def __init__(self, tiles, centers, weights, coefficients,
                 displacement_gradient, translation, center,
                 kernel_length_m, ridge, max_distance_m,
                 tile_m: float | None = None):
        self.tiles = list(tiles)
        self.east = np.asarray([tile["east"] for tile in self.tiles], float)
        self.north = np.asarray([tile["north"] for tile in self.tiles], float)
        self.cc = np.asarray([tile["cc"] for tile in self.tiles], float)
        self.residual_e = np.asarray([tile["residual_e"] for tile in self.tiles], float)
        self.residual_n = np.asarray([tile["residual_n"] for tile in self.tiles], float)
        self.centers = np.asarray(centers, float)
        self.weights = np.asarray(weights, float)
        self.coefficients = np.asarray(coefficients, float)
        self.center = np.asarray(center, float)
        self.displacement_gradient = np.asarray(displacement_gradient, float)
        self.translation = np.asarray(translation, float)
        self.kernel_length_m = float(kernel_length_m)
        self.ridge = float(ridge)
        self.max_distance_m = float(max_distance_m)
        self.tile_m = float(tile_m) if tile_m else self.kernel_length_m / 2.5
        self.tree = cKDTree(np.column_stack([self.east, self.north])) if len(self.east) else None

    @property
    def matrix(self):
        """Absolute similarity mapping A = I + B (compatibility view)."""
        return np.eye(2) + self.displacement_gradient

    @property
    def absolute_translation(self):
        """Absolute mapping translation t_abs = t0 - B p_c."""
        return self.translation - self.displacement_gradient @ self.center

    @classmethod
    def empty(cls, max_distance_m: float = 30.0):
        return cls(tiles=[], centers=np.empty((0, 2)), weights=np.empty(0),
                   coefficients=np.empty((0, 2)),
                   displacement_gradient=np.zeros((2, 2)),
                   translation=np.zeros(2), center=np.zeros(2),
                   kernel_length_m=1.0, ridge=0.1,
                   max_distance_m=max_distance_m, tile_m=1.0)

    @classmethod
    def from_model(cls, model: dict):
        residual = model["residual"]
        similarity = model["similarity"]
        absolute = np.asarray(similarity["matrix"], float)
        absolute_translation = np.asarray(similarity["translation"], float)
        if "center" in similarity:
            center = np.asarray(similarity["center"], float)
            gradient = np.asarray(
                similarity.get("displacement_gradient", absolute - np.eye(2)),
                float)
            translation = np.asarray(
                similarity.get("centered_translation",
                               absolute_translation + gradient @ center), float)
        else:
            # v1 compatibility: the old model used absolute A and t_abs with
            # no reference centre, which is the p_c=0 case.
            center = np.zeros(2)
            gradient = absolute - np.eye(2)
            translation = absolute_translation
        return cls(
            tiles=residual["tiles"],
            centers=np.asarray(residual["centers"], float),
            weights=np.asarray(residual["weights"], float),
            coefficients=np.column_stack([
                np.asarray(residual["coefficients_e"], float),
                np.asarray(residual["coefficients_n"], float)]),
            displacement_gradient=gradient,
            translation=translation,
            center=center,
            kernel_length_m=float(residual["kernel_length_m"]),
            ridge=float(residual["ridge"]),
            max_distance_m=float(model.get("max_distance_m", 30.0)),
            tile_m=float(model.get("tile_m", 0.0)) or None)

    def to_model(self) -> dict:
        rotation = float(np.degrees(np.arctan2(self.matrix[1, 0], self.matrix[0, 0])))
        scale = float(np.hypot(self.matrix[0, 0], self.matrix[1, 0]))
        return {
            "type": REGULARIZED_FIELD_TYPE,
            "similarity": {
                "matrix": self.matrix.tolist(),
                "translation": self.absolute_translation.tolist(),
                "rotation_deg": rotation,
                "scale": scale,
                "center": self.center.tolist(),
                "displacement_gradient": self.displacement_gradient.tolist(),
                "centered_translation": self.translation.tolist(),
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
            "tile_m": self.tile_m,
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
        """Return the mapping Jacobian J_T, det, and singular values."""
        points = np.column_stack([np.asarray(east, float), np.asarray(north, float)])
        kernel = self._kernel(east, north)
        offsets = points[:, None, :] - self.centers[None, :, :]
        correction = (1.0 / (self.kernel_length_m ** 2)) * np.einsum(
            "nm,ma,nmb->nab", kernel, self.coefficients, offsets)
        jacobian = (np.eye(2)[None, :, :]
                    + self.displacement_gradient[None, :, :] - correction)
        determinant = np.linalg.det(jacobian)
        singular = np.linalg.svd(jacobian, compute_uv=False)
        return {
            "jacobian": jacobian,
            "det": determinant,
            "sigma_min": singular[:, -1],
            "sigma_max": singular[:, 0],
        }


    def support_confidence(self, east, north):
        """Continuous tile-scale Gaussian support in [0, 1]."""
        east = np.asarray(east, float)
        north = np.asarray(north, float)
        if len(self.centers) == 0 or len(east) == 0:
            return np.zeros(len(east), float)
        sigma = max(self.tile_m / 2.0, 1e-6)
        total_weight = max(float(self.weights.sum()), 1e-12)
        result = np.empty(len(east), float)
        chunk = 250_000
        for start in range(0, len(east), chunk):
            stop = start + chunk
            points = np.column_stack([east[start:stop], north[start:stop]])
            squared = ((points[:, None, 0] - self.centers[None, :, 0]) ** 2
                       + (points[:, None, 1] - self.centers[None, :, 1]) ** 2)
            kernel = np.exp(-0.5 * squared / (sigma ** 2))
            result[start:stop] = (kernel @ self.weights) / total_weight
        return result

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
        centered = points - self.center
        displacement = centered @ self.displacement_gradient.T + self.translation
        residual = np.zeros_like(points)
        density = np.zeros(len(points), float)
        if len(self.centers):
            squared = ((points[:, None, 0] - self.centers[None, :, 0]) ** 2
                       + (points[:, None, 1] - self.centers[None, :, 1]) ** 2)
            kernel = np.exp(-0.5 * squared / (self.kernel_length_m ** 2))
            residual = kernel @ self.coefficients
            density = kernel.sum(axis=1) / max(len(self.centers), 1)
        delta = displacement + residual
        return delta[:, 0], delta[:, 1], density




"""Tangent-local boundary-correction field.

The boundary is divided into overlapping tangent bins.  A small normal/tangent
model is fitted per side/bin from accepted tile residuals.  Bins without enough
observations produce no correction instead of borrowing a global coefficient
from an unrelated part of the edge.
"""
from __future__ import annotations

import numpy as np

DEFAULT_BAND_M = 40.0
DEFAULT_BIN_M = 50.0
MIN_LOCAL_TILES = 4
DEFAULT_RIDGE = 0.01

_NORMALS = np.array([[-1.0, 0.0], [1.0, 0.0],
                     [0.0, -1.0], [0.0, 1.0]])
_TANGENTS = np.array([[0.0, 1.0], [0.0, 1.0],
                      [1.0, 0.0], [1.0, 0.0]])


def _geometry(east, north, bounds):
    west, east_max, south, north_max = (float(value) for value in bounds)
    distances = np.column_stack([
        np.asarray(east, float) - west,
        east_max - np.asarray(east, float),
        np.asarray(north, float) - south,
        north_max - np.asarray(north, float),
    ])
    side = np.argmin(distances, axis=1)
    inward = distances[np.arange(len(side)), side]
    return side, inward


def _features(east, north, bounds, side, inward, taper_m):
    phi = np.maximum(0.0, 1.0 - inward / float(taper_m)) ** 2
    size = len(np.asarray(east))
    design = np.zeros((2 * size, 6), float)
    for index in range(4):
        active = phi * (side == index)
        normal = _NORMALS[index]
        tangent = _TANGENTS[index]
        design[:size, 0] += active * normal[0]
        design[:size, 1] += active * tangent[0]
        design[:size, 2] += active * inward * normal[0]
        design[size:, 3] += active * normal[1]
        design[size:, 4] += active * tangent[1]
        design[size:, 5] += active * inward * normal[1]
    return design


def _tile_data(field):
    tiles = getattr(field, "tiles", None)
    if not tiles:
        return None
    centers = np.column_stack([
        [float(tile["east"]) for tile in tiles],
        [float(tile["north"]) for tile in tiles],
    ])
    residual = np.column_stack([
        [float(tile["residual_e"]) for tile in tiles],
        [float(tile["residual_n"]) for tile in tiles],
    ])
    gradient = np.asarray(field.displacement_gradient, float)
    translation = np.asarray(field.translation, float)
    centered = centers - np.asarray(field.center, float)
    delta = centered @ gradient.T + translation + residual
    weights = np.asarray([float(tile.get("cc", 1.0)) for tile in tiles], float)
    return centers, delta, weights


def _fit_local(field, bounds):
    data = _tile_data(field)
    if data is None:
        return []
    centers, delta, weights = data
    side, inward = _geometry(centers[:, 0], centers[:, 1], bounds)
    base_e, base_n, _, _ = field.evaluate(centers[:, 0], centers[:, 1])
    residual = delta - np.column_stack([base_e, base_n])
    tile_m = float(getattr(field, "tile_m", DEFAULT_BIN_M))
    bin_m = 2.0 * tile_m
    half_window_m = 2.0 * tile_m
    fit_taper_m = 4.0 * tile_m
    west, east_max, south, north_max = (float(value) for value in bounds)
    fits = []
    for index in range(4):
        tangent = centers[:, 0] if index in (2, 3) else centers[:, 1]
        tangent_min = west if index in (2, 3) else south
        tangent_max = east_max if index in (2, 3) else north_max
        centers_t = np.arange(tangent_min, tangent_max + 1e-6, bin_m)
        for center in centers_t:
            selected = ((side == index) & (inward <= fit_taper_m)
                        & (np.abs(tangent - center) <= half_window_m))
            if int(selected.sum()) < MIN_LOCAL_TILES:
                continue
            s = inward[selected]
            local_features = _features(
                centers[selected, 0], centers[selected, 1], bounds,
                np.full(int(selected.sum()), index),
                s, fit_taper_m)
            root = np.sqrt(np.clip(weights[selected], 1e-3, 1.0))
            row_weight = np.concatenate([root, root])[:, None]
            target = np.concatenate([
                residual[selected, 0] * root,
                residual[selected, 1] * root,
            ])
            augmented = np.vstack([
                local_features * row_weight,
                np.sqrt(float(DEFAULT_RIDGE)) * np.eye(6),
            ])
            coefficients = np.linalg.lstsq(
                augmented, np.concatenate([target, np.zeros(6)]),
                rcond=None)[0]
            fits.append({
                "side": index, "center": float(center),
                "coefficients": coefficients,
            })
    return fits


def evaluate_boundary_field(field, east, north, bounds, *,
                            band_m: float | None = None,
                            base_e=None, base_n=None):
    """Apply the tangent-local boundary model and return edge diagnostics.

    ``base_e``/``base_n`` let callers reuse a displacement already evaluated
    by the mapping layer instead of evaluating the whole field twice.
    """
    east = np.asarray(east, float)
    north = np.asarray(north, float)
    if base_e is None or base_n is None:
        base_e, base_n, _, _ = field.evaluate(east, north)
    base_e = np.asarray(base_e, float).copy()
    base_n = np.asarray(base_n, float).copy()
    size = len(east)
    edge_distance = np.full(size, np.inf, float)
    applied = np.zeros(size, bool)
    if size == 0:
        return base_e, base_n, edge_distance, applied
    fits = _fit_local(field, bounds)
    if not fits:
        return base_e, base_n, edge_distance, applied
    side, inward = _geometry(east, north, bounds)
    edge_distance = inward
    tile_m = float(getattr(field, "tile_m", DEFAULT_BAND_M))
    range_m = 2.0 * tile_m if band_m is None else float(band_m)
    bin_m = 2.0 * tile_m
    half_window_m = 2.0 * tile_m
    fit_taper_m = 4.0 * tile_m
    selected = (inward >= 0.0) & (inward < range_m)
    if not np.any(selected):
        return base_e, base_n, edge_distance, applied
    tangent = east if size else east
    # top/bottom use east tangent; left/right use north. Evaluate per side so
    # each query selects the nearest local bin centre.
    for index in range(4):
        local = selected & (side == index)
        if not np.any(local):
            continue
        candidates = [fit for fit in fits if fit["side"] == index]
        if not candidates:
            continue
        centers = np.asarray([fit["center"] for fit in candidates])
        idx = np.flatnonzero(local)
        t = (east[idx] if index in (2, 3) else north[idx])
        nearest_bin = np.argmin(np.abs(centers[None, :] - t[:, None]), axis=1)
        for position, (local_index, bin_index) in enumerate(zip(idx, nearest_bin)):
            fit = candidates[int(bin_index)]
            if abs(t[position] - fit["center"]) > (
                    bin_m / 2.0 + half_window_m):
                continue
            feature = _features(
                np.array([east[local_index]]), np.array([north[local_index]]),
                bounds, np.array([index]), np.array([inward[local_index]]),
                fit_taper_m)
            correction = (feature @ fit["coefficients"]).reshape(2)
            # Gaussian with sigma = T, truncated and normalized to zero at 2T.
            raw = np.exp(-0.5 * (inward[local_index] / tile_m) ** 2)
            edge = np.exp(-2.0)
            taper = max(0.0, (raw - edge) / (1.0 - edge))
            base_e[local_index] += taper * correction[0]
            base_n[local_index] += taper * correction[1]
            applied[local_index] = True
    return base_e, base_n, edge_distance, applied

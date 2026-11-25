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


def _side_distances(east, north, bounds):
    """Perpendicular distance to the left/right/bottom/top edge."""
    west, east_max, south, north_max = (float(value) for value in bounds)
    return np.column_stack([
        np.asarray(east, float) - west,
        east_max - np.asarray(east, float),
        np.asarray(north, float) - south,
        north_max - np.asarray(north, float),
    ])


def _geometry(east, north, bounds):
    distances = _side_distances(east, north, bounds)
    side = np.argmin(distances, axis=1)
    inward = distances[np.arange(len(side)), side]
    return side, inward


def _side_features(inward, index, taper_m):
    inward = np.asarray(inward, float)
    phi = np.maximum(0.0, 1.0 - inward / float(taper_m)) ** 2
    size = len(inward)
    design = np.zeros((2 * size, 6), float)
    normal = _NORMALS[index]
    tangent = _TANGENTS[index]
    design[:size, 0] = phi * normal[0]
    design[:size, 1] = phi * tangent[0]
    design[:size, 2] = phi * inward * normal[0]
    design[size:, 3] = phi * normal[1]
    design[size:, 4] = phi * tangent[1]
    design[size:, 5] = phi * inward * normal[1]
    return design


def _features(east, north, bounds, side, inward, taper_m):
    side = np.asarray(side, int)
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
    tile_m = float(getattr(field, "tile_m", DEFAULT_BAND_M))
    range_m = 2.0 * tile_m if band_m is None else float(band_m)
    bin_m = 2.0 * tile_m
    half_window_m = 2.0 * tile_m
    fit_taper_m = 4.0 * tile_m
    distances = _side_distances(east, north, bounds)
    side = np.argmin(distances, axis=1)
    edge_distance = distances[np.arange(size), side]
    in_band = (distances >= 0.0) & (distances < range_m)
    if not np.any(in_band):
        return base_e, base_n, edge_distance, applied
    edge_value = np.exp(-2.0)
    alpha = np.maximum(
        0.0, (np.exp(-0.5 * (distances / tile_m) ** 2) - edge_value)
        / (1.0 - edge_value))
    # Two edges share a point only where their distances are comparable, so the
    # inverse-square weights remove the jump at a corner bisector.  The envelope
    # is the taper of the strongest contributing edge, not a sum of tapers, so
    # a single-edge correction keeps its previous magnitude.
    weight = np.zeros(size)
    envelope = np.zeros(size)
    correction_e = np.zeros(size)
    correction_n = np.zeros(size)
    for index in range(4):
        candidates = sorted((fit for fit in fits if fit["side"] == index),
                            key=lambda fit: fit["center"])
        active = np.flatnonzero(in_band[:, index])
        if not candidates or len(active) == 0:
            continue
        centers = np.asarray([fit["center"] for fit in candidates], float)
        coefficients = np.asarray([fit["coefficients"] for fit in candidates],
                                  float)
        t = east[active] if index in (2, 3) else north[active]
        if len(centers) > 1:
            upper = np.clip(np.searchsorted(centers, t, side="right"), 1,
                            len(centers) - 1)
            lower = upper - 1
            span = centers[upper] - centers[lower]
            blend = np.divide(t - centers[lower], span,
                              out=np.zeros(len(t), float), where=span > 0.0)
            blend = np.clip(blend, 0.0, 1.0)
        else:
            lower = upper = np.zeros(len(t), int)
            blend = np.zeros(len(t), float)
        within = np.abs(t - np.clip(t, centers[0], centers[-1])) <= (
            bin_m / 2.0 + half_window_m)
        if not np.any(within):
            continue
        active, lower, upper, blend = (active[within], lower[within],
                                       upper[within], blend[within])
        inward_local = distances[active, index]
        count = len(active)
        features = _side_features(inward_local, index, fit_taper_m)
        east_features = features[:count]
        north_features = features[count:]
        lower_coefficients = coefficients[lower]
        upper_coefficients = coefficients[upper]
        # east rows only populate columns 0-2, north rows only 3-5, so the full
        # six-element coefficient vector applies to each half of the design
        part_e = ((1.0 - blend) * np.einsum(
            "nk,nk->n", east_features, lower_coefficients)
            + blend * np.einsum("nk,nk->n", east_features, upper_coefficients))
        part_n = ((1.0 - blend) * np.einsum(
            "nk,nk->n", north_features, lower_coefficients)
            + blend * np.einsum("nk,nk->n", north_features, upper_coefficients))
        local_weight = 1.0 / np.maximum(inward_local, 1e-6) ** 2
        weight[active] += local_weight
        envelope[active] = np.maximum(envelope[active], alpha[active, index])
        correction_e[active] += local_weight * part_e
        correction_n[active] += local_weight * part_n
    corrected = weight > 0.0
    if not np.any(corrected):
        return base_e, base_n, edge_distance, applied
    scale = envelope[corrected] / weight[corrected]
    base_e[corrected] += scale * correction_e[corrected]
    base_n[corrected] += scale * correction_n[corrected]
    applied[corrected] = True
    return base_e, base_n, edge_distance, applied

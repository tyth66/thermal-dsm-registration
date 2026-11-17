"""High-pass DSM registration and robust planimetric field estimation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from .io_utils import stats
from .residual_model import fit_regularized_field
from .tile_observations import score_shift, tile_candidates


def load_grid(visible: Path, thermal: Path, resolution: float,
              num_threads: int = 1):
    with rasterio.open(visible) as first, rasterio.open(thermal) as second:
        west = max(first.bounds.left, second.bounds.left)
        east = min(first.bounds.right, second.bounds.right)
        south = max(first.bounds.bottom, second.bounds.bottom)
        north = min(first.bounds.top, second.bounds.top)
        width = int((east - west) / resolution)
        height = int((north - south) / resolution)
        transform = from_origin(west, north, resolution, resolution)

        def sample(dataset):
            array = np.full((height, width), np.nan, np.float32)
            reproject(rasterio.band(dataset, 1), array,
                      src_transform=dataset.transform, src_crs=dataset.crs,
                      dst_transform=transform, dst_crs=dataset.crs,
                      resampling=Resampling.bilinear, src_nodata=dataset.nodata,
                      dst_nodata=np.nan, num_threads=max(1, int(num_threads)))
            valid = np.isfinite(array) & (array > -100) & (array < 80)
            return array, valid

        visible_z, visible_valid = sample(first)
        thermal_z, thermal_valid = sample(second)
    return visible_z, visible_valid, thermal_z, thermal_valid, transform, (west, north)


def highpass(array: np.ndarray, valid: np.ndarray, sigma_px: float,
             minimum_weight: float = 0.5) -> np.ndarray:
    """High-pass elevation with nodata-aware normalized Gaussian filtering.

    Invalid pixels are excluded from the local mean instead of being filled with
    an arbitrary median.  Pixels whose local valid weight is too low are
    protected (set to zero) so a nodata boundary cannot create a false edge.
    """
    valid = np.asarray(valid, bool)
    values = np.where(valid, np.nan_to_num(array, nan=0.0), 0.0).astype(np.float32)
    weights = valid.astype(np.float32)
    weighted_sum = cv2.GaussianBlur(values, (0, 0), sigma_px)
    weight_sum = cv2.GaussianBlur(weights, (0, 0), sigma_px)
    local_mean = np.divide(weighted_sum, weight_sum,
                           out=np.zeros_like(weighted_sum),
                           where=weight_sum > 1e-6)
    high = values - local_mean
    high[~valid] = 0.0
    high[weight_sum < float(minimum_weight)] = 0.0
    return high


def pixel_shift_to_world(shift_x: float, shift_y: float,
                        resolution_m: float):
    """Convert ECC/phase pixel shifts to world EN displacement.

    OpenCV returns the shift used by the inverse warp that aligns the input
    image (thermal) to the template (visible).  Raster rows increase downward
    while north increases upward, hence the sign change on y.
    """
    return shift_x * resolution_m, -shift_y * resolution_m


def ecc(first, second, warp=None):
    warp = np.eye(2, 3, dtype=np.float32) if warp is None else warp.astype(np.float32)
    try:
        correlation, warp = cv2.findTransformECC(
            first, second, warp, cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7), None, 5)
        return float(warp[0, 2]), float(warp[1, 2]), float(correlation)
    except cv2.error:
        return float("nan"), float("nan"), float("nan")


def _register_tile(visible_high, thermal_high, joint, transform, resolution,
                   tile: int, min_cc: float, max_shift_m: float,
                   row: int, column: int, top: int, left: int):
    mask = joint[top:top + tile, left:left + tile]
    if mask.mean() < 0.8:
        return None
    first = visible_high[top:top + tile, left:left + tile].astype(np.float32).copy()
    second = thermal_high[top:top + tile, left:left + tile].astype(np.float32).copy()
    if first.std() < 1e-6 or second.std() < 1e-6:
        return None
    first = (first - first.mean()) / (first.std() + 1e-9)
    second = (second - second.mean()) / (second.std() + 1e-9)
    coarse_first = cv2.resize(first, (tile // 2, tile // 2),
                              interpolation=cv2.INTER_AREA)
    coarse_second = cv2.resize(second, (tile // 2, tile // 2),
                               interpolation=cv2.INTER_AREA)
    coarse_x, coarse_y, _ = ecc(coarse_first, coarse_second)
    initial = None
    if np.isfinite(coarse_x):
        initial = np.array([[1.0, 0.0, 2.0 * coarse_x],
                            [0.0, 1.0, 2.0 * coarse_y]], np.float32)
    shift_x, shift_y, correlation = ecc(first, second, initial)
    if not np.isfinite(correlation):
        return None
    dx, dy = pixel_shift_to_world(shift_x, shift_y, resolution)
    if abs(dx) > max_shift_m or abs(dy) > max_shift_m:
        return None
    warp = np.array([[1.0, 0.0, -shift_x], [0.0, 1.0, -shift_y]], np.float32)
    aligned = cv2.warpAffine(second, warp, (tile, tile),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    center = transform * (left + tile / 2.0, top + tile / 2.0)
    candidates = [{
        "dx_m": float(dx), "dy_m": float(dy),
        "score": score_shift(first, second, mask, -shift_x, -shift_y),
        "phase_score": float(correlation),
        "shift_px": (float(-shift_x), float(-shift_y)),
        "source": "ecc",
    }]
    for candidate in tile_candidates(
            first, second, mask, resolution, top_k=5,
            min_separation_px=2):
        px, py = candidate.pop("shift_px")
        if any(abs(px - row["shift_px"][0]) <= 1
               and abs(py - row["shift_px"][1]) <= 1 for row in candidates):
            continue
        candidate["shift_px"] = (px, py)
        candidate["source"] = "phase"
        candidates.append(candidate)
    candidates.sort(key=lambda row: row["score"], reverse=True)
    candidates = candidates[:5]
    second_margin = (candidates[0]["score"] - candidates[1]["score"]
                     if len(candidates) > 1 else 1.0)
    separation_px = (np.hypot(candidates[0]["shift_px"][0] - candidates[1]["shift_px"][0],
                              candidates[0]["shift_px"][1] - candidates[1]["shift_px"][1])
                     if len(candidates) > 1 else 0.0)
    periodic_ambiguity = bool(
        len(candidates) > 1
        and second_margin <= 0.05
        and separation_px * resolution >= 1.0)
    unique = not periodic_ambiguity
    quality_weight = (1.0 if unique else
                      float(np.clip(second_margin / 0.05, 0.25, 0.75)))
    zero_x, zero_y, zero_cc = ecc(first, second)
    if np.isfinite(zero_x) and np.isfinite(zero_y):
        initial_delta_px = float(np.hypot(zero_x - shift_x, zero_y - shift_y))
    else:
        initial_delta_px = float("inf")
    initial_consistent = bool(
        np.isfinite(initial_delta_px)
        and initial_delta_px <= max(2.0, 0.02 * tile))
    return {
        "row": row, "column": column,
        "east": float(center[0]), "north": float(center[1]),
        "dx_m": dx, "dy_m": dy,
        "magnitude_m": float(np.hypot(dx, dy)),
        "cc": correlation,
        "ncc_before": float(np.mean(first * second)),
        "ncc_after": float(np.mean(first * aligned)),
        "valid_fraction": float(mask.mean()),
        "accepted": bool(correlation >= min_cc and unique),
        "outlier": False,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "second_margin": float(second_margin),
        "periodic_ambiguity": periodic_ambiguity,
        "unique": unique,
        "quality_weight": float(quality_weight),
        "initial_cc": float(zero_cc) if np.isfinite(zero_cc) else None,
        "initial_delta_px": (None if not np.isfinite(initial_delta_px)
                             else float(initial_delta_px)),
        "initial_consistent": initial_consistent,
    }


def register(visible_high, thermal_high, joint, transform, resolution,
             tile_m: float, step_m: float, min_cc: float, max_shift_m: float,
             workers: int = 1):
    tile = int(round(tile_m / resolution))
    step = int(round(step_m / resolution))
    height, width = joint.shape
    tasks = [(row, column, top, left)
             for row, top in enumerate(range(0, height - tile, step))
             for column, left in enumerate(range(0, width - tile, step))]
    def process(task):
        row, column, top, left = task
        return _register_tile(visible_high, thermal_high, joint, transform,
                              resolution, tile, min_cc, max_shift_m,
                              row, column, top, left)
    workers = max(1, int(workers))
    if workers > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(process, tasks))
    else:
        results = [process(task) for task in tasks]
    return [record for record in results if record is not None]


def reject_outliers(records: list[dict]) -> int:
    accepted = [record for record in records if record["accepted"]]
    if len(accepted) < 8:
        return 0
    centers = np.array([[record["east"], record["north"]] for record in accepted])
    shifts = np.array([[record["dx_m"], record["dy_m"]] for record in accepted])
    deviations = np.empty(len(accepted))
    for index in range(len(accepted)):
        distance = np.hypot(centers[:, 0] - centers[index, 0],
                            centers[:, 1] - centers[index, 1])
        neighbours = np.argsort(distance)[1:7]
        local = np.median(shifts[neighbours], axis=0)
        deviations[index] = float(np.hypot(*(shifts[index] - local)))
    median = float(np.median(deviations))
    mad = 1.4826 * float(np.median(np.abs(deviations - median)))
    threshold = max(0.60, median + 3.0 * mad)
    outliers = 0
    for index, record in enumerate(accepted):
        if deviations[index] > threshold:
            record["outlier"] = True
            outliers += 1
    return outliers


def _cross_validate_folds(records, folds, tile_m, max_distance_m,
                          kernel_length_m, ridge):
    errors_x = []
    errors_y = []
    for train_indices, test_indices in folds:
        train = [records[index] for index in train_indices]
        test = [records[index] for index in test_indices]
        if len(train) < 8 or not test:
            continue
        field = fit_regularized_field(
            train, tile_m=tile_m, max_distance_m=max_distance_m,
            kernel_length_m=kernel_length_m, ridge=ridge)
        east = np.asarray([record["east"] for record in test])
        north = np.asarray([record["north"] for record in test])
        dx, dy, _, _ = field.evaluate(east, north)
        errors_x.extend(record["dx_m"] - value for record, value in zip(test, dx))
        errors_y.extend(record["dy_m"] - value for record, value in zip(test, dy))
    return {"dx": stats(errors_x), "dy": stats(errors_y)}


def cross_validate_spatial(records: list[dict], blocks_x: int = 4,
                          blocks_y: int = 3, buffer_m: float = 4.0,
                          tile_m: float = 40.0, max_distance_m: float = 30.0,
                          kernel_length_m: float | None = None,
                          ridge: float = 0.1):
    """Spatial block holdout with a buffer around the held-out block."""
    usable = [index for index, record in enumerate(records)
              if record["accepted"] and not record["outlier"]]
    if not usable:
        return {"dx": stats([]), "dy": stats([])}
    east = np.asarray([records[index]["east"] for index in usable])
    north = np.asarray([records[index]["north"] for index in usable])
    x_edges = np.linspace(east.min(), east.max(), blocks_x + 1)
    y_edges = np.linspace(north.min(), north.max(), blocks_y + 1)
    folds = []
    for ix in range(blocks_x):
        for iy in range(blocks_y):
            in_block = ((east >= x_edges[ix]) & (east <= x_edges[ix + 1])
                        & (north >= y_edges[iy]) & (north <= y_edges[iy + 1]))
            test = [usable[i] for i, flag in enumerate(in_block) if flag]
            if len(test) < 3:
                continue
            x0, x1 = x_edges[ix] - buffer_m, x_edges[ix + 1] + buffer_m
            y0, y1 = y_edges[iy] - buffer_m, y_edges[iy + 1] + buffer_m
            train = [index for index in usable
                     if not (x0 <= records[index]["east"] <= x1
                             and y0 <= records[index]["north"] <= y1)]
            folds.append((train, test))
    return _cross_validate_folds(records, folds, tile_m, max_distance_m,
                                 kernel_length_m, ridge)


def cross_validate(records: list[dict], fold_kind: str,
                   tile_m: float = 40.0, max_distance_m: float = 30.0,
                   kernel_length_m: float | None = None, ridge: float = 0.1):
    """Evaluate fold-local fields without modifying the input observations."""
    errors_x = []
    errors_y = []
    if fold_kind == "checkerboard":
        folds = [(record["row"] % 2) * 2 + (record["column"] % 2) for record in records]
    else:
        east = np.asarray([record["east"] for record in records])
        north = np.asarray([record["north"] for record in records])
        folds = ((east > np.median(east)).astype(int) * 2
                 + (north > np.median(north)).astype(int)).tolist()
    for fold in sorted(set(folds)):
        train = [record for record, value in zip(records, folds)
                 if value != fold and record["accepted"] and not record["outlier"]]
        test = [record for record, value in zip(records, folds)
                if value == fold and record["accepted"] and not record["outlier"]]
        if len(train) < 8 or not test:
            continue
        field = fit_regularized_field(
            train, tile_m=tile_m, max_distance_m=max_distance_m,
            kernel_length_m=kernel_length_m, ridge=ridge)
        east = np.asarray([record["east"] for record in test])
        north = np.asarray([record["north"] for record in test])
        dx, dy, _, _ = field.evaluate(east, north)
        errors_x.extend(record["dx_m"] - value for record, value in zip(test, dx))
        errors_y.extend(record["dy_m"] - value for record, value in zip(test, dy))
    return {"dx": stats(errors_x), "dy": stats(errors_y)}

"""DSM registration and planimetric field estimation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path

import cv2
import numpy as np
import rasterio
from scipy.stats import chi2
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, reproject

from .dsm_sampling import elevation_range_for
from .masked_ecc import (
    CANDIDATE_MODE_MASS_MIN, SIGMA0_M, SOFTMAX_BETA,
    candidate_mode_quality, estimate_period_m, masked_ecc_candidates,
    masked_ecc_peaks, masked_ecc_surface,
)
# Reuse the single projected-metre CRS predicate instead of duplicating it: a
# DSM grid is only meaningful when both rasters share one metric, north-up frame.
from .visible_world_model import _is_metre_projected

# The station drift domain is a 2-D density mode estimated from the tile
# displacements themselves.  A joint mode is used instead of two marginal
# modes: a rectangular per-axis box can contain combinations of dx/dy that
# have no observations, and it ignores the correlation between axes.  The
# estimator is a deterministic multi-start mean-shift on a Gaussian KDE,
# followed by a 99.5% Mahalanobis ellipse.  The covariance floor keeps the
# domain elliptical instead of collapsing to a line when the drift is uniform.
DISPLACEMENT_DOMAIN_CONFIDENCE = 0.995
DISPLACEMENT_DOMAIN_BANDWIDTH = 0.8
DISPLACEMENT_DOMAIN_MIN_OBSERVATIONS = 8
DISPLACEMENT_DOMAIN_MIN_CLUSTER = 6
DISPLACEMENT_DOMAIN_MAD_FLOOR_M = 0.15
DISPLACEMENT_DOMAIN_COVARIANCE_FLOOR_M2 = 0.25
DISPLACEMENT_DOMAIN_SEED_GRID = 7


def load_grid(visible: Path, thermal: Path, resolution: float,
              num_threads: int = 1):
    with rasterio.open(visible) as first, rasterio.open(thermal) as second:
        # Both DSMs are reprojected onto one grid below, so they must share the
        # same projected-metre CRS.  A geographic CRS (degrees) or two different
        # CRSs would silently misplace every sample instead of failing here.
        if first.crs is None or second.crs is None:
            raise ValueError("visible and thermal DSM must both declare a CRS")
        if first.crs != second.crs:
            raise ValueError("visible and thermal DSM CRS differ: %s != %s"
                             % (first.crs, second.crs))
        if not _is_metre_projected(first.crs):
            raise ValueError("DSM registration requires a projected metre CRS, "
                             "got %s" % first.crs)
        visible_range = elevation_range_for(first)
        thermal_range = elevation_range_for(second)
        west = max(first.bounds.left, second.bounds.left)
        east = min(first.bounds.right, second.bounds.right)
        south = max(first.bounds.bottom, second.bounds.bottom)
        north = min(first.bounds.top, second.bounds.top)
        width = int((east - west) / resolution)
        height = int((north - south) / resolution)
        if width <= 0 or height <= 0:
            raise ValueError(
                "DSM overlap is empty: (west=%.3f, south=%.3f, east=%.3f, "
                "north=%.3f)" % (west, south, east, north))
        transform = from_origin(west, north, resolution, resolution)

        def sample(dataset, elevation_range):
            array = np.full((height, width), np.nan, np.float32)
            reproject(rasterio.band(dataset, 1), array,
                      src_transform=dataset.transform, src_crs=dataset.crs,
                      dst_transform=transform, dst_crs=dataset.crs,
                      resampling=Resampling.bilinear, src_nodata=dataset.nodata,
                      dst_nodata=np.nan, num_threads=max(1, int(num_threads)))
            valid = (np.isfinite(array)
                     & (array > elevation_range[0]) & (array < elevation_range[1]))
            return array, valid

        visible_z, visible_valid = sample(first, visible_range)
        thermal_z, thermal_valid = sample(second, thermal_range)
    return visible_z, visible_valid, thermal_z, thermal_valid, transform, (west, north)


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


def pixel_covariance_to_world(covariance_px, resolution):
    """Map pixel-shift covariance to EN metres: (dx,dy)=(r*sx,-r*sy)."""
    flip = np.diag([float(resolution), -float(resolution)])
    return flip @ np.asarray(covariance_px, float) @ flip.T


def _competing_separation(candidates, resolution, cc_margin=0.02):
    """Max spatial separation among peaks within cc_margin of the best."""
    if not candidates:
        return 0.0
    best = candidates[0]
    close = [item for item in candidates
             if item["cc"] >= best["cc"] - cc_margin]
    return max((float(np.hypot(item["sx"] - best["sx"],
                              item["sy"] - best["sy"]) * resolution)
                for item in close), default=0.0)


def _register_tile(visible, thermal, joint, transform, resolution,
                   tile: int, row: int, column: int, top: int, left: int,
                   prior_radius_m: float = 2.0,
                   min_valid_fraction: float = 0.5,
                   covariance_fn=None):
    """Register one tile and return it as an observation.

    The ECC search is computed over the candidate-dependent common mask, not
    over zero-filled nodata pixels.  The best forward peak is paired with the
    second forward peak (period ambiguity) and their spread (observation
    covariance).  A tile is still always a produced observation; the field-fit
    caller decides whether its flags permit admission.
    """
    mask = joint[top:top + tile, left:left + tile]
    valid_fraction = float(mask.mean())
    if valid_fraction < min_valid_fraction:
        return None
    first = visible[top:top + tile, left:left + tile].astype(np.float32).copy()
    second = thermal[top:top + tile, left:left + tile].astype(np.float32).copy()
    if first[mask].std() < 1e-6 or second[mask].std() < 1e-6:
        return None
    first_n = (first - first.mean()) / (first.std() + 1e-9)
    second_n = (second - second.mean()) / (second.std() + 1e-9)
    coarse_first = cv2.resize(first_n, (tile // 2, tile // 2),
                              interpolation=cv2.INTER_AREA)
    coarse_second = cv2.resize(second_n, (tile // 2, tile // 2),
                               interpolation=cv2.INTER_AREA)
    coarse_x, coarse_y, _ = ecc(coarse_first, coarse_second)
    if np.isfinite(coarse_x) and np.isfinite(coarse_y):
        prior = (2.0 * coarse_x, 2.0 * coarse_y)
    else:
        prior = (0.0, 0.0)
    radius_px = max(8.0, float(prior_radius_m) / resolution)

    surface = masked_ecc_surface(first, second, mask)
    peaks = masked_ecc_peaks(surface[1])
    forward = masked_ecc_candidates(
        first, second, mask, resolution, top_k=5,
        prior=prior, radius_px=radius_px, covariance_fn=covariance_fn,
        surface=surface, peaks=peaks)
    if not forward:
        return None
    global_forward = masked_ecc_candidates(
        first, second, mask, resolution, top_k=1, surface=surface, peaks=peaks)
    best = forward[0]
    if global_forward:
        global_best = global_forward[0]
        global_separation_m = float(np.hypot(
            global_best["sx"] - prior[0],
            global_best["sy"] - prior[1]) * resolution)
        global_cc = float(global_best["cc"])
    else:
        global_separation_m = 0.0
        global_cc = float("nan")
    alias_conflict = bool(
        np.isfinite(global_cc)
        and global_separation_m > radius_px * resolution
        and global_cc > best["cc"] + 0.05)
    shift_x = float(best["sx"])
    shift_y = float(best["sy"])
    dx, dy = pixel_shift_to_world(shift_x, shift_y, resolution)

    if len(forward) > 1:
        margin = float(forward[0]["cc"] - forward[1]["cc"])
        separation_m = float(np.hypot(forward[0]["sx"] - forward[1]["sx"],
                                      forward[0]["sy"] - forward[1]["sy"])
                             * resolution)
    else:
        margin = 1.0
        separation_m = 0.0
    mode_quality = candidate_mode_quality(forward, resolution)
    # Period estimation is an extra full-tile FFT and only affects the
    # ambiguity threshold when the candidate mode is already weak.  Skip it
    # for unique modes; period_m stays None as an explicit diagnostic.
    period_m = None
    if (mode_quality["top_mass"] < CANDIDATE_MODE_MASS_MIN
            and mode_quality["mode_separation_m"] >= 1.0):
        period_m = estimate_period_m(first, mask, resolution)
    separation_threshold = max(1.0, 0.5 * period_m) if period_m else 1.0
    candidate_ambiguous = bool(
        mode_quality["top_mass"] < CANDIDATE_MODE_MASS_MIN
        and mode_quality["mode_separation_m"] >= separation_threshold)
    ambiguous = bool(candidate_ambiguous or alias_conflict)
    if mode_quality["top_mass"] >= CANDIDATE_MODE_MASS_MIN:
        uniqueness = float(np.clip(
            (mode_quality["top_mass"] - CANDIDATE_MODE_MASS_MIN)
            / (1.0 - CANDIDATE_MODE_MASS_MIN), 0.0, 1.0))
    else:
        uniqueness = 0.0

    competing_separation_m = _competing_separation(forward, resolution)

    scores = np.asarray([item["cc"] for item in forward], float)
    shifts_px = np.asarray([[item["sx"], item["sy"]] for item in forward], float)
    logits = SOFTMAX_BETA * (scores - float(np.max(scores)))
    pi = np.exp(logits)
    pi /= max(float(np.sum(pi)), 1e-12)
    mean_shift = pi @ shifts_px
    centered = shifts_px - mean_shift
    covariance_px = (centered.T * pi) @ centered
    covariance_candidate = pixel_covariance_to_world(covariance_px, resolution)
    covariance = covariance_candidate + (SIGMA0_M ** 2) * np.eye(2)
    sigma_e = float(np.sqrt(max(covariance[0, 0], 0.0)))
    sigma_n = float(np.sqrt(max(covariance[1, 1], 0.0)))
    sigma_en = float(covariance[0, 1])

    warp = np.array([[1.0, 0.0, -shift_x], [0.0, 1.0, -shift_y]], np.float32)
    aligned = cv2.warpAffine(second_n, warp, (tile, tile),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    center = transform * (left + tile / 2.0, top + tile / 2.0)
    return {
        "row": row, "column": column,
        "east": float(center[0]), "north": float(center[1]),
        "dx_m": dx, "dy_m": dy,
        "magnitude_m": float(np.hypot(dx, dy)),
        "cc": float(best["cc"]),
        "ncc_before": float(np.mean(first_n * second_n)),
        "ncc_after": float(np.mean(first_n * aligned)),
        "valid_fraction": valid_fraction,
        "quality_weight": valid_fraction ** 1.5,
        "uniqueness": uniqueness,
        "prior_sx": float(prior[0]),
        "prior_sy": float(prior[1]),
        "prior_radius_m": float(radius_px * resolution),
        "global_cc": global_cc,
        "global_separation_m": global_separation_m,
        "alias_conflict": alias_conflict,
        "cc_best": float(best["cc"]),
        "cc_second": float(forward[1]["cc"]) if len(forward) > 1 else float(best["cc"]),
        "candidate_count": len(forward),
        "candidates": forward,
        "candidate_mode_count": int(mode_quality["mode_count"]),
        "candidate_top_mass": float(mode_quality["top_mass"]),
        "candidate_second_mass": float(mode_quality["second_mass"]),
        "candidate_mode_separation_m": float(
            mode_quality["mode_separation_m"]),
        "candidate_ambiguity": float(1.0 - mode_quality["top_mass"]),
        "uniqueness_observable": bool(len(forward) >= 2),
        "margin": margin,
        "separation_m": separation_m,
        "competing_separation_m": competing_separation_m,
        "period_m": (float(period_m) if period_m is not None else None),
        "ambiguous": ambiguous,
        "sigma_e": sigma_e,
        "sigma_n": sigma_n,
        "sigma_en": sigma_en,
        # Compatibility: every produced record is an observation.  The field
        # fit filters on ambiguous, not on these constants.
        "accepted": True,
        "outlier": False,
    }


def _finite_displacement_points(records):
    """Return the finite (dx, dy) observations as an (N, 2) array."""
    points = np.asarray(
        [[float(record["dx_m"]), float(record["dy_m"])]
         for record in records
         if np.isfinite(record["dx_m"]) and np.isfinite(record["dy_m"])],
        float)
    return points if points.size else np.empty((0, 2), float)


def _domain_radius() -> float:
    return float(np.sqrt(chi2.ppf(DISPLACEMENT_DOMAIN_CONFIDENCE, 2)))


def _domain_record(center, covariance, estimator, *, density=None,
                   bandwidth=None, observations=None) -> dict:
    """Build the public domain record and floor the covariance eigenvalues."""
    eigenvalue, eigenvector = np.linalg.eigh(np.asarray(covariance, float))
    eigenvalue = np.maximum(eigenvalue,
                            DISPLACEMENT_DOMAIN_COVARIANCE_FLOOR_M2)
    covariance = eigenvector @ np.diag(eigenvalue) @ eigenvector.T
    radius = _domain_radius()
    record = {
        "center_dx_m": float(center[0]),
        "center_dy_m": float(center[1]),
        "covariance_m2": covariance.tolist(),
        "semiaxis_m": (np.sqrt(eigenvalue) * radius).tolist(),
        "radius": radius,
        "confidence": DISPLACEMENT_DOMAIN_CONFIDENCE,
        "estimator": estimator,
    }
    if density is not None:
        record["mode_density"] = float(density)
    if bandwidth is not None:
        record["bandwidth"] = float(bandwidth)
    if observations is not None:
        record["observations"] = int(observations)
    return record


def _median_mad_domain(points):
    """Sparse-data fallback: a 2-D median centre with a floored ellipse."""
    if len(points):
        center = np.median(points, axis=0)
        scale = 1.4826 * np.median(np.abs(points - center), axis=0)
    else:
        center = np.zeros(2, float)
        scale = np.zeros(2, float)
    scale = np.maximum(scale, np.sqrt(
        DISPLACEMENT_DOMAIN_COVARIANCE_FLOOR_M2))
    return _domain_record(center, np.diag(scale ** 2),
                          "median_mad_fallback",
                          observations=len(points))


def _kde_density(points, center, covariance, bandwidth):
    inverse = np.linalg.inv(covariance)
    delta = points - center
    distance = np.einsum("ij,jk,ik->i", delta, inverse, delta)
    weights = np.exp(-0.5 * distance / (bandwidth * bandwidth))
    normalizer = (2.0 * np.pi * bandwidth * bandwidth
                  * np.sqrt(np.linalg.det(covariance)))
    return float(weights.sum() / (len(points) * normalizer))


def _mean_shift(points, covariance, bandwidth, start):
    """Iterate to a local KDE mode with a Gaussian kernel."""
    center = np.asarray(start, float).copy()
    inverse = np.linalg.inv(covariance)
    for _ in range(200):
        delta = points - center
        distance = np.einsum("ij,jk,ik->i", delta, inverse, delta)
        weights = np.exp(-0.5 * distance / (bandwidth * bandwidth))
        total = weights.sum()
        if total <= 0:
            break
        updated = (weights[:, None] * points).sum(axis=0) / total
        if np.linalg.norm(updated - center) < 1e-8:
            center = updated
            break
        center = updated
    return center


def _two_dimensional_domain(points):
    """Deterministic multi-start mean-shift, then a Mahalanobis ellipse."""
    count = len(points)
    median = np.median(points, axis=0)
    mad = 1.4826 * np.median(np.abs(points - median), axis=0)
    scale = np.maximum(mad, DISPLACEMENT_DOMAIN_MAD_FLOOR_M)
    initial = np.diag(scale ** 2)
    bandwidth = DISPLACEMENT_DOMAIN_BANDWIDTH * count ** (-1.0 / 6.0)

    low = np.percentile(points, 10.0, axis=0)
    high = np.percentile(points, 90.0, axis=0)
    seeds = [median]
    for x in np.linspace(low[0], high[0], DISPLACEMENT_DOMAIN_SEED_GRID):
        for y in np.linspace(low[1], high[1], DISPLACEMENT_DOMAIN_SEED_GRID):
            seeds.append(np.array([x, y], float))
    modes = []
    for seed in seeds:
        center = _mean_shift(points, initial, bandwidth, seed)
        modes.append((_kde_density(points, center, initial, bandwidth), center))
    density, center = max(modes, key=lambda item: item[0])

    covariance = initial.copy()
    radius_squared = chi2.ppf(DISPLACEMENT_DOMAIN_CONFIDENCE, 2)
    mask = np.ones(count, bool)
    for _ in range(8):
        delta = points - center
        distance = np.einsum("ij,jk,ik->i", delta,
                             np.linalg.inv(covariance), delta)
        mask = distance <= radius_squared
        if mask.sum() < DISPLACEMENT_DOMAIN_MIN_CLUSTER:
            break
        raw = np.cov(points[mask].T, bias=True)
        eigenvalue, eigenvector = np.linalg.eigh(raw)
        eigenvalue = np.maximum(
            eigenvalue, DISPLACEMENT_DOMAIN_COVARIANCE_FLOOR_M2)
        covariance = eigenvector @ np.diag(eigenvalue) @ eigenvector.T

    delta = points - center
    distance = np.einsum("ij,jk,ik->i", delta,
                         np.linalg.inv(covariance), delta)
    if (distance <= radius_squared).sum() < DISPLACEMENT_DOMAIN_MIN_CLUSTER:
        return _median_mad_domain(points)
    return _domain_record(center, covariance, "2d_kde_mean_shift",
                          density=density, bandwidth=bandwidth,
                          observations=count)


def displacement_domain(records) -> dict:
    """Estimate the joint station-drift mode and its 99.5% ellipse."""
    points = _finite_displacement_points(records)
    if len(points) < DISPLACEMENT_DOMAIN_MIN_OBSERVATIONS:
        return _median_mad_domain(points)
    return _two_dimensional_domain(points)


def apply_displacement_domain(records):
    """Discard observations outside the joint 2-D drift ellipse."""
    records = list(records)
    measured = [record for record in records
                if np.isfinite(record["dx_m"]) and np.isfinite(record["dy_m"])]
    domain = displacement_domain(measured)
    points = np.asarray([[record["dx_m"], record["dy_m"]]
                         for record in measured], float)
    if len(points):
        center = np.array([domain["center_dx_m"], domain["center_dy_m"]])
        covariance = np.asarray(domain["covariance_m2"], float)
        delta = points - center
        distance = np.einsum("ij,jk,ik->i", delta,
                             np.linalg.inv(covariance), delta)
        kept = [record for record, value in zip(measured, distance)
                if value <= domain["radius"] ** 2]
    else:
        kept = []
    return kept, {**domain, "rejected": len(records) - len(kept)}



def _contour_boundary_tasks(joint, tile, step, transform, resolution):
    """Sample the valid-data contour with full windows shifted inward.

    The sample point lies on the irregular valid-data boundary.  The window
    centre may move inward by up to half a tile so the window stays inside the
    data region; both points are recorded.
    """
    binary = joint.astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)
    tasks = []
    seen = set()
    directions = [(math.cos(2.0 * math.pi * i / 16.0),
                   math.sin(2.0 * math.pi * i / 16.0)) for i in range(16)]
    distances = (tile // 4, tile // 2)
    for contour in contours:
        points = contour.reshape(-1, 2).astype(np.float64)
        if len(points) < 2:
            continue
        deltas = np.diff(points, axis=0)
        lengths = np.hypot(deltas[:, 0], deltas[:, 1])
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        if cumulative[-1] <= 0:
            continue
        sample_positions = np.arange(0.0, cumulative[-1], float(step))
        for position in sample_positions:
            index = int(np.searchsorted(cumulative, position, side="right") - 1)
            index = min(max(index, 0), len(points) - 1)
            x, y = float(points[index, 0]), float(points[index, 1])
            best = None
            for distance in distances:
                for dx, dy in directions:
                    cx = x + dx * distance
                    cy = y + dy * distance
                    top = int(round(cy - tile / 2.0))
                    left = int(round(cx - tile / 2.0))
                    if top < 0 or left < 0:
                        continue
                    if top + tile > joint.shape[0] or left + tile > joint.shape[1]:
                        continue
                    fraction = float(joint[top:top + tile, left:left + tile].mean())
                    if fraction < 0.50:
                        continue
                    if best is None or fraction > best[0]:
                        best = (fraction, top, left, cx, cy,
                                math.hypot(cx - x, cy - y))
            if best is None:
                continue
            key = (best[1], best[2], round(best[3], 1), round(best[4], 1))
            if key in seen:
                continue
            seen.add(key)
            tasks.append({
                "top": best[1], "left": best[2],
                "sample_x": x, "sample_y": y,
                "window_x": best[3], "window_y": best[4],
                "shift_m": best[5] * resolution,
                "valid_fraction": best[0],
            })
    return tasks

def _edge_tasks(height, width, tile, step, pad):
    """Tile centres on each raster edge, expressed in padded coordinates."""
    tasks = set()
    for row in range(0, height, step):
        top = pad + row - tile // 2
        tasks.add((top, 0))
        tasks.add((top, pad + width - 1 - tile // 2))
    for column in range(0, width, step):
        left = pad + column - tile // 2
        tasks.add((0, left))
        tasks.add((pad + height - 1 - tile // 2, left))
    return sorted(tasks)


def register(visible, thermal, joint, transform, resolution,
             tile_m: float, step_m: float, workers: int = 1,
             prior_radius_m: float = 2.0, include_boundary: bool = False,
             apply_domain: bool = True, covariance_fn=None):
    tile = int(round(tile_m / resolution))
    step = int(round(step_m / resolution))
    height, width = joint.shape
    if height < tile or width < tile:
        print("register: raster %dx%d is smaller than one %dpx tile; no tiles"
              % (width, height, tile))
        if not apply_domain:
            return [], {"rejected": 0, "estimator": "raw"}
        return apply_displacement_domain([])
    # The +1 makes the final full tile eligible when the usable extent is an
    # exact multiple of the step; without it the last row/column was dropped.
    tasks = [(row, column, top, left)
             for row, top in enumerate(range(0, height - tile + 1, step))
             for column, left in enumerate(range(0, width - tile + 1, step))]
    def process(task):
        row, column, top, left = task
        return _register_tile(visible, thermal, joint, transform,
                              resolution, tile, row, column, top, left,
                              prior_radius_m=prior_radius_m,
                              covariance_fn=covariance_fn)
    workers = max(1, int(workers))
    if workers > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(process, tasks))
    else:
        results = [process(task) for task in tasks]
    if include_boundary:
        boundary_tasks = _contour_boundary_tasks(
            joint, tile, step, transform, resolution)
        if boundary_tasks:
            def process_boundary(task):
                record = _register_tile(
                    visible, thermal, joint, transform, resolution,
                    tile, 0, 0, task["top"], task["left"],
                    prior_radius_m=prior_radius_m, min_valid_fraction=0.50,
                    covariance_fn=covariance_fn)
                if record is None:
                    return None
                sample = transform * (task["sample_x"] + 0.5,
                                      task["sample_y"] + 0.5)
                record["sample_east"] = float(sample[0])
                record["sample_north"] = float(sample[1])
                record["boundary_shift_m"] = float(task["shift_m"])
                return record
            if workers > 1 and len(boundary_tasks) > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    boundary_results = list(pool.map(
                        process_boundary, boundary_tasks))
            else:
                boundary_results = [process_boundary(task)
                                    for task in boundary_tasks]
            results.extend(boundary_results)
        else:
            pad = tile // 2
            padded_visible = np.pad(visible, ((pad, pad), (pad, pad)),
                                    mode="constant", constant_values=0.0)
            padded_thermal = np.pad(thermal, ((pad, pad), (pad, pad)),
                                    mode="constant", constant_values=0.0)
            padded_joint = np.pad(joint, ((pad, pad), (pad, pad)),
                                  mode="constant", constant_values=False)
            padded_transform = transform * Affine.translation(-pad, -pad)
            edge_tasks = _edge_tasks(height, width, tile, step, pad)
            def process_edge(task):
                top, left = task
                return _register_tile(
                    padded_visible, padded_thermal, padded_joint,
                    padded_transform, resolution, tile, 0, 0, top, left,
                    prior_radius_m=prior_radius_m, min_valid_fraction=0.25,
                    covariance_fn=covariance_fn)
            if workers > 1 and len(edge_tasks) > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    edge_results = list(pool.map(process_edge, edge_tasks))
            else:
                edge_results = [process_edge(task) for task in edge_tasks]
            results.extend(edge_results)
    raw = [record for record in results if record is not None]
    if not apply_domain:
        return raw, {"rejected": 0, "estimator": "raw"}
    return apply_displacement_domain(raw)

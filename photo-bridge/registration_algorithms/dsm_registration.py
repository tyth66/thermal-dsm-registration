"""High-pass DSM registration and planimetric field estimation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from .dsm_sampling import elevation_range_for
from .masked_ecc import (
    AMBIGUITY_MARGIN, FB_TOLERANCE_M, SIGMA0_M, SOFTMAX_BETA,
    estimate_period_m, masked_ecc_candidates,
)
# Reuse the single projected-metre CRS predicate instead of duplicating it: a
# DSM grid is only meaningful when both rasters share one metric, north-up frame.
from .visible_world_model import _is_metre_projected

# The station's planimetric drift domain is estimated from the observations
# themselves (same precedent as the adaptive DSM elevation window): a fixed
# clip calibrated on one station silently discards real observations on a
# station whose drift is several metres. Each axis gets
# center = median(shift), scale = 1.4826 * MAD, bound = max(floor, 3 * scale)
# and the valid domain is [center - bound, center + bound]. The 1.5 m floor
# keeps the domain usable when the MAD is ~0 (uniform drift); a periodic
# self-similarity alias (e.g. the 17.67 m dy tile on the 141 station) still
# falls outside every axis domain, so it is discarded as an invalid
# measurement rather than a low-quality one. True drift above the floor is
# preserved because the domain is centred on the observed drift.
DISPLACEMENT_BOUND_FLOOR_M = 1.5
DISPLACEMENT_BOUND_SIGMA = 3.0


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


def pixel_covariance_to_world(covariance_px, resolution):
    """Map pixel-shift covariance to EN metres: (dx,dy)=(r*sx,-r*sy)."""
    flip = np.diag([float(resolution), -float(resolution)])
    return flip @ np.asarray(covariance_px, float) @ flip.T


def _register_tile(visible_high, thermal_high, joint, transform, resolution,
                   tile: int, row: int, column: int, top: int, left: int,
                   visible_mask=None, thermal_mask=None):
    """Register one tile and return it as an observation.

    The ECC search is computed over the candidate-dependent common mask, not
    over zero-filled nodata pixels.  The best forward peak is paired with the
    reverse peak (forward-backward consistency), the second forward peak
    (period ambiguity), and their spread (observation covariance).  A tile is
    still always a produced observation; the field-fit caller decides whether
    its flags permit admission.
    """
    mask = joint[top:top + tile, left:left + tile]
    mask_v = (mask if visible_mask is None
              else visible_mask[top:top + tile, left:left + tile])
    mask_t = (mask if thermal_mask is None
              else thermal_mask[top:top + tile, left:left + tile])
    valid_fraction = float(mask.mean())
    if valid_fraction < 0.5:
        return None
    first = visible_high[top:top + tile, left:left + tile].astype(np.float32).copy()
    second = thermal_high[top:top + tile, left:left + tile].astype(np.float32).copy()
    if first[mask_v].std() < 1e-6 or second[mask_t].std() < 1e-6:
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
    radius_px = max(8.0, 2.0 / resolution)

    forward = masked_ecc_candidates(
        first, second, mask, resolution, top_k=5,
        prior=prior, radius_px=radius_px,
        visible_mask=mask_v, thermal_mask=mask_t)
    if not forward:
        return None
    reverse = masked_ecc_candidates(
        second, first, mask, resolution, top_k=1,
        prior=(-prior[0], -prior[1]), radius_px=radius_px,
        visible_mask=mask_t, thermal_mask=mask_v)
    global_forward = masked_ecc_candidates(
        first, second, mask, resolution, top_k=1,
        visible_mask=mask_v, thermal_mask=mask_t)
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

    if reverse:
        reverse_best = reverse[0]
        fb_error = float(np.hypot((shift_x + reverse_best["sx"]) * resolution,
                                  (shift_y + reverse_best["sy"]) * resolution))
        reverse_cc = float(reverse_best["cc"])
    else:
        fb_error = float("inf")
        reverse_cc = float("nan")
    consistent = bool(np.isfinite(fb_error) and fb_error <= FB_TOLERANCE_M)

    if len(forward) > 1:
        margin = float(forward[0]["cc"] - forward[1]["cc"])
        separation_m = float(np.hypot(forward[0]["sx"] - forward[1]["sx"],
                                      forward[0]["sy"] - forward[1]["sy"])
                             * resolution)
    else:
        margin = 1.0
        separation_m = 0.0
    period_m = estimate_period_m(first, mask_v, resolution)
    separation_threshold = max(1.0, 0.5 * period_m) if period_m else 1.0
    ambiguous = bool((margin < AMBIGUITY_MARGIN
                      and separation_m >= separation_threshold)
                     or alias_conflict)
    uniqueness = float(np.clip(margin / AMBIGUITY_MARGIN, 0.0, 1.0))

    scores = np.asarray([item["cc"] for item in forward], float)
    shifts_px = np.asarray([[item["sx"], item["sy"]] for item in forward], float)
    logits = SOFTMAX_BETA * (scores - float(np.max(scores)))
    pi = np.exp(logits)
    pi /= max(float(np.sum(pi)), 1e-12)
    mean_shift = pi @ shifts_px
    centered = shifts_px - mean_shift
    covariance_px = (centered.T * pi) @ centered
    covariance_candidate = pixel_covariance_to_world(covariance_px, resolution)
    covariance_fb = (fb_error ** 2 if np.isfinite(fb_error) else 1.0) * np.eye(2)
    covariance = covariance_candidate + covariance_fb + (SIGMA0_M ** 2) * np.eye(2)
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
        "margin": margin,
        "separation_m": separation_m,
        "period_m": (float(period_m) if period_m is not None else None),
        "reverse_cc": reverse_cc,
        "fb_error_m": fb_error,
        "consistent": consistent,
        "ambiguous": ambiguous,
        "sigma_e": sigma_e,
        "sigma_n": sigma_n,
        "sigma_en": sigma_en,
        # Compatibility: every produced record is an observation.  The field
        # fit filters on consistent/ambiguous, not on these constants.
        "accepted": True,
        "outlier": False,
    }


def _joint_mode_domain(records):
    """Two-dimensional densest-cluster domain, used as an opt-in diagnostic.

    Per-axis modes can combine two different 2D clusters.  This estimator
    selects the densest 2D bin (plus its 3x3 neighbourhood), centres on that
    cluster, and returns a circular radius.  It returns None when the cluster
    is too small, letting the caller fall back to the per-axis mode.
    """
    valid = [(float(record["dx_m"]), float(record["dy_m"])) for record in records
             if np.isfinite(record["dx_m"]) and np.isfinite(record["dy_m"])]
    if len(valid) < 8:
        return None
    values = np.asarray(valid, float)
    x, y = values[:, 0], values[:, 1]
    span = float(np.hypot(np.ptp(x), np.ptp(y)))
    width = max(0.5, min(1.0, span / 20.0))
    x_edges = np.arange(float(x.min()) - width, float(x.max()) + 2 * width, width)
    y_edges = np.arange(float(y.min()) - width, float(y.max()) + 2 * width, width)
    histogram, x_edges, y_edges = np.histogram2d(x, y, bins=[x_edges, y_edges])
    ix, iy = np.unravel_index(int(np.argmax(histogram)), histogram.shape)
    cluster = ((x >= x_edges[max(0, ix - 1)]) & (x < x_edges[min(len(x_edges) - 1, ix + 2)])
               & (y >= y_edges[max(0, iy - 1)]) & (y < y_edges[min(len(y_edges) - 1, iy + 2)]))
    if cluster.sum() < max(4, int(round(0.15 * len(values)))):
        return None
    center_x = float(np.median(x[cluster]))
    center_y = float(np.median(y[cluster]))
    radial = np.hypot(x[cluster] - center_x, y[cluster] - center_y)
    scale = 1.4826 * float(np.median(np.abs(radial - np.median(radial))))
    radius = max(DISPLACEMENT_BOUND_FLOOR_M, DISPLACEMENT_BOUND_SIGMA * scale)
    return {"center_dx": center_x, "center_dy": center_y,
            "bound_dx": radius, "bound_dy": radius,
            "joint": True, "radius_m": radius}


def _mode_axis_domain(values):
    """Densest-cluster drift estimate, with the median/MAD estimator as fallback.

    A median/MAD domain can be inflated by a large periodic-alias cluster
    (the 247 dx observations contain both a near-zero and a +/-5 m mode).
    The mode estimator centres on the densest histogram bin and takes its
    local MAD; if the cluster is too small it falls back to median/MAD.
    """
    finite = np.asarray([value for value in values if np.isfinite(value)], float)
    if len(finite) < 8:
        return _robust_axis_domain(values)
    low, high = np.percentile(finite, [5.0, 95.0])
    span = float(high - low)
    width = max(0.25, min(1.0, span / 12.0)) if span > 0 else 0.5
    edges = np.arange(float(finite.min()) - width,
                      float(finite.max()) + 2.0 * width, width)
    histogram, edges = np.histogram(finite, bins=edges)
    peak = int(np.argmax(histogram))
    lower = edges[max(0, peak - 1)]
    upper = edges[min(len(edges) - 1, peak + 2)]
    cluster = finite[(finite >= lower) & (finite < upper)]
    if len(cluster) < max(4, int(round(0.15 * len(finite)))):
        return _robust_axis_domain(values)
    center = float(np.median(cluster))
    scale = 1.4826 * float(np.median(np.abs(cluster - center)))
    return center, max(DISPLACEMENT_BOUND_FLOOR_M,
                       DISPLACEMENT_BOUND_SIGMA * scale)


def _robust_axis_domain(values):
    """Return (centre, bound) for one displacement axis, self-estimated.

    Non-finite measurements are invalid and must not poison the median/MAD used
    to estimate the domain; they are dropped here and rejected as invalid
    measurements by apply_displacement_domain.
    """
    values = [value for value in values if np.isfinite(value)]
    if not values:
        return 0.0, DISPLACEMENT_BOUND_FLOOR_M
    array = np.asarray(values, float)
    center = float(np.median(array))
    scale = 1.4826 * float(np.median(np.abs(array - center)))
    return center, max(DISPLACEMENT_BOUND_FLOOR_M,
                       DISPLACEMENT_BOUND_SIGMA * scale)


def displacement_domain(records, mode: str = "mad") -> dict:
    """Estimate the per-axis station drift domain from its observations."""
    if mode not in ("mad", "mode", "joint"):
        raise ValueError("domain mode must be 'mad', 'mode' or 'joint'")
    if mode == "joint":
        joint = _joint_mode_domain(records)
        if joint is not None:
            return joint
        mode = "mode"
    estimator = _mode_axis_domain if mode == "mode" else _robust_axis_domain
    center_dx, bound_dx = estimator(
        [record["dx_m"] for record in records])
    center_dy, bound_dy = estimator(
        [record["dy_m"] for record in records])
    return {"center_dx": center_dx, "bound_dx": bound_dx,
            "center_dy": center_dy, "bound_dy": bound_dy}


def apply_displacement_domain(records, mode: str = "mad"):
    """Discard observations outside the self-estimated drift domain.

    Returns (kept_records, stats); stats carries the estimated per-axis
    centre/bound plus the number of discarded invalid measurements.  A record
    with a non-finite dx/dy is an invalid measurement (same semantics as a
    periodic self-similarity alias): it is excluded from the domain estimate
    and counted as rejected instead of being treated as a usable observation.
    """
    records = list(records)
    measured = [record for record in records
                if np.isfinite(record["dx_m"]) and np.isfinite(record["dy_m"])]
    domain = displacement_domain(measured, mode=mode)
    if domain.get("joint"):
        kept = [record for record in measured
                if np.hypot(record["dx_m"] - domain["center_dx"],
                            record["dy_m"] - domain["center_dy"])
                <= domain["radius_m"]]
    else:
        kept = [record for record in measured
                if (domain["center_dx"] - domain["bound_dx"] <= record["dx_m"]
                    <= domain["center_dx"] + domain["bound_dx"]
                    and domain["center_dy"] - domain["bound_dy"] <= record["dy_m"]
                    <= domain["center_dy"] + domain["bound_dy"])]
    return kept, {**domain, "rejected": len(records) - len(kept)}


def register(visible_high, thermal_high, joint, transform, resolution,
             tile_m: float, step_m: float, workers: int = 1,
             domain_mode: str = "mad",
             visible_mask=None, thermal_mask=None):
    tile = int(round(tile_m / resolution))
    step = int(round(step_m / resolution))
    height, width = joint.shape
    if height < tile or width < tile:
        print("register: raster %dx%d is smaller than one %dpx tile; no tiles"
              % (width, height, tile))
        return apply_displacement_domain([], mode=domain_mode)
    # The +1 makes the final full tile eligible when the usable extent is an
    # exact multiple of the step; without it the last row/column was dropped.
    tasks = [(row, column, top, left)
             for row, top in enumerate(range(0, height - tile + 1, step))
             for column, left in enumerate(range(0, width - tile + 1, step))]
    def process(task):
        row, column, top, left = task
        return _register_tile(visible_high, thermal_high, joint, transform,
                              resolution, tile, row, column, top, left,
                              visible_mask=visible_mask,
                              thermal_mask=thermal_mask)
    workers = max(1, int(workers))
    if workers > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(process, tasks))
    else:
        results = [process(task) for task in tasks]
    raw = [record for record in results if record is not None]
    # Second pass: the station drift domain is estimated from every produced
    # observation, then only measurements outside that domain are dropped.
    return apply_displacement_domain(raw, mode=domain_mode)

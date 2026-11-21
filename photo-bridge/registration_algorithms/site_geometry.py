"""Read-only site geometry profiler for the V3-A adaptation study."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from scipy.spatial import cKDTree

from .application.planimetric import adaptive_field_support, adaptive_tiling
from .dsm_registration import highpass, load_grid, register
from .masked_ecc import estimate_period_m
from .residual_model import fit_regularized_field


def _downsample(array, valid, max_dim):
    height, width = array.shape
    scale = max(1, int(np.ceil(max(height, width) / float(max_dim))))
    if scale == 1:
        return array, valid, 1.0
    target = (max(1, width // scale), max(1, height // scale))
    sampled = cv2.resize(array, target, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(valid.astype(np.uint8), target,
                      interpolation=cv2.INTER_NEAREST) > 0
    return sampled, mask, float(scale)


def _texture_stats(array, valid):
    values = array[valid]
    if len(values) < 10:
        return {"std": 0.0, "gradient_rms": 0.0, "edge_density": 0.0}
    gradient = cv2.Sobel(np.where(valid, array, 0.0).astype(np.float32),
                         cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(np.where(valid, array, 0.0).astype(np.float32),
                           cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gradient, gradient_y)
    threshold = float(np.percentile(magnitude[valid], 90.0))
    return {
        "std": float(np.std(values)),
        "gradient_rms": float(np.sqrt(np.mean(magnitude[valid] ** 2))),
        "edge_density": float(np.mean(magnitude[valid] > threshold)),
    }


def _semivariogram(points, values, bins=10, max_distance=None):
    if len(points) < 3:
        return {"bins_m": [], "gamma": [], "correlation_length_m": None}
    points = np.asarray(points, float)
    values = np.asarray(values, float)
    distances = np.hypot(points[:, None, 0] - points[None, :, 0],
                         points[:, None, 1] - points[None, :, 1])
    upper = float(np.max(distances)) if max_distance is None else float(max_distance)
    edges = np.linspace(0.0, upper, bins + 1)
    centers, gamma = [], []
    for index in range(bins):
        mask = (distances >= edges[index]) & (distances < edges[index + 1])
        pairs = np.argwhere(np.triu(mask, k=1))
        if not len(pairs):
            continue
        diff = values[pairs[:, 0]] - values[pairs[:, 1]]
        centers.append(float((edges[index] + edges[index + 1]) / 2.0))
        gamma.append(float(0.5 * np.mean(diff ** 2)))
    if not gamma:
        return {"bins_m": [], "gamma": [], "correlation_length_m": None}
    sill = max(float(np.var(values)), float(max(gamma)))
    target = 0.63 * sill
    correlation = None
    for center, value in zip(centers, gamma):
        if value >= target:
            correlation = center
            break
    return {"bins_m": centers, "gamma": gamma,
            "correlation_length_m": correlation}


def profile_site(visible_dsm, thermal_dsm, resolution_m=0.1,
                 visible_world=None, max_dim=1024):
    visible = Path(visible_dsm)
    thermal = Path(thermal_dsm)
    with rasterio.open(visible) as first, rasterio.open(thermal) as second:
        gsd_visible = [abs(first.transform.a), abs(first.transform.e)]
        gsd_thermal = [abs(second.transform.a), abs(second.transform.e)]
    visible_z, visible_valid, thermal_z, thermal_valid, transform, _ = load_grid(
        visible, thermal, resolution_m)
    joint = visible_valid & thermal_valid
    if not np.any(joint):
        raise ValueError("site has no joint DSM coverage")
    sigma_px = 1.5 / resolution_m
    visible_high = highpass(visible_z, visible_valid, sigma_px)
    thermal_high = highpass(thermal_z, thermal_valid, sigma_px)

    visible_ds, visible_ds_mask, visible_scale = _downsample(
        visible_high, visible_valid, max_dim)
    thermal_ds, thermal_ds_mask, thermal_scale = _downsample(
        thermal_high, thermal_valid, max_dim)
    joint_ds = visible_ds_mask & thermal_ds_mask
    period_visible = estimate_period_m(
        visible_ds, joint_ds, resolution_m * visible_scale, max_period_m=100.0)
    period_thermal = estimate_period_m(
        thermal_ds, joint_ds, resolution_m * thermal_scale, max_period_m=100.0)

    extent_height_m = joint.shape[0] * resolution_m
    extent_width_m = joint.shape[1] * resolution_m
    plan = adaptive_tiling(extent_width_m, extent_height_m, resolution_m,
                           40.0, 20.0)
    support = adaptive_field_support(extent_width_m, extent_height_m, 30.0)
    records, domain = register(
        visible_high, thermal_high, joint, transform, resolution_m,
        tile_m=plan.tile_m, step_m=plan.step_m, workers=1,
        domain_mode="mode")
    accepted = [record for record in records
                if record.get("consistent", True)
                and not record.get("ambiguous", False)]
    displacement = {}
    if accepted:
        field = fit_regularized_field(
            accepted, tile_m=plan.tile_m,
            max_distance_m=support.max_distance_m,
            kernel_length_m=None, ridge=0.1)
        east = np.asarray([row["east"] for row in accepted], float)
        north = np.asarray([row["north"] for row in accepted], float)
        de, dn, _, _ = field.evaluate(east, north)
        residual = np.hypot(
            np.asarray([row["dx_m"] for row in accepted]) - de,
            np.asarray([row["dy_m"] for row in accepted]) - dn)
        displacement = {
            "accepted": len(accepted),
            "records": len(records),
            "ambiguous": sum(bool(row["ambiguous"]) for row in records),
            "acceptance_rate": float(len(accepted) / max(len(records), 1)),
            "kernel_length_m": float(field.kernel_length_m),
            "residual_semivariogram": _semivariogram(
                np.column_stack([east, north]), residual),
        }

    result = {
        "resolution_m": float(resolution_m),
        "dsm_gsd_m": {"visible": gsd_visible, "thermal": gsd_thermal},
        "coverage": {
            "visible_valid_fraction": float(visible_valid.mean()),
            "thermal_valid_fraction": float(thermal_valid.mean()),
            "joint_valid_fraction": float(joint.mean()),
        },
        "texture": {
            "visible": _texture_stats(visible_high, visible_valid),
            "thermal": _texture_stats(thermal_high, thermal_valid),
        },
        "period_m": {"visible": period_visible, "thermal": period_thermal},
        "tiling": {"tile_m": float(plan.tile_m), "step_m": float(plan.step_m),
                   "support_m": float(support.max_distance_m)},
        "component_layout": None,
        "displacement": displacement,
    }
    if visible_world is not None and visible_world.get("panels"):
        panels = visible_world["panels"]
        centers = np.asarray([panel["center_xyz"][:2] for panel in panels], float)
        spacing = cKDTree(centers).query(centers, k=2)[0][:, 1]
        spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
        result["component_layout"] = {
            "panels": len(panels),
            "arrays": len({str(panel["array_id"]) for panel in panels}),
            "panel_spacing_median_m": float(np.median(spacing)) if len(spacing) else None,
        }
    return result

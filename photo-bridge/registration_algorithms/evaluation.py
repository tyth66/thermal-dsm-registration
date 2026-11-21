"""Regression-tested helpers for V3-B spatial evaluation."""
from __future__ import annotations

import numpy as np


def quadrant_ids(points, bounds):
    """Return 0..3 quadrant ids for fixed bounds (xmin, ymin, xmax, ymax)."""
    points = np.asarray(points, float)
    xmin, ymin, xmax, ymax = bounds
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    return ((points[:, 0] >= xmid).astype(int)
            + 2 * (points[:, 1] >= ymid).astype(int))


def footprint_overlap_matrix(left_points, right_points, tile_m, buffer_m=0.0):
    """Axis-aligned square footprint overlap, with optional per-tile buffer."""
    left = np.asarray(left_points, float)
    right = np.asarray(right_points, float)
    limit = float(tile_m) + 2.0 * float(buffer_m)
    dx = np.abs(left[:, None, 0] - right[None, :, 0])
    dy = np.abs(left[:, None, 1] - right[None, :, 1])
    return (dx < limit) & (dy < limit)


def training_mask(fold_ids, held_fold, train_points, held_points,
                  tile_m, buffer_m=0.0):
    """Training candidates outside the fold and without held-out footprint overlap."""
    fold_ids = np.asarray(fold_ids, int)
    base = fold_ids != int(held_fold)
    overlap = footprint_overlap_matrix(
        train_points, held_points, tile_m, buffer_m).any(axis=1)
    return base & ~overlap


def paired_supported_errors(error_a, support_a, error_b, support_b):
    """Paired statistics on the common supported set only."""
    error_a = np.asarray(error_a, float)
    error_b = np.asarray(error_b, float)
    support_a = np.asarray(support_a, bool)
    support_b = np.asarray(support_b, bool)
    common = (support_a & support_b
              & np.isfinite(error_a) & np.isfinite(error_b))
    if not common.any():
        return {"n": 0, "mean_a": None, "mean_b": None, "delta_mean": None,
                "abs_delta_p50": None, "abs_delta_p95": None}
    delta = error_a[common] - error_b[common]
    return {
        "n": int(common.sum()),
        "mean_a": float(np.mean(error_a[common])),
        "mean_b": float(np.mean(error_b[common])),
        "delta_mean": float(np.mean(delta)),
        "abs_delta_p50": float(np.median(np.abs(delta))),
        "abs_delta_p95": float(np.percentile(np.abs(delta), 95)),
    }


def uniqueness_observable(candidate_count):
    """A margin is only observable when at least two candidates exist."""
    return np.asarray(candidate_count, int) >= 2

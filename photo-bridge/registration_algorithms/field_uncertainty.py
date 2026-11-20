"""Empirical field uncertainty by spatial block bootstrap."""
from __future__ import annotations

import numpy as np

from .residual_model import fit_regularized_field


def block_bootstrap_uncertainty(rows, tile_m: float, max_distance_m: float,
                                query_east, query_north, blocks: int = 3,
                                draws: int = 100, seed: int = 0,
                                ridge: float = 0.1) -> dict:
    """Resample spatial blocks, refit the field, and return query covariance.

    Tiles inside one block share DSM pixels; resampling whole blocks preserves
    that dependence better than resampling individual tiles.  The result is an
    empirical spread, not a calibrated posterior; periodic aliases can still
    produce a small spread with a wrong mean.
    """
    rows = list(rows)
    if len(rows) < 8:
        raise ValueError("block bootstrap needs at least 8 observations")
    east = np.asarray([float(row["east"]) for row in rows], float)
    north = np.asarray([float(row["north"]) for row in rows], float)
    x_edges = np.quantile(east, np.linspace(0.0, 1.0, blocks + 1))
    y_edges = np.quantile(north, np.linspace(0.0, 1.0, blocks + 1))
    x_index = np.clip(np.digitize(east, x_edges[1:-1]), 0, blocks - 1)
    y_index = np.clip(np.digitize(north, y_edges[1:-1]), 0, blocks - 1)
    block_id = x_index * blocks + y_index
    groups = {int(block): np.where(block_id == block)[0]
              for block in np.unique(block_id)}
    group_ids = sorted(groups)
    if len(group_ids) < 2:
        raise ValueError("block bootstrap needs at least two occupied blocks")
    rng = np.random.default_rng(seed)
    query_east = np.asarray(query_east, float)
    query_north = np.asarray(query_north, float)
    samples = []
    for _ in range(int(draws)):
        chosen = rng.choice(group_ids, size=len(group_ids), replace=True)
        selected = np.concatenate([groups[int(block)] for block in chosen])
        if len(selected) < 6:
            continue
        subset = [rows[index] for index in selected]
        field = fit_regularized_field(
            subset, tile_m=tile_m, max_distance_m=max_distance_m,
            kernel_length_m=None, ridge=ridge)
        delta_e, delta_n, _, _ = field.evaluate(query_east, query_north)
        samples.append(np.column_stack([delta_e, delta_n]))
    if not samples:
        raise ValueError("block bootstrap produced no valid fits")
    samples = np.asarray(samples, float)
    mean = np.mean(samples, axis=0)
    centered = samples - mean[None, :, :]
    covariance = np.einsum("dni,dnj->nij", centered, centered) / len(samples)
    return {"samples": samples, "mean": mean, "covariance": covariance}

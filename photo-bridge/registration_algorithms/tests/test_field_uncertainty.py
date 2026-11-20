"""P1 block-bootstrap uncertainty contract."""
from __future__ import annotations

import numpy as np

from registration_algorithms.field_uncertainty import block_bootstrap_uncertainty


def test_block_bootstrap_returns_positive_semidefinite_covariance():
    rng = np.random.default_rng(4)
    rows = []
    for east in np.arange(0.0, 41.0, 8.0):
        for north in np.arange(0.0, 41.0, 8.0):
            rows.append({
                "east": float(east), "north": float(north),
                "dx_m": 0.2 + 0.01 * rng.normal(),
                "dy_m": -0.1 + 0.01 * rng.normal(),
                "cc": 0.95, "quality_weight": 1.0,
                "sigma_e": 0.05, "sigma_n": 0.05, "sigma_en": 0.0,
            })
    result = block_bootstrap_uncertainty(
        rows, tile_m=8.0, max_distance_m=30.0,
        query_east=[5.0, 20.0], query_north=[5.0, 20.0],
        blocks=3, draws=20, seed=1)
    covariance = result["covariance"]
    assert covariance.shape == (2, 2, 2)
    assert np.all(np.isfinite(covariance))
    for matrix in covariance:
        assert np.all(np.linalg.eigvalsh(matrix) >= -1e-12)

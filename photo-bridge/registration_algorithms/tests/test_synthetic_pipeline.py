"""Synthetic end-to-end P0 -> field recovery with a known shift."""
from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin

from registration_algorithms.dsm_registration import register
from registration_algorithms.residual_model import fit_regularized_field


def _texture(size=128, seed=3):
    rng = np.random.default_rng(seed)
    visible = rng.normal(0, 1, (size, size)).astype(np.float32)
    visible[30:50, 40:80] += 4.0
    visible[70:75, 10:100] += 3.0
    return visible


def test_synthetic_pipeline_recovers_known_shift():
    visible = _texture()
    thermal = np.roll(visible, shift=(2, -3), axis=(0, 1))
    joint = np.ones_like(visible, bool)
    transform = from_origin(0.0, 100.0, 0.1, 0.1)
    records, domain = register(
        visible, thermal, joint, transform, 0.1, 4.0, 2.0,
        workers=1, domain_mode="mode")
    assert len(records) >= 10
    assert domain["center_dx"] == pytest.approx(-0.30, abs=0.02)
    assert domain["center_dy"] == pytest.approx(-0.20, abs=0.02)
    field = fit_regularized_field(records, tile_m=4.0, max_distance_m=30.0,
                                  kernel_length_m=None, ridge=0.1)
    east = np.asarray([record["east"] for record in records])
    north = np.asarray([record["north"] for record in records])
    dx, dy, _, _ = field.evaluate(east, north)
    error = np.hypot(dx + 0.30, dy + 0.20)
    assert float(np.mean(error)) < 0.02

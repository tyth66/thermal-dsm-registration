"""Masked ECC, forward-backward consistency and covariance contracts."""
from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin

from registration_algorithms.dsm_registration import _register_tile
from registration_algorithms.masked_ecc import (
    masked_ecc_candidates,
    masked_ecc_surface,
)


def _structured_pair(size=64, roll=(2, -3)):
    rng = np.random.default_rng(3)
    visible = rng.normal(0, 1, (size, size)).astype(np.float32)
    visible[20:28, 18:46] += 4.0
    thermal = np.roll(visible, shift=roll, axis=(0, 1))
    return visible, thermal


def test_masked_ecc_recovers_known_roll():
    visible, thermal = _structured_pair()
    candidates = masked_ecc_candidates(
        visible, thermal, np.ones_like(visible, bool), 0.1, top_k=3)
    assert candidates
    best = candidates[0]
    assert best["cc"] > 0.9
    assert best["sx"] == pytest.approx(-3.0, abs=0.2)
    assert best["sy"] == pytest.approx(2.0, abs=0.2)
    # World convention: dx = sx*r, dy = -sy*r.
    assert best["sx"] * 0.1 == pytest.approx(-0.3, abs=0.02)
    assert -best["sy"] * 0.1 == pytest.approx(-0.2, abs=0.02)


def test_masked_ecc_excludes_nodata_hole():
    visible, thermal = _structured_pair()
    mask = np.ones_like(visible, bool)
    mask[25:35, 20:40] = False
    candidates = masked_ecc_candidates(visible, thermal, mask, 0.1, top_k=3)
    assert candidates
    assert candidates[0]["cc"] > 0.9
    assert candidates[0]["sx"] == pytest.approx(-3.0, abs=0.2)
    assert candidates[0]["sy"] == pytest.approx(2.0, abs=0.2)


def test_periodic_pattern_produces_competing_peaks():
    pattern = np.zeros((64, 64), np.float32)
    for column in range(0, 64, 12):
        pattern[:, column:column + 4] = 1.0
    for row in range(0, 64, 10):
        pattern[row:row + 4, :] += 0.5
    candidates = masked_ecc_candidates(
        pattern, pattern, np.ones((64, 64), bool), 0.1, top_k=5)
    assert len(candidates) >= 2
    separation = np.hypot(candidates[0]["sx"] - candidates[1]["sx"],
                          candidates[0]["sy"] - candidates[1]["sy"]) * 0.1
    assert separation > 0.0
    assert candidates[0]["cc"] - candidates[1]["cc"] < 0.5


def test_register_tile_records_observability_fields():
    rng = np.random.default_rng(9)
    size = 32
    visible = rng.normal(0, 1, (size, size)).astype(np.float32)
    thermal = np.roll(visible, shift=(2, -3), axis=(0, 1))
    transform = from_origin(596000.0, 3234000.0, 0.1, 0.1)
    record = _register_tile(visible, thermal, np.ones_like(visible, bool),
                            transform, 0.1, size,
                            row=0, column=0, top=0, left=0)
    assert record is not None
    for key in ("cc_best", "cc_second", "margin", "separation_m", "period_m",
                "reverse_cc", "fb_error_m", "consistent", "ambiguous",
                "sigma_e", "sigma_n", "sigma_en", "uniqueness"):
        assert key in record
    assert record["sigma_e"] > 0.0
    assert record["sigma_n"] > 0.0
    assert np.isfinite(record["fb_error_m"])
    assert record["consistent"] is True
    assert record["cc_best"] >= record["cc_second"]
    covariance = np.array([[record["sigma_e"] ** 2, record["sigma_en"]],
                           [record["sigma_en"], record["sigma_n"] ** 2]])
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


def test_separate_masks_recover_pixels_excluded_by_joint_mask():
    size = 24
    rng = np.random.default_rng(5)
    visible = rng.normal(0, 1, (size, size)).astype(np.float32)
    thermal = np.roll(visible, shift=(1, -2), axis=(0, 1))
    visible_mask = np.ones((size, size), bool)
    thermal_mask = np.ones((size, size), bool)
    thermal_mask[:, 0] = False
    joint = visible_mask & thermal_mask
    _, _, joint_area = masked_ecc_surface(
        visible, thermal, joint, max_shift_px=2)
    _, _, separate_area = masked_ecc_surface(
        visible, thermal, joint, max_shift_px=2,
        visible_mask=visible_mask, thermal_mask=thermal_mask)
    assert np.all(separate_area >= joint_area - 1e-9)
    assert np.any(separate_area > joint_area)


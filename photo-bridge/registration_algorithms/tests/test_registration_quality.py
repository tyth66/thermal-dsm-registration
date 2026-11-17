"""Regression tests for nodata highpass and ECC direction contracts."""
from __future__ import annotations

import numpy as np
from rasterio.transform import from_origin

from registration_algorithms.dsm_registration import (
    ecc, highpass, pixel_shift_to_world,
)


def test_pixel_shift_to_world_changes_row_sign():
    dx, dy = pixel_shift_to_world(2.0, -3.0, resolution_m=0.5)
    assert dx == 1.0
    assert dy == 1.5


def test_ecc_direction_for_known_thermal_roll():
    rng = np.random.default_rng(11)
    visible = rng.normal(0, 1, (64, 64)).astype(np.float32)
    visible[20:28, 18:46] += 4.0
    thermal = np.roll(visible, shift=(-6, 9), axis=(0, 1))
    shift_x, shift_y, cc = ecc(visible, thermal)
    dx, dy = pixel_shift_to_world(shift_x, shift_y, 0.1)
    assert cc > 0.9
    assert abs(dx - 0.9) < 0.2
    assert abs(dy - 0.6) < 0.2


def test_highpass_does_not_fill_nodata_with_median_edge():
    height, width = 80, 80
    east = np.arange(width, dtype=np.float32)[None, :]
    plane = np.repeat(east, height, axis=0) * 0.1
    valid = np.ones((height, width), bool)
    valid[25:55, 30:52] = False
    high = highpass(plane, valid, sigma_px=3.0)
    assert np.all(high[~valid] == 0.0)
    # A smooth plane has no high-frequency signal away from the hole.
    assert np.median(np.abs(high[5:20, 5:20])) < 1e-3
    # The normalized filter must not create a sign-flipped median edge at the
    # hole boundary.
    assert np.max(np.abs(high)) < 1.5


def test_register_tile_reports_quality_fields_on_simple_tile():
    from registration_algorithms.dsm_registration import _register_tile
    rng = np.random.default_rng(5)
    tile = 32
    visible = rng.normal(0, 1, (tile, tile)).astype(np.float32)
    thermal = np.roll(visible, shift=(2, -3), axis=(0, 1))
    transform = from_origin(596000.0, 3234000.0, 0.1, 0.1)
    record = _register_tile(visible, thermal, np.ones_like(visible, bool),
                            transform, 0.1, tile, 0.5, 2.0,
                            row=0, column=0, top=0, left=0)
    assert record is not None
    for key in ("candidates", "candidate_count", "second_margin",
                "periodic_ambiguity", "unique", "initial_consistent"):
        assert key in record

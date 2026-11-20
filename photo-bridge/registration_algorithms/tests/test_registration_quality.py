"""Regression tests for nodata highpass and ECC direction contracts."""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from registration_algorithms.dsm_registration import (
    DISPLACEMENT_BOUND_FLOOR_M, apply_displacement_domain, displacement_domain,
    ecc, highpass, load_grid, pixel_shift_to_world, register,
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


def test_register_tile_returns_every_finite_ecc_shift():
    from registration_algorithms.dsm_registration import _register_tile
    rng = np.random.default_rng(5)
    tile = 32
    visible = rng.normal(0, 1, (tile, tile)).astype(np.float32)
    thermal = np.roll(visible, shift=(2, -3), axis=(0, 1))
    transform = from_origin(596000.0, 3234000.0, 0.1, 0.1)
    record = _register_tile(visible, thermal, np.ones_like(visible, bool),
                            transform, 0.1, tile,
                            row=0, column=0, top=0, left=0)
    assert record is not None
    # No admission verdicts survive: a finite ECC solution is an observation.
    assert record["accepted"] is True
    assert record["outlier"] is False
    assert np.isfinite(record["cc"])
    for key in ("dx_m", "dy_m", "magnitude_m", "cc", "ncc_before",
                "ncc_after", "valid_fraction", "quality_weight"):
        assert key in record
    assert record["quality_weight"] == pytest.approx(1.0)
    for removed in ("candidates", "candidate_count", "second_margin",
                    "periodic_ambiguity", "unique"):
        assert removed not in record


def test_register_tile_accepts_half_support_and_carries_coverage_weight():
    from registration_algorithms.dsm_registration import _register_tile
    rng = np.random.default_rng(7)
    tile = 32
    visible = rng.normal(0, 1, (tile, tile)).astype(np.float32)
    thermal = np.roll(visible, shift=(2, -3), axis=(0, 1))
    mask = np.zeros((tile, tile), bool)
    mask[:, :19] = True
    transform = from_origin(596000.0, 3234000.0, 0.1, 0.1)
    record = _register_tile(visible, thermal, mask, transform, 0.1, tile,
                            row=0, column=0, top=0, left=0)
    assert record is not None
    assert record["valid_fraction"] == pytest.approx(19.0 / 32.0)
    assert record["quality_weight"] == pytest.approx((19.0 / 32.0) ** 1.5)


def _observation(dx, dy):
    return {"dx_m": float(dx), "dy_m": float(dy)}


def test_displacement_domain_keeps_real_drift_above_floor_and_drops_alias():
    # A station with ~3 m of real drift: the domain is estimated from the
    # observations, so the true drift is preserved instead of being clipped by
    # the fixed 1.5 m bound calibrated on the 141 station.
    offsets = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5] * 2
    dy_offsets = [0.0, 0.1, -0.1, 0.2, -0.2] * 4
    records = [_observation(3.0 + dx, 0.2 + dy)
               for dx, dy in zip(offsets, dy_offsets)]
    records.append(_observation(0.3, 17.67))   # periodic ECC alias
    records.append(_observation(3.2, 0.1))     # genuine drift edge

    domain = displacement_domain(records)
    kept, stats = apply_displacement_domain(records)

    assert domain["center_dx"] == pytest.approx(3.0, abs=0.1)
    assert domain["bound_dx"] == pytest.approx(DISPLACEMENT_BOUND_FLOOR_M)
    assert domain["center_dy"] == pytest.approx(0.2, abs=0.1)
    assert domain["bound_dy"] == pytest.approx(DISPLACEMENT_BOUND_FLOOR_M)
    kept_pairs = {(round(row["dx_m"], 6), round(row["dy_m"], 6)) for row in kept}
    assert (3.2, 0.1) in kept_pairs           # real 3.2 m observation survives
    assert (0.3, 17.67) not in kept_pairs     # alias is an invalid measurement
    assert abs(17.67 - domain["center_dy"]) > domain["bound_dy"]
    assert stats["rejected"] == 1


def test_displacement_domain_falls_back_to_floor_for_compact_station():
    # MAD ~ 0 (uniform drift): the floor keeps the domain wide enough for the
    # station's real spatial variation.
    records = [_observation(0.2, 0.1) for _ in range(20)]
    records.append(_observation(1.2, 0.1))    # true 1.2 m observation
    kept, stats = apply_displacement_domain(records)
    assert stats["bound_dx"] == pytest.approx(DISPLACEMENT_BOUND_FLOOR_M)
    assert stats["bound_dy"] == pytest.approx(DISPLACEMENT_BOUND_FLOOR_M)
    kept_pairs = {(round(row["dx_m"], 6), round(row["dy_m"], 6)) for row in kept}
    assert (1.2, 0.1) in kept_pairs
    assert stats["rejected"] == 0


def test_nan_displacement_is_rejected_without_poisoning_the_domain():
    # A single non-finite measurement must not turn the median/MAD (and hence
    # the whole domain) into NaN: that rejected every record instead of one.
    records = [_observation(0.10, 0.20), _observation(0.20, 0.10),
               _observation(0.15, 0.25), _observation(0.05, 0.15)]
    records.append(_observation(np.nan, 0.20))

    domain = displacement_domain(records)
    kept, stats = apply_displacement_domain(records)

    assert np.isfinite(domain["center_dx"])
    assert np.isfinite(domain["center_dy"])
    assert len(kept) == 4
    assert stats["rejected"] == 1
    assert all(np.isfinite(row["dx_m"]) and np.isfinite(row["dy_m"])
               for row in kept)


def _write_dsm(path, *, crs, west=596000.0, north=3234000.0,
               size=8, resolution=0.5):
    with rasterio.open(path, "w", driver="GTiff", width=size, height=size,
                       count=1, dtype="float32", crs=crs,
                       transform=from_origin(west, north, resolution,
                                             resolution)) as dataset:
        dataset.write(np.full((size, size), 20.0, np.float32), 1)
    return path


def test_load_grid_rejects_mismatched_dsm_crs(tmp_path):
    visible = _write_dsm(tmp_path / "visible.tif", crs="EPSG:32649")
    thermal = _write_dsm(tmp_path / "thermal.tif", crs="EPSG:32650")
    with pytest.raises(ValueError, match="CRS differ"):
        load_grid(visible, thermal, 0.5)


def test_load_grid_rejects_geographic_dsm_crs(tmp_path):
    visible = _write_dsm(tmp_path / "visible.tif", crs="EPSG:4326")
    thermal = _write_dsm(tmp_path / "thermal.tif", crs="EPSG:4326")
    with pytest.raises(ValueError, match="projected metre CRS"):
        load_grid(visible, thermal, 0.5)


def test_load_grid_rejects_empty_dsm_overlap(tmp_path):
    visible = _write_dsm(tmp_path / "visible.tif", crs="EPSG:32649")
    thermal = _write_dsm(tmp_path / "thermal.tif", crs="EPSG:32649",
                         west=596000.0 + 10000.0)
    with pytest.raises(ValueError, match="overlap is empty"):
        load_grid(visible, thermal, 0.5)


def test_register_includes_last_full_tile_when_extent_is_exact_multiple(monkeypatch):
    from registration_algorithms import dsm_registration
    calls = []

    def fake_tile(visible, thermal, joint, transform, resolution, tile,
                  row, column, top, left):
        calls.append((row, column, top, left))
        return {"dx_m": 0.0, "dy_m": 0.0}

    monkeypatch.setattr(dsm_registration, "_register_tile", fake_tile)
    size, tile, step = 96, 32, 16    # (96 - 32) % 16 == 0: the last tile is full
    joint = np.ones((size, size), bool)
    transform = from_origin(596000.0, 3234000.0, 0.1, 0.1)
    records, _ = register(np.zeros((size, size), np.float32),
                          np.zeros((size, size), np.float32),
                          joint, transform, 0.1, 3.2, 1.6)

    assert len(calls) == 25
    assert max(top for _, _, top, _ in calls) == size - tile
    assert (4, 4, size - tile, size - tile) in calls
    assert len(records) == 25


def test_register_returns_empty_for_raster_smaller_than_tile(monkeypatch, capsys):
    from registration_algorithms import dsm_registration
    monkeypatch.setattr(dsm_registration, "_register_tile",
                        lambda *args, **kwargs: pytest.fail("tile attempted"))
    joint = np.ones((20, 20), bool)
    transform = from_origin(596000.0, 3234000.0, 0.1, 0.1)
    records, stats = register(np.zeros((20, 20), np.float32),
                              np.zeros((20, 20), np.float32),
                              joint, transform, 0.1, 4.0, 2.0)
    assert records == []
    assert stats["rejected"] == 0
    assert "smaller than one" in capsys.readouterr().out



def test_mode_domain_rejects_large_alias_cluster():
    records = [_observation(0.2 + 0.02 * (i % 5), 0.1) for i in range(20)]
    records += [_observation(5.0, 0.1) for _ in range(8)]
    records += [_observation(-5.0, 0.1) for _ in range(5)]
    domain = displacement_domain(records, mode="mode")
    kept, stats = apply_displacement_domain(records, mode="mode")
    assert domain["center_dx"] == pytest.approx(0.2, abs=0.1)
    assert domain["bound_dx"] == pytest.approx(DISPLACEMENT_BOUND_FLOOR_M)
    kept_pairs = {(round(row["dx_m"], 6), round(row["dy_m"], 6)) for row in kept}
    assert (0.2, 0.1) in kept_pairs
    assert (5.0, 0.1) not in kept_pairs
    assert stats["rejected"] == 13


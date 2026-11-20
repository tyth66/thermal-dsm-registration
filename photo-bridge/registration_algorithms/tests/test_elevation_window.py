"""Site-adaptive elevation window: high-altitude DSMs stay usable.

Regression tests for the silent-failure mode fixed by elevation_range_for:
the legacy absolute window (-100, 80) m rejected every sample of a site
whose elevations sit outside it, so footprint sampling returned nodata and
tile registration died with "no accepted DSM tiles" without any diagnostic
pointing at the elevation window.
"""
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import from_origin

from registration_algorithms.dsm_sampling import (
    LEGACY_ELEVATION_RANGE_M,
    center_elevation,
    elevation_range_for,
    sample_footprint,
)
from registration_algorithms.dsm_registration import load_grid


def _write_band(path, values, *, transform=None, nodata=None, crs=None):
    shape = values.shape
    with rasterio.open(path, "w", driver="GTiff", width=shape[1], height=shape[0],
                       count=1, dtype="float32", nodata=nodata, crs=crs,
                       transform=transform if transform is not None else Affine.identity()) as ds:
        ds.write(values.astype(np.float32), 1)
    return path


def test_elevation_range_for_low_altitude_site_keeps_legacy_window(tmp_path):
    values = np.linspace(20.0, 30.0, 256).reshape(16, 16)
    path = _write_band(tmp_path / "low.tif", values)
    with rasterio.open(path) as ds:
        window = elevation_range_for(ds)
    assert window == (-100.0, 80.0)
    assert window == LEGACY_ELEVATION_RANGE_M


def test_elevation_range_for_high_altitude_site_recentres_window(tmp_path):
    values = np.linspace(2495.0, 2505.0, 256).reshape(16, 16)
    path = _write_band(tmp_path / "high.tif", values)
    with rasterio.open(path) as ds:
        window = elevation_range_for(ds)
    assert window == (2410.0, 2590.0)


def test_elevation_range_for_all_nodata_falls_back_to_legacy(tmp_path):
    values = np.full((16, 16), -9999.0)
    path = _write_band(tmp_path / "nodata.tif", values, nodata=-9999.0)
    with rasterio.open(path) as ds:
        window = elevation_range_for(ds)
    assert window == LEGACY_ELEVATION_RANGE_M


def test_sample_footprint_honours_explicit_elevation_range(tmp_path):
    corners = np.array([[2, 2], [10, 2], [10, 10], [2, 10]], float)
    inside = _write_band(tmp_path / "inside.tif", np.full((16, 16), 25.0))
    with rasterio.open(inside) as ds:
        summary = sample_footprint(ds, corners, 1., elevation_range=(10., 30.))
        assert summary["valid_pixels"] > 0
        assert summary["median_z"] == 25.0

    excluded = _write_band(tmp_path / "excluded.tif", np.full((16, 16), 35.0))
    with rasterio.open(excluded) as ds:
        summary = sample_footprint(ds, corners, 1., elevation_range=(10., 30.))
        assert summary["valid_pixels"] == 0
        assert summary["median_z"] is None


def test_sample_footprint_default_range_matches_legacy_behaviour(tmp_path):
    corners = np.array([[2, 2], [10, 2], [10, 10], [2, 10]], float)
    below = _write_band(tmp_path / "below.tif", np.full((16, 16), -150.0))
    with rasterio.open(below) as ds:
        summary = sample_footprint(ds, corners, 1.)
        assert summary["valid_pixels"] == 0
        assert summary["median_z"] is None

    within = _write_band(tmp_path / "within.tif", np.full((16, 16), 40.0))
    with rasterio.open(within) as ds:
        summary = sample_footprint(ds, corners, 1.)
        assert summary["valid_pixels"] > 0
        assert summary["median_z"] == 40.0


def test_center_elevation_and_footprint_samples_honour_explicit_range(tmp_path):
    corners = np.array([[2, 2], [10, 2], [10, 10], [2, 10]], float)
    low = _write_band(tmp_path / "low.tif", np.full((16, 16), 25.0))
    with rasterio.open(low) as ds:
        assert center_elevation(ds, np.array([6., 6.]), elevation_range=(10., 30.)) == 25.0
        summary = sample_footprint(ds, corners, 1., include_samples=True,
                                   elevation_range=(10., 30.))
        east, north, z = summary["samples"]
        assert len(z) > 0
    high = _write_band(tmp_path / "high.tif", np.full((16, 16), 35.0))
    with rasterio.open(high) as ds:
        assert center_elevation(ds, np.array([6., 6.]), elevation_range=(10., 30.)) is None
        summary = sample_footprint(ds, corners, 1., include_samples=True,
                                   elevation_range=(10., 30.))
        assert summary["valid_pixels"] == 0
        assert all(len(array) == 0 for array in summary["samples"])


def _write_high_altitude_pair(tmp_path):
    rng = np.random.default_rng(17)
    size, res = 64, 0.5
    transform = from_origin(500000.0, 3240000.0, res, res)
    crs = CRS.from_epsg(32649)
    ys, xs = np.mgrid[0:size, 0:size]
    visible = (2500.0 + 0.8 * np.sin(xs / 5.0) + 0.3 * np.cos(ys / 3.0)
               + rng.normal(0, 0.05, (size, size))).astype(np.float32)
    thermal = (2500.0 + 0.8 * np.sin((xs + 1) / 5.0) + 0.3 * np.cos(ys / 3.0)
               + rng.normal(0, 0.05, (size, size))).astype(np.float32)
    paths = []
    for name, values in (("visible.tif", visible), ("thermal.tif", thermal)):
        values[:4, :4] = -9999.0  # declared nodata block
        paths.append(_write_band(tmp_path / name, values, transform=transform,
                                 nodata=-9999.0, crs=crs))
    return paths


def test_load_grid_returns_valid_pixels_for_high_altitude_site(tmp_path):
    visible_path, thermal_path = _write_high_altitude_pair(tmp_path)
    visible_z, visible_valid, thermal_z, thermal_valid, _, _ = load_grid(
        visible_path, thermal_path, 0.5)
    assert visible_valid.any()
    assert thermal_valid.any()
    assert int((visible_valid & thermal_valid).sum()) > 0
    assert float(np.nanmedian(visible_z[visible_valid])) > 2000.0
    # The pre-fix hardcoded window (-100, 80) left the whole grid invalid --
    # the silent failure this regression test pins down.
    legacy = np.isfinite(visible_z) & (visible_z > -100) & (visible_z < 80)
    assert not legacy.any()
    assert int(visible_valid.sum()) > int(legacy.sum())

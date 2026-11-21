"""Site geometry profiler contracts."""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from registration_algorithms.site_geometry import _semivariogram, profile_site


def _write_pair(tmp_path, size=512, period_px=40):
    yy, xx = np.mgrid[0:size, 0:size]
    visible = (np.sin(2 * np.pi * xx / period_px)
               + np.sin(2 * np.pi * yy / period_px)).astype(np.float32)
    visible += np.random.default_rng(3).normal(0, 0.1, visible.shape).astype(np.float32)
    thermal = np.roll(visible, shift=(2, -3), axis=(0, 1))
    transform = from_origin(0.0, size * 0.1, 0.1, 0.1)
    paths = []
    for name, array in (("visible", visible), ("thermal", thermal)):
        path = tmp_path / (name + ".tif")
        with rasterio.open(path, "w", driver="GTiff", height=size, width=size,
                           count=1, dtype="float32", crs="EPSG:32650",
                           transform=transform, nodata=np.nan) as dataset:
            dataset.write(array, 1)
        paths.append(path)
    return paths


def test_profile_site_reports_period_coverage_and_texture(tmp_path):
    visible, thermal = _write_pair(tmp_path)
    result = profile_site(visible, thermal, resolution_m=0.1, max_dim=512)
    assert result["coverage"]["joint_valid_fraction"] == 1.0
    assert result["texture"]["visible"]["std"] > 0.0
    assert result["period_m"]["visible"] is not None
    assert result["period_m"]["visible"] == pytest.approx(4.0, abs=0.5)
    assert result["dimensionless"]["tile_period"] > 0.0
    assert result["dimensionless"]["prior_period"] > 0.0


def test_semivariogram_returns_sill_and_correlation_length():
    points = np.column_stack([np.arange(20.0), np.zeros(20)])
    values = np.exp(-points[:, 0] / 5.0)
    result = _semivariogram(points, values, bins=8)
    assert result["bins_m"]
    assert len(result["bins_m"]) == len(result["gamma"])
    assert result["correlation_length_m"] is not None

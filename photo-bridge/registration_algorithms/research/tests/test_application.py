"""End-to-end research API test with synthetic DSM rasters."""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from registration_algorithms.research import (
    ResearchPlanimetricInputs, estimate_research_planimetric,
)


def test_research_end_to_end_from_dsm_paths(tmp_path):
    rng = np.random.default_rng(5)
    visible = rng.normal(0.0, 1.0, (128, 128)).astype(np.float32)
    visible[30:50, 40:80] += 4.0
    visible[70:75, 10:100] += 3.0
    thermal = np.roll(visible, shift=(2, -3), axis=(0, 1))
    transform = from_origin(500000.0, 3234000.0, 0.1, 0.1)
    crs = CRS.from_epsg(32650)
    visible_path = tmp_path / "visible.tif"
    thermal_path = tmp_path / "thermal.tif"
    for path, array in ((visible_path, visible), (thermal_path, thermal)):
        with rasterio.open(
                path, "w", driver="GTiff", height=array.shape[0],
                width=array.shape[1], count=1, dtype="float32",
                crs=crs, transform=transform) as dataset:
            dataset.write(array, 1)
    inputs = ResearchPlanimetricInputs(
        visible_dsm=visible_path,
        thermal_dsm=thermal_path,
        visible_world={"panels": []},
        tile_m=4.0,
        step_m=2.0,
        max_iterations=2,
        sigma_prior_m=0.2,
    )
    result = estimate_research_planimetric(inputs)
    assert result.records
    assert result.inliers
    assert len(result.history) == 2
    assert result.sigma_f_m > 0.0
    assert np.isfinite(result.field.translation).all()
    assert 0.0 < result.tiling.tile_m <= 4.0
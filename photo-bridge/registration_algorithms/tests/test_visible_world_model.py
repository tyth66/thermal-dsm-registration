"""Tests for visible world model builder."""
from __future__ import annotations

import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

from registration_algorithms.visible_world_model import build_visible_world_model


def test_build_visible_world_model_from_panels_and_dsm(tmp_path):
    transform = from_origin(0.0, 100.0, 1.0, 1.0)
    profile = dict(driver="GTiff", width=100, height=100, count=1,
                   dtype="float32", crs="EPSG:32649", transform=transform)
    ortho_path = tmp_path / "ortho.tif"
    dsm_path = tmp_path / "dsm.tif"
    with rasterio.open(ortho_path, "w", **profile) as dataset:
        dataset.write(np.ones((100, 100), dtype="float32"), 1)
    with rasterio.open(dsm_path, "w", **profile) as dataset:
        dataset.write(np.full((100, 100), 25.0, dtype="float32"), 1)
    panels = {"schema_version": "standard-panels/v1",
              "panels": [{"id": "p%d" % index, "arrayId": "a0",
                          "x": 20.0 + 8.0 * (index % 5), "y": 20.0 + 8.0 * (index // 5),
                          "width": 4.0, "height": 3.0, "rotation": 0.0}
                         for index in range(10)]}
    panels_path = tmp_path / "panels.json"
    panels_path.write_text(json.dumps(panels), encoding="utf-8")
    model = build_visible_world_model(panels_path, ortho_path, dsm_path,
                                      dsm_erode_m=0.0, minimum_dsm_pixels=1)
    assert model["input_panel_count"] == 10
    assert len(model["panels"]) == 10
    assert model["crs"] == "EPSG:32649"
    assert set(model["surfaces"]) == {"a0"}
    assert all(panel["z_source"] == "m0_plane" for panel in model["panels"])
    assert all(abs(panel["center_xyz"][2] - 25.0) < 1e-6 for panel in model["panels"])

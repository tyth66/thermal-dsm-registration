"""Regression tests for the shared sampling/mapping planimetric field."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from registration_algorithms.dsm_registration import (
    cross_validate, cross_validate_spatial,
)
from registration_algorithms.residual_model import PlanimetricField
from registration_algorithms.vertical_estimation import sample_component_vertical


TOOLS = Path(__file__).resolve().parents[2] / "tools"


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules.
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiles():
    rows = []
    for row, north in enumerate(np.linspace(3234000, 3234100, 6)):
        for col, east in enumerate(np.linspace(596000, 596100, 6)):
            e, n = (east - 596000) / 100, (north - 3234000) / 100
            rows.append({
                "row": row, "column": col, "east": float(east), "north": float(north),
                "dx_m": float(0.2 + 0.1 * e + 0.05 * n * n),
                "dy_m": float(-0.1 + 0.08 * n + 0.03 * e * n),
                "cc": 0.9, "accepted": True, "outlier": False,
                "ncc_before": 0.6, "ncc_after": 0.9,
                "magnitude_m": 0.3,
            })
    return rows


def test_cross_validation_does_not_mutate_records(tiles):
    # Existing residual keys must be preserved as well as raw observations.
    for tile in tiles:
        tile.update(residual_x=12.0, residual_y=-8.0)
    before = copy.deepcopy(tiles)
    for kind in ("checkerboard", "quadrant"):
        result = cross_validate(tiles, kind)
        assert result["dx"]["n"] == len(tiles)
        assert tiles == before


def test_spatial_cross_validation_uses_held_out_tiles(tiles):
    block = cross_validate_spatial(tiles, blocks_x=3, blocks_y=3,
                                   buffer_m=4.0, tile_m=10.0,
                                   max_distance_m=100.0)
    assert block["dx"]["n"] > 0
    assert block["dy"]["n"] > 0



def test_sampling_uses_the_field_at_each_corner(tmp_path, monkeypatch, tiles):
    from registration_algorithms import vertical_estimation
    field = PlanimetricField.from_tiles(tiles)
    dsm_path = tmp_path / "dsm.tif"
    with rasterio.open(dsm_path, "w", driver="GTiff", width=2, height=2, count=1,
                       dtype="float32", transform=from_origin(596000, 3234100, 1, 1)) as ds:
        ds.write(np.full((2, 2), 20, np.float32), 1)
    corners = np.array([[596040, 3234040], [596060, 3234040],
                        [596060, 3234060], [596040, 3234060]], float)
    dx, dy, _, _ = field.evaluate(corners[:, 0], corners[:, 1])
    expected = corners + np.column_stack([dx, dy])
    def sample(ds, shifted, erode):
        np.testing.assert_allclose(shifted, expected, atol=1e-9, rtol=0)
        return {"median_z": 20.0, "mad_z": 0, "valid_pixels": 20}
    monkeypatch.setattr(vertical_estimation, "sample_footprint", sample)
    panel = {"panel_id": "p0", "array_id": "a0", "center_xyz": [596050, 3234050, 20],
             "corners_xyz": np.column_stack([corners, np.full(4, 20)]).tolist()}
    sample_component_vertical([panel], field, dsm_path, 0.1, 20)


def test_unsupported_footprints_still_enter_vertical_fit(tmp_path, monkeypatch, tiles):
    from registration_algorithms import vertical_estimation
    field = PlanimetricField.from_tiles(tiles, max_distance_m=0.1)
    dsm_path = tmp_path / "dsm.tif"
    with rasterio.open(dsm_path, "w", driver="GTiff", width=2, height=2, count=1,
                       dtype="float32", transform=from_origin(596000, 3234100, 1, 1)) as ds:
        ds.write(np.full((2, 2), 20, np.float32), 1)
    def sample(*args):
        return {"median_z": 21.0, "mad_z": 0.01, "valid_pixels": 20}
    monkeypatch.setattr(vertical_estimation, "sample_footprint", sample)
    panel = {"panel_id": "p0", "array_id": "a0", "center_xyz": [596050, 3234050, 20],
             "corners_xyz": [[596049, 3234049, 20], [596051, 3234049, 20],
                             [596051, 3234051, 20], [596049, 3234051, 20]]}
    rows = sample_component_vertical([panel], field, dsm_path, 0.1, 20)
    assert len(rows) == 1
    assert rows[0]["horizontal_valid"] == 0
    assert rows[0]["delta_z_measured_m"] == 1.0
    assert rows[0]["dsm_valid_pixels"] == 20


def test_unsupported_footprints_still_enter_thermal_array_fit(tmp_path, monkeypatch, tiles):
    from registration_algorithms import panel_thermal_model
    field = PlanimetricField.from_tiles(tiles, max_distance_m=0.1)
    visible_path, thermal_path = tmp_path / "visible.tif", tmp_path / "thermal.tif"
    for path in (visible_path, thermal_path):
        with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1,
                           dtype="float32", transform=from_origin(596000, 3234100, 1, 1)) as ds:
            ds.write(np.full((2, 2), 20, np.float32), 1)
    thermal_samples = []
    def sample(ds, *args):
        if Path(ds.name) == visible_path:
            return None
        thermal_samples.append(ds.name)
        e, n = np.meshgrid(np.linspace(596049, 596051, 8),
                           np.linspace(3234049, 3234051, 8))
        return e.ravel(), n.ravel(), np.full(e.size, 20.0)
    monkeypatch.setattr(panel_thermal_model, "sample_polygon_pixels", sample)
    panel = {"panel_id": "p0", "array_id": "a0", "center_xyz": [596050, 3234050, 20],
             "corners_xyz": [[596049, 3234049, 20], [596051, 3234049, 20],
                             [596051, 3234051, 20], [596049, 3234051, 20]]}
    component = {"p0": {"delta_e_thermal_vs_visible_m": 0.2,
                         "delta_n_thermal_vs_visible_m": 0.1}}
    arrays, panels, _, _ = panel_thermal_model.build_panel_thermal_model(
        {"panels": [panel]}, component, visible_path, thermal_path, 0.1, 20, 0.2, field)
    assert thermal_samples
    assert arrays["a0"]["support"] == 1
    assert panels["p0"]["z_source"] == "thermal_panel_plane"

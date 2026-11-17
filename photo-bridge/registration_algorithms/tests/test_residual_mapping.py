"""Tests for canonical residual model and forward mapping abstraction."""
from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from registration_algorithms.forward_mapping import grid_mapping, panel_points
from registration_algorithms.residual_model import (
    PlanimetricField, panels_from_model, validate_residual_model,
)


def _zero_field(radius: float = 1000.0) -> PlanimetricField:
    rows = []
    for east in np.linspace(0.0, 90.0, 10):
        for north in np.linspace(0.0, 90.0, 10):
            rows.append({"east": float(east), "north": float(north),
                         "cc": 1.0, "dx_m": 0.0, "dy_m": 0.0})
    return PlanimetricField.from_tiles(rows, radius=radius, power=2.0,
                                      max_distance_m=1000.0)


def _world_and_panels():
    panels = []
    for index, (east, north) in enumerate(((100.0, 100.0), (102.0, 100.0),
                                            (100.0, 102.0), (102.0, 102.0))):
        half = 1.0
        corners = [[east - half, north - half, 20.0],
                   [east + half, north - half, 20.0],
                   [east + half, north + half, 20.0],
                   [east - half, north + half, 20.0]]
        panels.append({"panel_id": "p%d" % index, "array_id": "a0",
                       "center_xyz": [east, north, 20.0],
                       "corners_xyz": corners})
    return {"panels": panels}


@pytest.fixture
def transforms():
    visible = from_origin(5.0, 205.0, 0.1, 0.1)
    thermal = from_origin(5.0, 205.0, 0.2, 0.2)
    return visible, thermal


def test_planimetric_field_model_roundtrip():
    field = _zero_field()
    model = field.to_model()
    restored = PlanimetricField.from_model(model)
    east = np.asarray([10.0, 40.0, 80.0])
    north = np.asarray([20.0, 50.0, 75.0])
    before = field.evaluate(east, north)
    after = restored.evaluate(east, north)
    np.testing.assert_allclose(before[0], after[0], atol=1e-9)
    np.testing.assert_allclose(before[1], after[1], atol=1e-9)
    np.testing.assert_allclose(before[2], after[2], atol=1e-9)


def test_panel_points_identity_mapping(transforms):
    visible, thermal = transforms
    world = _world_and_panels()
    field = _zero_field()
    panels = {panel["panel_id"]: {"visible_plane": None, "thermal_plane": None}
              for panel in world["panels"]}
    alpha = np.ones((250, 250), np.uint8)
    rows = panel_points(world, panels, field, visible, thermal, 1000, 1000,
                        alpha, 4, field_max_distance=1000.0)
    assert len(rows) == 20
    assert all(row["valid"] for row in rows)
    for row in rows:
        expected_u = (row["east_visible"] - 5.0) / 0.2
        expected_v = (205.0 - row["north_visible"]) / 0.2
        assert row["thermal_u"] == pytest.approx(expected_u, abs=1e-9)
        assert row["thermal_v"] == pytest.approx(expected_v, abs=1e-9)
        assert row["delta_e"] == pytest.approx(0.0, abs=1e-12)
        assert row["delta_n"] == pytest.approx(0.0, abs=1e-12)


def test_grid_mapping_shape_and_identity(transforms):
    visible, thermal = transforms
    world = _world_and_panels()
    field = _zero_field()
    panels = {panel["panel_id"]: {"visible_plane": None, "thermal_plane": None}
              for panel in world["panels"]}
    alpha = np.ones((250, 250), np.uint8)
    visible_z = np.zeros((100, 100), np.float64).ravel()
    result = grid_mapping(world, panels, field, visible, 2000, 2000, visible_z,
                          thermal, 1000, 1000, alpha, 4, 20, 1000.0)
    assert result["grid_height"] == 100
    assert result["grid_width"] == 100
    assert result["east"].shape == (10000,)
    assert result["valid"].all()
    np.testing.assert_allclose(result["delta_e"], 0.0, atol=1e-12)
    np.testing.assert_allclose(result["delta_n"], 0.0, atol=1e-12)


def test_panels_from_model_arrays():
    vertical = {"p0": {"array_id": "a0", "visible_plane": [0.1, 0.2, 0.3],
                       "thermal_plane": [0.4, 0.5, 0.6]}}
    panels = panels_from_model(vertical)
    assert panels["p0"]["array_id"] == "a0"
    assert panels["p0"]["visible_plane"].evaluate(1., 2.) == pytest.approx(0.8)
    assert panels["p0"]["thermal_plane"].evaluate(1., 2.) == pytest.approx(2.0)


def test_validate_residual_model_accepts_roundtrip():
    field = _zero_field()
    model = {"planimetric": field.to_model(),
             "vertical": {"panels": {"p0": {"array_id": "a0",
                                             "visible_plane": [0.0, 0.0, 1.0],
                                             "thermal_plane": None}}}}
    validate_residual_model(model)


def test_validate_residual_model_rejects_bad_panel():
    field = _zero_field()
    model = {"planimetric": field.to_model(),
             "vertical": {"panels": {"p0": {"visible_plane": [0.0, 0.0]}}}}
    with pytest.raises(ValueError):
        validate_residual_model(model)


class _FixedField:
    def __init__(self, dx=0.0, dy=0.0, weight=1.0, nearest=1.0, max_distance=30.0):
        self.dx = dx
        self.dy = dy
        self.weight = weight
        self.nearest = nearest
        self.max_distance_m = max_distance

    def evaluate(self, east, north):
        size = len(east)
        return (np.full(size, self.dx), np.full(size, self.dy),
                np.full(size, self.weight), np.full(size, self.nearest))


def _one_panel(east=100.0, north=100.0):
    panel = {"panel_id": "p0", "array_id": "a0",
             "center_xyz": [east, north, 20.0],
             "corners_xyz": [[east - 0.5, north - 0.5, 20.0],
                             [east + 0.5, north - 0.5, 20.0],
                             [east + 0.5, north + 0.5, 20.0],
                             [east - 0.5, north + 0.5, 20.0]]}
    return {"panels": [panel]}, {"p0": {"visible_plane": None, "thermal_plane": None}}


def _panel_rows(field, alpha, transforms, max_distance=None):
    visible, thermal = transforms
    world, panels = _one_panel()
    return panel_points(world, panels, field, visible, thermal, 1000, 1000,
                        alpha, 1, max_distance)


class _ShearField:
    max_distance_m = 30.0

    def evaluate(self, east, north):
        size = len(east)
        north = np.asarray(north, float)
        return (0.05 * (north - 100.0), np.zeros(size),
                np.ones(size), np.zeros(size))


def _panel_targets(panel_mapping):
    transforms = (from_origin(5.0, 205.0, 0.1, 0.1),
                  from_origin(5.0, 205.0, 0.2, 0.2))
    world, panels = _one_panel()
    alpha = np.ones((1000, 1000), np.uint8)
    rows = panel_points(world, panels, _ShearField(), transforms[0],
                        transforms[1], 1000, 1000, alpha, 1,
                        panel_mapping=panel_mapping)
    corners = [row for row in rows if row["point"].startswith("corner")]
    return np.asarray([[row["east_thermal"], row["north_thermal"]]
                       for row in corners])


def _edge_lengths(points):
    return np.linalg.norm(np.diff(np.vstack([points, points[:1]]), axis=0), axis=1)


def test_panel_rigid_mapping_preserves_shape():
    pointwise = _panel_targets("pointwise")
    rigid = _panel_targets("rigid")
    assert not np.allclose(_edge_lengths(pointwise), _edge_lengths(rigid))
    np.testing.assert_allclose(_edge_lengths(rigid), _edge_lengths(rigid)[0], atol=1e-9)


def test_panel_translation_alias_matches_legacy_rigid():
    np.testing.assert_allclose(_panel_targets("translation"),
                               _panel_targets("rigid"), atol=1e-12)


def test_panel_similarity_mapping_preserves_shape():
    pointwise = _panel_targets("pointwise")
    similarity = _panel_targets("similarity")
    assert not np.allclose(_edge_lengths(pointwise), _edge_lengths(similarity))
    # A similarity may rotate/scale, but opposite edges remain equal and adjacent
    # edges keep their ratio.
    lengths = _edge_lengths(similarity)
    assert lengths[0] == pytest.approx(lengths[2], abs=1e-9)
    assert lengths[1] == pytest.approx(lengths[3], abs=1e-9)


def test_any_originally_covered_panel_point_maps_all_five_points(transforms):
    alpha = np.ones((1000, 1000), np.uint8)
    # Only the center has original coverage; all five predictions are kept.
    alpha[:] = 0
    alpha[525, 475] = 255
    # The center's corrected target is nodata, but original coverage keeps the
    # whole component mapped under the requested component-level rule.
    alpha[525, 480] = 0
    rows = _panel_rows(_FixedField(dx=1.0), alpha, transforms)
    center = rows[0]
    assert center["naive_valid"] == 1
    assert center["corrected_valid"] == 0
    assert center["target_nodata"] == 1
    assert center["valid"] == 1
    assert center["residual_applied"] == 1
    assert all(row["valid"] == 1 for row in rows)


def test_originally_all_nodata_panel_is_excluded_even_if_corrected_covered(transforms):
    alpha = np.ones((1000, 1000), np.uint8)
    alpha[:] = 0
    alpha[525, 480] = 255
    rows = _panel_rows(_FixedField(dx=1.0), alpha, transforms)
    center = rows[0]
    assert center["naive_valid"] == 0
    assert center["corrected_valid"] == 1
    assert center["valid"] == 0
    assert all(row["valid"] == 0 for row in rows)


@pytest.mark.parametrize("field", [
    _FixedField(nearest=31.0),
    _FixedField(weight=0.0),
    _FixedField(dx=np.nan),
    _FixedField(dy=np.inf),
    _FixedField(nearest=np.inf),
])
def test_unsupported_field_does_not_block_originally_covered_panel(transforms, field):
    if not np.isfinite(field.dx) or not np.isfinite(field.dy):
        with pytest.raises(ValueError, match="non-finite planimetric prediction"):
            _panel_rows(field, np.ones((1000, 1000), np.uint8), transforms)
        return
    rows = _panel_rows(field, np.ones((1000, 1000), np.uint8), transforms)
    assert all(row["field_valid"] == 0 and row["valid"] == 1 for row in rows)


def test_explicit_field_distance_overrides_model_distance(transforms):
    field = _FixedField(nearest=31.0, max_distance=30.0)
    rows = _panel_rows(field, np.ones((1000, 1000), np.uint8), transforms,
                       max_distance=40.0)
    assert all(row["field_valid"] == 1 and row["valid"] == 1 for row in rows)


def test_empty_field_and_nonfinite_projected_coordinates_are_diagnostic(transforms):
    visible, thermal = transforms
    world, panels = _one_panel()
    empty = PlanimetricField.from_tiles([], max_distance_m=30.0)
    empty_rows = panel_points(world, panels, empty, visible, thermal, 1000, 1000,
                              np.ones((1000, 1000), np.uint8), 1, None)
    assert all(row["valid"] == 1 and row["field_valid"] == 0 for row in empty_rows)

    rows = _panel_rows(_FixedField(dx=1e9), np.ones((1000, 1000), np.uint8), transforms)
    assert all(row["corrected_valid"] == 0 and row["target_nodata"] == 1
               and row["valid"] == 1 for row in rows)


def test_panel_and_grid_validity_use_component_original_coverage(transforms):
    visible, thermal = transforms
    world, panels = _one_panel(east=100.0, north=100.0)
    field = _FixedField(dx=1.0, nearest=31.0)
    alpha = np.ones((1000, 1000), np.uint8)
    alpha[525, 475] = 0
    visible_z = np.zeros(10000, np.float64)
    result = grid_mapping(world, panels, field, visible, 2000, 2000, visible_z,
                          thermal, 1000, 1000, alpha, 1, 20, None)
    panel_rows = panel_points(world, panels, field, visible, thermal, 1000, 1000,
                              alpha, 1, None)
    assert all(row["valid"] == 1 for row in panel_rows)
    # Pixels inside the component inherit its five-point original-coverage rule.
    assert np.any(result["panel_id"] == "p0")
    assert np.all(result["valid"][result["panel_id"] == "p0"])
    assert np.all(result["vertical_source"][result["panel_id"] == "p0"]
                  == "no_vertical_surface_model")


def test_panel_without_vertical_model_still_maps(transforms):
    visible, thermal = transforms
    world, _ = _one_panel()
    rows = panel_points(world, {}, _FixedField(dx=0.25), visible, thermal,
                        1000, 1000, np.ones((1000, 1000), np.uint8), 1)
    assert len(rows) == 5
    assert all(row["valid"] == 1 for row in rows)
    assert all(row["vertical_source"].endswith("no_vertical_surface_model") for row in rows)
    assert any(row["thermal_u"] != row["visible_u"] for row in rows)


def _write_raster(path, data, transform, count, dtype="uint8"):
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[2],
                       height=data.shape[1], count=count, dtype=dtype,
                       crs="EPSG:32649", transform=transform) as dataset:
        dataset.write(data)

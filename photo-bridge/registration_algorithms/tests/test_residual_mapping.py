"""Tests for the planimetric field and the positions-only forward mapping."""
from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin

from registration_algorithms.forward_mapping import grid_mapping, panel_points
from registration_algorithms.residual_model import (
    PlanimetricField, fit_regularized_field,
)


def _zero_field() -> PlanimetricField:
    rows = []
    for east in np.linspace(0.0, 90.0, 10):
        for north in np.linspace(0.0, 90.0, 10):
            rows.append({"east": float(east), "north": float(north),
                         "cc": 1.0, "dx_m": 0.0, "dy_m": 0.0})
    return PlanimetricField.from_tiles(rows, max_distance_m=1000.0)


def _world():
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


def test_nonfinite_and_zero_cc_get_minimum_fit_weight():
    # A zero or non-finite ECC correlation is the least trustworthy observation,
    # so it must receive the minimum weight, not the maximum default 1.0.
    rows = []
    for east in np.linspace(0.0, 90.0, 10):
        for north in np.linspace(0.0, 90.0, 10):
            rows.append({"east": float(east), "north": float(north),
                         "cc": 1.0, "dx_m": 0.0, "dy_m": 0.0})
    rows.append({"east": 45.0, "north": 45.0, "cc": 0.0,
                 "dx_m": 0.0, "dy_m": 0.0})
    rows.append({"east": 45.0, "north": 45.0, "cc": np.nan,
                 "dx_m": 0.0, "dy_m": 0.0})

    field = fit_regularized_field(rows, tile_m=10.0, max_distance_m=1000.0)

    assert field.cc[-2] == pytest.approx(1e-3)
    assert field.cc[-1] == pytest.approx(1e-3)
    typical = float(np.median(field.weights[:-2]))
    assert field.weights[-2] < 0.01 * typical
    assert field.weights[-1] < 0.01 * typical


def test_panel_points_identity_mapping(transforms):
    visible, thermal = transforms
    world = _world()
    field = _zero_field()
    alpha = np.ones((250, 250), np.uint8)
    rows = panel_points(world, field, visible, thermal, 1000, 1000,
                        alpha, field_max_distance=1000.0)
    assert len(rows) == 20
    assert all(row["valid"] for row in rows)
    for row in rows:
        expected_u = (row["east_visible"] - 5.0) / 0.2
        expected_v = (205.0 - row["north_visible"]) / 0.2
        assert row["thermal_u"] == pytest.approx(expected_u, abs=1e-9)
        assert row["thermal_v"] == pytest.approx(expected_v, abs=1e-9)
        assert row["delta_e"] == pytest.approx(0.0, abs=1e-12)
        assert row["delta_n"] == pytest.approx(0.0, abs=1e-12)


def test_panel_rows_are_positions_only(transforms):
    visible, thermal = transforms
    alpha = np.ones((1000, 1000), np.uint8)
    rows = panel_points(_world(), _FixedField(dx=0.25), visible, thermal,
                        1000, 1000, alpha)
    for row in rows:
        assert "z_visible" not in row
        assert "z_thermal" not in row
        assert "delta_z" not in row
        assert "vertical_source" not in row
        assert "thermal_u" in row and "thermal_v" in row


def test_grid_mapping_shape_and_identity(transforms):
    visible, thermal = transforms
    world = _world()
    field = _zero_field()
    alpha = np.ones((250, 250), np.uint8)
    result = grid_mapping(world, field, visible, 2000, 2000,
                          thermal, 1000, 1000, alpha, 20, 1000.0)
    assert result["grid_height"] == 100
    assert result["grid_width"] == 100
    assert result["east"].shape == (10000,)
    assert result["valid"].all()
    assert result["panel_mask"].shape == (10000,)
    np.testing.assert_allclose(result["delta_e"], 0.0, atol=1e-12)
    np.testing.assert_allclose(result["delta_n"], 0.0, atol=1e-12)
    for removed in ("z_visible", "z_thermal", "delta_z", "vertical_source"):
        assert removed not in result


def test_grid_mapping_rejects_step_larger_than_visible_raster(transforms):
    visible, thermal = transforms
    alpha = np.ones((1000, 1000), np.uint8)
    with pytest.raises(ValueError, match="grid step exceeds visible raster dimensions"):
        grid_mapping(_world(), _zero_field(), visible, 10, 10,
                     thermal, 1000, 1000, alpha, 20, 1000.0)


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
    return {"panels": [panel]}


def _panel_rows(field, alpha, transforms, max_distance=None):
    visible, thermal = transforms
    return panel_points(_one_panel(), field, visible, thermal, 1000, 1000,
                        alpha, max_distance)


class _ShearField:
    max_distance_m = 30.0

    def evaluate(self, east, north):
        size = len(east)
        north = np.asarray(north, float)
        return (0.05 * (north - 100.0), np.zeros(size),
                np.ones(size), np.zeros(size))


def _panel_targets():
    transforms = (from_origin(5.0, 205.0, 0.1, 0.1),
                  from_origin(5.0, 205.0, 0.2, 0.2))
    alpha = np.ones((1000, 1000), np.uint8)
    rows = panel_points(_one_panel(), _ShearField(), transforms[0],
                        transforms[1], 1000, 1000, alpha)
    corners = [row for row in rows if row["point"].startswith("corner")]
    return np.asarray([[row["east_thermal"], row["north_thermal"]]
                       for row in corners])


def _edge_lengths(points):
    return np.linalg.norm(np.diff(np.vstack([points, points[:1]]), axis=0), axis=1)


def test_panel_rigid_mapping_preserves_shape():
    rigid = _panel_targets()
    np.testing.assert_allclose(_edge_lengths(rigid), _edge_lengths(rigid)[0], atol=1e-9)


def test_any_originally_covered_panel_point_maps_all_five_points(transforms):
    alpha = np.ones((1000, 1000), np.uint8)
    alpha[:] = 0
    alpha[525, 475] = 255
    alpha[525, 480] = 0
    rows = _panel_rows(_FixedField(dx=1.0), alpha, transforms)
    center = rows[0]
    assert center["naive_valid"] == 1
    assert center["corrected_valid"] == 0
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
    world = _one_panel()
    empty = PlanimetricField.from_tiles([], max_distance_m=30.0)
    empty_rows = panel_points(world, empty, visible, thermal, 1000, 1000,
                              np.ones((1000, 1000), np.uint8), None)
    assert all(row["valid"] == 1 and row["field_valid"] == 0 for row in empty_rows)

    rows = _panel_rows(_FixedField(dx=1e9), np.ones((1000, 1000), np.uint8), transforms)
    assert all(row["corrected_valid"] == 0 and row["valid"] == 1
               for row in rows)


def test_panel_and_grid_validity_use_component_original_coverage(transforms):
    visible, thermal = transforms
    world = _one_panel(east=100.0, north=100.0)
    field = _FixedField(dx=1.0, nearest=31.0)
    alpha = np.ones((1000, 1000), np.uint8)
    alpha[525, 475] = 0
    result = grid_mapping(world, field, visible, 2000, 2000,
                          thermal, 1000, 1000, alpha, 20, None)
    panel_rows = panel_points(world, field, visible, thermal, 1000, 1000,
                              alpha, None)
    assert all(row["valid"] == 1 for row in panel_rows)
    # Pixels inside the component inherit its five-point original-coverage rule.
    assert np.any(result["panel_id"] == "p0")
    assert np.all(result["valid"][result["panel_id"] == "p0"])
    assert np.all(result["panel_mask"][result["panel_id"] == "p0"])

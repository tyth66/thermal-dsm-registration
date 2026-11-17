import copy
import json

import numpy as np
import pytest
from rasterio.transform import from_origin

from registration_algorithms.array_vertical_bias import (
    build_bias_rows, fit_array_bias, validate_bias,
)
from registration_algorithms.forward_mapping import grid_mapping, panel_points
from registration_algorithms.local_plane import LocalPlane
from registration_algorithms.residual_model import (
    PlanimetricField, panels_from_model, vertical_model,
)


def scene(collinear=False):
    east, north = 596000., 3234000.
    offsets = [(x, 0) for x in range(-10, 10)] if collinear else [
        (x, y) for x in range(-10, 12, 2) for y in (-1., 1.)]
    residual = LocalPlane(east, north, .25, .03, -.02)
    panels, geometry, components = {}, {}, {}
    for index, (x, y) in enumerate(offsets):
        key = f"p{index}"
        e, n = east + x, north + y
        anchor_e, anchor_n = e + .6, n - .4
        plane = LocalPlane(anchor_e, anchor_n, 20 + float(residual.evaluate(anchor_e, anchor_n)), .03, -.02)
        corners = [[e + dx, n + dy, 20 + float(residual.evaluate(e + dx, n + dy))]
                   for dx, dy in [(-.4, -.4), (.4, -.4), (.4, .4), (-.4, .4)]]
        panels[key] = {"array_id": "a", "center_xyz": [e, n, 20 + float(residual.evaluate(e, n))],
                       "corners_xyz": corners, "panel_plane": dict(plane.to_dict(), rms_m=.01, support=100),
                       "z_source": "thermal_panel_plane"}
        geometry[key] = {"visible_plane_e0": east - 2, "visible_plane_n0": north + 1,
                         "visible_plane_z0": 20., "visible_plane_a": 0., "visible_plane_b": 0.}
        components[key] = {"delta_e_thermal_vs_visible_m": 2., "delta_n_thermal_vs_visible_m": -1.}
    model = {"panels": panels, "plane_representation": "local_en/v1", "footprint_sampling": "sample_footprint/v1"}
    return model, geometry, components, residual


@pytest.mark.parametrize("shrinkage", [0., .5, 1.])
def test_array_plane_corrects_each_point_and_surface_in_thermal_coordinates(shrinkage):
    model, geometry, components, truth = scene()
    original = copy.deepcopy(model)
    rows = build_bias_rows(model, geometry, components)
    median = np.median([row["residual"] for row in rows])
    biases, corrected, diagnostics = fit_array_bias(model, rows, 20, plane_shrinkage=shrinkage)
    assert model == original
    assert biases["a"]["bias_m"] == pytest.approx(median)
    expected = lambda e, n: median + shrinkage * (truth.evaluate(e, n) - median)
    for key, panel in corrected["panels"].items():
        for old_point, new_point in zip([model["panels"][key]["center_xyz"]] + model["panels"][key]["corners_xyz"],
                                        [panel["center_xyz"]] + panel["corners_xyz"]):
            assert new_point[:2] == old_point[:2]
            assert new_point[2] == pytest.approx(old_point[2] - expected(*old_point[:2]))
        old = LocalPlane.from_model(model["panels"][key]["panel_plane"])
        new = LocalPlane.from_model(panel["panel_plane"])
        assert (new.e0, new.n0) == (old.e0, old.n0)
        assert (new.a, new.b) == pytest.approx(((1 - shrinkage) * truth.a, (1 - shrinkage) * truth.b))
        assert new.evaluate(old.e0 + .17, old.n0 - .23) == pytest.approx(
            old.evaluate(old.e0 + .17, old.n0 - .23) - expected(old.e0 + .17, old.n0 - .23))
        assert panel["panel_plane"]["rms_m"] == .01
        assert panel["panel_plane"]["support"] == 100
    for row in diagnostics:
        assert row["array_bias_m"] == pytest.approx(expected(row["east"], row["north"]))
        assert row["residual_after_m"] == pytest.approx(row["residual_before_m"] - row["array_bias_m"])
    json.dumps(corrected, allow_nan=False)


def test_low_support_uses_global_median_and_rank_failure_uses_array_median():
    model, geometry, components, _ = scene()
    rows = build_bias_rows(model, geometry, components)
    rows[0]["array_id"] = "small"
    model["panels"][rows[0]["panel_id"]]["array_id"] = "small"
    biases, corrected, diagnostics = fit_array_bias(model, rows, 20)
    assert biases["small"]["method"] == "global_median_fallback"
    expected_global = np.median([row["residual"] for row in rows])
    assert diagnostics[0]["array_bias_m"] == pytest.approx(expected_global)
    assert len(corrected["panels"]) == len(model["panels"])
    model, geometry, components, _ = scene(collinear=True)
    rows = build_bias_rows(model, geometry, components)
    biases, _, diagnostics = fit_array_bias(model, rows, 20)
    assert biases["a"]["method"] == "per_array_median_fallback"
    assert "rank" in biases["a"]["fit_error"]
    np.testing.assert_allclose([row["array_bias_m"] for row in diagnostics], np.median([row["residual"] for row in rows]))


def test_no_observations_preserves_missing_surface_and_uses_finite_zero_correction():
    model = {"panels": {"missing": {"array_id": "none", "center_xyz": None, "corners_xyz": [], "z_source": "missing"}}}
    biases, corrected, rows = fit_array_bias(model, [], 20)
    assert biases == {} and rows == []
    assert corrected["panels"] == model["panels"]
    assert corrected["provenance"]["global_bias_m"] == 0.
    json.dumps(corrected, allow_nan=False)


@pytest.mark.parametrize("shrinkage", [-.1, 1.1, float("nan")])
def test_invalid_shrinkage_is_rejected(shrinkage):
    with pytest.raises(ValueError, match="shrinkage"):
        fit_array_bias({"panels": {}}, [], 20, plane_shrinkage=shrinkage)


def test_spatial_validation_uses_heldout_predictions_not_full_fit():
    model, geometry, components, _ = scene()
    rows = build_bias_rows(model, geometry, components)
    validation = validate_bias(rows, 20, plane_shrinkage=.5)
    assert validation["method"] == "five_contiguous_pca_slabs"
    assert validation["buffer_m"] == 1.5
    assert validation["evaluated_components"] == len(rows)
    assert validation["constant"]["n"] == validation["residual_plane"]["n"] == len(rows)
    assert validation["residual_plane"]["p95"] < validation["constant"]["p95"]
    assert validation["residual_plane"]["p95"] > .01


def test_serialized_correction_reaches_mapping_z_without_changing_xy_or_validity():
    model, geometry, components, _ = scene()
    rows = build_bias_rows(model, geometry, components)
    _, corrected, _ = fit_array_bias(model, rows, 20)
    _, constant, _ = fit_array_bias(model, rows, 20, plane_shrinkage=0)
    # Work on one component whose shape is unchanged by either height correction.
    key = "p10"
    p = model["panels"][key]
    world = {"panels": [{"panel_id": key, "array_id": "a", "center_xyz": [p["center_xyz"][0]-2, p["center_xyz"][1]+1, 20.],
                          "corners_xyz": [[e-2, n+1, 20.] for e, n, z in p["corners_xyz"]]}]}
    transform = from_origin(595980., 3234020., 1., 1.)
    field = PlanimetricField.from_tiles([{"east": 596000., "north": 3234000., "cc": 1., "dx_m": 2., "dy_m": -1.}])
    mapped = []
    grids = []
    for thermal in (constant, json.loads(json.dumps(corrected))):
        restored = panels_from_model(vertical_model(world["panels"], geometry, thermal["panels"]))
        mapped.append(panel_points(world, restored, field, transform, transform, 40, 40, np.ones((40,40), np.uint8), 1))
        grids.append(grid_mapping(world, restored, field, transform, 40, 40, np.full(1600,20.), transform, 40, 40, np.ones((40,40),np.uint8), 1, 1))
    for old, new in zip(*mapped):
        for field_name in ("east_thermal", "north_thermal", "thermal_u", "thermal_v", "valid"):
            assert old[field_name] == new[field_name]
        plane = LocalPlane.from_model(corrected["panels"][key]["panel_plane"])
        assert new["z_thermal"] == pytest.approx(plane.evaluate(new["east_thermal"], new["north_thermal"]))
    selected = grids[1]["panel_id"] == key
    assert selected.any()
    for field_name in ("thermal_u", "thermal_v", "valid"):
        np.testing.assert_array_equal(grids[0][field_name], grids[1][field_name])
    np.testing.assert_allclose(grids[1]["z_thermal"][selected], plane.evaluate(grids[1]["east_t"][selected], grids[1]["north_t"][selected]))
    assert set(grids[1]["vertical_source"][selected]) == {"panel_plane+array_residual_plane"}

import numpy as np
import pytest
from rasterio.transform import from_origin

from registration_algorithms.residual_model import panels_from_model, vertical_model, validate_residual_model, PlanimetricField
from registration_algorithms.forward_mapping import panel_points, grid_mapping
from registration_algorithms.array_vertical_bias import build_bias_rows, fit_array_bias


def test_local_plane_serialization_and_bias():
    plane = {"type": "local_plane", "e0": 596000., "n0": 3234000., "z0": 20., "a": .2, "b": -.1}
    geometry = {"p": {"visible_plane_" + k: v for k, v in plane.items() if k != "type"}}
    thermal = {"p": {"array_id": "a", "center_xyz": [596001., 3234002., 22.],
                      "corners_xyz": [[596001., 3234002., 22.]] * 4,
                      "panel_plane": dict(plane, z0=22.), "z_source": "thermal_panel_plane"}}
    components = {"p": {"delta_e_thermal_vs_visible_m": 0., "delta_n_thermal_vs_visible_m": 0.}}
    rows = build_bias_rows({"panels": thermal}, geometry, components)
    assert rows[0]["visible_z"] == pytest.approx(20.)
    _, corrected, _ = fit_array_bias({"panels": thermal}, rows, 1)
    assert corrected["panels"]["p"]["panel_plane"]["z0"] == pytest.approx(20.)
    assert corrected["panels"]["p"]["panel_plane"]["e0"] == plane["e0"]
    result = vertical_model([{"panel_id": "p", "array_id": "a"}], geometry, corrected["panels"])
    assert result["p"]["visible_plane"] == plane
    model = {"planimetric": PlanimetricField.from_tiles([]).to_model(), "vertical": {"panels": result}}
    validate_residual_model(model)
    restored = panels_from_model(result)
    assert restored["p"]["thermal_plane"].evaluate(596001., 3234002.) == pytest.approx(20.)


def test_local_plane_mapping_evaluates_at_separate_visible_and_thermal_positions():
    e, n = 596000., 3234000.
    plane = {"type": "local_plane", "e0": e, "n0": n, "z0": 20., "a": .2, "b": -.1}
    thermal = dict(plane, e0=e+1, z0=21.)
    panels = panels_from_model({"p": {"visible_plane": plane, "thermal_plane": thermal}})
    transform = from_origin(e-10, n+10, 1, 1)
    world = {"panels": [{"panel_id": "p", "array_id": "a", "center_xyz": [e, n, 20.],
                          "corners_xyz": [[e+x,n+y,20.] for x,y in [(-1,-1),(1,-1),(1,1),(-1,1)]]}]}
    field = PlanimetricField.from_tiles([
        {"east": e + dx, "north": n + dy, "cc": 1.0, "dx_m": 1.0, "dy_m": 0.0}
        for dx in (-1.0, 0.0, 1.0) for dy in (-1.0, 0.0, 1.0)])
    rows = panel_points(world, panels, field, transform, transform, 20, 20, np.ones((20,20),np.uint8),1)
    assert rows[0]["thermal_u"] == pytest.approx(11.)
    assert rows[0]["z_thermal"] == pytest.approx(21.)
    grid = grid_mapping(world, panels, field, transform, 20,20,np.full(400,20.),transform,20,20,np.ones((20,20),np.uint8),1,1)
    sel=grid["panel_id"]=="p"
    np.testing.assert_allclose(grid["z_thermal"][sel], 21.+.2*(grid["east_t"][sel]-e-1)-.1*(grid["north_t"][sel]-n))

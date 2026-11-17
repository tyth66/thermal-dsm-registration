import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from registration_algorithms.local_plane import LocalPlane
from registration_algorithms.panel_thermal_model import fit_plane_samples
from registration_algorithms.residual_model import PlanimetricField
from registration_algorithms.vertical_estimation import fit_vertical


def test_panel_samples_fit_and_evaluate_in_local_coordinates():
    east = np.array([596000.0, 596002.0, 596004.0, 596006.0] * 2)
    north = np.array([3234000.0] * 4 + [3234002.0] * 4)
    values = 18.0 + 0.003 * (east - 596003.0) - 0.002 * (north - 3234003.0)

    plane, rms = fit_plane_samples(east, north, values)

    assert isinstance(plane, LocalPlane)
    assert plane.evaluate(east, north) == pytest.approx(values, abs=1e-8)
    assert rms == pytest.approx(0.0, abs=1e-8)
    assert plane.to_dict()["type"] == "local_plane"
    assert abs(plane.z0) < 100.0


def test_vertical_fit_keeps_local_plane_evaluation_stable_at_large_coordinates():
    rows = []
    for i in range(8):
        east = 596000.0 + (i % 4) * 2
        north = 3234000.0 + (i // 4) * 3
        rows.append({
            "array_id": "a0", "east": east, "north": north,
            "delta_z_measured_m": 1.5 + 0.01 * (east - 596003.0)
            - 0.02 * (north - 3234001.5),
        })

    fit_vertical(rows)

    expected = [row["delta_z_measured_m"] for row in rows]
    actual = [row["delta_z_fitted_m"] for row in rows]
    assert actual == pytest.approx(expected, abs=1e-8)
    assert rows[0]["vertical_support"] == 8


def test_vertical_fit_uses_constant_local_fallback_for_collinear_support():
    rows = [{"array_id": "a0", "east": 596000.0 + i,
             "north": 3234000.0 + 2 * i, "delta_z_measured_m": float(i)}
            for i in range(8)]

    fit_vertical(rows)

    assert [row["delta_z_fitted_m"] for row in rows] == pytest.approx([3.5] * 8)
    assert all(row["vertical_support"] == 8 for row in rows)


def test_geometry_csv_planes_keep_local_origins(tmp_path, monkeypatch):
    from pathlib import Path

    from registration_algorithms.application import geometry as module

    paths = [tmp_path / "visible.tif", tmp_path / "thermal.tif"]
    for path in paths:
        with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1,
                           dtype="float32", transform=from_origin(596002, 3234002, 1, 1)) as ds:
            ds.write(np.full((2, 2), 20, np.float32), 1)
    east = np.array([596000.0, 596002.0, 596000.0, 596002.0])
    north = np.array([3234000.0, 3234000.0, 3234002.0, 3234002.0])

    def samples(ds, *_args):
        z0 = 20.0 if Path(ds.name) == paths[0] else 21.0
        return east, north, z0 + 0.01 * (east - 596001.0)

    monkeypatch.setattr(module, "_samples", samples)
    field = PlanimetricField.from_tiles([], max_distance_m=10.0)
    panel = {"panel_id": "p0", "array_id": "a0",
             "center_xyz": [596001.0, 3234001.0, 20.0],
             "corners_xyz": [[596000.0, 3234000.0, 20.0],
                             [596002.0, 3234000.0, 20.0],
                             [596002.0, 3234002.0, 20.0],
                             [596000.0, 3234002.0, 20.0]]}
    component = {"p0": {"delta_e_thermal_vs_visible_m": 0.0,
                         "delta_n_thermal_vs_visible_m": 0.0}}
    rows, _ = module.panel_layer(paths[0], paths[1], {"panels": [panel]},
                                 component, 0.1, field)

    row = rows[0]
    assert row["visible_plane_e0"] == pytest.approx(596001.0)
    assert row["visible_plane_n0"] == pytest.approx(3234001.0)
    assert row["visible_plane_z0"] == pytest.approx(20.0)
    assert row["thermal_plane_z0"] == pytest.approx(21.0)
    assert "visible_plane_c" not in row and "thermal_plane_c" not in row


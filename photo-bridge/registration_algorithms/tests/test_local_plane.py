"""Numerically stable local height plane primitives."""
from __future__ import annotations

import numpy as np
import pytest

from registration_algorithms.local_plane import LocalPlane, fit_local_plane, plane_from_row
from registration_algorithms.array_surface import fit_array_plane


def test_local_plane_evaluate_serialize_and_shift():
    plane = LocalPlane(596_000.0, 3_234_000.0, 42.0, 0.12, -0.07)
    east = np.array([596_000.0, 596_001.5])
    north = np.array([3_234_000.0, 3_233_998.0])
    assert np.allclose(plane.evaluate(east, north), [42.0, 42.32])
    assert plane.evaluate(east[0], north[0]) == 42.0
    assert LocalPlane.from_model(plane.to_dict()) == plane
    for invalid in (float("nan"), float("inf"), -float("inf")):
        bad = plane.to_dict()
        bad["a"] = invalid
        with pytest.raises(ValueError, match="finite"):
            LocalPlane.from_model(bad)
    assert plane.shifted(2.5).evaluate(east, north) == pytest.approx(
        plane.evaluate(east, north) + 2.5)


def test_fit_local_plane_is_stable_under_large_translation():
    east_offset = np.array([-1.1, -0.7, -0.2, 0.3, 0.8, 1.2, -0.9, 0.6, 1.1, 0.1])
    north_offset = np.array([-0.5, 0.8, -0.9, 0.3, 1.0, -0.2, 0.4, -1.1, 0.6, -0.4])
    values = 31.0 + 0.18 * east_offset - 0.11 * north_offset
    p1 = fit_local_plane(east_offset, north_offset, values)
    p2 = fit_local_plane(east_offset + 596_000.0,
                         north_offset + 3_234_000.0, values)
    assert p2.a == pytest.approx(0.18, abs=1e-9)
    assert p2.b == pytest.approx(-0.11, abs=1e-9)
    assert p2.a == pytest.approx(p1.a, abs=1e-9)
    assert p2.b == pytest.approx(p1.b, abs=1e-9)
    assert p2.evaluate(596_000.0 + east_offset, 3_234_000.0 + north_offset) == pytest.approx(values)


def test_fit_local_plane_uses_mad_clipping_and_requires_rank_three():
    east = np.array([0, 1, 0, 1, 2, 2, 3, 3], dtype=float) + 596_000
    north = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=float) + 3_234_000
    values = 20 + 0.2 * (east - 596_000) - 0.1 * (north - 3_234_000)
    values[-1] += 4
    plane = fit_local_plane(east, north, values)
    assert plane.a == pytest.approx(0.2, abs=1e-8)
    assert plane.b == pytest.approx(-0.1, abs=1e-8)
    with pytest.raises(ValueError, match="rank"):
        fit_local_plane(np.arange(4.0), np.zeros(4), np.arange(4.0))


def test_plane_from_row_reads_local_and_legacy_csv_fields():
    local = plane_from_row({"visible_plane_e0": "596000", "visible_plane_n0": "3234000",
                            "visible_plane_z0": "25", "visible_plane_a": ".2",
                            "visible_plane_b": "-.1"})
    assert local == LocalPlane(596000, 3234000, 25, .2, -.1)
    legacy = plane_from_row({"visible_plane_a": ".2", "visible_plane_b": "-.1",
                            "visible_plane_c": "-302375"})
    assert legacy == LocalPlane.from_model([.2, -.1, -302375])
    assert plane_from_row({}) is None
    with pytest.raises(ValueError, match="incomplete"):
        plane_from_row({"visible_plane_e0": "596000", "visible_plane_a": ".2"})


def test_array_surface_exports_and_uses_local_plane_at_utm_coordinates():
    east_offsets = np.tile(np.arange(5, dtype=float), 4)
    north_offsets = np.repeat(np.arange(4, dtype=float), 5)
    east = east_offsets + 596_000.0
    north = north_offsets + 3_234_000.0
    z = 26.0 + 0.14 * east_offsets - 0.09 * north_offsets
    surface = fit_array_plane(list(zip(east, north, z)), panel_count=len(z))
    local = LocalPlane.from_model(surface["plane"])
    assert local.a == pytest.approx(0.14, abs=1e-9)
    assert local.b == pytest.approx(-0.09, abs=1e-9)
    assert local.evaluate(east, north) == pytest.approx(z)
    assert surface["plane"]["type"] == "local_plane"

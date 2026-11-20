"""Numerically stable local height plane primitives."""
from __future__ import annotations

import numpy as np
import pytest

from registration_algorithms.local_plane import LocalPlane
from registration_algorithms.array_surface import fit_array_plane


def test_local_plane_evaluate_and_serialize():
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


def test_local_plane_from_legacy_three_coefficient_model():
    plane = LocalPlane.from_model([0.12, -0.07, 42.0])
    assert plane == LocalPlane(0.0, 0.0, 42.0, 0.12, -0.07)
    with pytest.raises(ValueError, match="three"):
        LocalPlane.from_model([1.0, 2.0])


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

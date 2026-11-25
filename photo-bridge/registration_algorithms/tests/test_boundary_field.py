"""Current tangent-local boundary field tests."""
from __future__ import annotations

import numpy as np

from registration_algorithms.boundary_field import evaluate_boundary_field
from registration_algorithms.residual_model import fit_regularized_field


def _field():
    rows = []
    for east in (40.0, 80.0, 120.0, 160.0):
        for north in (40.0, 80.0, 120.0, 160.0, 200.0):
            rows.append({"east": east, "north": north, "dx_m": 0.0,
                         "dy_m": 0.01 * north, "cc": 0.9,
                         "sigma_e": 0.05, "sigma_n": 0.05, "sigma_en": 0.0})
    return fit_regularized_field(rows, tile_m=20.0, max_distance_m=100.0,
                                 kernel_length_m=50.0, ridge=0.1)


class _StubField:
    """Test-local double: flat base field, residuals carried by the tiles."""

    tile_m = 20.0

    def __init__(self, tiles):
        self.tiles = list(tiles)
        self.center = np.zeros(2)
        self.displacement_gradient = np.zeros((2, 2))
        self.translation = np.zeros(2)

    def evaluate(self, east, north):
        size = len(np.asarray(east))
        return (np.zeros(size), np.zeros(size), np.ones(size), np.zeros(size))


def _tile(east, north, residual_e, residual_n, cc=0.9):
    return {"east": float(east), "north": float(north), "cc": cc,
            "residual_e": float(residual_e), "residual_n": float(residual_n)}


def test_boundary_field_applies_at_edge():
    field = _field()
    de, dn, distance, applied = evaluate_boundary_field(
        field, np.array([100.0]), np.array([199.0]),
        (0.0, 200.0, 0.0, 200.0))
    assert applied[0]
    assert distance[0] == 1.0
    assert np.isfinite(de[0]) and np.isfinite(dn[0])


def test_boundary_field_leaves_interior_unchanged():
    field = _field()
    east = np.array([100.0])
    north = np.array([100.0])
    base_e, base_n, _, _ = field.evaluate(east, north)
    de, dn, distance, applied = evaluate_boundary_field(
        field, east, north, (0.0, 200.0, 0.0, 200.0))
    assert not applied[0]
    np.testing.assert_allclose(de, base_e)
    np.testing.assert_allclose(dn, base_n)


def test_boundary_field_accepts_precomputed_base():
    field = _field()
    east = np.array([100.0])
    north = np.array([199.0])
    base_e, base_n, _, _ = field.evaluate(east, north)
    expected = evaluate_boundary_field(
        field, east, north, (0.0, 200.0, 0.0, 200.0))
    original_evaluate = field.evaluate

    def guarded(east_arg, north_arg):
        if len(east_arg) == 1:
            raise AssertionError("query displacement must be reused")
        return original_evaluate(east_arg, north_arg)

    field.evaluate = guarded
    actual = evaluate_boundary_field(
        field, east, north, (0.0, 200.0, 0.0, 200.0),
        base_e=base_e, base_n=base_n)
    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])


def test_boundary_field_zero_at_two_tile_range():
    field = _field()
    de, dn, _, applied = evaluate_boundary_field(
        field, np.array([100.0]), np.array([160.0]),
        (0.0, 200.0, 0.0, 200.0))
    assert not applied[0]


def test_boundary_correction_is_continuous_across_a_bin_seam():
    """Bracketing bins must blend; a nearest-bin switch jumps at the seam."""
    # tangent bin centres sit 40 m apart (tile_m=20); the residual grows along
    # the top edge, so adjacent bins carry different coefficients
    field = _StubField(
        _tile(east, north, 0.0, 0.003 * east)
        for north in (180.0, 185.0, 190.0, 195.0)
        for east in range(30, 170, 20))
    bounds = (0.0, 200.0, 0.0, 200.0)
    query = np.array([39.999, 40.0, 40.001, 59.999, 60.0, 60.001,
                      79.999, 80.0, 80.001])
    north = np.full(len(query), 199.0)  # 1 m inside the top edge
    _, corrected_n, _, applied = evaluate_boundary_field(
        field, query, north, bounds)
    assert applied.all()
    # a 1 mm step across a seam must not move the applied correction
    steps = np.abs(np.diff(corrected_n.reshape(3, 3), axis=1))
    assert steps.max() < 1e-3
    # at the seam midpoint the blend is the mean of the two adjacent bin values
    np.testing.assert_allclose(
        corrected_n[4], 0.5 * (corrected_n[1] + corrected_n[7]), rtol=1e-9)
    assert corrected_n[1] < corrected_n[4] < corrected_n[7]
    # the two bin values really differ, so a nearest-bin switch would jump
    assert corrected_n[7] - corrected_n[1] > 1e-2


def test_boundary_correction_is_continuous_across_a_corner():
    """Two edges must share a corner instead of switching abruptly."""
    bounds = (0.0, 200.0, 0.0, 200.0)
    left_tiles = [_tile(5.0, north, 0.2, 0.0) for north in range(10, 51, 10)]
    bottom_tiles = [_tile(east, 5.0, 0.0, 0.4) for east in range(10, 51, 10)]
    # straddle the left/bottom bisector, 5 m from both edges
    east = 5.0 + np.array([-0.001, 0.0, 0.001])
    north = np.full(3, 5.0)
    both_e, both_n, _, applied = evaluate_boundary_field(
        _StubField(left_tiles + bottom_tiles), east, north, bounds)
    left_e, left_n, _, _ = evaluate_boundary_field(
        _StubField(left_tiles), east, north, bounds)
    bottom_e, bottom_n, _, _ = evaluate_boundary_field(
        _StubField(bottom_tiles), east, north, bounds)
    assert applied.all()
    # on the bisector the blended correction is the mean of the two edges
    middle = 1
    np.testing.assert_allclose(both_e[middle],
                               0.5 * (left_e[middle] + bottom_e[middle]),
                               rtol=1e-9)
    np.testing.assert_allclose(both_n[middle],
                               0.5 * (left_n[middle] + bottom_n[middle]),
                               rtol=1e-9)
    # the two single-edge models differ, so a nearest-edge switch would jump
    assert abs(left_e[middle] - bottom_e[middle]) > 1e-2
    # and a 1 mm step across the bisector keeps the correction continuous
    assert np.abs(np.diff(both_e)).max() < 1e-3
    assert np.abs(np.diff(both_n)).max() < 1e-3

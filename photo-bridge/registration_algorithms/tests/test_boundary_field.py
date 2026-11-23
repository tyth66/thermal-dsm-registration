"""Current tangent-local boundary field tests."""
from __future__ import annotations

import numpy as np

from registration_algorithms.boundary_field import (
    EDGE_CONFIDENCE_MIN, evaluate_boundary_field,
)
from registration_algorithms.residual_model import fit_regularized_field


def _field():
    rows = []
    for east in (40.0, 80.0, 120.0, 160.0):
        for north in (40.0, 80.0, 120.0, 160.0, 200.0):
            rows.append({"east": east, "north": north, "dx_m": 0.0,
                         "dy_m": 0.01 * north, "cc": 0.9})
    return fit_regularized_field(rows, tile_m=20.0, max_distance_m=100.0,
                                 kernel_length_m=50.0, ridge=0.1)


def test_boundary_field_applies_at_edge():
    field = _field()
    de, dn, distance, applied, confidence = evaluate_boundary_field(
        field, np.array([100.0]), np.array([199.0]),
        (0.0, 200.0, 0.0, 200.0))
    assert applied[0]
    assert distance[0] == 1.0
    assert np.isfinite(de[0]) and np.isfinite(dn[0])
    assert 0.0 <= confidence[0] <= 1.0


def test_boundary_field_leaves_interior_unchanged():
    field = _field()
    east = np.array([100.0])
    north = np.array([100.0])
    base_e, base_n, _, _ = field.evaluate(east, north)
    de, dn, distance, applied, confidence = evaluate_boundary_field(
        field, east, north, (0.0, 200.0, 0.0, 200.0))
    assert not applied[0]
    np.testing.assert_allclose(de, base_e)
    np.testing.assert_allclose(dn, base_n)
    assert confidence[0] == 1.0


def test_boundary_field_zero_at_two_tile_range():
    field = _field()
    de, dn, _, applied, _ = evaluate_boundary_field(
        field, np.array([100.0]), np.array([160.0]),
        (0.0, 200.0, 0.0, 200.0))
    assert not applied[0]


def test_edge_confidence_constant_is_available():
    assert EDGE_CONFIDENCE_MIN == 0.65

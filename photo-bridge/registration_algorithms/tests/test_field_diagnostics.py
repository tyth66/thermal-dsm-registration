"""Planimetric field Jacobian and support diagnostics."""
from __future__ import annotations

import numpy as np
import pytest

from registration_algorithms.residual_model import fit_regularized_field


def _synthetic_rows():
    matrix = np.array([[1.001, -0.002], [0.002, 1.001]])
    translation = np.array([0.20, -0.10])
    rows = []
    for east in np.arange(0.0, 41.0, 10.0):
        for north in np.arange(0.0, 41.0, 10.0):
            target = matrix @ np.array([east, north]) + translation
            rows.append({
                "east": east, "north": north,
                "dx_m": target[0] - east, "dy_m": target[1] - north,
                "cc": 0.95, "quality_weight": 1.0,
            })
    return rows, matrix, translation


def test_field_jacobian_and_support_diagnostics():
    rows, _, _ = _synthetic_rows()
    field = fit_regularized_field(rows, tile_m=10.0, max_distance_m=30.0,
                                  kernel_length_m=20.0, ridge=0.1)
    east = np.array([5.0, 15.0, 25.0])
    north = np.array([5.0, 15.0, 25.0])
    jacobian = field.jacobian(east, north)
    assert jacobian["jacobian"].shape == (3, 2, 2)
    assert np.all(np.isfinite(jacobian["det"]))
    assert np.all(jacobian["sigma_min"] > 0.0)
    support = field.support_metrics(east, north)
    assert support["effective_count"].shape == (3,)
    assert np.all(support["effective_count"] >= 1.0)
    assert np.all(support["lambda_min"] >= -1e-12)


def test_analytic_jacobian_matches_finite_difference():
    rows, _, _ = _synthetic_rows()
    field = fit_regularized_field(rows, tile_m=10.0, max_distance_m=30.0,
                                  kernel_length_m=20.0, ridge=0.1)
    point = np.array([[12.0, 17.0]])
    analytic = field.jacobian(point[:, 0], point[:, 1])["jacobian"][0]
    step = 1e-4
    numeric = np.zeros((2, 2))
    for axis in range(2):
        plus = point.copy(); minus = point.copy()
        plus[0, axis] += step; minus[0, axis] -= step
        de_p, dn_p, _, _ = field.evaluate(plus[:, 0], plus[:, 1])
        de_m, dn_m, _, _ = field.evaluate(minus[:, 0], minus[:, 1])
        f_plus = plus[0] + np.array([de_p[0], dn_p[0]])
        f_minus = minus[0] + np.array([de_m[0], dn_m[0]])
        numeric[:, axis] = (f_plus - f_minus) / (2.0 * step)
    assert analytic == pytest.approx(numeric, abs=1e-6)

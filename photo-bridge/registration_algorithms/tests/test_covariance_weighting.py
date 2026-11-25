"""Generalized least-squares RBF weighting with 2x2 ECC covariances."""
from __future__ import annotations

import numpy as np
import pytest

from registration_algorithms.residual_model import (
    _observation_precision, fit_regularized_field,
)


def _rows():
    rows = []
    for index in range(20):
        east = 10.0 * (index % 5)
        north = 10.0 * (index // 5)
        rows.append({
            "east": east, "north": north,
            "dx_m": 0.2 + 0.01 * east - 0.005 * north,
            "dy_m": -0.1 + 0.004 * east + 0.003 * north,
            "cc": 0.9, "quality_weight": 1.0,
            "sigma_e": 0.05, "sigma_n": 0.05, "sigma_en": 0.0,
        })
    return rows


def _fit(rows):
    return fit_regularized_field(rows, tile_m=10.0, max_distance_m=30.0,
                                 kernel_length_m=20.0, ridge=0.1)


def test_isotropic_covariance_fit_is_finite():
    field = _fit(_rows())
    assert np.all(np.isfinite(field.coefficients))
    east = np.asarray([12.0, 30.0])
    north = np.asarray([8.0, 22.0])
    assert np.all(np.isfinite(field.rbf_component(east, north)))


def test_anisotropic_covariance_changes_the_fit_and_precision():
    isotropic = _fit(_rows())
    anisotropic_rows = _rows()
    for index, row in enumerate(anisotropic_rows):
        if index % 2 == 0:
            row["sigma_e"], row["sigma_n"] = 0.01, 0.2
        else:
            row["sigma_e"], row["sigma_n"] = 0.2, 0.01
    anisotropic = _fit(anisotropic_rows)
    east = np.asarray([12.0, 30.0])
    north = np.asarray([8.0, 22.0])
    difference = np.abs(isotropic.rbf_component(east, north)
                         - anisotropic.rbf_component(east, north))
    assert difference.max() > 1e-3
    a, _, d = _observation_precision(anisotropic_rows)
    assert np.max(a / d) > 10.0


def test_non_positive_covariance_is_rejected():
    rows = _rows()
    rows[0]["sigma_e"] = 0.1
    rows[0]["sigma_n"] = 0.1
    rows[0]["sigma_en"] = 0.2
    with pytest.raises(ValueError, match="positive definite"):
        _fit(rows)


def test_missing_covariance_fields_are_rejected():
    rows = [{key: value for key, value in row.items()
             if not key.startswith("sigma_")} for row in _rows()]
    with pytest.raises(ValueError, match="requires sigma_e"):
        _fit(rows)

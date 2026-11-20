"""P1 field diagnostics and covariance-weighted similarity contracts."""
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
                "sigma_e": 0.05, "sigma_n": 0.05, "sigma_en": 0.0,
            })
    # One gross outlier with a large covariance must not move the similarity.
    rows[12] = dict(rows[12], dx_m=rows[12]["dx_m"] + 5.0,
                    sigma_e=5.0, sigma_n=5.0, sigma_en=0.0)
    return rows, matrix, translation


def test_covariance_weighted_similarity_downweights_noisy_outlier():
    rows, matrix, translation = _synthetic_rows()
    field = fit_regularized_field(rows, tile_m=10.0, max_distance_m=30.0,
                                  kernel_length_m=20.0, ridge=0.1,
                                  use_observation_covariance=True)
    assert field.matrix == pytest.approx(matrix, abs=0.01)
    assert field.translation == pytest.approx(translation, abs=0.10)


def test_field_jacobian_support_and_low_frequency_diagnostics():
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
    coupling = field.low_frequency_coupling(east, north)
    assert 0.0 <= coupling["eta"] <= 1.0
    assert 0.0 <= coupling["eta_sim"] <= 1.0


def test_low_frequency_orthogonalization_reduces_coupling():
    rows, _, _ = _synthetic_rows()
    # Add a smooth low-order residual that an unconstrained RBF would absorb.
    for row in rows:
        row["dx_m"] += 0.02 * (row["east"] - 20.0) / 20.0
    query_e = np.linspace(2.0, 38.0, 19)
    query_n = np.linspace(2.0, 38.0, 19)
    free = fit_regularized_field(rows, tile_m=10.0, max_distance_m=30.0,
                                 kernel_length_m=20.0, ridge=0.1)
    constrained = fit_regularized_field(
        rows, tile_m=10.0, max_distance_m=30.0, kernel_length_m=20.0,
        ridge=0.1, orthogonal_low_frequency=True, low_frequency_degree=1)
    free_eta = free.low_frequency_coupling(query_e, query_n)["eta"]
    constrained_eta = constrained.low_frequency_coupling(query_e, query_n)["eta"]
    assert constrained_eta <= free_eta + 1e-9


def test_krr_ridge_runs_and_returns_finite_field():
    rows, _, _ = _synthetic_rows()
    field = fit_regularized_field(rows, tile_m=10.0, max_distance_m=30.0,
                                  kernel_length_m=20.0, ridge=0.1,
                                  kernel_ridge="krr")
    delta_e, delta_n, _, _ = field.evaluate([5.0, 15.0], [5.0, 15.0])
    assert np.all(np.isfinite(delta_e))
    assert np.all(np.isfinite(delta_n))



def test_soft_low_frequency_penalty_reduces_coupling_without_destroying_field():
    rows, _, _ = _synthetic_rows()
    for row in rows:
        row["dx_m"] += 0.02 * (row["east"] - 20.0) / 20.0
    query_e = np.linspace(2.0, 38.0, 19)
    query_n = np.linspace(2.0, 38.0, 19)
    free = fit_regularized_field(rows, tile_m=10.0, max_distance_m=30.0,
                                 kernel_length_m=20.0, ridge=0.1)
    penalized = fit_regularized_field(
        rows, tile_m=10.0, max_distance_m=30.0, kernel_length_m=20.0,
        ridge=0.1, low_frequency_penalty=1.0)
    free_eta = free.low_frequency_coupling(query_e, query_n)["eta"]
    penalized_eta = penalized.low_frequency_coupling(query_e, query_n)["eta"]
    assert penalized_eta <= free_eta + 1e-9
    delta_e, delta_n, _, _ = penalized.evaluate(query_e, query_n)
    assert np.all(np.isfinite(delta_e))
    assert np.all(np.isfinite(delta_n))


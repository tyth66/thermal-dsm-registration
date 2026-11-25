"""Tile-level outlier mass in the joint candidate-field model."""
from __future__ import annotations

import numpy as np
import pytest

from registration_algorithms.research.candidate_field import (
    candidate_log_likelihood, candidate_responsibilities,
    estimate_sigma_f, field_posterior_uncertainty, fit_candidate_field_em,
    predict_displacement_covariance,
)
from registration_algorithms.residual_model import RegularizedPlanimetricField


def _record(dx, dy, cc=0.9, candidates=None):
    return {
        "east": 0.0,
        "north": 0.0,
        "sigma_e": 0.05,
        "sigma_n": 0.05,
        "sigma_en": 0.0,
        "quality_weight": 1.0,
        "candidates": candidates or [{
            "sx": dx / 0.1,
            "sy": -dy / 0.1,
            "cc": cc,
            "covariance_px": np.eye(2).tolist(),
        }],
    }


def _field():
    return RegularizedPlanimetricField.empty(max_distance_m=30.0)


def test_tile_outlier_mass_rejects_all_far_candidates():
    record = _record(2.0, 0.0)
    result = candidate_responsibilities(
        [record], _field(), resolution_m=0.1, sigma_f_m=0.2,
        outlier_prior=0.1, outlier_radius_m=4.0)
    assert result["q0"][0] > 0.5
    assert result["tile_mass"][0] < 0.5


def test_tile_outlier_mass_keeps_near_candidate():
    record = _record(0.01, 0.0)
    result = candidate_responsibilities(
        [record], _field(), resolution_m=0.1, sigma_f_m=0.2,
        outlier_prior=0.1, outlier_radius_m=4.0)
    assert result["q0"][0] < 0.05
    assert result["tile_mass"][0] > 0.95


def test_candidate_weight_is_low_when_alternative_is_near():
    candidates = [
        {"sx": 0.0, "sy": 0.0, "cc": 0.90,
         "covariance_px": np.eye(2).tolist()},
        {"sx": 20.0, "sy": 0.0, "cc": 0.90,
         "covariance_px": np.eye(2).tolist()},
    ]
    record = _record(0.0, 0.0, candidates=candidates)
    result = candidate_responsibilities(
        [record], _field(), resolution_m=0.1, sigma_f_m=0.2,
        outlier_prior=0.1, outlier_radius_m=4.0)
    assert result["q"][0, 0] > 0.99
    assert result["q"][0, 1] < 0.01


def test_larger_background_density_raises_outlier_mass():
    record = _record(0.5, 0.0)
    small = candidate_responsibilities(
        [record], _field(), resolution_m=0.1, sigma_f_m=0.2,
        outlier_prior=0.1, background_density=1.0 / 16.0)
    large = candidate_responsibilities(
        [record], _field(), resolution_m=0.1, sigma_f_m=0.2,
        outlier_prior=0.1, background_density=1.0)
    assert large["q0"][0] > small["q0"][0]
    assert large["tile_mass"][0] < small["tile_mass"][0]


def test_full_density_includes_common_background_factor():
    record = _record(0.0, 0.0)
    field = _field()
    background = 0.25
    result = candidate_log_likelihood(
        [record], field, resolution_m=0.1, sigma_f_m=0.2,
        outlier_prior=0.1, background_density=background)
    covariance = np.eye(2) * (0.1 ** 2) + 0.05 ** 2 * np.eye(2) + 0.2 ** 2 * np.eye(2)
    log_normal = (-np.log(2.0 * np.pi)
                  - 0.5 * np.log(np.linalg.det(covariance)))
    expected = np.log(background) + np.logaddexp(
        np.log(0.1), np.log(0.9) + log_normal - np.log(background))
    assert result["log_likelihood"][0] == pytest.approx(expected)
    assert result["total"] == pytest.approx(expected)


def test_two_layer_em_decomposition():
    candidates = [
        {"sx": 0.0, "sy": 0.0, "cc": 0.90,
         "covariance_px": np.eye(2).tolist()},
        {"sx": 20.0, "sy": 0.0, "cc": 0.90,
         "covariance_px": np.eye(2).tolist()},
    ]
    record = _record(0.0, 0.0, candidates=candidates)
    result = candidate_responsibilities(
        [record], _field(), resolution_m=0.1, sigma_f_m=0.2,
        outlier_prior=0.1, outlier_radius_m=4.0)
    mass = result["tile_mass"][0]
    weights = result["candidate_weights"][0, :2]
    q = result["q"][0, :2]
    assert weights.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(q, mass * weights, atol=1e-12)
    assert result["q0"][0] == pytest.approx(1.0 - mass)
    expected_odds = np.log(0.9 / 0.1) + result["log_match_evidence"][0]
    assert result["log_odds"][0] == pytest.approx(expected_odds)
    assert result["match_evidence"][0] == pytest.approx(
        np.exp(result["log_match_evidence"][0]))


def _weighted_q(records, field, sigma_f_m=0.5):
    result = candidate_responsibilities(
        records, field, resolution_m=0.1, sigma_f_m=sigma_f_m,
        outlier_prior=0.1, outlier_radius_m=4.0)
    q = result["q"]
    positions = result["positions"]
    covariances = result["covariances"]
    counts = result["counts"]
    east = np.array([record["east"] for record in records])
    north = np.array([record["north"] for record in records])
    dx, dy, _, _ = field.evaluate(east, north)
    total = 0.0
    for index in range(len(records)):
        for column in range(int(counts[index])):
            covariance = (covariances[index, column]
                          + sigma_f_m ** 2 * np.eye(2))
            residual = positions[index, column] - np.array([dx[index], dy[index]])
            total += 0.5 * q[index, column] * residual @ np.linalg.inv(covariance) @ residual
    return total + 0.5 * 0.1 * np.sum(field.coefficients ** 2)


def test_exact_m_step_reduces_q_objective():
    records = []
    for east, north, dx in ((0.0, 0.0, 0.10), (30.0, 0.0, 0.12),
                            (0.0, 30.0, 0.08)):
        record = _record(dx, 0.0)
        record["east"] = east
        record["north"] = north
        records.append(record)
    initial = _field()
    updated, _ = fit_candidate_field_em(
        records, resolution_m=0.1, tile_m=30.0, max_distance_m=60.0,
        max_iterations=1)
    before = _weighted_q(records, initial)
    after = _weighted_q(records, updated)
    assert after < before


def test_sigma_mle_respects_bounds_and_mass():
    records = []
    for east, north, dx in ((0.0, 0.0, 0.10), (30.0, 0.0, 0.12),
                            (0.0, 30.0, 0.08)):
        record = _record(dx, 0.0)
        record["east"] = east
        record["north"] = north
        records.append(record)
    field = _field()
    responsibilities = candidate_responsibilities(
        records, field, resolution_m=0.1, sigma_f_m=0.5,
        outlier_prior=0.1, outlier_radius_m=4.0)
    estimate = estimate_sigma_f(
        records, field, responsibilities, sigma_bounds=(0.05, 0.8),
        prior_sigma_m=0.5, min_mass=0.0)
    assert 0.05 <= estimate["sigma_f_m"] <= 0.8
    assert estimate["frozen"] is False
    frozen = estimate_sigma_f(
        records, field, responsibilities, sigma_bounds=(0.05, 0.8),
        prior_sigma_m=0.5, min_mass=1e9)
    assert frozen["frozen"] is True
    assert frozen["sigma_f_m"] == pytest.approx(0.5)


def test_temperature_softens_tile_posterior():
    record = _record(0.01, 0.0)
    cold = candidate_responsibilities(
        [record], _field(), resolution_m=0.1, sigma_f_m=0.2,
        outlier_prior=0.1, outlier_radius_m=4.0, temperature=1.0)
    hot = candidate_responsibilities(
        [record], _field(), resolution_m=0.1, sigma_f_m=0.2,
        outlier_prior=0.1, outlier_radius_m=4.0, temperature=2.0)
    assert hot["q0"][0] > cold["q0"][0]
    assert 0.0 < cold["q0"][0] < hot["q0"][0] < 1.0


def test_periodic_ambiguity_can_make_information_negative():
    candidates = [
        {"sx": -15.0, "sy": 0.0, "cc": 0.9,
         "covariance_px": np.eye(2).tolist()},
        {"sx": 15.0, "sy": 0.0, "cc": 0.9,
         "covariance_px": np.eye(2).tolist()},
    ]
    record = _record(0.0, 0.0, candidates=candidates)
    result = candidate_responsibilities(
        [record], _field(), resolution_m=0.1, sigma_f_m=0.2,
        outlier_prior=1e-9, outlier_radius_m=4.0)
    information = result["information"][0]
    assert np.linalg.eigvalsh(information)[0] < 0.0


def test_posterior_uncertainty_is_positive_at_tile_centers():
    records = []
    for east, north in ((0.0, 0.0), (30.0, 0.0), (0.0, 30.0)):
        record = _record(0.01, 0.0)
        record["east"] = east
        record["north"] = north
        records.append(record)
    field = _field()
    responsibilities = candidate_responsibilities(
        records, field, resolution_m=0.1, sigma_f_m=0.2,
        outlier_prior=0.1, outlier_radius_m=4.0)
    uncertainty = field_posterior_uncertainty(
        records, field, responsibilities, ridge=0.1)
    covariance = uncertainty["covariance"]
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    east = np.array([record["east"] for record in records])
    north = np.array([record["north"] for record in records])
    prediction = predict_displacement_covariance(
        field, uncertainty, east, north, sigma_f_m=0.2)
    for matrix in prediction["displacement_covariance"]:
        assert np.all(np.linalg.eigvalsh(matrix) > 0.0)


def test_entropy_decomposition_separates_validity_and_alias():
    candidates = [
        {"sx": -15.0, "sy": 0.0, "cc": 0.9,
         "covariance_px": np.eye(2).tolist()},
        {"sx": 15.0, "sy": 0.0, "cc": 0.9,
         "covariance_px": np.eye(2).tolist()},
    ]
    record = _record(0.0, 0.0, candidates=candidates)
    result = candidate_responsibilities(
        [record], _field(), resolution_m=0.1, sigma_f_m=0.2,
        outlier_prior=0.1, outlier_radius_m=4.0)
    mass = result["tile_mass"][0]
    expected = (result["validity_entropy"][0]
                + mass * result["alias_entropy"][0])
    assert result["total_entropy"][0] == pytest.approx(expected)
    assert result["alias_entropy"][0] > 0.0
    assert result["validity_entropy"][0] >= 0.0


def test_em_returns_history_and_field():
    records = [_record(0.01, 0.0), _record(0.02, 0.01)]
    field, history = fit_candidate_field_em(
        records, resolution_m=0.1, tile_m=10.0, max_iterations=2,
        outlier_prior=0.1, outlier_radius_m=4.0)
    assert field is not None
    assert len(history) == 2
    assert history[-1]["outlier_tiles"] == 0
    assert history[-1]["temperature"] == 1.0
    assert history[-1]["sigma_estimated_m"] > 0.0

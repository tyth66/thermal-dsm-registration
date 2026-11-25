"""Tests for the similarity + regularized residual planimetric field."""
from __future__ import annotations

import math

import numpy as np

from registration_algorithms.residual_model import (
    PlanimetricField, REGULARIZED_FIELD_TYPE, fit_regularized_field,
)


def _rows():
    rows = []
    angle = math.radians(2.0)
    scale = 1.01
    matrix = np.array([[scale * math.cos(angle), -scale * math.sin(angle)],
                       [scale * math.sin(angle), scale * math.cos(angle)]])
    translation = np.array([0.4, -0.3])
    for north in np.linspace(3234000.0, 3234060.0, 7):
        for east in np.linspace(596000.0, 596080.0, 9):
            source = np.array([east, north])
            target = matrix @ source + translation
            rows.append({
                "east": float(east), "north": float(north), "cc": 0.9,
                "dx_m": float(target[0] - east),
                "dy_m": float(target[1] - north),
                "sigma_e": 0.05, "sigma_n": 0.05, "sigma_en": 0.0,
            })
    return rows, matrix, translation


def test_regularized_field_recovers_similarity_and_roundtrips():
    rows, matrix, translation = _rows()
    field = fit_regularized_field(
        rows, tile_m=10.0, max_distance_m=100.0,
        kernel_length_m=20.0, ridge=1e-6)
    assert field.to_model()["type"] == REGULARIZED_FIELD_TYPE
    restored = PlanimetricField.from_model(field.to_model())
    east = np.asarray([596000.0, 596040.0, 596080.0])
    north = np.asarray([3234000.0, 3234030.0, 3234060.0])
    before = field.evaluate(east, north)
    after = restored.evaluate(east, north)
    np.testing.assert_allclose(before[0], after[0], atol=1e-9)
    np.testing.assert_allclose(before[1], after[1], atol=1e-9)


def test_centered_displacement_parameterization_roundtrips():
    rows, matrix, translation = _rows()
    field = fit_regularized_field(
        rows, tile_m=10.0, max_distance_m=100.0,
        kernel_length_m=20.0, ridge=1e-6)
    center = np.array([np.mean([row["east"] for row in rows]),
                       np.mean([row["north"] for row in rows])])
    np.testing.assert_allclose(field.center, center, atol=1e-9)
    np.testing.assert_allclose(field.displacement_gradient,
                               matrix - np.eye(2), atol=1e-4)
    np.testing.assert_allclose(field.absolute_translation, translation,
                               atol=2e-3)
    np.testing.assert_allclose(field.translation,
                               translation + (matrix - np.eye(2)) @ center,
                               atol=2e-3)
    similarity = field.to_model()["similarity"]
    for key in ("center", "displacement_gradient", "centered_translation"):
        assert key in similarity
    restored = PlanimetricField.from_model(field.to_model())
    np.testing.assert_allclose(restored.center, field.center, atol=1e-9)
    np.testing.assert_allclose(restored.displacement_gradient,
                               field.displacement_gradient, atol=1e-12)


def test_evaluate_with_support_matches_separate_passes():
    rows, _, _ = _rows()
    field = fit_regularized_field(
        rows, tile_m=10.0, max_distance_m=100.0,
        kernel_length_m=20.0, ridge=1e-6)
    east = np.asarray([596000.0, 596040.0, 596080.0])
    north = np.asarray([3234000.0, 3234030.0, 3234060.0])
    dx, dy, weight, nearest = field.evaluate(east, north)
    support = field.support_confidence(east, north)
    fused = field.evaluate_with_support(east, north)
    np.testing.assert_allclose(fused[0], dx)
    np.testing.assert_allclose(fused[1], dy)
    np.testing.assert_allclose(fused[2], weight)
    np.testing.assert_allclose(fused[3], nearest)
    np.testing.assert_allclose(fused[4], support)


def test_parallel_chunk_evaluation_matches_serial():
    rows, _, _ = _rows()
    field = fit_regularized_field(
        rows, tile_m=10.0, max_distance_m=100.0,
        kernel_length_m=20.0, ridge=1e-6)
    east = np.linspace(596000.0, 596080.0, 120_000)
    north = np.linspace(3234000.0, 3234060.0, 120_000)
    serial = field.evaluate_with_support(east, north, workers=1)
    parallel = field.evaluate_with_support(east, north, workers=2)
    for left, right in zip(serial, parallel):
        np.testing.assert_allclose(left, right)


def test_regularized_residual_does_not_extrapolate_to_unbounded_shift():
    rows, _, _ = _rows()
    field = fit_regularized_field(
        rows, tile_m=10.0, max_distance_m=100.0,
        kernel_length_m=15.0, ridge=1e-6)
    # Far outside the tile cloud the smooth residual decays and the global
    # similarity remains as the safe fallback.
    east = np.asarray([597000.0])
    north = np.asarray([3235000.0])
    dx, dy, weight, nearest = field.evaluate(east, north)
    assert np.isfinite(dx[0]) and np.isfinite(dy[0])
    assert weight[0] < 1e-6
    assert nearest[0] > 100.0

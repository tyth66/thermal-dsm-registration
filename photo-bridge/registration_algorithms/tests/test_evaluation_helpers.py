"""Evaluator regression tests for V3-B spatial comparison."""
import numpy as np
import pytest

from registration_algorithms.evaluation import (
    evaluation_feasibility,
    footprint_overlap_matrix,
    paired_supported_errors,
    quadrant_ids,
    training_mask,
    uniqueness_observable,
)


def test_quadrants_are_assigned_from_fixed_bounds():
    points = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    ids = quadrant_ids(points, (0.0, 0.0, 10.0, 10.0))
    assert ids.tolist() == [0, 1, 2, 3]


def test_footprint_overlap_includes_diagonal_overlap():
    left = np.array([[0.0, 0.0], [0.0, 0.0]])
    right = np.array([[9.0, 9.0], [11.0, 11.0]])
    overlap = footprint_overlap_matrix(left, right, tile_m=10.4)
    assert overlap[0, 0]
    assert not overlap[0, 1]


def test_training_mask_removes_heldout_footprint_neighbours():
    train = np.array([[0.0, 0.0], [9.0, 9.0], [100.0, 100.0]])
    held = np.array([[0.0, 0.0]])
    mask = training_mask(np.array([1, 1, 1]), 0, train, held,
                         tile_m=10.4, buffer_m=0.0)
    assert mask.tolist() == [False, False, True]


def test_paired_errors_use_only_common_supported_points():
    result = paired_supported_errors(
        [1.0, 2.0, 3.0], [True, True, False],
        [0.5, 5.0, 0.5], [True, False, True])
    assert result["n"] == 1
    assert result["mean_a"] == pytest.approx(1.0)
    assert result["mean_b"] == pytest.approx(0.5)
    assert result["delta_mean"] == pytest.approx(0.5)


def test_single_candidate_margin_is_not_observable():
    assert uniqueness_observable([1, 2, 3]).tolist() == [False, True, True]


def test_evaluation_feasibility_requires_supported_folds():
    blocked = evaluation_feasibility(
        39, 26, 0, 0, 0, 4, "prior2 accepted", 6.0)
    assert blocked["evaluation_feasible"] is False
    viable = evaluation_feasibility(
        39, 26, 20, 4, 12, 4, "prior2 accepted", 6.0)
    assert viable["evaluation_feasible"] is True


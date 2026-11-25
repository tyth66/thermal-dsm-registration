"""Synthetic capability boundaries for the joint 2-D mode domain."""
from __future__ import annotations

import numpy as np
import pytest

from registration_algorithms.dsm_registration import (
    apply_displacement_domain,
    displacement_domain,
)


def _cluster(dx, dy, count, rng):
    return [{"east": float(i * 10.0), "north": float((i % 7) * 10.0),
             "dx_m": float(dx + rng.normal(0.0, 0.01)),
             "dy_m": float(dy + rng.normal(0.0, 0.01))}
            for i in range(count)]


def _records(clusters, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for dx, dy, count in clusters:
        rows.extend(_cluster(dx, dy, count, rng))
    return rows


def test_mode_selects_majority_cluster_and_rejects_alias():
    rows = _records([(0.2, 0.1, 70), (5.0, 0.1, 30)])
    domain = displacement_domain(rows)
    kept, stats = apply_displacement_domain(rows)
    assert domain["center_dx_m"] == pytest.approx(0.2, abs=0.1)
    assert domain["center_dy_m"] == pytest.approx(0.1, abs=0.1)
    assert domain["confidence"] == pytest.approx(0.995)
    assert len(kept) == 70
    assert stats["rejected"] == 30


def test_majority_alias_is_unresolvable_from_domain_alone():
    rows = _records([(0.2, 0.1, 30), (5.0, 0.1, 70)])
    domain = displacement_domain(rows)
    kept, _ = apply_displacement_domain(rows)
    assert domain["center_dx_m"] == pytest.approx(5.0, abs=0.1)
    assert len(kept) == 70
    assert all(row["dx_m"] > 4.0 for row in kept)


def test_joint_ellipse_rejects_corner_alias_that_axis_box_would_keep():
    # The alias is inside a +/-1.5 m per-axis box but outside the joint
    # 99.5% ellipse, so the 2-D domain is the stricter one here.
    rows = _records([(0.0, 0.0, 70), (1.4, 1.4, 30)])
    domain = displacement_domain(rows)
    kept, stats = apply_displacement_domain(rows)
    assert domain["center_dx_m"] == pytest.approx(0.0, abs=0.1)
    assert all(abs(row["dx_m"]) < 0.1 for row in kept)
    assert stats["rejected"] == 30

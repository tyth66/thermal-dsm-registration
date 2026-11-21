"""Synthetic capability boundaries for MAD, per-axis mode and joint mode."""
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
    domain = displacement_domain(rows, mode="mode")
    kept, stats = apply_displacement_domain(rows, mode="mode")
    assert domain["center_dx"] == pytest.approx(0.2, abs=0.1)
    assert len(kept) == 70
    assert stats["rejected"] == 30


def test_majority_alias_is_unresolvable_from_domain_alone():
    rows = _records([(0.2, 0.1, 30), (5.0, 0.1, 70)])
    domain = displacement_domain(rows, mode="mode")
    kept, _ = apply_displacement_domain(rows, mode="mode")
    assert domain["center_dx"] == pytest.approx(5.0, abs=0.1)
    assert len(kept) == 70
    assert all(row["dx_m"] > 4.0 for row in kept)


def test_joint_mode_selects_a_real_2d_cluster():
    rows = _records([(0.5, -1.0, 50), (2.0, 1.0, 50)])
    axis = displacement_domain(rows, mode="mode")
    joint = displacement_domain(rows, mode="joint")
    kept_joint, _ = apply_displacement_domain(rows, mode="joint")
    # Both estimators select one occupied cluster here; the joint estimator
    # additionally guarantees that the selected dx/dy pair is occupied.
    assert any(abs(axis["center_dx"] - value) < 0.1 for value in (0.5, 2.0))
    assert any(abs(axis["center_dy"] - value) < 0.1 for value in (-1.0, 1.0))
    assert joint["joint"] is True
    assert len(kept_joint) == 50
    assert any(abs(joint["center_dx"] - dx) < 0.1
               and abs(joint["center_dy"] - dy) < 0.1
               for dx, dy in ((0.5, -1.0), (2.0, 1.0)))


def test_mad_domain_is_inflated_by_balanced_alias_cluster():
    rows = _records([(0.2, 0.1, 50), (5.0, 0.1, 50)])
    mad = displacement_domain(rows, mode="mad")
    mode = displacement_domain(rows, mode="mode")
    assert mad["bound_dx"] > mode["bound_dx"]
    assert mad["bound_dx"] > 2.0


def test_domain_can_be_estimated_from_usable_subset():
    rows = _records([(0.2, 0.1, 50), (5.0, 0.1, 50)])
    usable = rows[:50]
    domain = displacement_domain(usable, mode="mode")
    kept, stats = apply_displacement_domain(
        rows, mode="mode", domain_records=usable)
    assert domain["center_dx"] == pytest.approx(0.2, abs=0.1)
    assert len(kept) == 50
    assert stats["rejected"] == 50


import pytest

from registration_algorithms.application import (
    PlanimetricInputs, adaptive_field_support, adaptive_tiling, estimate_planimetric,
)


def test_adaptive_field_support_keeps_baseline_scale():
    """202 m short edge: 0.30 * 202 = 60.6 m, capped at the 30 m ceiling."""
    plan = adaptive_field_support(202.0, 299.1, 30.0)
    assert plan.max_distance_m == pytest.approx(30.0)
    assert plan.short_overlap_m == pytest.approx(202.0)


def test_adaptive_field_support_shrinks_small_station():
    """Below a 100 m short edge the confirmed 0.30 ratio scales support."""
    plan = adaptive_field_support(98.7, 52.2, 30.0)
    assert plan.max_distance_m == pytest.approx(0.30 * 52.2)
    assert plan.max_distance_m < 30.0


def test_adaptive_field_support_caps_very_large_station():
    plan = adaptive_field_support(500.0, 800.0, 30.0)
    assert plan.max_distance_m == pytest.approx(30.0)


def test_adaptive_field_support_adaptation_starts_below_100m():
    """0.30 * short_edge crosses the 30 m cap exactly at a 100 m short edge."""
    at_threshold = adaptive_field_support(100.0, 150.0, 30.0)
    assert at_threshold.max_distance_m == pytest.approx(30.0)
    below_threshold = adaptive_field_support(99.0, 150.0, 30.0)
    assert below_threshold.max_distance_m == pytest.approx(0.30 * 99.0)


def test_adaptive_field_support_has_floor():
    """Very small overlaps are floored at 10 m instead of shrinking further."""
    plan = adaptive_field_support(20.0, 10.0, 30.0)
    assert plan.max_distance_m == pytest.approx(10.0)


def test_adaptive_tiling_uses_one_percent_area():
    plan = adaptive_tiling(98.7, 52.2, 0.1, 40.0, 20.0)
    assert plan.tile_m < 40.0
    assert plan.tile_m * plan.tile_m / (98.7 * 52.2) == pytest.approx(
        0.01, abs=0.0002)
    assert plan.step_m == pytest.approx(plan.tile_m / 2.0)
    assert plan.tile_px >= 8
    assert plan.step_px < plan.tile_px


def test_adaptive_tiling_uses_one_percent_for_large_station():
    plan = adaptive_tiling(202.0, 299.1, 0.1, 40.0, 20.0)
    assert plan.tile_m * plan.tile_m / (202.0 * 299.1) == pytest.approx(
        0.01, abs=0.0002)
    assert plan.step_m == pytest.approx(plan.tile_m / 2.0)
    assert plan.tile_px % 2 == 0
    assert plan.step_px == plan.tile_px // 2


def test_adaptive_tiling_has_pixel_floor():
    plan = adaptive_tiling(2.0, 1.0, 0.1, 40.0, 20.0)
    assert plan.tile_px == 8
    assert plan.step_px >= 1
    assert plan.tile_m == pytest.approx(0.8)
    with pytest.raises(ValueError):
        adaptive_tiling(0.0, 1.0, 0.1, 40.0, 20.0)


def test_estimate_planimetric_rejects_non_positive_scale_parameters(tmp_path):
    """F10: validation runs before any DSM work and names the bad parameter."""
    base = dict(visible_dsm=tmp_path / "visible.tif",
                thermal_dsm=tmp_path / "thermal.tif",
                visible_world={"panels": []})
    for name in ("resolution_m", "tile_m", "step_m",
                 "field_max_distance_m"):
        with pytest.raises(ValueError, match=name):
            estimate_planimetric(PlanimetricInputs(**base, **{name: 0.0}))
        with pytest.raises(ValueError, match=name):
            estimate_planimetric(PlanimetricInputs(**base, **{name: -1.0}))
    with pytest.raises(ValueError, match="resolution_m"):
        estimate_planimetric(PlanimetricInputs(**base,
                                               resolution_m=float("nan")))


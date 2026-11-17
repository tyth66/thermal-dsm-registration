import pytest

from registration_algorithms.application import adaptive_field_support, adaptive_tiling


def test_adaptive_field_support_keeps_baseline_scale():
    plan = adaptive_field_support(202.0, 299.1, 30.0)
    assert plan.max_distance_m == pytest.approx(30.0)
    assert plan.short_overlap_m == pytest.approx(202.0)
    assert plan.adaptive is True


def test_adaptive_field_support_shrinks_small_station():
    plan = adaptive_field_support(98.7, 52.2, 30.0)
    assert plan.max_distance_m == pytest.approx(0.30 * 52.2)
    assert plan.max_distance_m < 30.0


def test_adaptive_field_support_caps_very_large_station():
    plan = adaptive_field_support(500.0, 800.0, 30.0)
    assert plan.max_distance_m == pytest.approx(30.0)


def test_adaptive_field_support_has_floor():
    plan = adaptive_field_support(20.0, 10.0, 30.0)
    assert plan.max_distance_m == pytest.approx(10.0)


def test_fixed_field_support_uses_requested_value():
    plan = adaptive_field_support(98.7, 52.2, 25.0, adaptive=False)
    assert plan.max_distance_m == pytest.approx(25.0)
    assert plan.adaptive is False


def test_adaptive_tiling_caps_small_station_to_overlap():
    plan = adaptive_tiling(98.7, 52.2, 0.1, 40.0, 20.0)
    assert plan.tile_m < 40.0
    assert plan.tile_m <= 52.2 / 2.0 + 1e-9
    assert plan.tile_m / 52.2 == pytest.approx(0.199, abs=0.002)
    assert plan.tile_m == pytest.approx(10.4)
    assert plan.step_m == pytest.approx(5.2)
    assert plan.tile_px >= 8
    assert plan.step_px < plan.tile_px


def test_adaptive_tiling_keeps_requested_for_large_station():
    plan = adaptive_tiling(202.0, 299.1, 0.1, 40.0, 20.0)
    assert plan.tile_m == pytest.approx(40.0)
    assert plan.step_m == pytest.approx(20.0)
    assert plan.tile_px == 400
    assert plan.step_px == 200


def test_adaptive_tiling_has_pixel_floor():
    plan = adaptive_tiling(2.0, 1.0, 0.1, 40.0, 20.0)
    assert plan.tile_px == 8
    assert plan.step_px >= 1
    assert plan.tile_m == pytest.approx(0.8)
    with pytest.raises(ValueError):
        adaptive_tiling(0.0, 1.0, 0.1, 40.0, 20.0)

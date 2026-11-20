
from registration_algorithms.application import (
    MappingComputation, mapping_result_payload, nested_panel_points)


def _point(panel_id, name, valid):
    return {"panel_id": panel_id, "point": name, "east_thermal": 1.0,
            "north_thermal": 2.0, "thermal_u": 3.0, "thermal_v": 4.0,
            "valid": valid}


def test_mapping_result_payload_groups_panel_points():
    rows = [_point("p0", name, 1)
            for name in ("center", "corner_0", "corner_1", "corner_2", "corner_3")]
    grouped = nested_panel_points(rows)
    assert len(grouped) == 1
    assert grouped[0]["panel_id"] == "p0"
    assert grouped[0]["center"] is not None
    assert len(grouped[0]["corners"]) == 4
    payload = mapping_result_payload(MappingComputation(
        panel_rows=tuple(rows), grid={}, summary={"method": "test"},
        thermal_resolution=1.0))
    assert payload["schema"] == "visible_to_thermal_mapping_result/v3"
    assert payload["panel_count"] == 1
    for removed in ("z_visible", "z_thermal", "delta_z", "vertical_source"):
        assert removed not in payload["panels"][0]["center"]
    assert "3D residual" not in payload["method"]


def test_mapping_result_payload_excludes_all_original_nodata_panels():
    rows = []
    for panel_id, valid in (("p0", 1), ("p1", 0)):
        rows.extend(_point(panel_id, name, valid)
                    for name in ("center", "corner_0", "corner_1",
                                 "corner_2", "corner_3"))
    payload = mapping_result_payload(MappingComputation(
        panel_rows=tuple(rows), grid={}, summary={"method": "test"},
        thermal_resolution=1.0))
    assert payload["panel_count"] == 1
    assert [panel["panel_id"] for panel in payload["panels"]] == ["p0"]
    assert payload["summary"]["panel_ids_excluded_original_nodata"] == 1
    assert "all nodata are excluded" in payload["summary"]["mapping_policy"]

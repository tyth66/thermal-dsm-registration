
from registration_algorithms.application import (
    MappingComputation, mapping_result_payload, nested_panel_points)


def test_mapping_result_payload_groups_panel_points():
    rows = []
    for name in ("center", "corner_0", "corner_1", "corner_2", "corner_3"):
        rows.append({"panel_id": "p0", "point": name, "east_thermal": 1.0,
                     "north_thermal": 2.0, "z_thermal": 3.0, "valid": 1})
    grouped = nested_panel_points(rows)
    assert len(grouped) == 1
    assert grouped[0]["panel_id"] == "p0"
    assert grouped[0]["center"] is not None
    assert len(grouped[0]["corners"]) == 4
    payload = mapping_result_payload(MappingComputation(
        panel_rows=tuple(rows), grid={}, summary={"method": "test"},
        thermal_resolution=1.0))
    assert payload["schema"].endswith("/v2")
    assert payload["panel_count"] == 1

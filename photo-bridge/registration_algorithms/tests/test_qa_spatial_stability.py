"""Pure-computation tests for tools/qa_spatial_stability.py (no DSM needed)."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "qa_spatial_stability.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("qa_spatial_stability",
                                                  TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["qa_spatial_stability"] = module
    spec.loader.exec_module(module)
    return module


qa = _load_tool()

# production-scale parameters used by the synthetic fixtures
TILE_M = 40.0
MAX_DISTANCE_M = 30.0


def _grid_tiles(east_values, north_values):
    tiles = []
    for row, east in enumerate(east_values):
        for column, north in enumerate(north_values):
            tiles.append({"row": row, "column": column, "east": float(east),
                          "north": float(north), "dx_m": 0.0, "dy_m": 0.0,
                          "cc": 0.9})
    return tiles


def _synthetic_tiles(size=6, spacing=20.0):
    east = [index * spacing for index in range(size)]
    north = [index * spacing for index in range(size)]
    tiles = _grid_tiles(east, north)
    for tile in tiles:
        tile["dx_m"] = 0.2 + 0.001 * tile["east"]
        tile["dy_m"] = -0.1 + 0.0005 * tile["north"]
    return tiles


def _field_rows(tiles):
    return [{"east": tile["east"], "north": tile["north"],
             "dx_m": tile["dx_m"], "dy_m": tile["dy_m"],
             "cc": max(0.05, tile["cc"])} for tile in tiles]


def _panel_points(east, north):
    return np.asarray([[east, north], [east - 0.5, north - 0.5],
                       [east + 0.5, north - 0.5], [east - 0.5, north + 0.5],
                       [east + 0.5, north + 0.5]], float)


def _rotated_grid(rows=4, columns=6, spacing=20.0, angle_deg=-33.0):
    angle = np.radians(angle_deg)
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
    grid = np.asarray([[column * spacing, row * spacing]
                       for row in range(rows) for column in range(columns)], float)
    return grid @ rotation.T


def _block_counts(labels, blocks):
    return [int(np.sum(np.asarray(labels) == ("B%d" % index)))
            for index in range(blocks)]


def test_pca_axis_split_rotated_array_is_balanced():
    """F17: a -33 deg array splits along its own long axis, not EN quadrants."""
    centers = _rotated_grid(rows=4, columns=6, spacing=20.0, angle_deg=-33.0)
    scheme, assign = qa.partition_scheme(centers, blocks=4)
    assert scheme["method"] == "pca_axis"
    assert scheme["blocks"] == 4
    direction = np.asarray(scheme["axis_direction"], float)
    expected = np.asarray([np.cos(np.radians(-33.0)),
                           np.sin(np.radians(-33.0))])
    assert abs(float(direction @ expected)) == pytest.approx(1.0, abs=1e-9)
    assert scheme["axis_angle_deg"] == pytest.approx(-33.0, abs=1e-9)
    labels = assign(centers)
    assert _block_counts(labels, 4) == [6, 6, 6, 6]
    assert scheme["realized"] == {"B0": 6, "B1": 6, "B2": 6, "B3": 6}
    # Panels are assigned with the same axis and slice boundaries: the centroid
    # of every tile slice falls back into its own block.
    projection = (centers - np.asarray(scheme["center"])) @ direction
    order = np.argsort(projection)
    for index, group in enumerate(np.array_split(order, 4)):
        centroid = centers[group].mean(axis=0, keepdims=True)
        assert assign(centroid)[0] == "B%d" % index


def test_pca_axis_split_blocks_two_is_single_cut():
    centers = _rotated_grid(rows=2, columns=8, spacing=10.0, angle_deg=25.0)
    scheme, assign = qa.partition_scheme(centers, blocks=2)
    assert scheme["method"] == "pca_axis"
    assert len(scheme["edges"]) == 1
    assert _block_counts(assign(centers), 2) == [8, 8]
    assert scheme["realized"] == {"B0": 8, "B1": 8}


def test_partition_falls_back_to_en_split_when_pca_degenerate():
    centers = np.tile(np.asarray([[100.0, 200.0]]), (5, 1))
    scheme, assign = qa.partition_scheme(centers, blocks=4)
    assert scheme["method"] == "median_quadrants"
    assert scheme["realized"] == {"NE": 5}
    assert set(assign(centers)) == {"NE"}
    quantile_scheme, _ = qa.partition_scheme(centers, blocks=3)
    assert quantile_scheme["method"] == "axis_quantiles"


def test_partition_rejects_invalid_input():
    with pytest.raises(ValueError):
        qa.partition_scheme(np.asarray([[0.0, 0.0]]), blocks=1)
    with pytest.raises(ValueError):
        qa.partition_scheme(np.asarray([]), blocks=4)


def test_rigid_delta_is_five_point_mean():
    tiles = _synthetic_tiles()
    field = qa.fit_field(_field_rows(tiles), TILE_M, MAX_DISTANCE_M)
    points = _panel_points(tiles[7]["east"], tiles[7]["north"])
    delta = qa.rigid_delta(field, points)
    dx, dy, _, _ = field.evaluate(points[:, 0], points[:, 1])
    assert delta[0] == pytest.approx(float(np.mean(dx)), abs=1e-12)
    assert delta[1] == pytest.approx(float(np.mean(dy)), abs=1e-12)


def test_panel_sensitivity_matches_manual_rigid_computation():
    tiles = _synthetic_tiles()
    centers = np.asarray([[tile["east"], tile["north"]] for tile in tiles])
    scheme, assign = qa.partition_scheme(centers, blocks=4)
    labels = assign(centers)
    field_all = qa.fit_field(_field_rows(tiles), TILE_M, MAX_DISTANCE_M)
    folds, fold_fields = qa.fit_fold_fields(tiles, labels, labels, field_all,
                                            TILE_M, MAX_DISTANCE_M)
    assert not any(fold["skipped"] for fold in folds)
    assert set(fold_fields) == {"B0", "B1", "B2", "B3"}
    for fold in folds:
        assert fold["holdout_ratio"] == pytest.approx(
            fold["held_tiles"] / len(tiles))

    random = np.random.default_rng(7)
    points = [_panel_points(20.0 * random.uniform(0, 5),
                            20.0 * random.uniform(0, 5)) for _ in range(20)]
    panel_centers = np.asarray([item.mean(axis=0) for item in points])
    values, panel_labels, skipped = qa.panel_sensitivity(
        field_all, fold_fields, points, panel_centers, assign)
    assert skipped == []
    assert len(values) == len(points)
    assert np.all(np.isfinite(values))
    index = 3
    expected = float(np.linalg.norm(
        qa.rigid_delta(fold_fields[panel_labels[index]], points[index])
        - qa.rigid_delta(field_all, points[index])))
    assert values[index] == pytest.approx(expected, abs=1e-12)


def test_fold_skips_when_training_support_too_small():
    tiles = _grid_tiles([0.0, 20.0, 40.0, 0.0, 20.0, 40.0, 0.0, 20.0],
                        [0.0, 0.0, 0.0, 20.0, 20.0, 20.0, 40.0, 40.0])
    labels = np.asarray(["A"] * 7 + ["B"], dtype=object)
    field_all = qa.fit_field(_field_rows(tiles), TILE_M, MAX_DISTANCE_M)
    folds, fold_fields = qa.fit_fold_fields(tiles, labels, labels, field_all,
                                            TILE_M, MAX_DISTANCE_M)
    assert fold_fields == {}
    assert len(folds) == 2
    for fold in folds:
        assert fold["skipped"] is True
        assert "train tiles" in fold["skip_reason"]
        assert fold["holdout_ratio"] == pytest.approx(
            fold["held_tiles"] / len(tiles))

    healthy = _synthetic_tiles()
    healthy_centers = np.asarray([[tile["east"], tile["north"]]
                                  for tile in healthy])
    _, assign = qa.partition_scheme(healthy_centers, blocks=4)
    healthy_labels = assign(healthy_centers)
    folds, fold_fields = qa.fit_fold_fields(
        healthy, healthy_labels, healthy_labels,
        qa.fit_field(_field_rows(healthy), TILE_M, MAX_DISTANCE_M),
        TILE_M, MAX_DISTANCE_M)
    assert not any(fold["skipped"] for fold in folds)
    assert len(fold_fields) == 4


def test_high_sensitivity_clusters_reports_connected_components():
    centers = np.asarray([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5],
                          [10.0, 10.0], [10.5, 10.0], [5.0, 5.0]])
    values = np.asarray([0.9, 0.8, 0.85, 0.9, 0.8, 0.1])
    result = qa.high_sensitivity_clusters(centers, values, threshold_m=0.5,
                                          eps_m=3.0)
    assert result["panels"] == 5
    assert result["cluster_count"] == 2
    assert result["max_cluster_size"] == 3
    assert result["clusters"][0]["panels"] == 3

    empty = qa.high_sensitivity_clusters(centers, values, threshold_m=10.0)
    assert empty["panels"] == 0
    assert empty["cluster_count"] == 0
    assert empty["max_cluster_size"] == 0


def test_summarize_ignores_missing_values():
    summary = qa.summarize([1.0, None, 2.0, 3.0, float("nan")])
    assert summary["n"] == 3
    assert summary["p50"] == pytest.approx(2.0)
    assert summary["max"] == pytest.approx(3.0)
    assert qa.summarize([]) == {"n": 0}


def test_spatial_support_nearest_distances_and_coverage():
    tiles = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    panels = np.asarray([[1.0, 0.0], [5.0, 0.0]])
    support = qa.spatial_support(tiles, panels)
    assert support["tiles"]["count"] == 3
    assert support["panels"]["nearest_tile_distance_m"]["p50"] == pytest.approx(3.0)
    assert support["panels"]["nearest_tile_distance_m"]["max"] == pytest.approx(5.0)
    assert support["tiles"]["nearest_tile_distance_m"]["max"] == pytest.approx(10.0)
    balance = qa.coverage_balance([{"tiles": 3, "panels": 10},
                                   {"tiles": 5, "panels": 20}])
    assert balance["tile_count_max_over_min"] == pytest.approx(5.0 / 3.0)
    assert balance["panel_count_max_over_min"] == pytest.approx(2.0)


def test_fit_field_uses_the_passed_production_scale():
    """Kernel length follows the production tiling, not a fixed constant."""
    tiles = _synthetic_tiles()
    rows = _field_rows(tiles)
    coarse = qa.fit_field(rows, 40.0, MAX_DISTANCE_M)
    fine = qa.fit_field(rows, 10.4, MAX_DISTANCE_M)
    # auto kernel = max(2.5 * tile_m, 4 * median tile spacing); spacing is 20 m
    assert coarse.kernel_length_m == pytest.approx(100.0)
    assert fine.kernel_length_m == pytest.approx(80.0)
    assert coarse.kernel_length_m != fine.kernel_length_m


def test_estimate_panel_pitch_from_panel_lattice():
    centers = np.asarray([[float(column), float(row)]
                          for row in range(4) for column in range(4)]) * 2.0
    assert qa.estimate_panel_pitch(centers) == pytest.approx(2.0)
    assert qa.estimate_panel_pitch(np.asarray([[0.0, 0.0]])) is None


def _estimation(tiles):
    field = qa.fit_field(_field_rows(tiles), TILE_M, MAX_DISTANCE_M)
    return SimpleNamespace(
        accepted=tiles, records=tiles, field=field,
        tiling=SimpleNamespace(tile_m=TILE_M, step_m=20.0, tile_px=400,
                               step_px=200),
        field_support=SimpleNamespace(max_distance_m=MAX_DISTANCE_M))


def _payload_args(result_json="result.json"):
    return SimpleNamespace(blocks=4, visible_dsm="visible_dsm.tif",
                           thermal_dsm="thermal_dsm.tif",
                           result_json=result_json)


def test_empty_panel_payload_prints_summary_without_key_error(capsys):
    """F1: a panel-free payload must not raise KeyError in print_summary."""
    tiles = _synthetic_tiles()
    payload = qa.build_payload(_payload_args(), _estimation(tiles),
                               {"panels": []})
    assert payload["component_sensitivity"]["station"] == {"n": 0}
    out_path = Path("empty_qa.json")
    qa.print_summary(payload, out_path)
    printed = capsys.readouterr().out
    assert "station=empty" in printed
    assert "P50=n/a" in printed
    assert "pitch=n/a" in printed
    assert "U P50=n/a" in printed
    assert "partition: method=pca_axis" in printed
    assert "U_k depends on holdout_ratio" in printed


def test_payload_reports_pca_partition_and_holdout_ratio():
    """F17: the payload exposes the split method, realized blocks and ratios."""
    tiles = _synthetic_tiles()
    payload = qa.build_payload(_payload_args(), _estimation(tiles),
                               {"panels": []})
    partition = payload["partition"]
    assert partition["method"] == "pca_axis"
    assert partition["blocks"] == 4
    assert sum(partition["realized"].values()) == len(tiles)
    assert payload["component_sensitivity"]["note"] == (
        "U_k depends on holdout_ratio; compare blocks with similar held/total "
        "ratios")
    blocks = payload["spatial_support"]["blocks"]
    assert [block["block"] for block in blocks] == ["B0", "B1", "B2", "B3"]
    for block in blocks:
        assert block["holdout_ratio"] == pytest.approx(
            block["tiles"] / len(tiles))
    for fold in payload["global_parameter_stability"]["folds"]:
        assert fold["holdout_ratio"] == pytest.approx(
            fold["held_tiles"] / len(tiles))


def test_station_blocks_report_panel_only_block():
    """F2: a block with panels but no tiles stays in the block table."""
    tiles = _grid_tiles([0.0, 20.0, 40.0], [0.0, 20.0, 40.0])
    labels = np.asarray(["NE"] * 5 + ["NW"] * 4, dtype=object)
    panels = np.asarray([[0.0, 0.0], [10.0, 0.0]], float)
    blocks = qa.station_blocks(tiles, labels, panels,
                               np.asarray(["SW", "SW"], dtype=object))
    assert [block["block"] for block in blocks] == ["NE", "NW", "SW"]
    empty = blocks[2]
    assert empty["tiles"] == 0
    assert empty["panels"] == 2
    assert empty["skipped"] is True
    assert empty["skip_reason"] == "empty held-out block"
    assert empty["tile_bbox_east"] is None
    assert empty["panel_bbox_east"] == [0.0, 10.0]
    assert empty["holdout_ratio"] == 0.0
    assert blocks[0]["holdout_ratio"] == pytest.approx(5.0 / 9.0)
    assert blocks[1]["holdout_ratio"] == pytest.approx(4.0 / 9.0)
    balance = qa.coverage_balance(blocks)
    assert balance["tile_count_min"] == 0
    assert balance["zero_tile_blocks"] == ["SW"]
    assert balance["tile_count_max_over_min"] == pytest.approx(5.0 / 4.0)
    assert qa.coverage_balance([{"tiles": 3, "panels": 1}])["zero_tile_blocks"] == []


def test_fold_reports_panel_only_block_as_empty_held_out():
    """F2: the empty-held-out branch is reachable through a panel-only block."""
    tiles = _synthetic_tiles()
    centers = np.asarray([[tile["east"], tile["north"]] for tile in tiles])
    _, assign = qa.partition_scheme(centers, blocks=4)
    labels = assign(centers)
    empty_block = str(labels[0])
    keep = labels != empty_block
    kept_tiles = [tile for tile, flag in zip(tiles, keep) if flag]
    kept_labels = labels[keep]
    field_all = qa.fit_field(_field_rows(kept_tiles), TILE_M, MAX_DISTANCE_M)
    folds, fold_fields = qa.fit_fold_fields(
        kept_tiles, kept_labels, np.asarray([empty_block], dtype=object),
        field_all, TILE_M, MAX_DISTANCE_M)
    assert len(folds) == 4
    empty = next(fold for fold in folds if fold["block"] == empty_block)
    assert empty["skipped"] is True
    assert empty["skip_reason"] == "empty held-out block"
    assert empty["held_tiles"] == 0
    assert empty["holdout_ratio"] == 0.0
    assert empty_block not in fold_fields


def test_station_name_from_argument_or_result_parent(tmp_path):
    """F3: --station-name wins; otherwise the result parent directory name."""
    result_json = tmp_path / "run290" / "visible_to_thermal_mapping_result.json"
    explicit = SimpleNamespace(station_name="290", result_json=result_json)
    assert qa._station_name(explicit) == "290"
    inferred = SimpleNamespace(station_name=None, result_json=result_json)
    assert qa._station_name(inferred) == "run290"


def test_run_names_output_and_refuses_overwrite(tmp_path, monkeypatch):
    """F3: output is <station_name>_qa.json and existing files are protected."""
    result_dir = tmp_path / "run290"
    result_dir.mkdir()
    result_json = result_dir / "visible_to_thermal_mapping_result.json"
    result_json.write_text('{"panels": []}', encoding="utf-8")
    tiles = _synthetic_tiles()
    estimation = _estimation(tiles)
    monkeypatch.setattr(qa, "estimate_planimetric", lambda inputs: estimation)
    out_dir = tmp_path / "qa"
    args = SimpleNamespace(
        visible_dsm=tmp_path / "visible_dsm.tif",
        thermal_dsm=tmp_path / "thermal_dsm.tif",
        result_json=result_json, blocks=4, out_dir=out_dir,
        station_name=None, force=False)
    out_path = qa.run(args)
    assert out_path == out_dir / "run290_qa.json"
    assert out_path.exists()
    with pytest.raises(FileExistsError):
        qa.run(args)
    forced = SimpleNamespace(**{**vars(args), "force": True})
    assert qa.run(forced) == out_path


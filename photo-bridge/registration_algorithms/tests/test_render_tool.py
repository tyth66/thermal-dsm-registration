"""Pure/CLI tests for tools/render_full_array_mapping.py (no real data needed)."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "render_full_array_mapping.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("render_full_array_mapping",
                                                  TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["render_full_array_mapping"] = module
    spec.loader.exec_module(module)
    return module


render = _load_tool()


def _corner(east, north):
    return {"thermal_u": east, "thermal_v": north,
            "visible_u": east / 2.0, "visible_v": north / 2.0}


def _write_raster(path, width, height, resolution):
    transform = from_origin(500000.0, 3000000.0, resolution, resolution)
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width,
                       count=4, dtype="uint8", crs="EPSG:32650",
                       transform=transform) as dataset:
        dataset.write(np.full((4, height, width), 128, dtype=np.uint8))


def _write_payloads(tmp_path):
    world = tmp_path / "world_model.json"
    world.write_text(json.dumps({"panels": [{"panel_id": "p0",
                                             "array_id": "1"}]}),
                     encoding="utf-8")
    points = [{"thermal_u": 8.0, "thermal_v": 6.0,
               "visible_u": 4.0, "visible_v": 3.0, "corrected_valid": 1,
               "field_valid": 1, "valid": 1}]
    points += [_corner(east, north) for east, north in
               ((4.0, 4.0), (12.0, 4.0), (12.0, 8.0), (4.0, 8.0))]
    points = [{**point, "corrected_valid": 1, "field_valid": 1, "valid": 1}
              for point in points]
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"panels": [{"panel_id": "p0",
                                              "center": points[0],
                                              "corners": points[1:]}]}),
                      encoding="utf-8")
    return world, result


def test_panel_bbox_skips_non_finite_and_clamps_to_image():
    """F4: NaN/inf corners are ignored and the bbox stays inside the image."""
    normal = {"corners": [_corner(10.0, 10.0), _corner(20.0, 10.0),
                          _corner(20.0, 20.0), _corner(10.0, 20.0)]}
    outside = {"corners": [_corner(500.0, 500.0), _corner(510.0, 500.0),
                           _corner(510.0, 510.0), _corner(500.0, 510.0)]}
    assert render._panel_bbox([normal], "thermal", 1.0, 100, 80) == \
        (9, 9, 12, 12)
    # A far outside corner clamps the top-left to the last valid pixel.
    assert render._panel_bbox([outside], "thermal", 1.0, 100, 80) == \
        (99, 79, 1, 1)
    # Mixed input still yields a valid, in-bounds bbox instead of crashing.
    x, y, width, height = render._panel_bbox([normal, outside], "thermal",
                                             1.0, 100, 80)
    assert 0 <= x < 100 and 0 <= y < 80
    assert 1 <= width <= 100 - x and 1 <= height <= 80 - y
    nan_corners = {"corners": [{"thermal_u": float("nan"),
                                "thermal_v": float("nan")}]}
    inf_corners = {"corners": [{"thermal_u": float("inf"),
                                "thermal_v": float("-inf")}]}
    assert render._panel_bbox([nan_corners, inf_corners], "thermal", 1.0,
                              100, 80) == (0, 0, 100, 80)
    negative = {"corners": [{"thermal_u": -5.0, "thermal_v": -5.0}]}
    assert render._panel_bbox([negative], "thermal", 1.0, 100, 80) == \
        (0, 0, 1, 1)


def test_raster_metadata_reads_resolution_and_size(tmp_path):
    """F15: the visible scale comes from the raster, not a CLI width."""
    path = tmp_path / "visible.tif"
    _write_raster(path, width=16, height=12, resolution=0.5)
    resolution, width, height = render._raster_metadata(path)
    assert resolution == pytest.approx(0.5)
    assert (width, height) == (16, 12)


def test_cli_no_longer_accepts_visible_width(capsys):
    """F15: --visible-width was removed from the CLI contract."""
    with pytest.raises(SystemExit) as exit_info:
        render.main(["--help"])
    assert exit_info.value.code == 0
    assert "--visible-width" not in capsys.readouterr().out


def test_render_derives_visible_width_from_raster(tmp_path):
    """F15 regression: render() has no visible-width argument any more."""
    thermal = tmp_path / "thermal.tif"
    visible = tmp_path / "visible.tif"
    _write_raster(thermal, width=32, height=24, resolution=1.0)
    _write_raster(visible, width=16, height=12, resolution=2.0)
    world, result = _write_payloads(tmp_path)
    args = SimpleNamespace(
        station_name="s1", thermal_ortho=thermal, visible_ortho=visible,
        world_model=world, result_json=result, panel_csv=None,
        out_dir=tmp_path / "out", overview_width=64, side_width=48)
    summary = render.render(args)
    assert summary["panels"] == 1
    assert summary["station"] == "s1"
    for output in summary["outputs"]:
        assert Path(output).exists()

"""Regression tests: optimize tool summary reports the current field model.

The tool previously serialized a removed polynomial field model
(field.cx / field.cy / to_model()["normalization"]) and crashed on every run.
These tests lock the current contract: main() exits 0 and summary.json reports
the similarity + regularized RBF model.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from registration_algorithms.residual_model import fit_regularized_field


def _load_tool():
    path = (Path(__file__).resolve().parents[2] / "tools"
            / "optimize_dsm_3d_residual.py")
    spec = importlib.util.spec_from_file_location("optimize_tool_test", path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def _synthetic_field():
    rows = [{"east": 1000.0 + 20.0 * i, "north": 2000.0 + 20.0 * j,
             "dx_m": 0.30 + 0.002 * i - 0.001 * j,
             "dy_m": 0.10 - 0.001 * i + 0.002 * j, "cc": 0.9}
            for i in range(3) for j in range(2)]
    return fit_regularized_field(rows, tile_m=40.0, max_distance_m=30.0)


def _estimation(field):
    return SimpleNamespace(
        field=field,
        records=(),
        accepted=(),
        components=(),
        outliers=0,
        joint_shape=(2, 2),
        overlap_extent_m=(1.0, 1.0),
        tiling=SimpleNamespace(tile_m=40.0, step_m=20.0, tile_px=400,
                                step_px=200),
        field_support=SimpleNamespace(short_overlap_m=1.0,
                                       max_distance_m=30.0,
                                       requested_max_distance_m=30.0),
        attempts=(),
    )


def test_optimize_tool_summary_reports_similarity_model(monkeypatch, tmp_path):
    tool = _load_tool()
    field = _synthetic_field()
    monkeypatch.setattr(tool, "estimate_planimetric",
                        lambda inputs: _estimation(field))

    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "world_model.json").write_text('{"panels": []}',
                                                 encoding="utf-8")
    out_dir = tmp_path / "out"

    code = tool.main([
        "--result-dir", str(result_dir),
        "--visible-dsm", str(tmp_path / "visible_dsm.tif"),
        "--thermal-dsm", str(tmp_path / "thermal_dsm.tif"),
        "--out-dir", str(out_dir),
    ])

    assert code == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "polynomial_coefficients" not in summary
    assert "IDW" not in summary["method"]
    similarity = summary["planimetric_similarity"]
    expected = field.to_model()["similarity"]
    assert similarity["rotation_deg"] == pytest.approx(expected["rotation_deg"])
    assert similarity["scale"] == pytest.approx(expected["scale"])
    assert similarity["translation"] == pytest.approx(expected["translation"],
                                                      abs=1e-12)
    residual = summary["regularized_residual"]
    assert residual["kernel_length_m"] == pytest.approx(field.kernel_length_m)
    assert residual["ridge"] == pytest.approx(field.ridge)
    assert residual["centers"] == len(field.centers)
    field_model = json.loads((out_dir / "planimetric_field.json")
                             .read_text(encoding="utf-8"))
    assert field_model["type"] == "similarity_regularized_rbf/v1"
    # The tile admission verdicts are gone from the estimator and its report.
    assert summary["tile_candidates"] == {"tiles": 0, "accepted": 0}
    for removed in ("periodic_ambiguity", "unique", "candidate_count",
                    "second_margin"):
        assert removed not in summary["tile_candidates"]


def test_optimize_tool_writes_candidate_artifacts(monkeypatch, tmp_path):
    tool = _load_tool()
    field = _synthetic_field()
    monkeypatch.setattr(tool, "estimate_planimetric",
                        lambda inputs: _estimation(field))

    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "world_model.json").write_text('{"panels": []}',
                                                 encoding="utf-8")
    out_dir = tmp_path / "out"

    code = tool.main([
        "--result-dir", str(result_dir),
        "--visible-dsm", str(tmp_path / "visible_dsm.tif"),
        "--thermal-dsm", str(tmp_path / "thermal_dsm.tif"),
        "--out-dir", str(out_dir),
    ])

    assert code == 0
    candidates = json.loads((out_dir / "tile_candidates.json")
                           .read_text(encoding="utf-8"))
    assert candidates == []
    # Empty tile records leave no tiles.csv (write_csv skips empty row sets),
    # but the field model and summary are always written.
    assert not (out_dir / "tiles.csv").exists()


def test_optimize_tool_requires_result_dir_and_dropped_dead_flags():
    """F5/F6: dead flags are gone and the stale result default was removed."""
    tool = _load_tool()
    assert not hasattr(tool, "DEFAULT_RESULT")
    with pytest.raises(SystemExit) as missing:
        tool.main(["--visible-dsm", "visible.tif"])
    assert missing.value.code == 2
    for removed in (["--erode-m", "0.35"], ["--minimum-pixels", "20"]):
        with pytest.raises(SystemExit) as error:
            tool.main(removed + ["--result-dir", "result"])
        assert error.value.code == 2


def test_tool_default_paths_use_outputs_directory():
    """F6: no tool default points at the non-existent output/ directory."""
    tool = _load_tool()
    assert "photo-bridge/outputs/" in str(tool.DEFAULT_OUT)
    path = (Path(__file__).resolve().parents[2] / "tools"
            / "build_visible_world_model.py")
    spec = importlib.util.spec_from_file_location("world_model_tool_test", path)
    world_tool = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = world_tool
    spec.loader.exec_module(world_tool)
    assert "photo-bridge/outputs/" in str(world_tool.DEFAULT_OUT)


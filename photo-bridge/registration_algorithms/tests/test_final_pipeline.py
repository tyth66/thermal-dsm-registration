
from types import SimpleNamespace

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from registration_algorithms.application import (
    FieldSupportPlan, FinalPipelineInputs, TilingPlan,
)
from registration_algorithms.application import final_pipeline as module


def _pipeline_module():
    path = Path(__file__).resolve().parents[2] / "tools" / "run_3d_residual_mapping_pipeline.py"
    spec = importlib.util.spec_from_file_location("pipeline_config_test", path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def _planimetric_result():
    """Minimal PlanimetricResult-like object for provenance injection."""
    return SimpleNamespace(
        field=SimpleNamespace(kernel_length_m=80.0, ridge=0.1),
        tiling=TilingPlan(
            tile_m=40.0, step_m=20.0, tile_px=400, step_px=200,
            overlap_width_m=202.0, overlap_height_m=299.1,
            requested_tile_m=40.0, requested_step_m=20.0),
        field_support=FieldSupportPlan(
            max_distance_m=30.0, short_overlap_m=202.0,
            requested_max_distance_m=30.0),
        attempts=({"attempt": 1, "tile_m": 40.0, "step_m": 20.0},
                  {"attempt": 2, "tile_m": 20.0, "step_m": 10.0}),
    )


def test_pipeline_config_validates_parameters(tmp_path):
    pipeline = _pipeline_module()
    values = dict(
        result_dir=tmp_path, mapping_root=tmp_path,
        visible_dsm=tmp_path / "v.tif", thermal_dsm=tmp_path / "t.tif",
        visible_ortho=tmp_path / "vo.tif", thermal_ortho=tmp_path / "to.tif",
        out_dir=tmp_path / "out",
    )
    config = pipeline.PipelineConfig(**values)
    assert not hasattr(config, "array_plane_shrinkage")
    assert not hasattr(config, "planimetric_highpass_sigma_m")
    with pytest.raises(ValueError):
        pipeline.PipelineConfig(**values, grid_step=0)
    with pytest.raises(ValueError):
        pipeline.PipelineConfig(**values, planimetric_workers=0)
    # F10: the planimetric scale contract is validated at config construction.
    for name, invalid in (("planimetric_resolution_m", 0.0),
                          ("planimetric_tile_m", -1.0),
                          ("planimetric_step_m", 0.0),
                          ("planimetric_field_max_distance_m", float("nan"))):
        with pytest.raises(ValueError, match=name):
            pipeline.PipelineConfig(**values, **{name: invalid})


def test_pipeline_cli_requires_result_dir(tmp_path):
    # F6: the stale raw_registration default was removed; the caller must name
    # the result directory instead of silently resolving a missing path.
    pipeline = _pipeline_module()
    assert not hasattr(pipeline, "DEFAULT_RESULT")
    with pytest.raises(SystemExit):
        pipeline.main(["--visible-dsm", str(tmp_path / "v.tif")])


def test_final_pipeline_runs_two_stages_in_memory(monkeypatch, tmp_path):
    estimation = _planimetric_result()
    captured = {}
    computation = SimpleNamespace(summary={"method": "test"}, panel_rows=())

    def fake_estimate(inputs):
        captured["planimetric"] = inputs
        return estimation

    def fake_mapping(inputs):
        captured["mapping"] = inputs
        return computation

    monkeypatch.setattr(module, "estimate_planimetric", fake_estimate)
    monkeypatch.setattr(module, "compute_mapping", fake_mapping)

    output = tmp_path / "final.json"
    result = module.run_final_pipeline(FinalPipelineInputs(
        world={"panels": [{"panel_id": "p0"}]},
        planimetric=object(),
        visible_ortho=tmp_path / "visible_ortho.tif",
        thermal_ortho=tmp_path / "thermal_ortho.tif",
        output_json=output,
    ))

    assert output.exists()
    assert result.final_json == output
    assert [stage["stage"] for stage in result.stages] == [
        "planimetric_residual", "visible_to_thermal_mapping",
    ]
    assert all(stage.get("in_memory") for stage in result.stages)
    # Mapping consumes the planimetric field directly; no vertical/canonical
    # stage and no DSM path plumbing survives on the final-pipeline inputs.
    assert captured["mapping"].field is estimation.field
    assert captured["mapping"].grid_mode == "summary"
    assert not hasattr(captured["mapping"], "canonical")
    assert not hasattr(module.FinalPipelineInputs(
        world={}, planimetric=object(), visible_ortho=tmp_path / "v.tif",
        thermal_ortho=tmp_path / "t.tif", output_json=output), "visible_dsm")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "visible_to_thermal_mapping_result/v3"
    assert "3D residual" not in payload["method"]
    assert payload["summary"]["planimetric"]["tiling"]["tile_m"] == pytest.approx(40.0)


def test_final_pipeline_payload_carries_planimetric_provenance(monkeypatch,
                                                              tmp_path):
    """F14: the delivered payload records how the planimetric field was fit."""
    estimation = _planimetric_result()
    computation = SimpleNamespace(summary={"method": "test"}, panel_rows=())
    monkeypatch.setattr(module, "estimate_planimetric",
                        lambda inputs: estimation)
    monkeypatch.setattr(module, "compute_mapping", lambda inputs: computation)
    output = tmp_path / "final.json"
    result = module.run_final_pipeline(FinalPipelineInputs(
        world={"panels": []}, planimetric=object(),
        visible_ortho=tmp_path / "v.tif", thermal_ortho=tmp_path / "t.tif",
        output_json=output))

    payload = json.loads(output.read_text(encoding="utf-8"))
    provenance = payload["summary"]["planimetric"]
    tiling = provenance["tiling"]
    assert tiling["tile_m"] == pytest.approx(40.0)
    assert tiling["step_m"] == pytest.approx(20.0)
    assert (tiling["tile_px"], tiling["step_px"]) == (400, 200)
    assert tiling["overlap_width_m"] == pytest.approx(202.0)
    assert tiling["overlap_height_m"] == pytest.approx(299.1)
    assert tiling["requested_tile_m"] == pytest.approx(40.0)
    assert tiling["requested_step_m"] == pytest.approx(20.0)
    support = provenance["field_support"]
    assert support["max_distance_m"] == pytest.approx(30.0)
    assert support["short_overlap_m"] == pytest.approx(202.0)
    assert support["requested_max_distance_m"] == pytest.approx(30.0)
    assert [attempt["tile_m"] for attempt in provenance["attempts"]] == [40.0, 20.0]
    assert provenance["kernel_length_m"] == pytest.approx(80.0)
    assert provenance["ridge"] == pytest.approx(0.1)
    assert result.summary["planimetric"] == provenance

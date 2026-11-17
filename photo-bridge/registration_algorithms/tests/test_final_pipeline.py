
from types import SimpleNamespace

import importlib.util
import sys
from pathlib import Path

import pytest

from registration_algorithms.application import FinalPipelineInputs
from registration_algorithms.application import final_pipeline as module


def _pipeline_module():
    path = Path(__file__).resolve().parents[2] / "tools" / "run_3d_residual_mapping_pipeline.py"
    spec = importlib.util.spec_from_file_location("pipeline_config_test", path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def test_pipeline_config_validates_parameters(tmp_path):
    pipeline = _pipeline_module()
    values = dict(
        result_dir=tmp_path, mapping_root=tmp_path,
        visible_dsm=tmp_path / "v.tif", thermal_dsm=tmp_path / "t.tif",
        visible_ortho=tmp_path / "vo.tif", thermal_ortho=tmp_path / "to.tif",
        out_dir=tmp_path / "out",
    )
    pipeline.PipelineConfig(**values)
    with pytest.raises(ValueError):
        pipeline.PipelineConfig(**values, grid_step=0)
    with pytest.raises(ValueError):
        pipeline.PipelineConfig(**values, array_plane_shrinkage=1.5)
    with pytest.raises(ValueError):
        pipeline.PipelineConfig(**values, planimetric_workers=0)

from registration_algorithms.application import FinalPipelineInputs
from registration_algorithms.application import final_pipeline as module


def test_final_pipeline_runs_use_cases_in_memory(monkeypatch, tmp_path):
    estimation = SimpleNamespace(
        field=object(), components=({"panel_id": "p0"},))
    panel = SimpleNamespace(model=object(), samples={"p0": {}}, plane_cache={})
    geometry = SimpleNamespace(rows=({"panel_id": "p0", "array_id": "a0"},))
    bias = SimpleNamespace(corrected=SimpleNamespace(
        provenance={"array_height_correction": "per_array"}))
    stages = []

    class FakeContext:
        def __init__(self, **kwargs):
            stages.append(("context", kwargs))

        def compose(self):
            return "canonical"

    class FakeComputation:
        summary = {"method": "test"}
        panel_rows = ()

    monkeypatch.setattr(module, "estimate_planimetric", lambda inputs: estimation)
    monkeypatch.setattr(module, "build_panel_model", lambda inputs: panel)
    monkeypatch.setattr(module, "build_geometry", lambda inputs: geometry)
    monkeypatch.setattr(module, "fit_bias", lambda inputs: bias)
    monkeypatch.setattr(module, "PipelineContext", FakeContext)
    monkeypatch.setattr(module, "compute_mapping", lambda inputs: FakeComputation())

    output = tmp_path / "final.json"
    result = module.run_final_pipeline(FinalPipelineInputs(
        world={"panels": [{"panel_id": "p0"}]},
        planimetric=object(),
        visible_dsm=tmp_path / "visible_dsm.tif",
        thermal_dsm=tmp_path / "thermal_dsm.tif",
        visible_ortho=tmp_path / "visible_ortho.tif",
        thermal_ortho=tmp_path / "thermal_ortho.tif",
        output_json=output,
        array_plane_shrinkage=0.5,
    ))

    assert output.exists()
    assert result.final_json == output
    assert [stage["stage"] for stage in result.stages] == [
        "planimetric_residual", "panel_vertical_model", "vertical_geometry",
        "array_vertical_bias", "compose_residual_model",
        "visible_to_thermal_mapping",
    ]
    assert all(stage.get("in_memory") for stage in result.stages)
    assert stages and stages[0][0] == "context"
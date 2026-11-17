
import pytest

from registration_algorithms.application import PipelineContext
from registration_algorithms.domain import (
    CanonicalResidualModel, GeometryModel, PanelResidual, PanelVerticalBundle,
    PanelVerticalModel)
from registration_algorithms.local_plane import LocalPlane
from registration_algorithms.residual_model import PlanimetricField


def _field():
    return PlanimetricField.from_tiles([
        {"east": 596000.0, "north": 3234000.0, "cc": 1.0, "dx_m": 0.2, "dy_m": -0.1},
        {"east": 596020.0, "north": 3234000.0, "cc": 0.9, "dx_m": 0.1, "dy_m": 0.0},
        {"east": 596000.0, "north": 3234020.0, "cc": 0.9, "dx_m": 0.3, "dy_m": 0.1},
    ])


def test_panel_residual_roundtrip():
    row = {
        "array_id": 7,
        "visible_plane": {"type": "local_plane", "e0": 596000.0, "n0": 3234000.0,
                          "z0": 20.0, "a": 0.1, "b": 0.0},
        "thermal_plane": {"type": "local_plane", "e0": 596001.0, "n0": 3234000.0,
                          "z0": 21.0, "a": 0.1, "b": 0.0},
        "thermal_correction_method": "array_bias",
    }
    panel = PanelResidual.from_row(row)
    assert panel.array_id == "7"
    assert panel.visible_plane is not None
    assert panel.to_row() == {
        "array_id": "7",
        "visible_plane": row["visible_plane"],
        "thermal_plane": row["thermal_plane"],
        "thermal_correction_method": "array_bias",
    }


def test_canonical_model_roundtrip_and_mapping_view():
    field = _field()
    rows = {
        "p0": {"array_id": "a0",
               "visible_plane": {"type": "local_plane", "e0": 596000.0,
                                 "n0": 3234000.0, "z0": 20.0, "a": 0.1, "b": 0.0},
               "thermal_plane": None,
               "thermal_correction_method": "array_bias"},
    }
    model = CanonicalResidualModel.from_parts(
        field, rows, {"visible_panels": 1}, {"source": "test"}, version="2.0")
    data = model.to_dict()
    restored = CanonicalResidualModel.from_dict(data)
    assert restored.version == "2.0"
    assert restored.coverage == {"visible_panels": 1}
    assert restored.provenance == {"source": "test"}
    assert restored.planimetric.to_model() == field.to_model()
    assert restored.panels["p0"].array_id == "a0"
    assert restored.panels["p0"].thermal_plane is None
    mapping = restored.mapping_panels()
    assert isinstance(mapping["p0"]["visible_plane"], LocalPlane)
    assert mapping["p0"]["thermal_correction_method"] == "array_bias"
def test_panel_vertical_model_roundtrip():
    arrays = {
        "a0": {"plane": {"type": "local_plane", "e0": 596000.0, "n0": 3234000.0,
                         "z0": 20.0, "a": 0.0, "b": 0.0},
               "quality_status": "surface_valid", "support": 36},
    }
    panels = {
        "p0": {"array_id": "a0", "center_xyz": [596000.0, 3234000.0, 20.0],
               "corners_xyz": [[596000.0 - 1, 3234000.0 - 1, 20.0],
                               [596000.0 + 1, 3234000.0 - 1, 20.0],
                               [596000.0 + 1, 3234000.0 + 1, 20.0],
                               [596000.0 - 1, 3234000.0 + 1, 20.0]],
               "z_source": "thermal_panel_plane",
               "panel_plane": {"type": "local_plane", "e0": 596000.0, "n0": 3234000.0,
                               "z0": 21.0, "a": 0.1, "b": -0.1,
                               "rms_m": 0.01, "support": 100}},
    }
    model = PanelVerticalModel.from_parts(arrays, panels,
                                          inputs={"source": "test"},
                                          provenance={"kind": "test"})
    data = model.to_dict()
    restored = PanelVerticalModel.from_dict(data)
    assert restored.inputs == {"source": "test"}
    assert restored.arrays["a0"].quality_status == "surface_valid"
    assert restored.panels["p0"].array_id == "a0"
    assert restored.panels["p0"].panel_plane is not None
    assert restored.panels["p0"].panel_plane.rms_m == pytest.approx(0.01)
    assert restored.to_dict() == data
def test_geometry_bundle_and_bias_roundtrip():
    geometry = GeometryModel.from_rows([{
        "panel_id": "p0", "array_id": "a0",
        "visible_plane_e0": "596000", "visible_plane_n0": "3234000",
        "visible_plane_z0": "20", "visible_plane_a": "0.1", "visible_plane_b": "0.0",
        "visible_plane_rms_m": "0.01", "visible_samples": "100",
        "thermal_plane_e0": "596001", "thermal_plane_n0": "3234000",
        "thermal_plane_z0": "21", "thermal_plane_a": "0.1", "thermal_plane_b": "0.0",
        "thermal_plane_rms_m": "0.02", "thermal_samples": "90",
    }])
    assert geometry.panels["p0"].visible_plane is not None
    bias = {
        "plane_representation": "local_en/v1",
        "footprint_sampling": "sample_footprint/v1",
        "inputs": {"source": "test"},
        "arrays": {},
        "panels": {
            "p0": {"array_id": "a0", "center_xyz": [596000.0, 3234000.0, 21.0],
                   "corners_xyz": [[596000.0, 3234000.0, 21.0]] * 4,
                   "z_source": "thermal_panel_plane",
                   "panel_plane": {"type": "local_plane", "e0": 596001.0,
                                   "n0": 3234000.0, "z0": 21.0, "a": 0.1, "b": 0.0,
                                   "rms_m": 0.02, "support": 90},
                   "vertical_bias_m": 0.5,
                   "vertical_correction_method": "per_array_residual_plane"},
        },
        "provenance": {"kind": "test"},
        "array_height_corrections": {"version": "test"},
    }
    model = PanelVerticalModel.from_dict(bias)
    assert model.panels["p0"].vertical_bias_m == pytest.approx(0.5)
    assert model.to_dict()["array_height_corrections"] == {"version": "test"}
    field = _field()
    canonical = PanelVerticalBundle(panel_model=model, geometry=geometry).to_canonical(
        field, ["p0"], {"p0": "a0"}, {"visible_panels": 1}, {"source": "test"})
    assert canonical.panels["p0"].visible_plane is not None
    assert canonical.panels["p0"].thermal_plane is not None
    assert canonical.panels["p0"].thermal_correction_method == "per_array_residual_plane"
def test_pipeline_context_composes_once():
    field = _field()
    world = {"panels": [{"panel_id": "p0", "array_id": "a0"}]}
    geometry = GeometryModel.from_rows([{
        "panel_id": "p0", "array_id": "a0",
        "visible_plane_e0": "596000", "visible_plane_n0": "3234000",
        "visible_plane_z0": "20", "visible_plane_a": "0.0", "visible_plane_b": "0.0",
    }])
    model = PanelVerticalModel.from_dict({
        "plane_representation": "local_en/v1",
        "arrays": {},
        "panels": {"p0": {"array_id": "a0", "center_xyz": [596000.0, 3234000.0, 20.0],
                          "corners_xyz": None, "z_source": "thermal_m0",
                          "panel_plane": None}},
        "provenance": {},
    })
    context = PipelineContext(world=world, planimetric=field, panel_model=model,
                              geometry=geometry, provenance={"source": "test"})
    canonical = context.compose()
    assert context.canonical is canonical
    assert context.compose() is canonical
    assert context.available_panels == 1
    assert canonical.panels["p0"].array_id == "a0"




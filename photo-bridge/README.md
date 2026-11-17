# Photo Bridge — Current Production Algorithm

Visible-to-thermal ortho mapping with a hierarchical 3D residual model.

## Pipeline

    visible ortho + visible DSM + panels
        -> visible_world_model
        -> planimetric_residual
        -> panel_vertical_model
        -> vertical_geometry
        -> array_vertical_bias
        -> compose_residual_model
        -> visible_to_thermal_mapping

The pipeline runs in memory and writes only the final mapping JSON.

## Current production choices

- Planimetric field: global similarity + regularized RBF residual.
- Tile ECC: adaptive tiling, nodata-aware highpass, top-5 candidates.
- Quality gate: `CC >= 0.5` and uniqueness.
- Outlier rejection: local kNN median, threshold `max(0.60 m, median + 3 MAD)`.
- Vertical model: panel planes on shifted thermal DSM footprints, array
  residual-plane shrinkage 0.5, minimum array support 20.
- Panel mapping: `rigid` = per-panel translation (default).
- Output: Thermal ortho pixel coordinates, not Thermal RAW projection.

## Main modules

    registration_algorithms/
      application/          use-case pipeline and mapping
      dsm_registration.py   highpass, tile ECC, candidates, CV
      residual_model.py     similarity + regularized RBF field
      panel_thermal_model.py
      array_vertical_bias.py
      forward_mapping.py
      visible_world_model.py

## Run

    cd analysis/photo-bridge
    python tools/run_3d_residual_mapping_pipeline.py \
      --panels <panels.json.standard.json> \
      --visible-ortho <visible.tif> --visible-dsm <visible_dsm.tif> \
      --thermal-ortho <thermal.tif> --thermal-dsm <thermal_dsm.tif> \
      --out-dir <output-dir>

For an already-built world model:

    python tools/run_3d_residual_mapping_pipeline.py \
      --result-dir <dir containing world_model.json> \
      --visible-ortho ... --visible-dsm ... \
      --thermal-ortho ... --thermal-dsm ... \
      --out-dir <output-dir>

Output:

    <output-dir>/visible_to_thermal_mapping/visible_to_thermal_mapping_result.json

## Tests

    PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q registration_algorithms/tests

## Preserved outputs

- `outputs/141/visible_to_thermal_mapping/visible_to_thermal_mapping_result.json`
- `outputs/247/visible_to_thermal_mapping/visible_to_thermal_mapping_result.json`
- `outputs/visualizations/` — full-array thermal overlays and visible/thermal
  side-by-side images for stations 141 and 247.

## Visualization

    python tools/render_full_array_mapping.py \
      --station-name 247 \
      --thermal-ortho <thermal.tif> --visible-ortho <visible.tif> \
      --visible-width <visible width> \
      --world-model <world_model.json> \
      --result-json <visible_to_thermal_mapping_result.json> \
      --out-dir outputs/visualizations


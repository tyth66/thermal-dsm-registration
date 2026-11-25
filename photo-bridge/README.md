# Visible–thermal DSM planimetric registration

Planimetrically register a thermal DSM to a visible DSM with masked ECC
candidate matching and a continuous displacement field.

## Status

- **Production:** two-stage hard-gated estimator, panel mapping and
  conservative delivery.
- **Research prototype:** all-raw candidate / field joint MAP-EM in
  `registration_algorithms/research/`; not imported by production.

## Production API

```python
import json
from registration_algorithms.application import (
    PlanimetricInputs, estimate_planimetric,
)

world = json.load(open("world_model.json"))
result = estimate_planimetric(PlanimetricInputs(
    visible_dsm="visible_dsm.tif",
    thermal_dsm="thermal_dsm.tif",
    visible_world=world,
))
```

For the full panel-mapping pipeline, use `run_final_pipeline`.

## Research API

```python
from registration_algorithms.research import (
    ResearchPlanimetricInputs, estimate_research_planimetric,
)

result = estimate_research_planimetric(ResearchPlanimetricInputs(
    visible_dsm="visible_dsm.tif",
    thermal_dsm="thermal_dsm.tif",
    visible_world=world,
))
```

The research API returns the field, inliers, panel components, EM history,
responsibilities and posterior diagnostics. For in-memory panel/grid mapping,
use `run_research_pipeline`, which adds the production mapping layer without
writing a final JSON.

## Package boundary

```text
registration_algorithms/
  application/          production planimetric + panel-mapping use cases
  dsm_registration.py    shared DSM grid and raw tile observations
  masked_ecc.py          shared ECC candidate extraction
  residual_model.py      shared centered displacement field
  boundary_field.py      production boundary correction
  research/              research-only estimator and DSM covariance
```

Dependency direction:

```text
research -> production base
production -X-> research
```

## Documentation

- [CURRENT_ALGORITHM.md](CURRENT_ALGORITHM.md): production algorithm.
- [CANDIDATE_FIELD_EM.md](CANDIDATE_FIELD_EM.md): research model, status
  and limitations.
- [ALGORITHM_MATH_MODEL.md](ALGORITHM_MATH_MODEL.md): production math.
- [ALGORITHM_DEEP_DIVE.md](ALGORITHM_DEEP_DIVE.md): design rationale.
- [BOUNDARY_EXTRAPOLATION.md](BOUNDARY_EXTRAPOLATION.md): boundary support.

## Tests

```bash
python3 -m pytest -q
```

Current: `113 passed`.
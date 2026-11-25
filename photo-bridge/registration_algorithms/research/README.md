# Research-only estimators

This package is **not imported by the production planimetric pipeline**.
Production code and parameters remain in the parent package and are documented
in `CURRENT_ALGORITHM.md`.

## Contents

- `candidate_field.py`: all-raw candidate / field joint MAP-EM prototype.
- `dsm_covariance.py`: optional DSM-profile candidate covariance
  `C=(beta_eff Q)^-1`.
- `tests/`: research tests, including the double-peak information-loss check.

## Dependency direction

```text
production base (masked_ecc, dsm_registration, residual_model)
        ↑
research candidate field
```

The production pipeline does not import this package. Raw observation
collection uses the generic low-level `register(..., apply_domain=False)`
and an optional covariance callback.

## Status

The candidate-field model is a research prototype. Exact W/h M-step,
sigma_f MAP-MLE, DSM covariance and posterior-curvature diagnostics are
implemented, but none are in production. See
`CANDIDATE_FIELD_EM.md` for derivation and current limitations.
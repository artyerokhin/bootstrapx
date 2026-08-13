# Changelog

## 0.4.1

### Statistical correctness

- Studentized estimates and nested standard errors now come from the same
  outer resample.
- Bayesian bootstrap evaluates functionals directly under Dirichlet weights;
  custom statistics provide a weighted callable.
- Subsampling intervals use centered, rate-scaled roots.
- Bernoulli subsets include a finite-population correction.
- Statistical regression tests cover scaling, weighting, cluster uncertainty,
  and nested-resample pairing.

## 0.4.0

### Reliability

- Strict validation for public parameters and resampled statistics.
- Python 3.10–3.13 CI on Linux, macOS, and Windows.
- Lint, documentation, package-build, and release-publishing guardrails.

## 0.3.0

### Fixed

- Bayesian bootstrap reproducibility bug.
- Studentized interval quantile ordering.
- Over-aggressive batch sizing heuristic.
- Memory-heavy jackknife implementation.
- Time-series sieve implementation performance bottleneck.
- Misleading unimplemented CUDA backend story.

### Added

- `BootstrapCV` for scikit-learn.
- pandas accessor for `Series` and `DataFrame`.
- Expanded README examples and optional dependency model.
- New notebook/documentation directions for ML metrics, A/B testing, and time series.

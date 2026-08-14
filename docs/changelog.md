# Changelog

## 0.4.4

### Practical workflow polish

- Results export to compact dictionaries and one-row pandas DataFrames.
- The documented `numba` install extra now exists and has its own CI check.
- Numba guidance now states exactly which methods it accelerates, its JIT
  startup tradeoff, and how to reproduce the fallback comparison.
- A limitations guide makes one-sample, scalar-output, dependent-data, and
  pre-1.0 boundaries explicit.
- DataFrame documentation distinguishes separate column intervals from an
  actual treatment-effect comparison.
- The guide now starts from data shape and separates supported paired/grouped
  workflows from unsupported unpaired A/B effects.
- Versioned 0.4.4 runtime, memory, Numba, and 160-cell coverage evidence is
  published with exact metadata and generated plots.
- Coverage benchmark random streams, failure accounting, metadata, and CI smoke
  validation have been corrected.
- Release benchmark profiles now preserve separate outputs and support safe
  checkpoint/resume for long coverage studies.
- Optional-dependency errors, issue forms, and contribution guidance are more
  actionable for practitioners.

## 0.4.3

### Engineering reliability

- Strict `mypy` validation now covers the complete source tree and runs in CI.
- Time-series tests cover all six methods across batch sizes, optional Numba
  fallback, taper variants, wild multipliers, and sieve edge cases.
- Invalid taper-window errors are stable across supported SciPy versions, and
  strict typing uses NumPy stubs compatible with the Python 3.10 target.
- GitHub Actions use immutable commit pins maintained by Dependabot.
- Publishing can be rerun safely after a partial release: an existing complete
  PyPI version is skipped and the GitHub Release is created or updated.
- Documentation is automatically deployed to GitHub Pages when relevant
  changes reach `main`, and can also be deployed manually.
- Release checks include an 85% coverage floor, dependency consistency,
  byte-compilation, and core import smoke tests.

## 0.4.2

### Correctness and release safety

- Tapered block bootstrap now preserves constant series, location shifts, and
  variance scale by tapering centered observations with energy normalization.
- NumPy fast paths honor `batch_size` instead of allocating all resamples at
  once.
- `BootstrapCV` always produces the requested number of nonempty OOB splits
  and validates small samples and constructor parameters.
- Constant time series produce a degenerate sieve-bootstrap distribution
  instead of failing with a singular matrix.
- Release publication now runs tests, lint, documentation, and coverage gates
  before uploading to PyPI and creates a GitHub Release.

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

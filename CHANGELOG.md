# Changelog

All notable changes to **bootstrapx** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Automatic block length selection (Politis & White, 2004)
- Permutation-based confidence intervals
- Registry-based method dispatch

## [0.2.0] — 2026-03-06

### Added
- **Bayesian bootstrap** method via Dirichlet(1,...,1) weights (Rubin, 1981).
- **`vectorized`** parameter — call `statistic(batch, axis=1)` for 10-50x speedup.
- **`ci_method`** parameter for generator-based methods (percentile or basic).
- **`pandas.DataFrame`** and `pandas.Series` input support in `validate_data()`.
- **Parallel jackknife** via joblib for BCa on large datasets (`n_jobs` parameter).
- **`ConfidenceInterval.width`** property and `__contains__` method.
- **`py.typed`** PEP 561 marker for mypy.
- **CI workflow** (`.github/workflows/ci.yml`) — Python 3.9–3.13 matrix.

### Fixed
- **Version sync** — `__version__` now uses `importlib.metadata` (single source).
- **Poisson/Bernoulli weights** — use `np.repeat` for correct multiplicity handling
  instead of binary mask filtering.
- **Studentized bootstrap** — batched outer loop (no full `(n_resamples, n)` allocation).
- **CBB validation** — added `block_length >= n` check.

### Changed
- `auto_batch_size()` target increased from 32K to 64K elements.
- Type hints added throughout all modules.

## [0.1.0] — 2026-02-16

### Added
- Initial release with 14 bootstrap methods, Numba acceleration, generator batching.

[Unreleased]: https://github.com/artyerokhin/bootstrapx/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/artyerokhin/bootstrapx/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/artyerokhin/bootstrapx/releases/tag/v0.1.0

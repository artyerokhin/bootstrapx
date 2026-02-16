# Changelog

All notable changes to **bootstrapx** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Bayesian bootstrap method
- Permutation-based confidence intervals
- `pandas.DataFrame` input support
- Automatic block length selection (Politis & White, 2004)
- Parallel jackknife for BCa on large datasets

## [0.1.0] — 2026-02-16

### Added

#### Core API
- `bootstrap()` — unified entry point with 14 methods, generator-based batching,
  and automatic backend selection.
- `BootstrapResult` dataclass with `.confidence_interval`, `.bootstrap_distribution`,
  `.theta_hat`, `.standard_error`.

#### IID Resampling (7 methods)
- **Percentile** — standard quantile-based CI.
- **Basic** (reverse percentile) — `2θ̂ − q` formulation.
- **BCa** (Bias-Corrected and Accelerated) — jackknife acceleration `â` and
  bias correction `z₀` via inverse normal CDF.
- **Studentized** (Bootstrap-t) — nested bootstrap for SE estimation.
- **Poisson Bootstrap** — weighted resampling with `W ~ Poisson(1)`.
- **Bernoulli Bootstrap** — binary weights `W ~ Bernoulli(p)`.
- **Subsampling** (m-out-of-n) — without-replacement sampling for extreme statistics.

#### Time Series (6 methods)
- **Moving Block Bootstrap (MBB)** — overlapping fixed-length blocks.
- **Circular Block Bootstrap (CBB)** — wrapping to eliminate edge effects.
- **Stationary Bootstrap** (Politis & Romano) — geometric random block lengths
  with Numba-optimized index generation loop.
- **Tapered Block Bootstrap** — window function applied to blocks for spectral estimation.
- **AR-Sieve Bootstrap** — AR(p) fit via Yule-Walker, residual resampling,
  recursive reconstruction.
- **Wild Bootstrap** — Rademacher and Mammen two-point distributions for
  heteroskedastic data.

#### Hierarchical (2 methods)
- **Cluster Bootstrap** — resample entire groups/clusters.
- **Stratified Bootstrap** — within-stratum resampling to preserve class proportions.

#### Infrastructure
- Numba `@njit` acceleration for MBB, CBB, Stationary index generation
  and batch resampling (`prange` parallelism).
- Automatic backend dispatcher: `numba_cpu` → `numba_cuda` → `vanilla`.
- Generator-based batching — memory capped at ~64 MiB per batch.
- `validate_data()` utility — checks NaN, dimensionality, minimum observations.
- `auto_batch_size()` — heuristic batch sizing.

#### Testing
- 49 pytest tests covering all methods, edge cases, validation, backend dispatch,
  CI math, and reproducibility.

#### Documentation
- MkDocs Material site with Getting Started, API Reference, Methods Guide,
  Benchmarks, and Changelog.
- Jupyter notebooks: Quick Start, Benchmark vs SciPy.
- GitHub Actions CI for Python 3.9–3.12.

### Dependencies
- `numpy>=1.23`
- `scipy>=1.10`
- `numba>=0.57`
- `joblib>=1.3`

[Unreleased]: https://github.com/artyerokhin/bootstrapx/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/artyerokhin/bootstrapx/releases/tag/v0.1.0

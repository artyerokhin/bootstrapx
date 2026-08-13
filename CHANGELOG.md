# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.4.0] — 2026-08-13

### Fixed
- Public parameter validation now rejects invalid resample counts, batch sizes,
  confidence levels, block sizes and method-specific parameters before a
  calculation begins. In particular, zero batch or block sizes can no longer
  cause generator loops to hang.
- Unsupported method keyword arguments and inappropriate `ci_method` values
  now fail explicitly instead of being silently ignored or coerced.
- Cluster and strata identifiers are validated for shape and cardinality, and
  categorical string identifiers are supported.
- Data and observed statistics must be finite, preventing undefined intervals
  from being returned for NaN or infinite input.

### Changed
- Minimum supported Python version is now 3.10. The package uses modern union
  type syntax which cannot be parsed by Python 3.9.
- CI now checks formatting and linting, tests Python 3.10–3.13 on Linux,
  macOS and Windows, publishes coverage separately, and verifies that a built
  wheel can be installed and imported.
- Documentation now accurately lists 16 methods and supported integrations.

## [0.3.2] — 2026-05-27

### Fixed
- **studentized bootstrap**: `inner_idx` now generated independently per
  outer sample instead of once per batch. Fixes correlated SE* estimates
  that caused empirical CI coverage ~5–10% below nominal (issue #2).
- **backend**: removed dead `numba_cpu` stub. Passing `backend='numba_cpu'`
  now raises `ValueError` with a clear message instead of silently running
  vanilla Python (issue #3). Use `backend='auto'` or `backend='vanilla'`.

### Changed
- `_FAST_PATH_N` raised from 500 to 1000: numpy built-in statistics now
  use the single-matrix fast path for samples up to n=1000.

### Removed
- `BackendKind.NUMBA_CPU` and `BackendKind.NUMBA_CUDA` enum values removed.
- `numba` optional dependency removed from `pyproject.toml`.

## [0.3.1] — 2026-05-27

### Added (benchmarks)
- **`benchmarks/` suite** — first public benchmark suite for bootstrapx:
  - `bench_speed.py`: wall-clock time (median of 5 runs) and peak memory
    via `tracemalloc` vs `scipy.stats.bootstrap` for BCa and percentile;
    covers small-sample fast path (n=20–500), large-n memory (n=500–10 000),
    and arbitrary callable statistics (`trimmed_mean`, `iqr`).
  - `bench_coverage_accuracy.py`: empirical coverage on 4 distributions
    (Normal, LogNormal, Pareto, Bernoulli) × 3 statistics × 2 sample sizes.
    CLI: `--fast` (N_SIM=100, ~40s), default (N_SIM=200, ~3min),
    `--full` (N_SIM=1000, ~15min).
  - `plot_results.py`: generates `fig_fastpath.png`, `fig_memory.png`,
    `fig_coverage.png` from CSV results.

### Fixed (performance)
- **Small-sample fast path**: `apply_statistic_batched()` now detects
  `n < 500` combined with NumPy built-ins supporting `axis=`
  (`mean`, `median`, `std`, `var`, `sum`, `min`, `max`, nan-variants)
  and runs a single vectorized reduction over a preallocated
  `(n_resamples, n)` matrix. Removes per-sample Python loop overhead.

## [0.3.0] — 2026-05-24

### Fixed (correctness)
- **Bayesian bootstrap**: `_collect_bayesian` was creating a new unseeded
  `np.random.default_rng()` on every sample, making results non-reproducible
  even when `random_state` was set.  Now uses the caller's RNG throughout.
- **Studentized interval**: `t_q_lo` and `t_q_hi` quantile assignments were
  swapped, reflecting the interval around θ̂ for asymmetric distributions.
  Fixed per Hall (1992) §3.5.
- **CUDA backend**: was advertised but never implemented.  Now raises
  `NotImplementedError` with a clear message instead of a misleading
  `RuntimeError("no GPU found")`.

### Fixed (performance)
- **Sieve bootstrap**: replaced nested Python AR(p) loop with
  `scipy.signal.lfilter`.  Measured speedup: **~16×** for n=300,
  n_resamples=9 999 (691 ms → 43 ms).
- **`auto_batch_size`**: was dividing `target_elements=65536` by `n` without
  accounting for `itemsize`, producing float64 batches **8× too large** for
  L2 cache.  Fixed to `target_bytes // (n * itemsize)`.
- **`_jackknife`**: replaced `N` calls to `np.concatenate` (O(n²) total
  allocations, 191 MB for n=5 000) with a single preallocated buffer (39 KB).

### Fixed (thread safety)
- **Numba `_batch_indices_numba`**: called `np.random.seed()` inside `prange`
  — a write to global legacy state from multiple threads.  Removed from index
  generation entirely; index generation now always uses NumPy PCG64 (already
  SIMD-vectorised and statistically stronger).

### Changed
- `numba` moved from required to optional dependency (`pip install bootstrapx-lib[numba]`).
- `parallel=True` removed from remaining Numba kernels in `timeseries.py` to
  prevent seed-racing on machines with multiple cores.
- Time-series `_batch_gen` now uses `np.random.SeedSequence.spawn` for
  per-sample seeds instead of `seed_base+i` (which could collide for small seeds).

### Added
- **`BootstrapCV`**: scikit-learn-compatible cross-validator using OOB bootstrap
  splits.  Works with `cross_val_score`, `GridSearchCV`, etc.
- **pandas accessor**: `pd.Series.bootstrap.bca()`, `.percentile()`, `.ci()`;
  `pd.DataFrame.bootstrap.summary()` for column-wise CI tables.
- **CI workflow** (`.github/workflows/ci.yml`): matrix across Python 3.9–3.13,
  Ubuntu/Windows/macOS, with Codecov coverage upload.
- **Notebooks**: `03_ml_model_uncertainty.ipynb`, `04_ab_test_cluster_bootstrap.ipynb`,
  `05_timeseries_finance.ipynb`.
- **Keywords** expanded in `pyproject.toml` for PyPI search discoverability.

## [0.2.0] — 2025-12-01

### Added
- 14 bootstrap methods (iid, time-series, hierarchical)
- Numba JIT acceleration
- Memory-safe batched generation
- MkDocs documentation site

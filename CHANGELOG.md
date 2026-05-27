# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

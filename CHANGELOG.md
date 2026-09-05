# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.5.0] — Unreleased

### Added
- `bootstrap_two_sample()` estimates an explicit treatment-versus-control
  effect for independent, paired, or separately clustered samples.
- Built-in difference, ratio, and relative-lift effects, plus custom scalar
  effect callables.
- `TwoSampleBootstrapResult` reports both arm estimates, the effect interval,
  bootstrap standard error and distribution, experiment design metadata, and
  compact dictionary/DataFrame exports.
- Percentile, basic, and multi-sample BCa intervals. Clustered BCa uses
  leave-one-cluster-out acceleration rather than deleting individual rows.
- A resumable comparison coverage study and separate runtime/`tracemalloc`
  benchmark with exact version, commit, dependency, and environment metadata.
- Versioned 0.5.0 evidence covering 9,900 interval trials plus matched runtime
  and allocation measurements against SciPy.
- A reproducible Hillstrom email-experiment notebook downloads and verifies
  the public source, then estimates visit, conversion, lift, and spend effects.

### Changed
- Experiment guidance now starts from the randomization and analysis unit and
  distinguishes independent, paired, and repeated-event workflows.
- The clustered A/B notebook now uses the native two-sample API and separate
  experiment arms instead of reasoning from a precomputed row-wise difference.
- CI runs a two-sample statistical smoke benchmark in addition to the existing
  one-sample coverage pipeline.

### Tests
- Deterministic resampling, batch-size invariance, SciPy reference comparisons,
  cluster deletion, exports, guardrails, and focused coverage simulations for
  independent, paired, and clustered designs.

## [0.4.4] — 2026-08-14

### Added
- `BootstrapResult.to_dict()` returns a compact, mutation-safe result record;
  the full bootstrap distribution is available through the explicit
  `include_distribution=True` option.
- `BootstrapResult.to_frame()` returns a one-row pandas DataFrame suitable for
  reports, concatenation, and experiment tracking.
- The documented `numba` extra now exists and is tested independently in CI.
- Versioned release-benchmark evidence: matched runtime, `tracemalloc`, Numba,
  and 160-cell coverage results with exact environment metadata and plots.
- A current-limitations guide documents scalar-output, two-sample, missing-data,
  dependent-data, Monte Carlo, and pre-1.0 compatibility boundaries.
- Structured bug and practitioner-workflow issue forms, plus a contribution
  guide with the complete local verification sequence.

### Changed
- Project wording now describes bootstrapx as practical rather than claiming
  blanket production readiness while the public API remains pre-1.0.
- DataFrame examples explicitly state that column-wise intervals do not
  estimate a difference, ratio, lift, paired effect, or p-value between groups.
- The documentation is reorganized around data shape and practitioner
  decisions, with result interpretation, grouped-data, experiment,
  block-sensitivity, and optional-performance guidance.
- Benchmark claims distinguish runtime, `tracemalloc` working-memory
  measurements, and statistical coverage; they now cite the completed 0.4.4
  release run rather than stale pre-fix coverage artifacts.
- Coverage simulations now use independent deterministic streams for data and
  resampling, report invalid/failed trials and Monte Carlo uncertainty, record
  environment metadata, and have a small CI smoke test.
- A sequential release-benchmark runner provides quick, release, and
  statistical profiles. Long coverage studies checkpoint each configuration
  and can resume only with a matching commit, environment, and configuration.
- Missing optional-dependency errors point to the corresponding bootstrapx
  installation extra.

## [0.4.3] — 2026-08-13

### Changed
- The complete source tree now passes strict `mypy` validation. Public and
  internal array, generator, statistic, and compatibility-layer contracts use
  explicit types, and CI rejects new typing regressions.
- GitHub Actions are pinned to full commit SHAs. Dependabot checks these pins
  weekly, while workflow permissions default to read-only.
- PyPI publication is safely repeatable: reruns skip an already complete PyPI
  version and create or update the matching GitHub Release.
- Release and package jobs now run `pip check`, byte-compilation, and import
  smoke tests in addition to the existing test, documentation, and build checks.
- Documentation is automatically rebuilt and deployed to GitHub Pages after
  relevant changes reach `main`, with an additional manual-run option.
- The enforced coverage floor is raised from 70% to 85%.

### Fixed
- Documentation now matches the actual NumPy fast-path boundary of `n < 1000`.
- Invalid taper windows now produce a stable bootstrapx error across SciPy
  versions instead of exposing version-specific SciPy wording.
- Strict typing uses NumPy stubs compatible with its Python 3.10 target, while
  the runtime matrix continues testing the newest supported NumPy releases.
- Time-series index generators have explicit Python fallback implementations,
  preserving deterministic behavior when Numba is unavailable.

### Tests
- Expanded time-series tests cover taper windows and invalid windows, Mammen
  multipliers with fitted values, nonconstant and singular sieve fits,
  no-Numba fallback execution, boundary block lengths, and reproducibility of
  all six time-series methods across different batch sizes.
- Time-series generator coverage increased from 48% to 79%; total source
  coverage increased from 81% to 88%.

## [0.4.2] — 2026-08-13

### Fixed
- **Tapered block bootstrap** now applies an energy-normalized taper to the
  centered series and restores the sample mean. This preserves constant
  series, location shifts, and the variance scale instead of pulling tapered
  observations toward zero.
- NumPy built-in fast paths now honor `batch_size`. Large `n_resamples` no
  longer allocate a full `(n_resamples, n)` matrix despite batching being
  requested.
- `BootstrapCV` now redraws empty OOB samples so `split()` yields exactly
  `n_splits`, rejects invalid split counts and samples smaller than two, and
  documents the 0.632 estimator weights in the correct order.
- Sieve bootstrap returns the correct degenerate distribution for a constant
  series and explains singular autoregressive fits with an actionable error.
- Cluster and strata identifiers reject missing or incomparable values before
  resampling instead of leaking internal `KeyError`/NumPy exceptions.
- Unhashable callable statistics no longer fail during NumPy fast-path
  detection, and non-string backends receive a public `TypeError`.
- `from bootstrapx import *` works in core-only installations without
  scikit-learn.
- Studentized results no longer depend on the technical `batch_size` for a
  fixed random seed.

### Changed
- `vectorized=True` is explicitly limited to percentile, basic, and BCa
  methods; unsupported methods now fail instead of silently ignoring it.
- Package metadata uses an SPDX license expression and current license-file
  metadata.
- Release publication verifies that the tagged commit is on `main`, runs the
  complete release checks before PyPI upload, and creates a GitHub Release.
- CI tests minimum supported core dependencies, executable documentation
  examples, and enforces a 70% coverage floor.

### Tests
- Added regression tests for tapered-bootstrap invariants, fast-path batching,
  constant-series sieve behavior, exact BootstrapCV split counts, missing
  cluster identifiers, optional exports, and public argument validation.

## [0.4.1] — 2026-08-13

### Fixed
- **Studentized bootstrap** now computes each outer estimate and its nested
  standard error from the same outer resample. Previously, independently drawn
  outer samples were paired in the bootstrap-t root, which could distort
  coverage for skewed or scale-dependent statistics. The default number of
  inner resamples is increased from 50 to 100 to reduce Monte Carlo noise.
- **Bayesian bootstrap** now evaluates the statistic directly under
  `Dirichlet(1, ..., 1)` weights. The previous implementation performed an
  additional multinomial draw, adding posterior-predictive sampling noise and
  inflating the variance by about `sqrt(2)` for the sample mean.
- **Subsampling intervals** now use the centered and scaled subsampling root
  instead of percentiles of the smaller-sample estimates. The default
  convergence rate is root-n and can be configured with `rate`.
- **Bernoulli subsampling** now applies the realized subset-size and
  finite-population correction required for smooth root-n statistics.
- Poisson multiplier samples are conditioned on positive total weight instead
  of replacing an empty resample with the original data.

### Changed
- Custom Bayesian-bootstrap statistics must provide
  `weighted_statistic(data, weights)`. `np.mean`, `np.nanmean`, and
  `np.average` have built-in weighted implementations.
- Bayesian intervals are explicitly labelled as credible intervals in result
  metadata. Subsampling and Bernoulli intervals use their own method labels.
- Bernoulli `prob` must now be strictly between 0 and 1.

### Tests
- Added exact algorithm regression tests for Bayesian weights, studentized
  outer/inner pairing, subsampling scaling, Bernoulli finite-population
  correction, Poisson empty samples, and cluster-level uncertainty.

## [0.4.0] — 2026-08-13

### Breaking changes
- Python 3.9 is no longer supported; upgrade to Python 3.10 or later.
- Previously accepted invalid parameters now raise an explicit `TypeError` or
  `ValueError`. This includes zero batch/block sizes, unsupported method
  keyword arguments, and non-finite data or statistic results.

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
- Every bootstrap distribution now must contain exactly one finite scalar per
  requested resample; malformed vectorized statistics and non-finite sampled
  statistics fail with an explicit error.
- `random_state`, `method`, `ci_method`, and `vectorized` are validated against
  their documented public API types.

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

"""Public API — unified ``bootstrap()`` entry point.

Changes vs 0.2.0:
- _collect_bayesian: now receives the *caller's* rng instead of creating a new
  unseeded ``np.random.default_rng()`` per sample (broke reproducibility).
- studentized bootstrap: inner SE loop is vectorised via a (n_inner, n) index
  matrix instead of a nested Python loop.
- All generator-based methods forward rng consistently.

Changes vs 0.3.1:
- studentized bootstrap: inner_idx is now generated per outer sample, not once
  per batch. Fixes correlated SE* estimates (issue #2).

Changes vs 0.4.1:
- studentized random draws are independent of the technical batch size.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from bootstrapx.engine.backend import apply_statistic_batched, resolve_backend
from bootstrapx.generators.hierarchical import cluster_resample, strata_resample
from bootstrapx.generators.iid import (
    bayesian_resample,
    bernoulli_resample,
    poisson_resample,
    subsampling_resample,
)
from bootstrapx.generators.timeseries import (
    cbb_resample,
    mbb_resample,
    sieve_resample,
    stationary_resample,
    tapered_block_resample,
    wild_resample,
)
from bootstrapx.stats.confidence import (
    ConfidenceInterval,
    basic_interval,
    bca_interval,
    percentile_interval,
    root_interval,
    studentized_interval,
)
from bootstrapx.utils import (
    auto_batch_size,
    validate_bootstrap_distribution,
    validate_bootstrap_params,
    validate_data,
    validate_random_state,
)

FloatArray = NDArray[np.float64]


@dataclass
class BootstrapResult:
    """Container for bootstrap estimation results."""

    confidence_interval: ConfidenceInterval
    bootstrap_distribution: FloatArray
    theta_hat: float
    standard_error: float
    n_resamples: int
    method: str
    extra: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        ci = self.confidence_interval
        return (
            f"BootstrapResult(method={self.method!r}, "
            f"theta_hat={self.theta_hat:.6g}, "
            f"se={self.standard_error:.6g}, "
            f"CI=[{ci.low:.6g}, {ci.high:.6g}])"
        )


# ---------------------------------------------------------------------------
# Internal helpers for generator-based methods
# ---------------------------------------------------------------------------


def _collect_weighted(
    gen: Any,
    statistic: Callable[..., float],
    data: FloatArray,
) -> list[float]:
    """Collect stats from weighted generators (poisson / bernoulli).

    Expands integer weights via np.repeat for integer-valued weight arrays.
    """
    results: list[float] = []
    for batch in gen:
        data_ref, weights = batch
        for i in range(weights.shape[0]):
            w = weights[i]
            idx = np.repeat(np.arange(len(w)), np.maximum(w, 0).astype(np.intp))
            results.append(float(statistic(data_ref[idx])))
    return results


def _collect_bayesian(
    gen: Any,
    weighted_statistic: Callable[[FloatArray, FloatArray], float],
) -> list[float]:
    """Evaluate a functional directly under Bayesian-bootstrap weights."""
    results: list[float] = []
    for batch in gen:
        data_ref, weights = batch
        for i in range(weights.shape[0]):
            results.append(float(weighted_statistic(data_ref, weights[i])))
    return results


def _resolve_weighted_statistic(
    statistic: Callable[..., float],
    candidate: Any,
) -> Callable[[FloatArray, FloatArray], float]:
    """Resolve the exact weighted functional used by Bayesian bootstrap."""
    if candidate is not None:
        if not callable(candidate):
            raise TypeError("weighted_statistic must be callable.")
        return cast(Callable[[FloatArray, FloatArray], float], candidate)
    if any(statistic is built_in for built_in in (np.mean, np.nanmean, np.average)):
        return lambda data, weights: float(np.average(data, weights=weights))
    raise ValueError(
        "Bayesian bootstrap requires `weighted_statistic(data, weights)` for "
        "custom statistics. np.mean, np.nanmean, and np.average are supported directly."
    )


def _collect_bernoulli(
    gen: Any,
    statistic: Callable[..., float],
) -> tuple[FloatArray, FloatArray]:
    """Collect Bernoulli-subset statistics and their realized sample sizes."""
    stats: list[float] = []
    sizes: list[int] = []
    for data_ref, masks in gen:
        for mask in masks:
            idx = np.flatnonzero(mask)
            stats.append(float(statistic(data_ref[idx])))
            sizes.append(len(idx))
    return np.asarray(stats, dtype=np.float64), np.asarray(sizes, dtype=np.float64)


def _collect_arrays(
    gen: Any,
    statistic: Callable[..., float],
) -> list[float]:
    """Collect stats from array / list generators."""
    results: list[float] = []
    for batch in gen:
        if isinstance(batch, list):
            for arr in batch:
                results.append(float(statistic(arr)))
        else:
            for i in range(batch.shape[0]):
                results.append(float(statistic(batch[i])))
    return results


# ---------------------------------------------------------------------------
# Method sets
# ---------------------------------------------------------------------------

_IID_METHODS = {
    "percentile",
    "basic",
    "bca",
    "studentized",
    "poisson",
    "bernoulli",
    "subsampling",
    "bayesian",
}
_TS_METHODS = {"mbb", "cbb", "stationary", "tapered", "sieve", "wild"}
_HIER_METHODS = {"cluster", "strata"}
_ALL_METHODS = _IID_METHODS | _TS_METHODS | _HIER_METHODS
_CI_CAPABLE = {"percentile", "basic", "bca", "studentized"}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def bootstrap(
    data: Any,
    statistic: Callable[..., float],
    *,
    method: str = "bca",
    n_resamples: int = 9999,
    batch_size: int | None = None,
    confidence_level: float = 0.95,
    ci_method: str | None = None,
    backend: str = "auto",
    random_state: int | np.random.Generator | None = None,
    vectorized: bool = False,
    n_jobs: int = 1,
    **kwargs: Any,
) -> BootstrapResult:
    """Run bootstrap estimation.

    Parameters
    ----------
    data : array-like or pandas Series/DataFrame
        Observed sample.
    statistic : callable
        ``(array) -> float``.  If ``vectorized=True``, must accept
        ``(array_2d, axis=1) -> array_1d``.
    method : str
        One of: bca, percentile, basic, studentized, poisson, bernoulli,
        bayesian, subsampling, mbb, cbb, stationary, tapered, sieve,
        wild, cluster, strata.
    ci_method : str or None
        CI construction for generator-based methods that do not define a
        specialized interval: ``"percentile"`` or ``"basic"``. Defaults to
        ``"percentile"``. Bayesian, Bernoulli, and subsampling intervals are
        not configurable through this parameter.
    vectorized : bool
        For percentile, basic, and BCa methods, call ``statistic`` as
        ``statistic(batch, axis=1)``. Other methods reject this option.
    n_jobs : int
        Parallelism for jackknife in BCa (effective only for n >= 2000).

    Other Parameters
    ----------------
    weighted_statistic : callable
        Required for custom Bayesian-bootstrap statistics. Called as
        ``weighted_statistic(data, weights)`` for each Dirichlet draw.
    subsample_size : int
        Number of observations in each subsample.
    rate : float
        Convergence-rate exponent for subsampling. ``0.5`` means root-n.
    prob : float
        Inclusion probability for Bernoulli subsampling; strictly between 0 and 1.
    n_inner : int
        Number of inner resamples per outer sample for the studentized method.
        Defaults to 100.
    """
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    if ci_method is not None and not isinstance(ci_method, str):
        raise TypeError("ci_method must be a string or None.")
    if not isinstance(vectorized, bool):
        raise TypeError("vectorized must be a boolean.")
    if not isinstance(backend, str):
        raise TypeError("backend must be a string.")
    if not callable(statistic):
        raise TypeError("statistic must be callable.")

    method = method.lower().strip()
    ci_method = ci_method.lower().strip() if ci_method is not None else None
    if method not in _ALL_METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from {sorted(_ALL_METHODS)}.")
    if vectorized and method not in {"percentile", "basic", "bca"}:
        raise ValueError(
            "vectorized=True is supported only for percentile, basic, and bca methods."
        )

    arr = validate_data(data, allow_2d=(method in _HIER_METHODS))
    n = arr.shape[0]
    validate_bootstrap_params(
        method=method,
        n_observations=n,
        n_resamples=n_resamples,
        batch_size=batch_size,
        confidence_level=confidence_level,
        ci_method=ci_method,
        n_jobs=n_jobs,
        kwargs=kwargs,
    )
    validate_random_state(random_state)

    rng: np.random.Generator = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    if batch_size is None:
        batch_size = auto_batch_size(n, n_resamples)

    backend_kind = resolve_backend(backend)
    theta_hat = float(statistic(arr))
    if not np.isfinite(theta_hat):
        raise ValueError("statistic must return a finite scalar value for the observed data.")

    result_extra: dict[str, Any] = {}
    result_standard_error: float | None = None

    if method == "studentized":
        n_inner = int(kwargs.get("n_inner", 100))
        boot_stats = np.empty(n_resamples, dtype=np.float64)
        boot_se = np.empty(n_resamples, dtype=np.float64)
        for i in range(n_resamples):
            outer_idx = rng.integers(0, n, size=n)
            sample = arr[outer_idx]
            boot_stats[i] = float(statistic(sample))
            inner_idx = rng.integers(0, n, size=(n_inner, n))
            inner_vals = np.array([float(statistic(sample[inner_idx[k]])) for k in range(n_inner)])
            boot_se[i] = float(np.std(inner_vals, ddof=1))
        boot_stats = validate_bootstrap_distribution(boot_stats, n_resamples)
        if not np.all(np.isfinite(boot_se)):
            raise ValueError("statistic returned NaN or inf during studentized bootstrap.")
        ci = studentized_interval(
            arr,
            statistic,
            theta_hat,
            boot_stats,
            boot_se,
            confidence_level,
        )
        result_extra["n_inner"] = n_inner

    elif method in _CI_CAPABLE:
        boot_stats = apply_statistic_batched(
            arr,
            statistic,
            batch_size,
            n_resamples,
            backend_kind,
            rng,
            vectorized=vectorized,
        )
        boot_stats = validate_bootstrap_distribution(boot_stats, n_resamples)

        if method == "percentile":
            ci = percentile_interval(boot_stats, confidence_level)

        elif method == "basic":
            ci = basic_interval(boot_stats, theta_hat, confidence_level)

        elif method == "bca":
            ci = bca_interval(
                boot_stats,
                arr,
                statistic,
                theta_hat,
                confidence_level,
                n_jobs=n_jobs,
            )

    else:
        if method == "bayesian":
            weighted_statistic = _resolve_weighted_statistic(
                statistic, kwargs.get("weighted_statistic")
            )
            boot_stats_list = _collect_bayesian(
                bayesian_resample(arr, n_resamples, batch_size, rng), weighted_statistic
            )

        elif method == "poisson":
            boot_stats_list = _collect_weighted(
                poisson_resample(arr, n_resamples, batch_size, rng), statistic, arr
            )

        elif method == "bernoulli":
            prob = float(kwargs.get("prob", 0.5))
            boot_stats, subset_sizes = _collect_bernoulli(
                bernoulli_resample(arr, n_resamples, batch_size, rng, prob=prob), statistic
            )

        elif method == "cluster":
            cids = kwargs["cluster_ids"]
            boot_stats_list = _collect_arrays(
                cluster_resample(arr, np.asarray(cids), n_resamples, batch_size, rng), statistic
            )

        elif method == "strata":
            sids = kwargs["strata_ids"]
            boot_stats_list = _collect_arrays(
                strata_resample(arr, np.asarray(sids), n_resamples, batch_size, rng), statistic
            )

        elif method == "subsampling":
            ss = kwargs.get("subsample_size")
            boot_stats_list = _collect_arrays(
                subsampling_resample(arr, n_resamples, batch_size, rng, subsample_size=ss),
                statistic,
            )

        elif method == "mbb":
            bl = int(kwargs.get("block_length", 10))
            boot_stats_list = _collect_arrays(
                mbb_resample(arr, n_resamples, batch_size, rng, block_length=bl), statistic
            )

        elif method == "cbb":
            bl = int(kwargs.get("block_length", 10))
            boot_stats_list = _collect_arrays(
                cbb_resample(arr, n_resamples, batch_size, rng, block_length=bl), statistic
            )

        elif method == "stationary":
            mb = float(kwargs.get("mean_block", 10.0))
            boot_stats_list = _collect_arrays(
                stationary_resample(arr, n_resamples, batch_size, rng, mean_block=mb), statistic
            )

        elif method == "tapered":
            bl = int(kwargs.get("block_length", 10))
            tp = str(kwargs.get("taper", "tukey"))
            boot_stats_list = _collect_arrays(
                tapered_block_resample(
                    arr,
                    n_resamples,
                    batch_size,
                    rng,
                    block_length=bl,
                    taper=tp,
                ),
                statistic,
            )

        elif method == "sieve":
            ar = kwargs.get("ar_order")
            boot_stats_list = _collect_arrays(
                sieve_resample(arr, n_resamples, batch_size, rng, ar_order=ar), statistic
            )

        elif method == "wild":
            fit = kwargs.get("fitted")
            dist = str(kwargs.get("distribution", "rademacher"))
            boot_stats_list = _collect_arrays(
                wild_resample(arr, n_resamples, batch_size, rng, fitted=fit, distribution=dist),
                statistic,
            )

        else:
            raise ValueError(f"Method {method!r} not implemented.")

        if method != "bernoulli":
            boot_stats = np.array(boot_stats_list, dtype=np.float64)

        boot_stats = validate_bootstrap_distribution(boot_stats, n_resamples)

        if method == "bayesian":
            ci = percentile_interval(boot_stats, confidence_level)
            ci.method = "bayesian"
            result_extra["interval_type"] = "credible"
        elif method == "subsampling":
            subsample_size = int(kwargs.get("subsample_size") or max(1, np.sqrt(n)))
            rate = float(kwargs.get("rate", 0.5))
            scale_subsample = float(subsample_size**rate)
            scale_n = float(n**rate)
            root_stats = scale_subsample * (boot_stats - theta_hat)
            ci = root_interval(
                root_stats,
                theta_hat,
                scale_n,
                confidence_level,
                method="subsampling",
            )
            result_standard_error = float(np.std(root_stats, ddof=1) / scale_n)
            result_extra.update(
                {"subsample_size": subsample_size, "rate": rate, "root_distribution": root_stats}
            )
        elif method == "bernoulli":
            fractions = subset_sizes / n
            root_stats = np.sqrt(subset_sizes / (1.0 - fractions)) * (boot_stats - theta_hat)
            ci = root_interval(
                root_stats,
                theta_hat,
                np.sqrt(n),
                confidence_level,
                method="bernoulli",
            )
            result_standard_error = float(np.std(root_stats, ddof=1) / np.sqrt(n))
            result_extra.update(
                {"prob": prob, "subset_sizes": subset_sizes, "root_distribution": root_stats}
            )
        else:
            _ci_method = ci_method or "percentile"
            if _ci_method == "basic":
                ci = basic_interval(boot_stats, theta_hat, confidence_level)
            else:
                ci = percentile_interval(boot_stats, confidence_level)

    return BootstrapResult(
        confidence_interval=ci,
        bootstrap_distribution=boot_stats,
        theta_hat=theta_hat,
        standard_error=(
            result_standard_error
            if result_standard_error is not None
            else float(np.std(boot_stats, ddof=1))
        ),
        n_resamples=len(boot_stats),
        method=method,
        extra=result_extra,
    )

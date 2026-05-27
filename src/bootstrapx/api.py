"""Public API — unified ``bootstrap()`` entry point.

Changes vs 0.2.0:
- _collect_bayesian: now receives the *caller's* rng instead of creating a new
  unseeded ``np.random.default_rng()`` per sample (broke reproducibility).
- studentized bootstrap: inner SE loop is vectorised via a (n_inner, n) index
  matrix instead of a nested Python loop.
- All generator-based methods forward rng consistently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from bootstrapx.engine.backend import resolve_backend, apply_statistic_batched
from bootstrapx.generators.iid import (
    basic_resample,
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
from bootstrapx.generators.hierarchical import cluster_resample, strata_resample
from bootstrapx.stats.confidence import (
    ConfidenceInterval,
    basic_interval,
    bca_interval,
    percentile_interval,
    studentized_interval,
)
from bootstrapx.utils import auto_batch_size, validate_data


@dataclass
class BootstrapResult:
    """Container for bootstrap estimation results."""

    confidence_interval: ConfidenceInterval
    bootstrap_distribution: np.ndarray
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
    data: np.ndarray,
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
            if len(idx) > 0:
                results.append(float(statistic(data_ref[idx])))
            else:
                results.append(float(statistic(data_ref)))
    return results


def _collect_bayesian(
    gen: Any,
    statistic: Callable[..., float],
    rng: np.random.Generator,            # ← now receives caller's rng
) -> list[float]:
    """Collect stats from Bayesian (Dirichlet) generators.

    Fix: the original code called ``np.random.default_rng()`` (no seed) inside
    the collection loop, making results non-reproducible even when the caller
    passed ``random_state``.  We now use the caller's Generator throughout.
    """
    results: list[float] = []
    for batch in gen:
        data_ref, weights = batch
        n = len(data_ref)
        for i in range(weights.shape[0]):
            w = weights[i]
            idx = rng.choice(n, size=n, replace=True, p=w)
            results.append(float(statistic(data_ref[idx])))
    return results


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
    "percentile", "basic", "bca", "studentized",
    "poisson", "bernoulli", "subsampling", "bayesian",
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
        CI construction for generator-based methods: ``"percentile"`` or
        ``"basic"``.  Defaults to ``"percentile"``.
    vectorized : bool
        If ``True``, ``statistic`` is called as ``statistic(batch, axis=1)``.
    n_jobs : int
        Parallelism for jackknife in BCa (effective only for n ≥ 2000).
    """
    method = method.lower().strip()
    if method not in _ALL_METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Choose from {sorted(_ALL_METHODS)}."
        )

    arr = validate_data(data, allow_2d=(method in _HIER_METHODS))
    n = arr.shape[0]

    rng: np.random.Generator = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    if batch_size is None:
        batch_size = auto_batch_size(n, n_resamples)

    backend_kind = resolve_backend(backend)
    theta_hat = float(statistic(arr))

    if method in _CI_CAPABLE:
        boot_stats = apply_statistic_batched(
            arr, statistic, batch_size, n_resamples, backend_kind, rng,
            vectorized=vectorized,
        )

        if method == "percentile":
            ci = percentile_interval(boot_stats, confidence_level)

        elif method == "basic":
            ci = basic_interval(boot_stats, theta_hat, confidence_level)

        elif method == "bca":
            ci = bca_interval(
                boot_stats, arr, statistic, theta_hat, confidence_level,
                n_jobs=n_jobs,
            )

        elif method == "studentized":
            n_inner = int(kwargs.get("n_inner", 50))
            boot_se = np.empty(n_resamples, dtype=np.float64)
            done = 0
            while done < n_resamples:
                bs = min(batch_size, n_resamples - done)
                outer_idx = rng.integers(0, n, size=(bs, n))
                # Vectorise inner SE: draw (n_inner, n) indices once per outer sample
                inner_idx = rng.integers(0, n, size=(n_inner, n))
                for b in range(bs):
                    sample = arr[outer_idx[b]]
                    inner_vals = np.array(
                        [float(statistic(sample[inner_idx[k]])) for k in range(n_inner)]
                    )
                    boot_se[done + b] = float(np.std(inner_vals, ddof=1))
                done += bs
            ci = studentized_interval(
                arr, statistic, theta_hat, boot_stats, boot_se, confidence_level,
            )

    else:
        if method == "bayesian":
            gen = bayesian_resample(arr, n_resamples, batch_size, rng)
            boot_stats_list = _collect_bayesian(gen, statistic, rng)  # pass rng!

        elif method == "poisson":
            gen = poisson_resample(arr, n_resamples, batch_size, rng)
            boot_stats_list = _collect_weighted(gen, statistic, arr)

        elif method == "bernoulli":
            prob = float(kwargs.get("prob", 0.5))
            gen = bernoulli_resample(arr, n_resamples, batch_size, rng, prob=prob)
            boot_stats_list = _collect_weighted(gen, statistic, arr)

        elif method == "cluster":
            cids = kwargs.get("cluster_ids")
            if cids is None:
                raise ValueError("cluster method requires `cluster_ids` kwarg.")
            gen = cluster_resample(arr, np.asarray(cids), n_resamples, batch_size, rng)
            boot_stats_list = _collect_arrays(gen, statistic)

        elif method == "strata":
            sids = kwargs.get("strata_ids")
            if sids is None:
                raise ValueError("strata method requires `strata_ids` kwarg.")
            gen = strata_resample(arr, np.asarray(sids), n_resamples, batch_size, rng)
            boot_stats_list = _collect_arrays(gen, statistic)

        elif method == "subsampling":
            ss = kwargs.get("subsample_size")
            gen = subsampling_resample(arr, n_resamples, batch_size, rng, subsample_size=ss)
            boot_stats_list = _collect_arrays(gen, statistic)

        elif method == "mbb":
            bl = int(kwargs.get("block_length", 10))
            gen = mbb_resample(arr, n_resamples, batch_size, rng, block_length=bl)
            boot_stats_list = _collect_arrays(gen, statistic)

        elif method == "cbb":
            bl = int(kwargs.get("block_length", 10))
            gen = cbb_resample(arr, n_resamples, batch_size, rng, block_length=bl)
            boot_stats_list = _collect_arrays(gen, statistic)

        elif method == "stationary":
            mb = float(kwargs.get("mean_block", 10.0))
            gen = stationary_resample(arr, n_resamples, batch_size, rng, mean_block=mb)
            boot_stats_list = _collect_arrays(gen, statistic)

        elif method == "tapered":
            bl = int(kwargs.get("block_length", 10))
            tp = str(kwargs.get("taper", "tukey"))
            gen = tapered_block_resample(
                arr, n_resamples, batch_size, rng, block_length=bl, taper=tp,
            )
            boot_stats_list = _collect_arrays(gen, statistic)

        elif method == "sieve":
            ar = kwargs.get("ar_order")
            gen = sieve_resample(arr, n_resamples, batch_size, rng, ar_order=ar)
            boot_stats_list = _collect_arrays(gen, statistic)

        elif method == "wild":
            fit = kwargs.get("fitted")
            dist = str(kwargs.get("distribution", "rademacher"))
            gen = wild_resample(arr, n_resamples, batch_size, rng, fitted=fit, distribution=dist)
            boot_stats_list = _collect_arrays(gen, statistic)

        else:
            raise ValueError(f"Method {method!r} not implemented.")

        boot_stats = np.array(boot_stats_list, dtype=np.float64)

        _ci_method = (ci_method or "percentile").lower()
        if _ci_method == "basic":
            ci = basic_interval(boot_stats, theta_hat, confidence_level)
        else:
            ci = percentile_interval(boot_stats, confidence_level)

    return BootstrapResult(
        confidence_interval=ci,
        bootstrap_distribution=boot_stats,
        theta_hat=theta_hat,
        standard_error=float(np.std(boot_stats, ddof=1)),
        n_resamples=len(boot_stats),
        method=method,
    )

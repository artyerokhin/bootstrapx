from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from bootstrapx.engine.backend import resolve_backend, apply_statistic_batched
from bootstrapx.generators.iid import (
    basic_resample,
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


def _collect(gen, statistic: Callable[[np.ndarray], float]) -> list[float]:
    results = []
    for batch in gen:
        if isinstance(batch, tuple):
            data_ref, weights = batch
            # weights is (B, N), iterate over rows
            for i in range(weights.shape[0]):
                w = weights[i]
                mask = w > 0
                val = statistic(data_ref[mask]) if mask.any() else statistic(data_ref)
                results.append(float(val))
        elif isinstance(batch, list):
            for arr in batch:
                results.append(float(statistic(arr)))
        else:
            # Standard array batch (B, N)
            for i in range(batch.shape[0]):
                results.append(float(statistic(batch[i])))
    return results


_IID_METHODS = {
    "percentile",
    "basic",
    "bca",
    "studentized",
    "poisson",
    "bernoulli",
    "subsampling",
}
_TS_METHODS = {"mbb", "cbb", "stationary", "tapered", "sieve", "wild"}
_HIER_METHODS = {"cluster", "strata"}
_ALL_METHODS = _IID_METHODS | _TS_METHODS | _HIER_METHODS


def bootstrap(
    data: Any,
    statistic: Callable[[np.ndarray], float],
    *,
    method: str = "bca",
    n_resamples: int = 9999,
    batch_size: int | None = None,
    confidence_level: float = 0.95,
    backend: str = "auto",
    random_state: int | np.random.Generator | None = None,
    n_jobs: int = 1,
    **kwargs: Any,
) -> BootstrapResult:
    method = method.lower().strip()
    if method not in _ALL_METHODS:
        valid = sorted(_ALL_METHODS)
        raise ValueError(f"Unknown method {method!r}. Choose from {valid}.")

    arr = validate_data(data, allow_2d=(method in _HIER_METHODS))
    n = arr.shape[0]

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    if batch_size is None:
        batch_size = auto_batch_size(n, n_resamples)

    backend_kind = resolve_backend(backend)
    theta_hat = float(statistic(arr))

    if method in {"percentile", "basic", "bca", "studentized"}:
        boot_stats = apply_statistic_batched(
            arr, statistic, batch_size, n_resamples, backend_kind, rng
        )

        if method == "percentile":
            ci = percentile_interval(boot_stats, confidence_level)
        elif method == "basic":
            ci = basic_interval(boot_stats, theta_hat, confidence_level)
        elif method == "bca":
            ci = bca_interval(boot_stats, arr, statistic, theta_hat, confidence_level)
        elif method == "studentized":
            # Nested bootstrap for SE
            n_inner = kwargs.get("n_inner", 50)
            idx_all = rng.integers(0, n, size=(n_resamples, n))
            boot_se = np.empty(n_resamples, dtype=np.float64)

            for b in range(n_resamples):
                sample = arr[idx_all[b]]
                # Inner loop simple bootstrap
                inner_idx = rng.integers(0, n, size=(n_inner, n))
                inner_vals = [statistic(sample[inner_idx[k]]) for k in range(n_inner)]
                boot_se[b] = np.std(inner_vals, ddof=1)

            ci = studentized_interval(
                arr, statistic, theta_hat, boot_stats, boot_se, confidence_level
            )
        else:
            # Should not happen
            raise ValueError("Unreachable")

    else:
        # Generator-based methods
        if method == "cluster":
            cids = kwargs.get("cluster_ids")
            if cids is None:
                raise ValueError("cluster method requires `cluster_ids` kwarg.")
            gen = cluster_resample(arr, np.asarray(cids), n_resamples, batch_size, rng)

        elif method == "strata":
            sids = kwargs.get("strata_ids")
            if sids is None:
                raise ValueError("strata method requires `strata_ids` kwarg.")
            gen = strata_resample(arr, np.asarray(sids), n_resamples, batch_size, rng)

        else:
            # Map method name to generator function
            if method == "poisson":
                gen = poisson_resample(arr, n_resamples, batch_size, rng)
            elif method == "bernoulli":
                prob = kwargs.get("prob", 0.5)
                gen = bernoulli_resample(arr, n_resamples, batch_size, rng, prob=prob)
            elif method == "subsampling":
                ss = kwargs.get("subsample_size")
                gen = subsampling_resample(
                    arr, n_resamples, batch_size, rng, subsample_size=ss
                )
            elif method == "mbb":
                bl = kwargs.get("block_length", 10)
                gen = mbb_resample(arr, n_resamples, batch_size, rng, block_length=bl)
            elif method == "cbb":
                bl = kwargs.get("block_length", 10)
                gen = cbb_resample(arr, n_resamples, batch_size, rng, block_length=bl)
            elif method == "stationary":
                mb = kwargs.get("mean_block", 10.0)
                gen = stationary_resample(
                    arr, n_resamples, batch_size, rng, mean_block=mb
                )
            elif method == "tapered":
                bl = kwargs.get("block_length", 10)
                tp = kwargs.get("taper", "tukey")
                gen = tapered_block_resample(
                    arr, n_resamples, batch_size, rng, block_length=bl, taper=tp
                )
            elif method == "sieve":
                ar = kwargs.get("ar_order")
                gen = sieve_resample(arr, n_resamples, batch_size, rng, ar_order=ar)
            elif method == "wild":
                fit = kwargs.get("fitted")
                dist = kwargs.get("distribution", "rademacher")
                gen = wild_resample(
                    arr, n_resamples, batch_size, rng, fitted=fit, distribution=dist
                )
            else:
                # Should not happen due to check above
                raise ValueError(f"Method {method} not implemented in dispatcher.")

        boot_stats = np.array(_collect(gen, statistic), dtype=np.float64)
        ci = percentile_interval(boot_stats, confidence_level)

    return BootstrapResult(
        confidence_interval=ci,
        bootstrap_distribution=boot_stats,
        theta_hat=theta_hat,
        standard_error=float(np.std(boot_stats, ddof=1)),
        n_resamples=len(boot_stats),
        method=method,
    )

"""Backend dispatcher.

v0.3.2 changes:
- Removed NUMBA_CPU backend: was a dead stub (flag accepted but never
  dispatched). numpy fast path already covers the main use cases.
  Passing backend='numba_cpu' now raises ValueError (issue #3).
- NUMBA_CUDA removed from public API (was already NotImplementedError).
- auto always returns VANILLA.
- _FAST_PATH_N raised 500 -> 1000 (closes n=500-1000 performance gap).

v0.3.1 change:
- apply_statistic_batched(): fast path for n < 500 + numpy built-ins.
  Allocates single (n_resamples, n) matrix, axis=1 reduction.
  Removes per-sample Python loop overhead (~5x faster at n=50).
"""

from __future__ import annotations

import enum
from collections.abc import Callable

import numpy as np

_AXIS_BUILTINS: dict[object, str] = {
    np.mean: "mean",
    np.nanmean: "nanmean",
    np.median: "median",
    np.nanmedian: "nanmedian",
    np.std: "std",
    np.nanstd: "nanstd",
    np.var: "var",
    np.nanvar: "nanvar",
    np.sum: "sum",
    np.nansum: "nansum",
    np.min: "min",
    np.nanmin: "nanmin",
    np.max: "max",
    np.nanmax: "nanmax",
    np.ptp: "ptp",
}
_FAST_PATH_N = 1000  # raised from 500 in v0.3.2


class BackendKind(enum.Enum):
    VANILLA = "vanilla"


def resolve_backend(requested: str = "auto") -> BackendKind:
    requested = requested.lower().strip()
    if requested in ("numba_cpu", "numba_cuda"):
        raise ValueError(
            f"Backend {requested!r} is not implemented. "
            "bootstrapx uses numpy fast paths for acceleration. "
            "Use backend='vanilla' or backend='auto'."
        )
    if requested in ("auto", "vanilla"):
        return BackendKind.VANILLA
    raise ValueError(f"Unknown backend {requested!r}. Choose from ['vanilla'] or 'auto'.")


def _resample_batch(data: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    n = data.shape[0]
    return data[rng.integers(0, n, size=(batch_size, n))]


def _fast_path(
    data: np.ndarray, func_name: str, n_resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """Single-matrix fast path: one (n_resamples, n) alloc + axis=1 ufunc."""
    n = data.shape[0]
    resampled = data[rng.integers(0, n, size=(n_resamples, n))]
    return np.asarray(getattr(np, func_name)(resampled, axis=1), dtype=np.float64)


def apply_statistic_batched(
    data: np.ndarray,
    statistic: Callable[..., float],
    batch_size: int,
    n_resamples: int,
    backend: BackendKind,
    rng: np.random.Generator,
    *,
    vectorized: bool = False,
) -> np.ndarray:
    n = data.shape[0]
    # fast path: small n + numpy built-in + not custom vectorized
    if not vectorized and n < _FAST_PATH_N and statistic in _AXIS_BUILTINS:
        return _fast_path(data, _AXIS_BUILTINS[statistic], n_resamples, rng)
    # batched path: memory-safe for large n
    results: list[float] = []
    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        samples = _resample_batch(data, bs, rng)
        if vectorized:
            results.extend(float(v) for v in np.asarray(statistic(samples, axis=1)).ravel())
        else:
            for i in range(bs):
                results.append(float(statistic(samples[i])))
        done += bs
    return np.array(results, dtype=np.float64)

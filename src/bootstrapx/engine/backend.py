"""Backend dispatcher — v0.3.1 fast path for small samples.

Compatibility:
- BackendKind.NUMBA_CUDA retained for API compatibility only.
- Explicitly raises NotImplementedError at resolve time.
- auto never returns NUMBA_CUDA.

v0.3.1 change:
- apply_statistic_batched(): fast path for n < 500 + numpy built-ins.
  Allocates single (n_resamples, n) matrix, axis=1 reduction.
  Removes per-sample Python loop overhead (~5x faster at n=50).
"""
from __future__ import annotations

import enum
import warnings
from typing import Callable

import numpy as np

_AXIS_BUILTINS: dict[object, str] = {
    np.mean:    "mean",    np.nanmean:    "nanmean",
    np.median:  "median",  np.nanmedian:  "nanmedian",
    np.std:     "std",     np.nanstd:     "nanstd",
    np.var:     "var",     np.nanvar:     "nanvar",
    np.sum:     "sum",     np.nansum:     "nansum",
    np.min:     "min",     np.nanmin:     "nanmin",
    np.max:     "max",     np.nanmax:     "nanmax",
    np.ptp:     "ptp",
}
_FAST_PATH_N = 500


class BackendKind(enum.Enum):
    NUMBA_CPU  = "numba_cpu"
    NUMBA_CUDA = "numba_cuda"
    VANILLA    = "vanilla"


def _numba_available() -> bool:
    try:
        import numba  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_backend(requested: str = "auto") -> BackendKind:
    requested = requested.lower().strip()
    if requested == "numba_cuda":
        raise NotImplementedError(
            "CUDA backend retained for compatibility but not implemented. "
            "Use backend='numba_cpu' or backend='vanilla'."
        )
    if requested == "auto":
        return BackendKind.NUMBA_CPU if _numba_available() else BackendKind.VANILLA
    mapping = {"numba_cpu": BackendKind.NUMBA_CPU, "vanilla": BackendKind.VANILLA}
    if requested not in mapping:
        raise ValueError(
            f"Unknown backend {requested!r}. "
            "Choose from ['numba_cpu', 'vanilla'] or 'auto'."
        )
    kind = mapping[requested]
    if kind is BackendKind.NUMBA_CPU and not _numba_available():
        warnings.warn(
            "numba_cpu requested but numba not installed; falling back to vanilla.",
            RuntimeWarning, stacklevel=3,
        )
        return BackendKind.VANILLA
    return kind


def _resample_batch(data: np.ndarray, batch_size: int,
                    rng: np.random.Generator) -> np.ndarray:
    n = data.shape[0]
    return data[rng.integers(0, n, size=(batch_size, n))]


def _fast_path(data: np.ndarray, func_name: str,
               n_resamples: int, rng: np.random.Generator) -> np.ndarray:
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
            results.extend(float(v) for v in np.asarray(
                statistic(samples, axis=1)).ravel())
        else:
            for i in range(bs):
                results.append(float(statistic(samples[i])))
        done += bs
    return np.array(results, dtype=np.float64)

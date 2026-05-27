"""Backend dispatcher.

Compatibility hotfix:
- Keep BackendKind.NUMBA_CUDA symbol for test and API compatibility.
- Explicitly reject numba_cuda at resolve time with NotImplementedError.
- auto never returns NUMBA_CUDA.
"""
from __future__ import annotations

import enum
import warnings
from typing import Callable

import numpy as np


class BackendKind(enum.Enum):
    NUMBA_CPU = "numba_cpu"
    NUMBA_CUDA = "numba_cuda"  # kept for backward compatibility only
    VANILLA = "vanilla"


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
            "CUDA backend symbol is retained for compatibility, but GPU execution "
            "is not implemented. Use backend='numba_cpu' or backend='vanilla'."
        )

    if requested == "auto":
        return BackendKind.NUMBA_CPU if _numba_available() else BackendKind.VANILLA

    mapping = {
        "numba_cpu": BackendKind.NUMBA_CPU,
        "vanilla": BackendKind.VANILLA,
    }
    if requested not in mapping:
        raise ValueError(
            f"Unknown backend {requested!r}. Choose from ['numba_cpu', 'vanilla'] or 'auto'."
        )
    kind = mapping[requested]
    if kind is BackendKind.NUMBA_CPU and not _numba_available():
        warnings.warn(
            "numba_cpu requested but numba is not installed; falling back to vanilla.",
            RuntimeWarning,
            stacklevel=3,
        )
        return BackendKind.VANILLA
    return kind


def _resample_batch(data: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    n = data.shape[0]
    idx = rng.integers(0, n, size=(batch_size, n))
    return data[idx]


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
    results: list[float] = []
    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        samples = _resample_batch(data, bs, rng)
        if vectorized:
            batch_results = statistic(samples, axis=1)
            results.extend(float(v) for v in np.asarray(batch_results).ravel())
        else:
            for i in range(bs):
                results.append(float(statistic(samples[i])))
        done += bs
    return np.array(results, dtype=np.float64)

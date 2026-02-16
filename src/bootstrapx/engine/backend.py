"""Backend dispatcher with Numba acceleration."""
from __future__ import annotations

import enum
import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class BackendKind(enum.Enum):
    NUMBA_CPU = "numba_cpu"
    NUMBA_CUDA = "numba_cuda"
    VANILLA = "vanilla"


def _cuda_available() -> bool:
    try:
        from numba import cuda

        return bool(cuda.is_available())
    except Exception:
        return False


def _numba_available() -> bool:
    try:
        import numba  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_backend(requested: str = "auto") -> BackendKind:
    requested = requested.lower().strip()
    if requested == "auto":
        if _cuda_available():
            return BackendKind.NUMBA_CUDA
        elif _numba_available():
            return BackendKind.NUMBA_CPU
        else:
            return BackendKind.VANILLA

    mapping = {
        "numba_cpu": BackendKind.NUMBA_CPU,
        "numba_cuda": BackendKind.NUMBA_CUDA,
        "vanilla": BackendKind.VANILLA,
    }

    if requested not in mapping:
        valid = list(mapping.keys())
        raise ValueError(f"Unknown backend {requested!r}. Choose from {valid}.")

    kind = mapping[requested]
    if kind is BackendKind.NUMBA_CUDA and not _cuda_available():
        raise RuntimeError("CUDA backend requested but no GPU found.")
    return kind


try:
    from numba import njit, prange

    @njit(cache=True, parallel=True)
    def _batch_indices_numba(n: int, batch_size: int, seed_base: int) -> np.ndarray:
        out = np.empty((batch_size, n), dtype=np.int64)
        for b in prange(batch_size):
            np.random.seed(seed_base + b)
            for j in range(n):
                out[b, j] = np.random.randint(0, n)
        return out

    _HAS_NUMBA = True
except ImportError:
    # Dummy for type checking
    def _batch_indices_numba(n: int, batch_size: int, seed_base: int) -> np.ndarray:  # type: ignore
        return np.empty(0)

    _HAS_NUMBA = False


def resample_batch_vanilla(
    data: np.ndarray, batch_size: int, rng: np.random.Generator
) -> np.ndarray:
    n = data.shape[0]
    return data[rng.integers(0, n, size=(batch_size, n))]


def apply_statistic_batched(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    batch_size: int,
    n_resamples: int,
    backend: BackendKind,
    rng: np.random.Generator,
) -> np.ndarray:
    n = data.shape[0]
    results = []
    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        if backend is BackendKind.NUMBA_CPU and _HAS_NUMBA:
            # Cast explicit types for Numba signature if needed, or just pass int
            idx = _batch_indices_numba(n, bs, int(rng.integers(0, 2**31)))
            samples = data[idx]
        else:
            samples = resample_batch_vanilla(data, bs, rng)

        for i in range(bs):
            results.append(float(statistic(samples[i])))
        done += bs
    return np.array(results, dtype=np.float64)

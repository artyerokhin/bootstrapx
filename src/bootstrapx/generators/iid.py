from __future__ import annotations

from typing import Generator

import numpy as np


def basic_resample(
    data: np.ndarray, n_resamples: int, batch_size: int, rng: np.random.Generator
) -> Generator[np.ndarray, None, None]:
    n = data.shape[0]
    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        yield data[rng.integers(0, n, size=(bs, n))]
        done += bs


def poisson_resample(
    data: np.ndarray, n_resamples: int, batch_size: int, rng: np.random.Generator
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    n = data.shape[0]
    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        yield data, rng.poisson(1.0, size=(bs, n)).astype(np.float64)
        done += bs


def bernoulli_resample(
    data: np.ndarray,
    n_resamples: int,
    batch_size: int,
    rng: np.random.Generator,
    prob: float = 0.5,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    n = data.shape[0]
    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        yield data, (rng.random(size=(bs, n)) < prob).astype(np.float64)
        done += bs


def subsampling_resample(
    data: np.ndarray,
    n_resamples: int,
    batch_size: int,
    rng: np.random.Generator,
    subsample_size: int | None = None,
) -> Generator[np.ndarray, None, None]:
    n = data.shape[0]
    m = subsample_size or max(1, int(np.sqrt(n)))
    if m >= n:
        raise ValueError(f"subsample_size={m} must be < n={n}.")
    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        batch = np.empty((bs, m), dtype=data.dtype)
        for i in range(bs):
            batch[i] = data[rng.choice(n, size=m, replace=False)]
        yield batch
        done += bs

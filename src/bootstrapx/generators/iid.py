"""IID bootstrap generators."""

from __future__ import annotations

from collections.abc import Generator

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
    """Generate Poisson(1) multiplier weights conditional on positive total weight."""
    n = data.shape[0]
    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        weights = rng.poisson(1.0, size=(bs, n))
        empty = weights.sum(axis=1) == 0
        while np.any(empty):
            weights[empty] = rng.poisson(1.0, size=(int(empty.sum()), n))
            empty = weights.sum(axis=1) == 0
        yield data, weights.astype(np.float64)
        done += bs


def bernoulli_resample(
    data: np.ndarray,
    n_resamples: int,
    batch_size: int,
    rng: np.random.Generator,
    prob: float = 0.5,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Generate Bernoulli subsets conditional on being nonempty and nonfull."""
    n = data.shape[0]
    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        masks = rng.random(size=(bs, n)) < prob
        invalid = (masks.sum(axis=1) == 0) | (masks.sum(axis=1) == n)
        attempts = 0
        while np.any(invalid):
            masks[invalid] = rng.random(size=(int(invalid.sum()), n)) < prob
            invalid = (masks.sum(axis=1) == 0) | (masks.sum(axis=1) == n)
            attempts += 1
            if attempts >= 1000:
                raise ValueError(
                    "prob is too close to 0 or 1 to generate nondegenerate Bernoulli subsets."
                )
        yield data, masks.astype(np.float64)
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


def bayesian_resample(
    data: np.ndarray, n_resamples: int, batch_size: int, rng: np.random.Generator
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Bayesian bootstrap: Dirichlet(1,...,1) weights (Rubin, 1981)."""
    n = data.shape[0]
    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        weights = rng.dirichlet(np.ones(n), size=bs)
        yield data, weights
        done += bs

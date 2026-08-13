"""Hierarchical (cluster & stratified) bootstrap generators."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
AnyArray = NDArray[Any]


def cluster_resample(
    data: FloatArray,
    cluster_ids: AnyArray,
    n_resamples: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Generator[list[FloatArray], None, None]:
    unique = np.unique(cluster_ids)
    nc = len(unique)
    cmap = {c: np.where(cluster_ids == c)[0] for c in unique}
    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        batch: list[FloatArray] = []
        for _ in range(bs):
            chosen = rng.choice(unique, size=nc, replace=True)
            batch.append(data[np.concatenate([cmap[c] for c in chosen])])
        yield batch
        done += bs


def strata_resample(
    data: FloatArray,
    strata_ids: AnyArray,
    n_resamples: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Generator[list[FloatArray], None, None]:
    unique = np.unique(strata_ids)
    smap = {s: np.where(strata_ids == s)[0] for s in unique}
    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        batch: list[FloatArray] = []
        for _ in range(bs):
            parts = [data[rng.choice(smap[s], size=len(smap[s]), replace=True)] for s in unique]
            batch.append(np.concatenate(parts))
        yield batch
        done += bs

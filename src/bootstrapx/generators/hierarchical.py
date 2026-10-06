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


def cluster_strata_resample(
    data: FloatArray,
    cluster_ids: AnyArray,
    strata_ids: AnyArray,
    n_resamples: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Generator[list[FloatArray], None, None]:
    """Draw complete clusters within each fixed stratum.

    The public API validates nesting and at least two clusters per stratum
    before entering this generator. Draws are independent of batch size.
    """
    _, cluster_codes = np.unique(np.asarray(cluster_ids, dtype=object), return_inverse=True)
    _, stratum_codes = np.unique(np.asarray(strata_ids, dtype=object), return_inverse=True)
    cluster_order = np.argsort(cluster_codes, kind="stable")
    cluster_counts = np.bincount(cluster_codes)
    groups = np.split(cluster_order, np.cumsum(cluster_counts)[:-1])
    maps: list[list[NDArray[np.intp]]] = [[] for _ in range(int(stratum_codes.max()) + 1)]
    for rows in groups:
        maps[int(stratum_codes[rows[0]])].append(rows)

    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        batch: list[FloatArray] = []
        for _ in range(bs):
            selected_parts = [
                rows[index] for rows in maps for index in rng.integers(0, len(rows), size=len(rows))
            ]
            batch.append(data[np.concatenate(selected_parts)])
        yield batch
        done += bs

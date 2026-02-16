from __future__ import annotations

import numpy as np


def cluster_resample(data, cluster_ids, n_resamples, batch_size, rng):
    unique = np.unique(cluster_ids)
    nc = len(unique)
    cmap = {int(c): np.where(cluster_ids == c)[0] for c in unique}
    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        batch = []
        for _ in range(bs):
            chosen = rng.choice(unique, size=nc, replace=True)
            batch.append(data[np.concatenate([cmap[int(c)] for c in chosen])])
        yield batch
        done += bs


def strata_resample(data, strata_ids, n_resamples, batch_size, rng):
    unique = np.unique(strata_ids)
    smap = {int(s): np.where(strata_ids == s)[0] for s in unique}
    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        batch = []
        for _ in range(bs):
            parts = [
                data[rng.choice(smap[int(s)], size=len(smap[int(s)]), replace=True)]
                for s in unique
            ]
            batch.append(np.concatenate(parts))
        yield batch
        done += bs

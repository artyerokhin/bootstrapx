"""Time-series bootstrap generators with optional Numba acceleration."""
from __future__ import annotations

import numpy as np

try:
    from numba import njit

    @njit(cache=True)
    def _mbb_idx(n, bl, seed):
        np.random.seed(seed)
        out = np.empty(n, dtype=np.int64)
        pos = 0
        ns = n - bl + 1
        while pos < n:
            s = np.random.randint(0, ns)
            for j in range(bl):
                if pos >= n:
                    break
                out[pos] = s + j
                pos += 1
        return out

    @njit(cache=True)
    def _cbb_idx(n, bl, seed):
        np.random.seed(seed)
        out = np.empty(n, dtype=np.int64)
        pos = 0
        while pos < n:
            s = np.random.randint(0, n)
            for j in range(bl):
                if pos >= n:
                    break
                out[pos] = (s + j) % n
                pos += 1
        return out

    @njit(cache=True)
    def _stat_idx(n, mb, seed):
        np.random.seed(seed)
        p = 1.0 - 1.0 / mb
        out = np.empty(n, dtype=np.int64)
        out[0] = np.random.randint(0, n)
        for i in range(1, n):
            if np.random.random() < p:
                out[i] = (out[i - 1] + 1) % n
            else:
                out[i] = np.random.randint(0, n)
        return out

    _NUMBA = True

except ImportError:
    _NUMBA = False

    def _mbb_idx(n, bl, seed):
        r = np.random.RandomState(seed)
        out = np.empty(n, dtype=np.int64)
        pos = 0
        while pos < n:
            s = r.randint(0, n - bl + 1)
            for j in range(bl):
                if pos >= n:
                    break
                out[pos] = s + j
                pos += 1
        return out

    def _cbb_idx(n, bl, seed):
        r = np.random.RandomState(seed)
        out = np.empty(n, dtype=np.int64)
        pos = 0
        while pos < n:
            s = r.randint(0, n)
            for j in range(bl):
                if pos >= n:
                    break
                out[pos] = (s + j) % n
                pos += 1
        return out

    def _stat_idx(n, mb, seed):
        r = np.random.RandomState(seed)
        p = 1.0 - 1.0 / mb
        out = np.empty(n, dtype=np.int64)
        out[0] = r.randint(0, n)
        for i in range(1, n):
            if r.random() < p:
                out[i] = (out[i - 1] + 1) % n
            else:
                out[i] = r.randint(0, n)
        return out


def _batch_gen(data, n_resamples, batch_size, rng, idx_fn, **kw):
    n = data.shape[0]
    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        sb = int(rng.integers(0, 2**31))
        batch = np.empty((bs, n), dtype=data.dtype)
        for i in range(bs):
            batch[i] = data[idx_fn(n, seed=sb + i, **kw)]
        yield batch
        done += bs


def mbb_resample(data, n_resamples, batch_size, rng, block_length=10):
    n = data.shape[0]
    if block_length >= n:
        raise ValueError("block_length must be < len(data).")
    return _batch_gen(
        data,
        n_resamples,
        batch_size,
        rng,
        lambda n, seed, **k: _mbb_idx(n, block_length, seed),
    )


def cbb_resample(data, n_resamples, batch_size, rng, block_length=10):
    return _batch_gen(
        data,
        n_resamples,
        batch_size,
        rng,
        lambda n, seed, **k: _cbb_idx(n, block_length, seed),
    )


def stationary_resample(data, n_resamples, batch_size, rng, mean_block=10.0):
    return _batch_gen(
        data,
        n_resamples,
        batch_size,
        rng,
        lambda n, seed, **k: _stat_idx(n, mean_block, seed),
    )


def tapered_block_resample(
    data, n_resamples, batch_size, rng, block_length=10, taper="tukey"
):
    from scipy.signal import windows as sw

    n = data.shape[0]
    if block_length >= n:
        raise ValueError("block_length must be < len(data).")

    win = sw.get_window(taper, block_length)
    win = win / win.sum() * block_length

    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        sb = int(rng.integers(0, 2**31))
        batch = np.empty((bs, n), dtype=np.float64)
        for i in range(bs):
            raw = data[_mbb_idx(n, block_length, sb + i)].astype(np.float64)
            for s in range(0, n, block_length):
                e = min(s + block_length, n)
                raw[s:e] *= win[: e - s]
            batch[i] = raw
        yield batch
        done += bs


def sieve_resample(data, n_resamples, batch_size, rng, ar_order=None):
    n = data.shape[0]
    if ar_order is None:
        ar_order = min(int(np.round(np.log(n))), n // 3)
    if ar_order < 1:
        ar_order = 1

    mu = data.mean()
    c = data - mu
    ac = np.correlate(c, c, mode="full")[n - 1 :][: ar_order + 1]

    # Construct Yule-Walker matrix R
    R = np.empty((ar_order, ar_order), dtype=np.float64)
    for i in range(ar_order):
        for j in range(ar_order):
            R[i, j] = ac[abs(i - j)]

    phi = np.linalg.solve(R, ac[1 : ar_order + 1])

    # Compute fitted values
    ft = np.zeros(n, dtype=np.float64)
    for t in range(ar_order, n):
        for k in range(ar_order):
            ft[t] += phi[k] * c[t - k - 1]

    res = c.copy()
    res[ar_order:] -= ft[ar_order:]

    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        batch = np.empty((bs, n), dtype=np.float64)
        for i in range(bs):
            eps = res[rng.integers(0, n, size=n)]
            y = np.zeros(n, dtype=np.float64)
            y[:ar_order] = c[:ar_order]
            for t in range(ar_order, n):
                s = eps[t]
                for k in range(ar_order):
                    s += phi[k] * y[t - k - 1]
                y[t] = s
            batch[i] = y + mu
        yield batch
        done += bs


def wild_resample(
    data, n_resamples, batch_size, rng, fitted=None, distribution="rademacher"
):
    n = data.shape[0]
    if fitted is None:
        fitted = np.full(n, data.mean(), dtype=np.float64)

    resid = data - fitted
    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        batch = np.empty((bs, n), dtype=np.float64)
        for i in range(bs):
            if distribution == "rademacher":
                v = rng.choice(np.array([-1.0, 1.0]), size=n)
            elif distribution == "mammen":
                s5 = np.sqrt(5.0)
                p = (s5 + 1) / (2 * s5)
                v = np.where(rng.random(n) < p, -(s5 - 1) / 2, (s5 + 1) / 2)
            else:
                raise ValueError(f"Unknown distribution: {distribution!r}")
            batch[i] = fitted + resid * v
        yield batch
        done += bs

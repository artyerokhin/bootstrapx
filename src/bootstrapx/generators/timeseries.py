"""Time-series bootstrap generators.

Hotfix:
- Fix SeedSequence.spawn usage: child.entropy is identical for all children, so
  using it as the actual seed made all MBB/CBB/stationary samples identical.
- Use child.generate_state(1)[0] to obtain a unique 32-bit seed per child.
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
from scipy.signal import lfilter

try:
    from numba import njit

    @njit(cache=True, parallel=False)
    def _mbb_idx(n: int, bl: int, seed: int) -> np.ndarray:
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

    @njit(cache=True, parallel=False)
    def _cbb_idx(n: int, bl: int, seed: int) -> np.ndarray:
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

    @njit(cache=True, parallel=False)
    def _stat_idx(n: int, mb: float, seed: int) -> np.ndarray:
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
except ImportError:

    def _mbb_idx(n: int, bl: int, seed: int) -> np.ndarray:
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

    def _cbb_idx(n: int, bl: int, seed: int) -> np.ndarray:
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

    def _stat_idx(n: int, mb: float, seed: int) -> np.ndarray:
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


def _spawn_uint32_seeds(rng: np.random.Generator, n_resamples: int) -> list[int]:
    ss = np.random.SeedSequence(int(rng.integers(0, 2**63)))
    children = ss.spawn(n_resamples)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


def _batch_gen(
    data: np.ndarray,
    n_resamples: int,
    batch_size: int,
    rng: np.random.Generator,
    idx_fn,
    **kw,
) -> Generator[np.ndarray, None, None]:
    n = data.shape[0]
    done = 0
    child_seeds = _spawn_uint32_seeds(rng, n_resamples)
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        batch = np.empty((bs, n), dtype=data.dtype)
        for i in range(bs):
            seed_i = child_seeds[done + i]
            batch[i] = data[idx_fn(n, seed=seed_i, **kw)]
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
    n = data.shape[0]
    if block_length >= n:
        raise ValueError("block_length must be < len(data).")
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


def tapered_block_resample(data, n_resamples, batch_size, rng, block_length=10, taper="tukey"):
    from scipy.signal import windows as sw

    n = data.shape[0]
    if block_length >= n:
        raise ValueError("block_length must be < len(data).")
    win = np.asarray(sw.get_window(taper, block_length, fftbins=False), dtype=np.float64)
    rms = float(np.sqrt(np.mean(win**2)))
    if not np.isfinite(rms) or rms == 0.0:
        win = np.ones(block_length, dtype=np.float64)
    else:
        win /= rms
    mean = float(data.mean())
    centered = data.astype(np.float64) - mean
    child_seeds = _spawn_uint32_seeds(rng, n_resamples)
    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        batch = np.empty((bs, n), dtype=np.float64)
        for i in range(bs):
            raw = centered[_mbb_idx(n, block_length, child_seeds[done + i])].copy()
            for s in range(0, n, block_length):
                e = min(s + block_length, n)
                raw[s:e] *= win[: e - s]
            batch[i] = raw + mean
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
    if np.all(c == 0.0):
        done = 0
        while done < n_resamples:
            bs = min(batch_size, n_resamples - done)
            yield np.broadcast_to(data, (bs, n)).copy()
            done += bs
        return
    ac = np.correlate(c, c, mode="full")[n - 1 :][: ar_order + 1]
    R = np.empty((ar_order, ar_order), dtype=np.float64)
    for i in range(ar_order):
        for j in range(ar_order):
            R[i, j] = ac[abs(i - j)]
    try:
        phi = np.linalg.solve(R, ac[1 : ar_order + 1])
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Cannot fit the sieve autoregression because its covariance matrix is singular. "
            "Try a smaller ar_order or another time-series bootstrap method."
        ) from exc
    ft = np.zeros(n, dtype=np.float64)
    for t in range(ar_order, n):
        for k in range(ar_order):
            ft[t] += phi[k] * c[t - k - 1]
    residuals = (c - ft)[ar_order:]
    residuals -= residuals.mean()
    a_coef = np.concatenate([[1.0], -phi])
    burnin = max(50, ar_order * 5)
    n_draw = n + burnin
    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        batch = np.empty((bs, n), dtype=np.float64)
        for i in range(bs):
            eps = residuals[rng.integers(0, len(residuals), size=n_draw)]
            y_full = lfilter([1.0], a_coef, eps)
            batch[i] = y_full[burnin:] + mu
        yield batch
        done += bs


def wild_resample(data, n_resamples, batch_size, rng, fitted=None, distribution="rademacher"):
    n = data.shape[0]
    if fitted is None:
        fitted = np.full(n, data.mean(), dtype=np.float64)
    resid = (data - fitted).astype(np.float64)
    if distribution not in ("rademacher", "mammen"):
        raise ValueError(f"Unknown distribution {distribution!r}. Choose 'rademacher' or 'mammen'.")
    done = 0
    while done < n_resamples:
        bs = min(batch_size, n_resamples - done)
        batch = np.empty((bs, n), dtype=np.float64)
        for i in range(bs):
            if distribution == "rademacher":
                v = rng.choice(np.array([-1.0, 1.0]), size=n)
            else:
                s5 = np.sqrt(5.0)
                p = (s5 + 1.0) / (2.0 * s5)
                v = np.where(rng.random(n) < p, -(s5 - 1.0) / 2.0, (s5 + 1.0) / 2.0)
            batch[i] = fitted + resid * v
        yield batch
        done += bs

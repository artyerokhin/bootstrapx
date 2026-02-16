#!/usr/bin/env python3
"""
Speed benchmark: bootstrapx vs scipy.stats.bootstrap.

Usage:
    python benchmarks/bench_speed.py
"""
import time
import numpy as np
from scipy import stats as sp_stats

import sys, os
# Ensure we import the local src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bootstrapx import bootstrap

def bench_scipy(data, n_resamples, method):
    # Map method names for scipy
    sp_method = "BCa" if method == "bca" else "percentile"
    t0 = time.perf_counter()
    sp_stats.bootstrap((data,), np.mean, n_resamples=n_resamples,
                       method=sp_method, random_state=0)
    return time.perf_counter() - t0

def bench_bootstrapx(data, n_resamples, method, backend):
    t0 = time.perf_counter()
    bootstrap(data, np.mean, method=method, n_resamples=n_resamples,
              random_state=0, backend=backend)
    return time.perf_counter() - t0

if __name__ == "__main__":
    n_resamples = 9999

    # Use smaller N for BCa (O(N^2) complexity due to Jackknife)
    # Use larger N for Percentile to show raw resampling speed
    configs = [
        # (N, Method)
        (500, "bca"),
        (1_000, "bca"),
        (5_000, "bca"),
        # Switch to percentile for large N to benchmark the resampling engine speed
        (10_000, "percentile"),
        (50_000, "percentile"),
        (100_000, "percentile"),
    ]

    print(f"{'N':>8s} | {'Method':>10s} | {'Scipy (s)':>10s} | {'Bx-Vanilla':>10s} | {'Bx-Numba':>10s} | {'Speedup':>7s}")
    print("-" * 75)

    for n, method in configs:
        data = np.random.default_rng(0).normal(0, 1, n)

        # Scipy
        try:
            t_sp = bench_scipy(data, n_resamples, method)
        except Exception:
            t_sp = float('nan')

        # Bootstrapx Vanilla
        t_bx_van = bench_bootstrapx(data, n_resamples, method, "vanilla")

        # Bootstrapx Numba (force if possible)
        try:
            t_bx_num = bench_bootstrapx(data, n_resamples, method, "numba_cpu")
        except Exception:
            t_bx_num = float('nan')

        speedup = t_sp / t_bx_num if t_bx_num > 0 else 0.0

        print(f"{n:>8d} | {method:>10s} | {t_sp:>10.2f} | {t_bx_van:>10.2f} | {t_bx_num:>10.2f} | {speedup:>6.1f}x")
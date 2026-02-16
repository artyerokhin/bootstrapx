#!/usr/bin/env python3
"""
Coverage accuracy benchmark: Monte Carlo simulation.

Tests whether the actual coverage of each CI method matches the
nominal confidence level across repeated samples.

Usage:
    python benchmarks/bench_coverage.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bootstrapx import bootstrap


def coverage_study(dist_fn, true_param, methods, n_obs=1000,
                   n_mc=500, n_boot=2000, confidence=0.95):
    results = {m: 0 for m in methods}

    for trial in range(n_mc):
        data = dist_fn(np.random.default_rng(trial), n_obs)
        for m in methods:
            r = bootstrap(data, np.mean, method=m, n_resamples=n_boot,
                         random_state=trial, backend="vanilla",
                         confidence_level=confidence)
            if r.confidence_interval.low <= true_param <= r.confidence_interval.high:
                results[m] += 1

    return {m: count / n_mc for m, count in results.items()}


if __name__ == "__main__":
    methods = ["percentile", "basic", "bca"]

    print("=" * 50)
    print("NORMAL DATA — N(5, 4), true mean = 5.0")
    print("=" * 50)
    cov_normal = coverage_study(
        lambda rng, n: rng.normal(5.0, 2.0, n),
        true_param=5.0, methods=methods,
    )
    for m, c in cov_normal.items():
        print(f"  {m:>12s}: {c*100:.1f}%")

    print()
    print("=" * 50)
    print("SKEWED DATA — Exp(2), true mean = 2.0")
    print("=" * 50)
    cov_skewed = coverage_study(
        lambda rng, n: rng.exponential(2.0, n),
        true_param=2.0, methods=methods,
    )
    for m, c in cov_skewed.items():
        print(f"  {m:>12s}: {c*100:.1f}%")

    print()
    print("=" * 50)
    print("HEAVY-TAILED — t(3), true mean = 0.0")
    print("=" * 50)
    cov_heavy = coverage_study(
        lambda rng, n: rng.standard_t(3, n),
        true_param=0.0, methods=methods,
    )
    for m, c in cov_heavy.items():
        print(f"  {m:>12s}: {c*100:.1f}%")

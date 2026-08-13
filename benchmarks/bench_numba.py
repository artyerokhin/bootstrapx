"""Compare optional Numba acceleration with the Python fallback.

Run from the repository root after ``pip install -e ".[numba]"``. Private
index functions are used intentionally so both implementations can be measured
through the same public ``bootstrap`` workflow.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import numba
import numpy as np

from bootstrapx import bootstrap
from bootstrapx.generators import timeseries

MethodSpec = tuple[str, str, str, dict[str, Any]]
METHODS: tuple[MethodSpec, ...] = (
    ("mbb", "_mbb_idx", "_mbb_idx_python", {"block_length": 20}),
    ("cbb", "_cbb_idx", "_cbb_idx_python", {"block_length": 20}),
    ("stationary", "_stat_idx", "_stat_idx_python", {"mean_block": 20.0}),
)


def _median_runtime(call: Callable[[], Any], repeats: int) -> float:
    runtimes = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        runtimes.append(time.perf_counter() - started)
    return statistics.median(runtimes)


def _run(
    data: np.ndarray[Any, np.dtype[np.float64]],
    method: str,
    n_resamples: int,
    kwargs: dict[str, Any],
) -> Any:
    return bootstrap(
        data,
        np.mean,
        method=method,
        n_resamples=n_resamples,
        random_state=42,
        **kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-resamples", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    print(f"system={platform.platform()}")
    print(f"python={platform.python_version()} numpy={np.__version__} numba={numba.__version__}")
    print(f"n_resamples={args.n_resamples} warm_repeats={args.repeats}")

    print("\nFirst process call (includes JIT compilation or cache loading)")
    startup_data = np.sin(np.arange(1_000, dtype=np.float64) / 10.0)
    for method, compiled_name, _, kwargs in METHODS:
        compiled = getattr(timeseries, compiled_name)
        setattr(timeseries, compiled_name, compiled)
        started = time.perf_counter()
        _run(startup_data, method, args.n_resamples, kwargs)
        first_call = time.perf_counter() - started
        print(f"{method:10s} first_call={first_call:7.4f}s")

    for n in (100, 1_000, 10_000):
        data = np.sin(np.arange(n, dtype=np.float64) / 10.0)
        print(f"\nn={n}")
        for method, compiled_name, fallback_name, kwargs in METHODS:
            compiled = getattr(timeseries, compiled_name)
            fallback = getattr(timeseries, fallback_name)
            try:
                setattr(timeseries, compiled_name, compiled)
                call = partial(_run, data, method, args.n_resamples, kwargs)
                warm = _median_runtime(call, args.repeats)

                setattr(timeseries, compiled_name, fallback)
                python = _median_runtime(call, args.repeats)
            finally:
                setattr(timeseries, compiled_name, compiled)

            print(
                f"{method:10s} warm={warm:7.4f}s "
                f"fallback={python:7.4f}s warm_speedup={python / warm:6.2f}x"
            )


if __name__ == "__main__":
    main()

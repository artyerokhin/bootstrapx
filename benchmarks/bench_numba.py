"""Compare optional Numba time-series acceleration with the Python fallback."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numba
import numpy as np
import scipy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bootstrapx import __version__, bootstrap
from bootstrapx.generators import timeseries

MethodSpec = tuple[str, str, str, dict[str, Any]]
METHODS: tuple[MethodSpec, ...] = (
    ("mbb", "_mbb_idx", "_mbb_idx_python", {"block_length": 20}),
    ("cbb", "_cbb_idx", "_cbb_idx_python", {"block_length": 20}),
    ("stationary", "_stat_idx", "_stat_idx_python", {"mean_block": 20.0}),
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


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
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_runs/numba"))
    args = parser.parse_args()
    if args.n_resamples < 1 or args.repeats < 1:
        parser.error("--n-resamples and --repeats must be at least 1")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()

    print(f"system={platform.platform()}")
    print(f"python={platform.python_version()} numpy={np.__version__} numba={numba.__version__}")
    print(f"n_resamples={args.n_resamples} warm_repeats={args.repeats}")

    first_calls: dict[str, float] = {}
    print("\nFirst process call (includes JIT compilation or cache loading)")
    startup_data = np.sin(np.arange(1_000, dtype=np.float64) / 10.0)
    for method, compiled_name, _, kwargs in METHODS:
        compiled = getattr(timeseries, compiled_name)
        setattr(timeseries, compiled_name, compiled)
        started = time.perf_counter()
        _run(startup_data, method, args.n_resamples, kwargs)
        first_calls[method] = time.perf_counter() - started
        print(f"{method:10s} first_call={first_calls[method]:7.4f}s")

    rows: list[dict[str, Any]] = []
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

            row = {
                "n": n,
                "method": method,
                "numba_warm_seconds": round(warm, 6),
                "python_fallback_seconds": round(python, 6),
                "fallback_over_numba": round(python / warm, 2),
                "first_process_call_seconds": round(first_calls[method], 6),
            }
            rows.append(row)
            print(
                f"{method:10s} warm={warm:7.4f}s "
                f"fallback={python:7.4f}s warm_speedup={python / warm:6.2f}x"
            )

    with (output_dir / "numba.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "benchmark": "optional-numba",
        "bootstrapx": __version__,
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "n_resamples": args.n_resamples,
        "warm_repeats": args.repeats,
        "statistic": "numpy.mean",
        "elapsed_seconds": round(time.perf_counter() - started_all, 2),
        "scope": "mbb, cbb, and stationary index generation only",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()

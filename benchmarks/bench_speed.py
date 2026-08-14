#!/usr/bin/env python3
"""Runtime and tracemalloc benchmark: bootstrapx versus SciPy."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import stats as sp_stats
from scipy.stats import iqr as sp_iqr
from scipy.stats import trim_mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import bootstrapx as bx

N_RESAMPLES = 4_999
CI = 0.95


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _timeit(call: Callable[[], Any], repeats: int) -> float:
    times = []
    call()  # warm-up is deliberately excluded from the reported median
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        call()
        times.append((time.perf_counter() - started) * 1_000)
    return float(np.median(times))


def _peak_mb(call: Callable[[], Any]) -> float:
    gc.collect()
    tracemalloc.start()
    call()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1_024 / 1_024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use the smaller development grid.")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_runs/speed"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()

    def bx_call(data: np.ndarray[Any, Any], statistic: Callable[..., float], method: str) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return bx.bootstrap(
                data,
                statistic=statistic,
                method=method,
                n_resamples=N_RESAMPLES,
                confidence_level=CI,
                random_state=42,
            )

    def scipy_call(data: np.ndarray[Any, Any], statistic: Callable[..., float], method: str) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sp_stats.bootstrap(
                (data,),
                statistic=statistic,
                n_resamples=N_RESAMPLES,
                confidence_level=CI,
                method="BCa" if method == "bca" else "percentile",
                random_state=42,
            )

    speed_rows: list[dict[str, Any]] = []
    grids = {
        "bca": [200, 500, 1_000, 2_000] if args.quick else [200, 500, 1_000, 2_000, 5_000, 10_000],
        "percentile": [1_000, 5_000, 10_000]
        if args.quick
        else [1_000, 5_000, 10_000, 50_000, 100_000],
    }
    for method, sizes in grids.items():
        print(f"\n{method.upper()} speed (np.mean)", flush=True)
        for n in sizes:
            data = np.random.default_rng(0).standard_normal(n)
            bx_ms = _timeit(partial(bx_call, data, np.mean, method), args.repeats)
            scipy_ms = _timeit(partial(scipy_call, data, np.mean, method), args.repeats)
            ratio = scipy_ms / bx_ms
            row = {
                "n": n,
                "method": method,
                "bootstrapx_ms": round(bx_ms, 2),
                "scipy_ms": round(scipy_ms, 2),
                "scipy_over_bootstrapx": round(ratio, 2),
            }
            speed_rows.append(row)
            print(
                f"n={n:>6d} bootstrapx={bx_ms:8.2f} ms "
                f"scipy={scipy_ms:8.2f} ms ratio={ratio:5.2f}x",
                flush=True,
            )
            _write_csv(output_dir / "speed.csv", list(row), speed_rows)

    memory_rows: list[dict[str, Any]] = []
    memory_sizes = [500, 1_000, 2_000] if args.quick else [500, 1_000, 2_000, 5_000, 10_000]
    print("\nBCa peak allocations visible to tracemalloc", flush=True)
    for n in memory_sizes:
        data = np.random.default_rng(1).standard_normal(n)
        bx_mb = _peak_mb(partial(bx_call, data, np.mean, "bca"))
        scipy_mb = _peak_mb(partial(scipy_call, data, np.mean, "bca"))
        row = {
            "n": n,
            "bootstrapx_tracemalloc_mb": round(bx_mb, 3),
            "scipy_tracemalloc_mb": round(scipy_mb, 3),
            "scipy_over_bootstrapx": round(scipy_mb / bx_mb, 2),
        }
        memory_rows.append(row)
        print(f"n={n:>6d} bootstrapx={bx_mb:8.3f} MB scipy={scipy_mb:8.3f} MB", flush=True)
        _write_csv(output_dir / "memory.csv", list(row), memory_rows)

    def trimmed_mean(values: np.ndarray[Any, Any]) -> float:
        return float(trim_mean(values, 0.1))

    def iqr(values: np.ndarray[Any, Any]) -> float:
        return float(sp_iqr(values))

    callable_rows: list[dict[str, Any]] = []
    callable_sizes = [500, 1_000] if args.quick else [500, 1_000, 2_000]
    print("\nArbitrary callables (BCa)", flush=True)
    for n in callable_sizes:
        data = np.random.default_rng(2).standard_normal(n)
        for name, statistic in (("trimmed_mean", trimmed_mean), ("iqr", iqr)):
            bx_ms = _timeit(partial(bx_call, data, statistic, "bca"), args.repeats)
            scipy_ms = _timeit(partial(scipy_call, data, statistic, "bca"), args.repeats)
            row = {
                "n": n,
                "statistic": name,
                "bootstrapx_ms": round(bx_ms, 2),
                "scipy_ms": round(scipy_ms, 2),
            }
            callable_rows.append(row)
            print(
                f"n={n:>6d} {name:>12s} bootstrapx={bx_ms:8.2f} ms scipy={scipy_ms:8.2f} ms",
                flush=True,
            )
            _write_csv(output_dir / "callables.csv", list(row), callable_rows)

    metadata = {
        "benchmark": "speed-memory-callables",
        "bootstrapx": bx.__version__,
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "n_resamples": N_RESAMPLES,
        "confidence_level": CI,
        "runtime_repeats": args.repeats,
        "runtime_summary": "median after one unmeasured warm-up",
        "memory_tool": "tracemalloc (not process RSS)",
        "quick": args.quick,
        "elapsed_seconds": round(time.perf_counter() - started_all, 2),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()

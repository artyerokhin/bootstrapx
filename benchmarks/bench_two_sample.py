#!/usr/bin/env python3
"""Runtime and tracemalloc benchmark for two-sample comparisons."""

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import bootstrapx as bx

CI = 0.95


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty() -> bool:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return True


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _timeit(call: Callable[[], Any], repeats: int) -> float:
    call()
    timings = []
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        call()
        timings.append((time.perf_counter() - started) * 1_000)
    return float(np.median(timings))


def _peak_mb(call: Callable[[], Any]) -> float:
    gc.collect()
    tracemalloc.start()
    call()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1_024 / 1_024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_runs/two-sample"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    n_resamples = 999 if args.quick else 4_999
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()

    def data(n: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        rng = np.random.default_rng(n)
        return rng.normal(size=n), rng.normal(0.2, 1.1, size=int(n * 1.25))

    def bx_call(control: np.ndarray[Any, Any], treatment: np.ndarray[Any, Any], method: str) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return bx.bootstrap_two_sample(
                control,
                treatment,
                np.mean,
                method=method,
                n_resamples=n_resamples,
                confidence_level=CI,
                random_state=42,
            )

    def scipy_scalar_call(
        control: np.ndarray[Any, Any], treatment: np.ndarray[Any, Any], method: str
    ) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sp_stats.bootstrap(
                (control, treatment),
                lambda control_sample, treatment_sample: (
                    np.mean(treatment_sample) - np.mean(control_sample)
                ),
                vectorized=False,
                method="BCa" if method == "bca" else method,
                n_resamples=n_resamples,
                confidence_level=CI,
                random_state=42,
            )

    def scipy_vectorized_bounded_call(
        control: np.ndarray[Any, Any], treatment: np.ndarray[Any, Any], method: str
    ) -> Any:
        def difference_of_means(
            control_sample: np.ndarray[Any, Any],
            treatment_sample: np.ndarray[Any, Any],
            *,
            axis: int = -1,
        ) -> np.ndarray[Any, Any]:
            return np.mean(treatment_sample, axis=axis) - np.mean(control_sample, axis=axis)

        matched_batch = bx.utils.auto_batch_size(
            len(control) + len(treatment),
            n_resamples,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sp_stats.bootstrap(
                (control, treatment),
                difference_of_means,
                vectorized=True,
                batch=matched_batch,
                method="BCa" if method == "bca" else method,
                n_resamples=n_resamples,
                confidence_level=CI,
                random_state=42,
            )

    runtime_grid = [("percentile", 200), ("bca", 200), ("percentile", 1_000)]
    if not args.quick:
        runtime_grid.extend([("bca", 1_000), ("percentile", 10_000)])
    runtime_rows: list[dict[str, Any]] = []
    for method, n in runtime_grid:
        control, treatment = data(n)
        bx_ms = _timeit(partial(bx_call, control, treatment, method), args.repeats)
        scipy_scalar_ms = _timeit(
            partial(scipy_scalar_call, control, treatment, method), args.repeats
        )
        scipy_vectorized_bounded_ms = _timeit(
            partial(scipy_vectorized_bounded_call, control, treatment, method), args.repeats
        )
        row = {
            "n_control": len(control),
            "n_treatment": len(treatment),
            "method": method,
            "bootstrapx_ms": round(bx_ms, 2),
            "scipy_scalar_ms": round(scipy_scalar_ms, 2),
            "scipy_vectorized_bounded_ms": round(scipy_vectorized_bounded_ms, 2),
            "scipy_scalar_over_bootstrapx": round(scipy_scalar_ms / bx_ms, 2),
            "scipy_vectorized_bounded_over_bootstrapx": round(
                scipy_vectorized_bounded_ms / bx_ms, 2
            ),
        }
        runtime_rows.append(row)
        _write_csv(output_dir / "runtime.csv", runtime_rows)
        print(
            f"{method:>10} n={n:>6} bootstrapx={bx_ms:8.2f} ms "
            f"scipy-scalar={scipy_scalar_ms:8.2f} ms "
            f"scipy-vectorized-bounded={scipy_vectorized_bounded_ms:8.2f} ms",
            flush=True,
        )

    memory_rows: list[dict[str, Any]] = []
    for n in [500] if args.quick else [500, 2_000]:
        control, treatment = data(n)
        bx_mb = _peak_mb(partial(bx_call, control, treatment, "percentile"))
        scipy_scalar_mb = _peak_mb(partial(scipy_scalar_call, control, treatment, "percentile"))
        scipy_vectorized_bounded_mb = _peak_mb(
            partial(scipy_vectorized_bounded_call, control, treatment, "percentile")
        )
        row = {
            "n_control": len(control),
            "n_treatment": len(treatment),
            "method": "percentile",
            "bootstrapx_tracemalloc_mb": round(bx_mb, 3),
            "scipy_scalar_tracemalloc_mb": round(scipy_scalar_mb, 3),
            "scipy_vectorized_bounded_tracemalloc_mb": round(scipy_vectorized_bounded_mb, 3),
            "scipy_scalar_over_bootstrapx": round(scipy_scalar_mb / bx_mb, 2),
            "scipy_vectorized_bounded_over_bootstrapx": round(
                scipy_vectorized_bounded_mb / bx_mb, 2
            ),
        }
        memory_rows.append(row)
        _write_csv(output_dir / "memory.csv", memory_rows)
        print(
            f"memory n={n:>6} bootstrapx={bx_mb:8.3f} MB "
            f"scipy-scalar={scipy_scalar_mb:8.3f} MB "
            f"scipy-vectorized-bounded={scipy_vectorized_bounded_mb:8.3f} MB",
            flush=True,
        )

    cluster_rows: list[dict[str, Any]] = []
    for n_clusters in [40] if args.quick else [40, 200]:
        rng = np.random.default_rng(n_clusters)
        control_ids = np.repeat(np.arange(n_clusters), 5)
        treatment_ids = np.repeat(np.arange(int(n_clusters * 1.2)), 5)
        control = rng.normal(size=len(control_ids))
        treatment = rng.normal(0.2, size=len(treatment_ids))

        cluster_call = partial(
            bx.bootstrap_two_sample,
            control,
            treatment,
            np.mean,
            control_cluster_ids=control_ids,
            treatment_cluster_ids=treatment_ids,
            method="percentile",
            n_resamples=n_resamples,
            random_state=42,
        )
        runtime_ms = _timeit(cluster_call, args.repeats)
        row = {
            "n_control_clusters": n_clusters,
            "n_treatment_clusters": int(n_clusters * 1.2),
            "rows_per_cluster": 5,
            "method": "percentile",
            "bootstrapx_ms": round(runtime_ms, 2),
        }
        cluster_rows.append(row)
        _write_csv(output_dir / "cluster_runtime.csv", cluster_rows)
        print(f"cluster n={n_clusters:>4} runtime={runtime_ms:8.2f} ms", flush=True)

    metadata = {
        "benchmark": "two-sample-runtime-memory",
        "bootstrapx": bx.__version__,
        "git_commit": _git_commit(),
        "git_worktree_dirty": _git_dirty(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "n_resamples": n_resamples,
        "confidence_level": CI,
        "runtime_repeats": args.repeats,
        "runtime_summary": "median after one unmeasured warm-up",
        "memory_tool": "tracemalloc (not process RSS)",
        "scipy_scalar_configuration": "vectorized=False; SciPy default batch",
        "scipy_vectorized_configuration": (
            "vectorized=True; batch matched to bootstrapx auto_batch_size"
        ),
        "quick": args.quick,
        "elapsed_seconds": round(time.perf_counter() - started_all, 2),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

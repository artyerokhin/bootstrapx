#!/usr/bin/env python3
"""Monte Carlo coverage study: bootstrapx versus scipy.stats.bootstrap.

The script checkpoints after every configuration. Re-run with ``--resume`` and
the same output directory after an interruption.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import stats as sp_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import bootstrapx as bx

CI = 0.95
FIELDNAMES = [
    "library",
    "method",
    "statistic",
    "n",
    "distribution",
    "coverage",
    "gap_from_nominal",
    "n_simulations",
    "valid_trials",
    "invalid_trials",
    "failed_trials",
    "coverage_mc_se",
    "coverage_mc_low",
    "coverage_mc_high",
    "first_failure",
    "elapsed_seconds",
]

TRUE_PARAMS = {
    ("normal", "mean"): 0.0,
    ("normal", "median"): 0.0,
    ("normal", "std"): 1.0,
    ("lognormal", "mean"): float(np.exp(0.5)),
    ("lognormal", "median"): 1.0,
    ("lognormal", "std"): float(np.sqrt((np.e - 1) * np.e)),
    ("exponential", "mean"): 2.0,
    ("exponential", "median"): float(2.0 * np.log(2)),
    ("exponential", "std"): 2.0,
    ("t3", "mean"): 0.0,
    ("t3", "median"): 0.0,
    ("t3", "std"): float(np.sqrt(3.0)),
}
STD_SKIP = {("lognormal", "std"), ("t3", "std")}
STATISTICS = {"mean": np.mean, "median": np.median, "std": np.std}


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


def _make_data(name: str, n: int, rng: np.random.Generator) -> np.ndarray[Any, Any]:
    if name == "normal":
        return rng.standard_normal(n)
    if name == "lognormal":
        return rng.lognormal(0, 1, n)
    if name == "exponential":
        return rng.exponential(2.0, n)
    if name == "t3":
        return rng.standard_t(3, n)
    raise ValueError(f"Unknown distribution: {name}")


def _wilson_interval(hits: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return float("nan"), float("nan")
    proportion = hits / trials
    z = 1.96
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return float(center - half_width), float(center + half_width)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _task_key(row: dict[str, Any]) -> tuple[str, str, str, int, str]:
    return (
        str(row["library"]),
        str(row["method"]),
        str(row["statistic"]),
        int(row["n"]),
        str(row["distribution"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--smoke", action="store_true", help="Eight calls; pipeline test only.")
    profile.add_argument("--fast", action="store_true", help="300 simulations per cell.")
    profile.add_argument("--full", action="store_true", help="2,000 simulations per cell.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_runs/coverage"))
    args = parser.parse_args()

    if args.smoke:
        profile_name, n_simulations, n_resamples = "smoke", 2, 99
        sample_sizes, statistics, distributions = [200], ["mean"], ["normal"]
    else:
        profile_name = "full" if args.full else "fast" if args.fast else "standard"
        n_simulations = 2_000 if args.full else 300 if args.fast else 1_000
        n_resamples = 4_999
        sample_sizes = [200, 500, 1_000, 2_000]
        statistics = ["mean", "median", "std"]
        distributions = ["normal", "lognormal", "exponential", "t3"]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "coverage.csv"
    metadata_path = output_dir / "metadata.json"
    commit = _git_commit()
    config = {
        "profile": profile_name,
        "n_simulations": n_simulations,
        "n_resamples": n_resamples,
        "confidence_level": CI,
        "sample_sizes": sample_sizes,
        "statistics": statistics,
        "distributions": distributions,
        "git_commit": commit,
        "bootstrapx": bx.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }

    if result_path.exists() and not args.resume:
        parser.error(f"{result_path} exists; use a new --output-dir or pass --resume")
    if args.resume and result_path.exists() and not metadata_path.exists():
        parser.error("cannot resume: coverage.csv exists without matching metadata.json")
    if args.resume and metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        mismatches = [key for key, value in config.items() if previous.get(key) != value]
        if mismatches:
            parser.error(f"cannot resume with changed configuration: {', '.join(mismatches)}")

    rows = _read_rows(result_path) if args.resume else []
    completed = {_task_key(row) for row in rows}
    tasks = [
        (library, method, statistic, n, distribution)
        for library in ("bootstrapx", "scipy")
        for method in ("bca", "percentile")
        for statistic in statistics
        for n in sample_sizes
        for distribution in distributions
        if (distribution, statistic) not in STD_SKIP
    ]
    started_all = time.perf_counter()
    metadata = {
        **config,
        "benchmark": "coverage",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_worktree_dirty": _git_dirty(),
        "random_streams": "data=SeedSequence([simulation,0]); resampling=[simulation,1]",
        "status": "running",
        "total_cells": len(tasks),
        "completed_cells": len(completed),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"profile={profile_name} cells={len(tasks)} already_completed={len(completed)} "
        f"n_simulations={n_simulations} n_resamples={n_resamples}",
        flush=True,
    )

    for task in tasks:
        if task in completed:
            continue
        library, method, statistic_name, n, distribution = task
        statistic = STATISTICS[statistic_name]
        true_value = TRUE_PARAMS[(distribution, statistic_name)]
        hits = valid = invalid = failures = 0
        first_failure = ""
        started_cell = time.perf_counter()

        for simulation in range(n_simulations):
            data_rng = np.random.default_rng(np.random.SeedSequence([simulation, 0]))
            resample_seed = np.random.SeedSequence([simulation, 1])
            data = _make_data(distribution, n, data_rng)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if library == "bootstrapx":
                        result = bx.bootstrap(
                            data,
                            statistic=statistic,
                            method=method,
                            n_resamples=n_resamples,
                            confidence_level=CI,
                            random_state=np.random.default_rng(resample_seed),
                        )
                    else:
                        result = sp_stats.bootstrap(
                            (data,),
                            statistic=statistic,
                            n_resamples=n_resamples,
                            confidence_level=CI,
                            method="BCa" if method == "bca" else "percentile",
                            random_state=np.random.default_rng(resample_seed),
                        )
                low = float(result.confidence_interval.low)
                high = float(result.confidence_interval.high)
                if not (np.isfinite(low) and np.isfinite(high)) or high <= low:
                    invalid += 1
                    continue
                valid += 1
                hits += int(low <= true_value <= high)
            except Exception as error:  # recorded explicitly in the output
                failures += 1
                if not first_failure:
                    first_failure = f"{type(error).__name__}: {error}"[:300]

        coverage = hits / valid if valid else float("nan")
        mc_se = np.sqrt(coverage * (1.0 - coverage) / valid) if valid else float("nan")
        mc_low, mc_high = _wilson_interval(hits, valid)
        row = {
            "library": library,
            "method": method,
            "statistic": statistic_name,
            "n": n,
            "distribution": distribution,
            "coverage": round(coverage, 4) if np.isfinite(coverage) else "nan",
            "gap_from_nominal": round(abs(coverage - CI), 4) if np.isfinite(coverage) else "nan",
            "n_simulations": n_simulations,
            "valid_trials": valid,
            "invalid_trials": invalid,
            "failed_trials": failures,
            "coverage_mc_se": round(float(mc_se), 5) if np.isfinite(mc_se) else "nan",
            "coverage_mc_low": round(mc_low, 4) if np.isfinite(mc_low) else "nan",
            "coverage_mc_high": round(mc_high, 4) if np.isfinite(mc_high) else "nan",
            "first_failure": first_failure,
            "elapsed_seconds": round(time.perf_counter() - started_cell, 2),
        }
        rows.append(row)
        completed.add(task)
        _write_rows(result_path, rows)
        metadata["completed_cells"] = len(completed)
        metadata["elapsed_seconds_this_run"] = round(time.perf_counter() - started_all, 2)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            f"[{len(completed):>3}/{len(tasks)}] {library:>10s} {method:>10s} "
            f"{statistic_name:>6s} n={n:<4d} {distribution:<11s} "
            f"coverage={coverage:.3f} invalid={invalid} failures={failures}",
            flush=True,
        )

    metadata["status"] = "complete"
    metadata["completed_cells"] = len(completed)
    metadata["elapsed_seconds_this_run"] = round(time.perf_counter() - started_all, 2)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nSaved complete results to {output_dir}")


if __name__ == "__main__":
    main()

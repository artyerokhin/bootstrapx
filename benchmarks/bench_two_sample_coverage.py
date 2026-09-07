#!/usr/bin/env python3
"""Coverage study for independent, paired, and clustered comparisons."""

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
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import stats as sp_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import bootstrapx as bx

CI = 0.95
FloatArray = np.ndarray[Any, np.dtype[np.float64]]
Statistic = Callable[[FloatArray], float]

FIELDNAMES = [
    "library",
    "design",
    "scenario",
    "method",
    "effect",
    "statistic",
    "n_control",
    "n_treatment",
    "true_effect",
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


@dataclass(frozen=True)
class Scenario:
    name: str
    design: str
    effect: str
    statistic_name: str
    statistic: Statistic
    n_control: int
    n_treatment: int
    true_effect: float


SCENARIOS = [
    Scenario("normal_mean", "independent", "difference", "mean", np.mean, 200, 250, 0.3),
    Scenario(
        "lognormal_mean",
        "independent",
        "relative_lift",
        "mean",
        np.mean,
        250,
        300,
        0.1,
    ),
    Scenario(
        "bernoulli_conversion",
        "independent",
        "difference",
        "mean",
        np.mean,
        300,
        400,
        0.02,
    ),
    Scenario(
        "exponential_median",
        "independent",
        "difference",
        "median",
        np.median,
        200,
        300,
        float(0.2 * np.log(2.0)),
    ),
    Scenario("paired_normal", "paired", "difference", "mean", np.mean, 200, 200, 0.3),
    Scenario("cluster_normal", "cluster", "difference", "mean", np.mean, 200, 240, 0.3),
]


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


def _make_data(
    scenario: Scenario,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray, np.ndarray[Any, Any] | None, np.ndarray[Any, Any] | None]:
    if scenario.name == "normal_mean":
        return (
            rng.normal(0.0, 1.0, scenario.n_control),
            rng.normal(0.3, 1.2, scenario.n_treatment),
            None,
            None,
        )
    if scenario.name == "lognormal_mean":
        return (
            rng.lognormal(0.0, 1.0, scenario.n_control),
            1.1 * rng.lognormal(0.0, 1.0, scenario.n_treatment),
            None,
            None,
        )
    if scenario.name == "bernoulli_conversion":
        return (
            rng.binomial(1, 0.10, scenario.n_control).astype(np.float64),
            rng.binomial(1, 0.12, scenario.n_treatment).astype(np.float64),
            None,
            None,
        )
    if scenario.name == "exponential_median":
        return (
            rng.exponential(2.0, scenario.n_control),
            rng.exponential(2.2, scenario.n_treatment),
            None,
            None,
        )
    if scenario.name == "paired_normal":
        control = rng.normal(0.0, 1.5, scenario.n_control)
        treatment = control + rng.normal(0.3, 0.8, scenario.n_treatment)
        return control, treatment, None, None
    if scenario.name == "cluster_normal":
        control_clusters = 40
        treatment_clusters = 48
        control_size = scenario.n_control // control_clusters
        treatment_size = scenario.n_treatment // treatment_clusters
        control_ids = np.repeat(np.arange(control_clusters), control_size)
        treatment_ids = np.repeat(np.arange(treatment_clusters), treatment_size)
        control_effects = rng.normal(0.0, 1.0, control_clusters)
        treatment_effects = rng.normal(0.3, 1.0, treatment_clusters)
        control = np.repeat(control_effects, control_size) + rng.normal(0.0, 0.4, len(control_ids))
        treatment = np.repeat(treatment_effects, treatment_size) + rng.normal(
            0.0, 0.4, len(treatment_ids)
        )
        return control, treatment, control_ids, treatment_ids
    raise ValueError(f"Unknown scenario: {scenario.name}")


def _effect(name: str, control: float, treatment: float) -> float:
    if name == "difference":
        return treatment - control
    if name == "relative_lift":
        return (treatment - control) / control
    raise ValueError(f"Unknown effect: {name}")


def _scipy_statistic(
    scenario: Scenario,
    control_sample: FloatArray,
    treatment_sample: FloatArray,
) -> float:
    return _effect(
        scenario.effect,
        float(scenario.statistic(control_sample)),
        float(scenario.statistic(treatment_sample)),
    )


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


def _task_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["library"]), str(row["scenario"]), str(row["method"])


def main() -> None:
    parser = argparse.ArgumentParser()
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--smoke", action="store_true")
    profile.add_argument("--release", action="store_true")
    profile.add_argument("--statistical", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_runs/two-sample-coverage")
    )
    args = parser.parse_args()

    if args.smoke:
        profile_name, n_simulations, n_resamples = "smoke", 2, 99
    elif args.release:
        profile_name, n_simulations, n_resamples = "release", 300, 4_999
    elif args.statistical:
        profile_name, n_simulations, n_resamples = "statistical", 1_000, 4_999
    else:
        profile_name, n_simulations, n_resamples = "fast", 100, 999

    tasks = [
        (library, scenario, method)
        for scenario in SCENARIOS
        for method in ("percentile", "basic", "bca")
        for library in (
            ("bootstrapx",) if scenario.design == "cluster" else ("bootstrapx", "scipy")
        )
    ]
    if args.smoke:
        tasks = [
            task
            for task in tasks
            if task[2] == "percentile"
            and task[1].name in {"normal_mean", "paired_normal", "cluster_normal"}
        ]

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
    metadata = {
        **config,
        "benchmark": "two-sample-coverage",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_worktree_dirty": _git_dirty(),
        "random_streams": "data=[scenario,simulation,0]; resampling=[scenario,simulation,1]",
        "status": "running",
        "total_cells": len(tasks),
        "completed_cells": len(completed),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"profile={profile_name} cells={len(tasks)} completed={len(completed)} "
        f"n_simulations={n_simulations} n_resamples={n_resamples}",
        flush=True,
    )
    started_all = time.perf_counter()

    for library, scenario, method in tasks:
        key = (library, scenario.name, method)
        if key in completed:
            continue
        hits = valid = invalid = failures = 0
        first_failure = ""
        started_cell = time.perf_counter()
        scenario_index = SCENARIOS.index(scenario)

        for simulation in range(n_simulations):
            data_rng = np.random.default_rng(
                np.random.SeedSequence([scenario_index, simulation, 0])
            )
            resample_rng = np.random.default_rng(
                np.random.SeedSequence([scenario_index, simulation, 1])
            )
            control, treatment, control_ids, treatment_ids = _make_data(scenario, data_rng)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if library == "bootstrapx":
                        result = bx.bootstrap_two_sample(
                            control,
                            treatment,
                            scenario.statistic,
                            effect=scenario.effect,
                            paired=scenario.design == "paired",
                            control_cluster_ids=control_ids,
                            treatment_cluster_ids=treatment_ids,
                            method=method,
                            n_resamples=n_resamples,
                            confidence_level=CI,
                            random_state=resample_rng,
                        )
                    else:
                        result = sp_stats.bootstrap(
                            (control, treatment),
                            partial(_scipy_statistic, scenario),
                            vectorized=False,
                            paired=scenario.design == "paired",
                            method="BCa" if method == "bca" else method,
                            n_resamples=n_resamples,
                            confidence_level=CI,
                            random_state=resample_rng,
                        )
                low = float(result.confidence_interval.low)
                high = float(result.confidence_interval.high)
                if not (np.isfinite(low) and np.isfinite(high)) or high < low:
                    invalid += 1
                    continue
                valid += 1
                hits += int(low <= scenario.true_effect <= high)
            except Exception as error:  # recorded explicitly in the result table
                failures += 1
                if not first_failure:
                    first_failure = f"{type(error).__name__}: {error}"[:300]

        coverage = hits / valid if valid else float("nan")
        mc_se = np.sqrt(coverage * (1.0 - coverage) / valid) if valid else float("nan")
        mc_low, mc_high = _wilson_interval(hits, valid)
        row = {
            "library": library,
            "design": scenario.design,
            "scenario": scenario.name,
            "method": method,
            "effect": scenario.effect,
            "statistic": scenario.statistic_name,
            "n_control": scenario.n_control,
            "n_treatment": scenario.n_treatment,
            "true_effect": round(scenario.true_effect, 8),
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
        _write_rows(result_path, rows)
        completed.add(key)
        metadata.update({"completed_cells": len(completed), "status": "running"})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(
            f"[{len(completed):>2}/{len(tasks)}] {library:>10} {scenario.name:>22} "
            f"{method:>10} coverage={coverage:.3f} valid={valid} invalid={invalid} "
            f"failed={failures}",
            flush=True,
        )

    metadata.update(
        {
            "status": "complete",
            "completed_cells": len(completed),
            "elapsed_seconds": round(time.perf_counter() - started_all, 2),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

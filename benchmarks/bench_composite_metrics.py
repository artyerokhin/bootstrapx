#!/usr/bin/env python3
"""Checkpointed composite-metric evidence; quick runs are pipeline smoke only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import tracemalloc
from functools import partial
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bootstrapx as bx  # noqa: E402
from bootstrapx.utils import auto_batch_size  # noqa: E402

SCENARIOS = (
    "independent",
    "paired",
    "cluster",
    "skew",
    "denominator_change",
    "sparse",
    "cluster_large",
    "activity_price",
    "activity_price_covariance",
    "cluster_activity_price",
    "cluster_activity_price_large",
)
METRIC = bx.RatioOfSums()


def make_data(name, rng, n_control=200, n_treatment=250, cluster_counts=None):
    """Generate ratios with analytically known population effects.

    Baseline order counts/cluster sizes are independent of prices. Activity-price
    scenarios instead share a normal latent variable: high-activity units have
    higher prices. Cluster sizes remain independent of that latent variable.
    The paired case shares price/count components across arms.
    """
    sigma = 1.3 if name == "skew" else 0.5
    m_treatment = 2.0 if name == "denominator_change" else 2.08
    true_effect = np.exp(m_treatment + sigma**2 / 2) - np.exp(2.0 + sigma**2 / 2)
    if name == "paired":
        n_treatment = n_control
        orders = rng.poisson(2, n_control)
        t_orders = orders + rng.poisson(0.2, n_control)
        z = rng.normal(size=n_control)
        c_price = np.exp(2 + sigma * z)
        t_price = np.exp(2.08 + sigma * z + 0.1 * rng.normal(size=n_control))
        true_effect = np.exp(2.08 + (sigma**2 + 0.1**2) / 2) - np.exp(2 + sigma**2 / 2)
        return (
            np.column_stack((orders * c_price, orders)),
            np.column_stack((t_orders * t_price, t_orders)),
            None,
            None,
            float(true_effect),
        )
    samples, ids = [], []
    if name.startswith("cluster"):
        n_control, n_treatment = cluster_counts or (
            (100, 120) if name in {"cluster_large", "cluster_activity_price_large"} else (24, 30)
        )
    if name == "sparse":
        n_control, n_treatment = 12, 15
    correlated = "activity_price" in name
    activity_loadings = (0.35, 0.65 if name == "activity_price_covariance" else 0.35)
    price_loading, noise_sigma = 0.45, 0.25
    if correlated:
        # E[orders | Z] = rate * exp(b*Z - b**2/2),
        # price = exp(m + c*Z + s*epsilon), with independent standard normals.
        # Thus E[orders*price]/E[orders] = exp(m + b*c + (c**2+s**2)/2).
        # Covariance-only treatment leaves marginal price AND mean order rate
        # unchanged, but changes the order-weighted population price.
        if name == "activity_price_covariance":
            m_treatment = 2.0
        ratios = [
            np.exp(m + b * price_loading + (price_loading**2 + noise_sigma**2) / 2)
            for m, b in zip((2.0, m_treatment), activity_loadings, strict=True)
        ]
        true_effect = ratios[1] - ratios[0]
    for arm, (n, m) in enumerate(((n_control, 2.0), (n_treatment, m_treatment))):
        cluster_ids = (
            np.repeat(np.arange(n), rng.integers(1, 8, n)) if name.startswith("cluster") else None
        )
        if correlated:
            latent = rng.normal(size=n)
            b = activity_loadings[arm]
            price = np.exp(m + price_loading * latent + noise_sigma * rng.normal(size=n))
            activity = np.exp(b * latent - b**2 / 2)
        else:
            price = rng.lognormal(m, sigma, n)
        if cluster_ids is not None:
            price = price[cluster_ids]
            if correlated:
                activity = activity[cluster_ids]
        rate = (0.04 if arm == 0 else 0.06) if name == "sparse" else (1.8 if arm == 0 else 2.2)
        if name == "activity_price_covariance":
            rate = 1.8
        if name == "denominator_change" and arm == 1:
            rate = 3.6
        orders = rng.poisson(rate * activity if correlated else rate, len(price))
        samples.append(np.column_stack((orders * price, orders)).astype(float))
        ids.append(cluster_ids)
    return *samples, *ids, float(true_effect)


def reference_problem(control, treatment, c_ids, t_ids):
    """SciPy resamples unit indices, never numeric features independently."""
    maps = []
    for ids in (c_ids, t_ids):
        maps.append(
            None if ids is None else [np.flatnonzero(ids == label) for label in np.unique(ids)]
        )

    def statistic(c_indices, t_indices):
        c_rows = (
            c_indices.astype(np.intp)
            if maps[0] is None
            else np.concatenate([maps[0][int(i)] for i in c_indices])
        )
        t_rows = (
            t_indices.astype(np.intp)
            if maps[1] is None
            else np.concatenate([maps[1][int(i)] for i in t_indices])
        )
        return METRIC(treatment[t_rows]) - METRIC(control[c_rows])

    return (
        np.arange(len(control) if maps[0] is None else len(maps[0])),
        np.arange(len(treatment) if maps[1] is None else len(maps[1])),
    ), statistic


def delta_interval(control, treatment, c_ids, t_ids, paired):
    influences, estimates = [], []
    for sample, ids in ((control, c_ids), (treatment, t_ids)):
        totals = (
            sample
            if ids is None
            else np.array([sample[ids == label].sum(axis=0) for label in np.unique(ids)])
        )
        estimate = METRIC(totals)
        influences.append((totals[:, 0] - estimate * totals[:, 1]) / totals[:, 1].mean())
        estimates.append(estimate)
    variance = (
        np.var(influences[1] - influences[0], ddof=1) / len(control)
        if paired
        else sum(np.var(x, ddof=1) / len(x) for x in influences)
    )
    half = stats.norm.ppf(0.975) * np.sqrt(variance)
    estimate = estimates[1] - estimates[0]
    return float(estimate - half), float(estimate + half)


def interval(library, method, data, count, rng, paired):
    c, t, c_ids, t_ids, _ = data
    if library == "delta":
        return delta_interval(c, t, c_ids, t_ids, paired)
    if library == "bootstrapx":
        result = bx.bootstrap_two_sample(
            c,
            t,
            METRIC,
            allow_2d=True,
            paired=paired,
            control_cluster_ids=c_ids,
            treatment_cluster_ids=t_ids,
            method=method,
            n_resamples=count,
            random_state=rng,
        )
    elif library == "scipy_vectorized":
        if c_ids is not None or t_ids is not None:
            raise ValueError("The vectorized reference supports row-level, not clustered input.")

        def statistic(c_indices, t_indices, axis=-1):
            c_values = c[c_indices.astype(np.intp)]
            t_values = t[t_indices.astype(np.intp)]
            c_denominator = c_values[..., 1].sum(axis=axis)
            t_denominator = t_values[..., 1].sum(axis=axis)
            if np.any(c_denominator == 0) or np.any(t_denominator == 0):
                raise ValueError("Vectorized ratio reference has a zero denominator sum.")
            return (
                t_values[..., 0].sum(axis=axis) / t_denominator
                - c_values[..., 0].sum(axis=axis) / c_denominator
            )

        result = stats.bootstrap(
            (np.arange(len(c)), np.arange(len(t))),
            statistic,
            vectorized=True,
            paired=paired,
            method=method,
            n_resamples=count,
            random_state=rng,
            batch=auto_batch_size(c.size + t.size, count),
        )
    else:
        units, statistic = reference_problem(c, t, c_ids, t_ids)
        result = stats.bootstrap(
            units,
            statistic,
            vectorized=False,
            paired=paired,
            method=method,
            n_resamples=count,
            random_state=rng,
            batch=128,
        )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def wilson(hits, count):
    p, z = hits / count, 1.96
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    half = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return max(0.0, float(center - half)), min(1.0, float(center + half))


def source_digest():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src" / "bootstrapx").rglob("*.py")) + [Path(__file__)]:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def checkpoint(root, metadata, rows):
    temporary = root / "results.csv.tmp"
    with temporary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "results.csv")
    metadata["completed_cells"] = len(rows)
    temporary = root / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(root / "metadata.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("quick", "validation", "release", "statistical"), default="quick"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        action="append",
        help="Restrict to selected scenarios (repeatable).",
    )
    args = parser.parse_args()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    )
    if args.profile in {"release", "statistical"} and dirty:
        parser.error(
            "Long evidence runs require a clean worktree, including untracked source/tests. "
            "Commit locally first; no push required."
        )
    trials, resamples = {
        "quick": (2, 99),
        "validation": (30, 499),
        "release": (300, 4999),
        "statistical": (1000, 4999),
    }[args.profile]
    config = {
        "profile": args.profile,
        "n_simulations": trials,
        "n_resamples": resamples,
        "bootstrapx": bx.__version__,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": source_digest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "confidence_level": 0.95,
        "schema_version": 1,
        "scenarios": list(dict.fromkeys(args.scenario or SCENARIOS)),
    }
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "metadata.json"
    rows = []
    if (root / "results.csv").exists() and not args.resume:
        parser.error("results.csv already exists; use --resume or a new output directory")
    if args.resume and (root / "results.csv").exists() and not metadata_path.exists():
        parser.error("cannot resume results without metadata")
    if args.resume and metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if any(previous.get(key) != value for key, value in config.items()):
            parser.error("cannot resume: source, environment, commit or configuration changed")
        if (root / "results.csv").exists():
            with (root / "results.csv").open(newline="") as input_file:
                rows = list(csv.DictReader(input_file))
    metadata = {
        **config,
        "git_worktree_dirty": dirty,
        "release_evidence": args.profile in {"release", "statistical"},
        "random_streams": "data=[scenario,trial,0], resampling=[scenario,trial,1]",
        "status": "running",
        "memory_measure": "tracemalloc allocations, not process RSS",
        "checkpoint": "per cell; interrupted cell restarts",
    }
    tasks = [
        (name, library, method)
        for name in config["scenarios"]
        for library in ("bootstrapx", "scipy", "delta")
        for method in (("normal",) if library == "delta" else ("percentile", "basic", "bca"))
    ]
    metadata["total_cells"] = len(tasks)
    done = {(r["scenario"], r["library"], r["method"]) for r in rows}
    print(
        f"{args.profile}: {len(tasks)} cells, {trials} trials/cell, {resamples} resamples. "
        "Quick/validation are NOT release coverage evidence.",
        flush=True,
    )
    for name, library, method in tasks:
        if (name, library, method) in done:
            continue
        hits = valid = failed = invalid = 0
        first_failure, widths = "", []
        started = time.perf_counter()
        scenario = SCENARIOS.index(name)
        for trial in range(trials):
            data = make_data(
                name, np.random.default_rng(np.random.SeedSequence([scenario, trial, 0]))
            )
            rng = np.random.default_rng(np.random.SeedSequence([scenario, trial, 1]))
            try:
                low, high = interval(library, method, data, resamples, rng, name == "paired")
                if not (np.isfinite(low) and np.isfinite(high) and high >= low):
                    invalid += 1
                else:
                    valid += 1
                    hits += int(low <= data[-1] <= high)
                    widths.append(high - low)
            except Exception as error:
                failed += 1
                first_failure = first_failure or f"{type(error).__name__}: {error}"[:300]
            if trials > 2 and (trial + 1) % 10 == 0:
                print(f"  {name}/{library}/{method}: {trial + 1}/{trials}", flush=True)
        mc_low, mc_high = wilson(hits, trials)
        rows.append(
            {
                "scenario": name,
                "library": library,
                "method": method,
                "true_effect": data[-1],
                "n_simulations": trials,
                "covered_trials": hits,
                "valid_trials": valid,
                "invalid_trials": invalid,
                "failed_trials": failed,
                "coverage_all_trials": hits / trials,
                "coverage_valid_trials": hits / valid if valid else "nan",
                "coverage_mc_low": mc_low,
                "coverage_mc_high": mc_high,
                "mean_valid_width": float(np.mean(widths)) if widths else "nan",
                "first_failure": first_failure,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        checkpoint(root, metadata, rows)
        print(
            f"{len(rows)}/{len(tasks)} {name}/{library}/{method}: "
            f"valid={valid}, invalid={invalid}, failed={failed}",
            flush=True,
        )
    # Timing is separate from simulation coverage. Repeatable seed, one warmup,
    # median of measurements; allocation peak is measured in a separate call.
    timings = []
    runtime_sizes = (100,) if args.profile in {"quick", "validation"} else (200, 1000, 10000)
    runtime_tasks = [(n, design) for n in runtime_sizes for design in ("independent", "cluster")]
    for n, design in runtime_tasks:
        data = make_data(
            design,
            np.random.default_rng(2026),
            n,
            n + 50,
            cluster_counts=(max(3, n // 8), max(3, n // 8) + 5),
        )
        for method in ("percentile", "bca"):
            libraries = (
                ("bootstrapx", "scipy", "scipy_vectorized")
                if design == "independent"
                else ("bootstrapx", "scipy")
            )
            for library in libraries:
                call = partial(interval, library, method, data, resamples, paired=False)

                def run(call=call):
                    return call(rng=np.random.default_rng(42))

                run()
                elapsed = []
                for _ in range(2 if args.profile == "quick" else 5):
                    start = time.perf_counter()
                    run()
                    elapsed.append(time.perf_counter() - start)
                tracemalloc.start()
                try:
                    run()
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                timings.append(
                    {
                        "library": library,
                        "design": design,
                        "method": method,
                        "n_control": len(data[0]),
                        "n_treatment": len(data[1]),
                        "n_control_units": len(data[0])
                        if data[2] is None
                        else len(np.unique(data[2])),
                        "n_treatment_units": len(data[1])
                        if data[3] is None
                        else len(np.unique(data[3])),
                        "n_resamples": resamples,
                        "median_seconds": float(np.median(elapsed)),
                        "peak_tracemalloc_bytes": peak,
                    }
                )
    with (root / "runtime.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(timings[0]))
        writer.writeheader()
        writer.writerows(timings)
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if source_digest() != config["source_sha256"] or current_commit != config["git_commit"]:
        metadata["status"] = "source_changed_during_run"
        checkpoint(root, metadata, rows)
        raise SystemExit("Source/commit changed during the run; results are not complete evidence.")
    metadata["status"] = "complete"
    checkpoint(root, metadata, rows)
    print(
        f"Complete: {root}. Coverage uses scalar unit-index references; "
        "runtime also includes bounded-vectorized SciPy.",
        flush=True,
    )


if __name__ == "__main__":
    main()

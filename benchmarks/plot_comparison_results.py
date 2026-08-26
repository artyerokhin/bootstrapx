#!/usr/bin/env python3
"""Render 0.5.0 comparison figures from a reviewed benchmark directory."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _save_coverage(rows: list[dict[str, str]], output_dir: Path) -> None:
    scipy = {
        (row["scenario"], row["method"]): float(row["coverage"])
        for row in rows
        if row["library"] == "scipy"
    }
    matched = [
        row
        for row in rows
        if row["library"] == "bootstrapx" and (row["scenario"], row["method"]) in scipy
    ]
    methods = ("percentile", "basic", "bca")
    colors = {"percentile": "#2563eb", "basic": "#d97706", "bca": "#0f766e"}

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    figure.patch.set_facecolor("#f8fafc")
    for method in methods:
        selected = [row for row in matched if row["method"] == method]
        axes[0].scatter(
            [scipy[(row["scenario"], method)] for row in selected],
            [float(row["coverage"]) for row in selected],
            color=colors[method],
            label=method.upper(),
            s=48,
            alpha=0.85,
        )
    axes[0].plot([0.82, 1.0], [0.82, 1.0], "--", color="#64748b", linewidth=1)
    axes[0].axhline(0.95, color="#a21caf", linestyle=":", linewidth=1.3)
    axes[0].axvline(0.95, color="#a21caf", linestyle=":", linewidth=1.3)
    axes[0].set(
        xlim=(0.82, 1.0),
        ylim=(0.82, 1.0),
        xlabel="SciPy empirical coverage",
        ylabel="bootstrapx empirical coverage",
        title="Matched independent and paired cells",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    bootstrapx_rows = [row for row in rows if row["library"] == "bootstrapx"]
    scenarios = list(dict.fromkeys(row["scenario"] for row in bootstrapx_rows))
    positions = np.arange(len(scenarios))
    width = 0.25
    for offset, method in enumerate(methods):
        selected = {row["scenario"]: row for row in bootstrapx_rows if row["method"] == method}
        coverage = [float(selected[scenario]["coverage"]) for scenario in scenarios]
        low = [float(selected[scenario]["coverage_mc_low"]) for scenario in scenarios]
        high = [float(selected[scenario]["coverage_mc_high"]) for scenario in scenarios]
        errors = np.array(
            [
                [value - lower for value, lower in zip(coverage, low, strict=True)],
                [upper - value for value, upper in zip(coverage, high, strict=True)],
            ]
        )
        axes[1].bar(
            positions + (offset - 1) * width,
            coverage,
            width,
            yerr=errors,
            capsize=2,
            color=colors[method],
            label=method.upper(),
        )
    axes[1].axhline(0.95, color="#a21caf", linestyle=":", linewidth=1.3)
    axes[1].set_xticks(positions, [name.replace("_", "\n") for name in scenarios])
    axes[1].set_ylim(0.75, 1.01)
    axes[1].set_ylabel("empirical coverage with 95% Wilson interval")
    axes[1].set_title("bootstrapx by scenario")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    figure.suptitle("Two-sample confidence-interval coverage")
    figure.tight_layout()
    figure.savefig(output_dir / "coverage.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_runtime(rows: list[dict[str, str]], output_dir: Path) -> None:
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    figure.patch.set_facecolor("#f8fafc")
    colors = {"bootstrapx": "#0f766e", "scipy": "#475569"}
    for axis, method in zip(axes, ("percentile", "bca"), strict=True):
        ordered = sorted(by_method[method], key=lambda row: int(row["n_control"]))
        n = [int(row["n_control"]) for row in ordered]
        axis.plot(
            n,
            [float(row["bootstrapx_ms"]) for row in ordered],
            "o-",
            label="bootstrapx",
            color=colors["bootstrapx"],
            linewidth=2,
        )
        axis.plot(
            n,
            [float(row["scipy_ms"]) for row in ordered],
            "s-",
            label="SciPy",
            color=colors["scipy"],
            linewidth=2,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(method.upper())
        axis.set_xlabel("control sample size")
        axis.set_ylabel("median runtime (ms)")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Two-sample mean difference runtime")
    figure.tight_layout()
    figure.savefig(output_dir / "runtime.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_coverage(_read_csv(input_dir / "coverage" / "coverage.csv"), output_dir)
    _save_runtime(_read_csv(input_dir / "runtime" / "runtime.csv"), output_dir)


if __name__ == "__main__":
    main()

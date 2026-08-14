#!/usr/bin/env python3
"""Render documented figures from a versioned release benchmark directory."""

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


def _save_speed(rows: list[dict[str, str]], output_dir: Path) -> None:
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    figure.patch.set_facecolor("#f8fafc")
    for axis, (method, method_rows) in zip(axes, sorted(by_method.items()), strict=True):
        ordered = sorted(method_rows, key=lambda row: int(row["n"]))
        n = [int(row["n"]) for row in ordered]
        bx = [float(row["bootstrapx_ms"]) for row in ordered]
        scipy = [float(row["scipy_ms"]) for row in ordered]
        axis.plot(n, bx, "o-", label="bootstrapx", color="#0f766e", linewidth=2)
        axis.plot(n, scipy, "s-", label="SciPy", color="#475569", linewidth=2)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(method.upper())
        axis.set_xlabel("sample size n")
        axis.set_ylabel("median runtime (ms)")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("bootstrapx vs SciPy — 4,999 resamples, np.mean")
    figure.tight_layout()
    figure.savefig(output_dir / "speed-vs-scipy.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_coverage(rows: list[dict[str, str]], output_dir: Path) -> None:
    pairs = []
    for row in rows:
        if row["library"] != "bootstrapx":
            continue
        matching = next(
            candidate
            for candidate in rows
            if candidate["library"] == "scipy"
            and all(
                candidate[field] == row[field]
                for field in ("method", "statistic", "n", "distribution")
            )
        )
        pairs.append((row, matching))

    figure, axis = plt.subplots(figsize=(6.5, 6.5))
    figure.patch.set_facecolor("#f8fafc")
    for method, color in (("bca", "#0f766e"), ("percentile", "#2563eb")):
        subset = [pair for pair in pairs if pair[0]["method"] == method]
        axis.scatter(
            [float(pair[1]["coverage"]) for pair in subset],
            [float(pair[0]["coverage"]) for pair in subset],
            label=method.upper(),
            color=color,
            alpha=0.8,
            s=38,
        )
    axis.plot([0.88, 0.98], [0.88, 0.98], "--", color="#64748b", linewidth=1)
    axis.axhline(0.95, color="#a21caf", linestyle=":", linewidth=1.4, label="nominal 0.95")
    axis.axvline(0.95, color="#a21caf", linestyle=":", linewidth=1.4)
    axis.set(xlim=(0.88, 0.98), ylim=(0.88, 0.98))
    axis.set_xlabel("SciPy empirical coverage")
    axis.set_ylabel("bootstrapx empirical coverage")
    axis.set_title("Matched coverage cells — 300 datasets per cell")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "coverage-vs-scipy.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_numba(rows: list[dict[str, str]], output_dir: Path) -> None:
    sizes = sorted({int(row["n"]) for row in rows})
    methods = ("mbb", "cbb", "stationary")
    figure, axis = plt.subplots(figsize=(8, 4.8))
    figure.patch.set_facecolor("#f8fafc")
    width = 0.24
    positions = np.arange(len(sizes))
    colors = {"mbb": "#0f766e", "cbb": "#2563eb", "stationary": "#7c3aed"}
    for offset, method in enumerate(methods):
        values = [
            float(
                next(row for row in rows if row["n"] == str(n) and row["method"] == method)[
                    "fallback_over_numba"
                ]
            )
            for n in sizes
        ]
        axis.bar(
            positions + (offset - 1) * width,
            values,
            width,
            label=method.upper(),
            color=colors[method],
        )
    axis.axhline(1, color="#64748b", linestyle="--", linewidth=1)
    axis.set_xticks(positions, [f"n={n:,}" for n in sizes])
    axis.set_ylabel("Python fallback / Numba warm runtime")
    axis.set_title("Optional Numba acceleration — 500 resamples, np.mean")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "numba-speedup.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_speed(_read_csv(input_dir / "speed" / "speed.csv"), output_dir)
    _save_coverage(_read_csv(input_dir / "coverage" / "coverage.csv"), output_dir)
    _save_numba(_read_csv(input_dir / "numba" / "numba.csv"), output_dir)


if __name__ == "__main__":
    main()

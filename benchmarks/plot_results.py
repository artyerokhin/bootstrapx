#!/usr/bin/env python3
"""benchmarks/plot_results.py

Reads CSVs produced by bench_coverage_accuracy.py and bench_speed.py,
outputs PNGs into benchmarks/.

Usage:
    python benchmarks/plot_results.py
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DIR = "benchmarks"

COLORS = {
    ("bootstrapx", "bca"):        "#1a8a8f",
    ("bootstrapx", "percentile"): "#4fc3d0",
    ("scipy",      "bca"):        "#5a5a6a",
    ("scipy",      "percentile"): "#aaaabc",
}
DIST_LABELS = {
    "normal":      "Normal",
    "lognormal":   "LogNormal",
    "exponential": "Exponential",
    "t3":          "t(3)",
}
LIB_METHODS = [
    ("bootstrapx", "bca"),
    ("bootstrapx", "percentile"),
    ("scipy",      "bca"),
    ("scipy",      "percentile"),
]

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

# ── 1. Coverage heatmaps — one per (stat, n) ─────────────────────────────────
cov_path = os.path.join(DIR, "results_coverage.csv")
if os.path.exists(cov_path):
    rows = read_csv(cov_path)
    stats   = sorted(set(r["stat"] for r in rows))
    ns      = sorted(set(int(r["n"]) for r in rows))
    dists   = list(DIST_LABELS.keys())

    for stat in stats:
        for n in ns:
            subset = [r for r in rows if r["stat"] == stat and int(r["n"]) == n]
            if not subset:
                continue

            x      = np.arange(len(dists))
            n_bars = len(LIB_METHODS)
            width  = 0.18

            fig, ax = plt.subplots(figsize=(11, 5))
            fig.patch.set_facecolor("#f7f6f2")
            ax.set_facecolor("#f7f6f2")

            for i, (lib, method) in enumerate(LIB_METHODS):
                vals = []
                for d in dists:
                    match = [r for r in subset
                             if r["library"] == lib and r["method"] == method
                             and r["distribution"] == d]
                    v = float(match[0]["coverage"]) if match and match[0]["coverage"] != "nan" else float("nan")
                    vals.append(v)
                offset = (i - (n_bars - 1) / 2) * width
                bars = ax.bar(x + offset, vals, width=width * 0.92,
                              color=COLORS[(lib, method)],
                              label=f"{lib} {method}")
                for bar, v in zip(bars, vals):
                    if np.isfinite(v):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.003,
                                f"{v:.3f}", ha="center", va="bottom",
                                fontsize=7.5, color="#222")

            ax.axhline(0.95, color="#a12c7b", linestyle="--", linewidth=1.5,
                       label="Target 0.95")
            ax.set_ylim(0.60, 1.02)
            ax.set_xticks(x)
            ax.set_xticklabels([DIST_LABELS.get(d, d) for d in dists], fontsize=11)
            ax.set_xlabel("Distribution", fontsize=12)
            ax.set_ylabel("Empirical coverage", fontsize=12)
            ax.set_title(f"Coverage accuracy — stat={stat}, n={n}  (target 0.95)", fontsize=13)
            ax.legend(fontsize=9, loc="lower right")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()

            out = os.path.join(DIR, f"fig_coverage_{stat}_n{n}.png")
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: {out}")

# ── 2. Speed: BCa log-log ─────────────────────────────────────────────────────
speed_path = os.path.join(DIR, "results_speed_bca.csv")
if os.path.exists(speed_path):
    rows     = read_csv(speed_path)
    ns       = [int(r["n"])        for r in rows]
    bx_ms    = [float(r["bx_ms"]) for r in rows]
    sp_ms    = [float(r["scipy_ms"]) for r in rows]
    speedups = [float(r["speedup"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#f7f6f2")
    for ax in (ax1, ax2):
        ax.set_facecolor("#f7f6f2")
        ax.spines[["top", "right"]].set_visible(False)

    ax1.plot(ns, bx_ms, "o-", color="#1a8a8f", linewidth=2.2, label="bootstrapx BCa")
    ax1.plot(ns, sp_ms, "s-", color="#5a5a6a", linewidth=2.2, label="scipy BCa")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.set_xlabel("n", fontsize=12)
    ax1.set_ylabel("Median time (ms, 5 runs)", fontsize=12)
    ax1.set_title("BCa speed vs scipy (log-log)", fontsize=13)
    ax1.legend(fontsize=10); ax1.grid(alpha=0.3)

    bar_colors = ["#1a8a8f" if s >= 1.0 else "#a12c7b" for s in speedups]
    bars = ax2.bar([str(n) for n in ns], speedups, color=bar_colors)
    ax2.axhline(1.0, color="#888", linestyle="--", linewidth=1.3)
    for bar, s in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05,
                 f"{s:.2f}×", ha="center", va="bottom", fontsize=9)
    ax2.set_xlabel("n", fontsize=12)
    ax2.set_ylabel("Speedup (scipy / bootstrapx)", fontsize=12)
    ax2.set_title("Speedup — red = bootstrapx slower", fontsize=13)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(DIR, "fig_speed_bca.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

# ── 3. Memory ─────────────────────────────────────────────────────────────────
mem_path = os.path.join(DIR, "results_memory.csv")
if os.path.exists(mem_path):
    rows  = read_csv(mem_path)
    ns    = [int(r["n"])           for r in rows]
    bx_mb = [float(r["bx_mb"])    for r in rows]
    sp_mb = [float(r["scipy_mb"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("#f7f6f2"); ax.set_facecolor("#f7f6f2")
    ax.plot(ns, bx_mb, "o-", color="#1a8a8f", linewidth=2.2, label="bootstrapx")
    ax.plot(ns, sp_mb, "s-", color="#5a5a6a", linewidth=2.2, label="scipy")
    ax.set_xlabel("n", fontsize=12); ax.set_ylabel("Peak memory (MB)", fontsize=12)
    ax.set_title("Memory usage — BCa bootstrap", fontsize=13)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = os.path.join(DIR, "fig_memory.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

print("\nDone.")

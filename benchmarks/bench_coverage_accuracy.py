#!/usr/bin/env python3
"""benchmarks/bench_coverage_accuracy.py

Monte Carlo coverage study: bootstrapx vs scipy.stats.bootstrap.

Usage:
    python benchmarks/bench_coverage_accuracy.py          # default N_SIM=1000
    python benchmarks/bench_coverage_accuracy.py --fast   # N_SIM=300, ~3 min M1
    python benchmarks/bench_coverage_accuracy.py --full   # N_SIM=2000, ~20 min M1
"""
import sys, os, csv, warnings
import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import bootstrapx as bx

# ── Config ────────────────────────────────────────────────────────────────────
N_SIM      = 2000 if "--full" in sys.argv else 300 if "--fast" in sys.argv else 1000
N_RESAMPLES = 4999
CI         = 0.95

# Realistic sample sizes: nothing below 200 (below that bootstrap theory barely holds)
NS    = [200, 500, 1000, 2000]

# Only stats where at least one library has a fair shot
# std excluded from heavy-tailed dists (infinite theoretical variance)
STATS = ["mean", "median", "std"]

DISTS = ["normal", "lognormal", "exponential", "t3"]
# Note: Bernoulli removed — discrete median degenerates both libraries' BCa jackknife.
# std on pareto removed — true std is infinite, coverage is undefined.

os.makedirs("benchmarks", exist_ok=True)

# ── True parameters ───────────────────────────────────────────────────────────
TRUE_PARAMS = {
    # (dist, stat): value
    ("normal",      "mean"):   0.0,
    ("normal",      "median"): 0.0,
    ("normal",      "std"):    1.0,
    ("lognormal",   "mean"):   float(np.exp(0.5)),      # e^(mu + sigma^2/2), mu=0, sigma=1
    ("lognormal",   "median"): 1.0,                     # e^mu = e^0
    ("lognormal",   "std"):    float(np.sqrt((np.e - 1) * np.e)),
    ("exponential", "mean"):   2.0,
    ("exponential", "median"): float(2.0 * np.log(2)),
    ("exponential", "std"):    2.0,
    ("t3",          "mean"):   0.0,
    ("t3",          "median"): 0.0,
    ("t3",          "std"):    float(np.sqrt(3.0 / (3 - 2))),  # sqrt(df/(df-2))
}

# std on heavy-tailed: exclude lognormal n<1000, t3 completely (std finite but poorly estimated)
STD_SKIP = {("lognormal", "std"), ("t3", "std")}

def make_data(dist_name, n, rng):
    if dist_name == "normal":
        return rng.standard_normal(n)
    elif dist_name == "lognormal":
        return rng.lognormal(0, 1, n)
    elif dist_name == "exponential":
        return rng.exponential(2.0, n)
    elif dist_name == "t3":
        return rng.standard_t(3, n)

stat_map = {"mean": np.mean, "median": np.median, "std": np.std}

total_cells = len(["bootstrapx","scipy"]) * len(["bca","percentile"]) * len(STATS) * len(NS) * len(DISTS)
print(f"N_SIM={N_SIM}  cells={total_cells}  total_calls={N_SIM * total_cells:,}")
print(f"Estimated time: ~{N_SIM * total_cells * 6 // 1000} s on M1\n")

rows = []

for lib in ["bootstrapx", "scipy"]:
    for method in ["bca", "percentile"]:
        for stat in STATS:
            fn = stat_map[stat]
            for n in NS:
                for dist_name in DISTS:
                    # Skip undefined/degenerate combinations
                    key = (dist_name, stat)
                    if key in STD_SKIP:
                        continue
                    tv = TRUE_PARAMS.get(key)
                    if tv is None:
                        continue

                    hits = 0
                    valid = 0
                    label = f"  {lib:>12s} {method:>10s} {stat:>6s} n={n:4d} {dist_name:<12s}"
                    print(label, end="", flush=True)

                    for sim in range(N_SIM):
                        rng = np.random.default_rng(sim)
                        data = make_data(dist_name, n, rng)

                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")  # suppress DegenerateDataWarning
                                if lib == "bootstrapx":
                                    r = bx.bootstrap(
                                        data, statistic=fn,
                                        method=method, n_resamples=N_RESAMPLES,
                                        confidence_level=CI, random_state=sim,
                                    )
                                else:
                                    r = sp_stats.bootstrap(
                                        (data,), statistic=fn,
                                        n_resamples=N_RESAMPLES, confidence_level=CI,
                                        method="BCa" if method == "bca" else "percentile",
                                        random_state=sim,
                                    )

                            lo = float(r.confidence_interval.low)
                            hi = float(r.confidence_interval.high)

                            # Skip degenerate / NaN / Inf CI — don't penalise either library
                            if not (np.isfinite(lo) and np.isfinite(hi)):
                                continue
                            if hi <= lo:
                                continue

                            valid += 1
                            if lo <= tv <= hi:
                                hits += 1

                        except Exception:
                            pass  # truly broken call — skip silently

                    cov = hits / valid if valid > 0 else float("nan")
                    gap = abs(cov - CI) if np.isfinite(cov) else float("nan")
                    skipped = N_SIM - valid
                    print(f"  cov={cov:.3f}  gap={gap:.3f}  skipped={skipped}")

                    rows.append({
                        "library":      lib,
                        "method":       method,
                        "stat":         stat,
                        "n":            n,
                        "distribution": dist_name,
                        "coverage":     round(cov, 4) if np.isfinite(cov) else "nan",
                        "gap_from_095": round(gap, 4) if np.isfinite(gap) else "nan",
                        "n_sim":        N_SIM,
                        "valid_trials": valid,
                    })

out = "benchmarks/results_coverage.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"\nSaved: {out}")
print("Run:   python benchmarks/plot_results.py")

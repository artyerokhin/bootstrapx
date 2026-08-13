#!/usr/bin/env python3
"""benchmarks/bench_speed.py — bootstrapx vs scipy: speed & memory.

Usage:
    python benchmarks/bench_speed.py           # full (~5 min M1)
    python benchmarks/bench_speed.py --quick   # quick (~1 min M1)
"""
import sys, os, time, tracemalloc, csv, gc, warnings, json, platform
import numpy as np
import scipy
from scipy import stats as sp_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import bootstrapx as bx

QUICK      = "--quick" in sys.argv
N_RESAMPLES = 4999
CI         = 0.95
os.makedirs("benchmarks", exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def timeit(fn, repeats=5):
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))

def peak_mb(fn):
    gc.collect()
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024

# ── Wrappers — detect correct confidence_level param name ────────────────────
def _bx_call(data, statistic=np.mean, method="bca"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # bootstrapx uses confidence_level (same as scipy)
        return bx.bootstrap(
            data, statistic=statistic,
            method=method, n_resamples=N_RESAMPLES,
            confidence_level=CI, random_state=42,
        )

def _sp_call(data, statistic=np.mean, method="bca"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp_stats.bootstrap(
            (data,), statistic=statistic,
            n_resamples=N_RESAMPLES, confidence_level=CI,
            method="BCa" if method == "bca" else "percentile",
            random_state=42,
        )

# ── 1. BCa speed: realistic n=200..10000 ─────────────────────────────────────
print("\n── BCa speed (np.mean) ──────────────────────────────────────────────")
NS_BCa = [200, 500, 1000, 2000, 5000, 10_000] if not QUICK else [200, 500, 1000, 2000]
bca_rows = []
print(f"{'n':>8}  {'bx_ms':>8}  {'scipy_ms':>10}  {'speedup':>8}")
print("-" * 45)
for n in NS_BCa:
    d = np.random.default_rng(0).standard_normal(n)
    t_bx = timeit(lambda: _bx_call(d))
    t_sp = timeit(lambda: _sp_call(d))
    sp   = t_sp / t_bx if t_bx > 0 else float("nan")
    bca_rows.append({"n": n, "bx_ms": round(t_bx, 2), "scipy_ms": round(t_sp, 2),
                     "speedup": round(sp, 2), "method": "bca"})
    marker = "  ✓ faster" if sp > 1.05 else ("  ✗ slower" if sp < 0.95 else "  ~ equal")
    print(f"{n:>8d}  {t_bx:>7.1f}ms  {t_sp:>9.1f}ms  {sp:>7.2f}×{marker}")

with open("benchmarks/results_speed_bca.csv", "w", newline="") as f:
    w = csv.DictWriter(f, ["n", "bx_ms", "scipy_ms", "speedup", "method"])
    w.writeheader(); w.writerows(bca_rows)

# ── 2. Percentile speed: larger n where it shines ────────────────────────────
print("\n── Percentile speed (np.mean) ───────────────────────────────────────")
NS_pct = [1000, 5000, 10_000, 50_000, 100_000] if not QUICK else [1000, 5000, 10_000]
pct_rows = []
print(f"{'n':>8}  {'bx_ms':>8}  {'scipy_ms':>10}  {'speedup':>8}")
print("-" * 45)
for n in NS_pct:
    d = np.random.default_rng(0).standard_normal(n)
    t_bx = timeit(lambda: _bx_call(d, method="percentile"))
    t_sp = timeit(lambda: _sp_call(d, method="percentile"))
    sp   = t_sp / t_bx if t_bx > 0 else float("nan")
    pct_rows.append({"n": n, "bx_ms": round(t_bx, 2), "scipy_ms": round(t_sp, 2),
                     "speedup": round(sp, 2), "method": "percentile"})
    print(f"{n:>8d}  {t_bx:>7.1f}ms  {t_sp:>9.1f}ms  {sp:>7.2f}×")

with open("benchmarks/results_speed_percentile.csv", "w", newline="") as f:
    w = csv.DictWriter(f, ["n", "bx_ms", "scipy_ms", "speedup", "method"])
    w.writeheader(); w.writerows(pct_rows)

# combined for plot_results.py
all_speed = bca_rows + pct_rows
with open("benchmarks/results_speed.csv", "w", newline="") as f:
    w = csv.DictWriter(f, ["n", "method", "bx_ms", "scipy_ms", "speedup"])
    w.writeheader(); w.writerows(all_speed)

# ── 3. Memory: BCa, n=500..10000 ─────────────────────────────────────────────
print("\n── Memory peak (BCa, np.mean) ───────────────────────────────────────")
NS_M = [500, 1000, 2000, 5000, 10_000] if not QUICK else [500, 1000, 2000]
mem_rows = []
print(f"{'n':>8}  {'bx_mb':>8}  {'scipy_mb':>10}  {'ratio':>7}")
print("-" * 40)
for n in NS_M:
    d = np.random.default_rng(1).standard_normal(n)
    bm = peak_mb(lambda: _bx_call(d))
    sm = peak_mb(lambda: _sp_call(d))
    ratio = sm / bm if bm > 0 else float("nan")
    mem_rows.append({"n": n, "bx_mb": round(bm, 3), "scipy_mb": round(sm, 3),
                     "ratio": round(ratio, 2)})
    print(f"{n:>8d}  {bm:>7.2f}MB  {sm:>9.2f}MB  {ratio:>6.2f}×")

with open("benchmarks/results_memory.csv", "w", newline="") as f:
    w = csv.DictWriter(f, ["n", "bx_mb", "scipy_mb", "ratio"])
    w.writeheader(); w.writerows(mem_rows)

# ── 4. Arbitrary callables ────────────────────────────────────────────────────
print("\n── Arbitrary callables (BCa) ────────────────────────────────────────")
from scipy.stats import trim_mean, iqr as sp_iqr
def trimmed_mean(x): return trim_mean(x, 0.1)
def iqr(x):          return float(sp_iqr(x))

NS_C = [500, 1000, 2000] if not QUICK else [500, 1000]
call_rows = []
for n in NS_C:
    d = np.random.default_rng(2).standard_normal(n)
    for name, fn in [("trimmed_mean", trimmed_mean), ("iqr", iqr)]:
        t_bx = timeit(lambda: _bx_call(d, statistic=fn))
        t_sp = timeit(lambda: _sp_call(d, statistic=fn))
        call_rows.append({"n": n, "stat": name,
                          "bx_ms": round(t_bx, 2), "scipy_ms": round(t_sp, 2)})
        print(f"  n={n:<5d} {name:<16} bx={t_bx:6.1f}ms  scipy={t_sp:6.1f}ms")

with open("benchmarks/results_callable.csv", "w", newline="") as f:
    w = csv.DictWriter(f, ["n", "stat", "bx_ms", "scipy_ms"])
    w.writeheader(); w.writerows(call_rows)

print("\nDone.")
metadata = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "bootstrapx": bx.__version__,
    "n_resamples": N_RESAMPLES,
    "confidence_level": CI,
    "runtime_repeats": 5,
    "runtime_summary": "median",
    "memory_tool": "tracemalloc",
    "quick": QUICK,
}
with open("benchmarks/results_speed_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Run: python benchmarks/plot_results.py")

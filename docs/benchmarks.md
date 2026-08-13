# Benchmarks

Performance claims are meaningful only when the method, statistic, sample size,
resample count, warm-up state, and dependency set match.

## General comparison

```bash
python benchmarks/bench_speed.py
```

## Optional Numba comparison

```bash
pip install -e ".[numba]"
python benchmarks/bench_numba.py
```

The Numba benchmark compares the same time-series resampling workflow against
bootstrapx's Python fallback in one process. It reports cold-call latency and
warm median runtime separately. It is intentionally limited to methods whose
index generators actually use Numba.

When publishing numbers, include CPU, OS, Python, NumPy, SciPy and Numba
versions, `n`, `n_resamples`, statistic, method, block setting, and whether the
measurement includes JIT startup.

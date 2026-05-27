# Benchmarks

`bootstrapx` focuses on CPU-portable performance that works on most machines.

## What is fast

- Batched iid resampling for scalar statistics.
- Vectorized statistics where the user provides a batch-aware callable.
- Time-series sieve implementation with efficient filtering-based generation.

## Benchmark philosophy

Compare methods under the same:

- sample size
- number of resamples
- statistic
- confidence method
- random seed when possible

## Reproducible benchmark command

```bash
python benchmarks/bench_speed.py
```

## What to report

When publishing benchmark numbers, include:

- CPU model
- Python version
- NumPy / SciPy version
- method name
- `n_resamples`
- sample size `n`

This keeps performance claims credible and repeatable.

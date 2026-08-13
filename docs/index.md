# bootstrapx

**bootstrapx** is a Python library for practical bootstrap uncertainty estimation.

It is designed for cases where `scipy.stats.bootstrap` is not enough: time-series dependence, cluster-aware resampling, weighted bootstrap, studentized intervals, and production-safe reproducibility.

The current API focuses on one-sample scalar statistics. Read
[Current limitations](limitations.md) before using it for experiment
comparisons, multi-output statistics, or complex survey designs.

## Why bootstrapx

- 15+ bootstrap methods in one API.
- Reproducible random-state handling.
- Memory-safe batched computation.
- Integrations for pandas and scikit-learn.
- Focus on real-world analytics, ML evaluation, and time-series work.

## Where it fits

Use `bootstrapx` when you need one of these:

- BCa or studentized intervals for a custom statistic.
- Block bootstrap for dependent time series.
- Cluster or stratified bootstrap for grouped data.
- Bootstrap uncertainty around model metrics.
- A consistent API that works from notebooks to backend services.

## Quick example

```python
import numpy as np
from bootstrapx import bootstrap

data = np.random.default_rng(42).normal(5, 2, size=300)
result = bootstrap(data, np.mean, method="bca", n_resamples=4999, random_state=42)

print(result)
print(result.confidence_interval.low, result.confidence_interval.high)
```

## Main sections

- **Getting Started**: install and first examples.
- **Methods**: which bootstrap method to choose and why.
- **Integrations**: pandas and scikit-learn workflows.
- **A/B Testing**: cluster-aware patterns and practical guidance.
- **Time Series**: block, stationary, sieve, and wild bootstrap.
- **API Reference**: public interfaces and examples.

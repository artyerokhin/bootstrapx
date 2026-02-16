# Getting Started

## Installation

```bash
pip install bootstrapx-lib

# With GPU support
pip install "bootstrapx-lib[cuda]"

# For development
pip install -e ".[dev,docs]"
```

## Basic Usage

```python
import numpy as np
from bootstrapx import bootstrap

data = np.random.default_rng(42).normal(5, 2, size=200)

# BCa (default) — best general-purpose method
result = bootstrap(data, np.mean)
print(f"Mean: {result.theta_hat:.3f}")
print(f"95% CI: [{result.confidence_interval.low:.3f}, {result.confidence_interval.high:.3f}]")
print(f"SE: {result.standard_error:.3f}")
```

## Comparing CI Methods

```python
data = np.random.default_rng(0).exponential(2.0, size=150)

for method in ["percentile", "basic", "bca"]:
    r = bootstrap(data, np.mean, method=method, n_resamples=9999, random_state=42)
    ci = r.confidence_interval
    print(f"{method:>12s}: [{ci.low:.3f}, {ci.high:.3f}]  se={r.standard_error:.3f}")
```

## Time Series

```python
ts = np.cumsum(np.random.default_rng(0).standard_normal(500))
result = bootstrap(ts, np.mean, method="stationary", mean_block=15.0)
```

## Custom Statistics

```python
def iqr(x):
    return float(np.percentile(x, 75) - np.percentile(x, 25))

result = bootstrap(data, iqr, method="bca", n_resamples=9999)
```

## Reproducibility

```python
r1 = bootstrap(data, np.mean, random_state=42)
r2 = bootstrap(data, np.mean, random_state=42)
assert np.array_equal(r1.bootstrap_distribution, r2.bootstrap_distribution)
```

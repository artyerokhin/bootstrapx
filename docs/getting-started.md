# Getting Started

## Installation

Choose the smallest install that fits your workflow.

```bash
pip install bootstrapx-lib
```

Optional extras:

```bash
pip install "bootstrapx-lib[pandas]"
pip install "bootstrapx-lib[sklearn]"
pip install "bootstrapx-lib[numba]"
pip install "bootstrapx-lib[pandas,sklearn,numba]"
```

## First bootstrap interval

```python
import numpy as np
from bootstrapx import bootstrap

data = np.random.default_rng(0).normal(loc=5.0, scale=2.0, size=200)
result = bootstrap(data, np.mean, method="bca", n_resamples=4999, random_state=42)

print(result.theta_hat)
print(result.standard_error)
print(result.confidence_interval)

# Export a compact record for reports or experiment tracking
record = result.to_dict()
frame = result.to_frame()  # requires the pandas extra
```

## Choosing a method

- `bca`: best default for general statistics.
- `percentile`: simplest and fast.
- `basic`: useful when symmetry assumptions are acceptable.
- `studentized`: higher compute cost, useful when bootstrap-t is desired.
- `mbb`, `stationary`, `sieve`: for dependent time series.
- `cluster`, `strata`: for grouped or sampled data.

## Reproducibility

Always pass `random_state` in production code:

```python
result = bootstrap(data, np.mean, random_state=123)
```

This is especially important for pipelines, tests, and repeated model evaluation.

## Local development

```bash
git clone https://github.com/artyerokhin/bootstrapx.git
cd bootstrapx
pip install -e ".[dev]"
pytest tests/ -v
```

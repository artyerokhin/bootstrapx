# Time Series

Time-series data violates the iid assumption behind ordinary bootstrap.
Use one of the dependent-data methods instead.

## Quick guidance

- `mbb`: simple and reliable for local dependence.
- `cbb`: useful when edge effects matter.
- `stationary`: good default for stationary dependent series.
- `sieve`: useful when an AR approximation is reasonable.
- `wild`: useful for heteroscedastic residuals.

## Moving block bootstrap

```python
import numpy as np
from bootstrapx import bootstrap

rng = np.random.default_rng(0)
y = np.zeros(500)
for t in range(1, 500):
    y[t] = 0.7 * y[t - 1] + rng.normal()

result = bootstrap(y, np.mean, method="mbb", block_length=15, n_resamples=4999, random_state=42)
print(result)
```

## Sieve bootstrap

```python
result = bootstrap(y, np.mean, method="sieve", n_resamples=4999, random_state=42)
print(result)
```

Use sieve when the series is reasonably represented by an autoregressive process. If dependence is complex and local, `stationary` or `mbb` is usually safer.

## Choosing block length

Block length is problem-dependent. Start with a moderate value, validate stability, and compare intervals across nearby choices.

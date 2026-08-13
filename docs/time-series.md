# Time Series

Ordinary IID resampling destroys serial dependence. Time-series methods create
new series while preserving a chosen dependence structure.

## Choose the model you can defend

| Method | Dependence model | Parameter |
|---|---|---|
| `stationary` | stationary series, random block lengths | `mean_block=` |
| `mbb` | contiguous fixed-length blocks | `block_length=` |
| `cbb` | fixed blocks with circular wraparound | `block_length=` |
| `tapered` | fixed blocks with softened boundaries | `block_length=`, `taper=` |
| `sieve` | autoregressive approximation | `ar_order=` |
| `wild` | heteroscedastic residual structure | `fitted=`, `distribution=` |

## Block-bootstrap example

```python
import numpy as np
from bootstrapx import bootstrap

rng = np.random.default_rng(0)
y = np.zeros(500)
for t in range(1, len(y)):
    y[t] = 0.7 * y[t - 1] + rng.normal()

result = bootstrap(
    y,
    np.mean,
    method="stationary",
    mean_block=15,
    n_resamples=4999,
    random_state=42,
)
print(result.confidence_interval)
```

The method assumes the observed sequence is ordered correctly and a stationary
dependence model is scientifically plausible. It does not detect trends,
seasonality, structural breaks, or leakage for you.

## Check block-length sensitivity

There is no universal block length. Compare a small range and report the
choice when endpoints matter:

```python
for mean_block in (10, 15, 20, 30):
    result = bootstrap(
        y,
        np.mean,
        method="stationary",
        mean_block=mean_block,
        n_resamples=4999,
        random_state=42,
    )
    ci = result.confidence_interval
    print(mean_block, ci.low, ci.high)
```

Large changes are evidence that the inference depends strongly on an unresolved
modeling choice; they are not a reason to select the narrowest interval.

For repeated block-bootstrap runs, see the optional
[Numba acceleration](integrations.md#when-numba-helps).

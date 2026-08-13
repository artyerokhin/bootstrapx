# bootstrapx

`bootstrapx` estimates uncertainty for a **single scalar statistic** when
ordinary IID resampling is not enough. Its strongest practical use cases today
are custom confidence intervals, dependent time series, and grouped data.

## Start with the shape of your data

| Your observations | Start with | Required argument |
|---|---|---|
| Independent rows | `bca` or `percentile` | none |
| Repeated rows per user, account, or store | `cluster` | `cluster_ids=` |
| Stationary time series | `stationary` or `mbb` | `mean_block=` or `block_length=` |
| Known sampling strata | `strata` | `strata=` |
| Paired treatment/control outcomes | bootstrap the row-wise difference | none |

```python
import numpy as np
from bootstrapx import bootstrap

x = np.random.default_rng(42).normal(size=300)
result = bootstrap(
    x,
    np.mean,
    method="bca",
    n_resamples=4999,
    random_state=42,
)

print(result.theta_hat)
print(result.confidence_interval)
print(result.standard_error)
```

## What you get

- `theta_hat`: statistic calculated on the observed data.
- `confidence_interval`: lower and upper bounds plus the interval method.
- `standard_error`: spread of the bootstrap estimate.
- `bootstrap_distribution`: all resampled estimates for diagnostics.
- `to_dict()` / `to_frame()`: compact output for reports and tracking.

## Important scope boundary

bootstrapx 0.4 estimates one scalar from one input array. It does **not** yet
provide a native unpaired two-sample A/B effect, lift, p-value, vector-valued
statistic, or automatic missing-value policy. Two separate group intervals are
not an interval for their difference.

See [Current limitations](limitations.md) before using the result for a
decision-critical analysis.

## Read next

1. [First analysis](getting-started.md) — installation, result interpretation,
   and basic checks.
2. [Choose a method](methods.md) — decision table and assumptions.
3. [Grouped and experiment data](ab-testing.md) or
   [time series](time-series.md) — complete workflow examples.
4. [Integrations and performance](integrations.md) — pandas, sklearn, and when
   the optional Numba extra is useful.

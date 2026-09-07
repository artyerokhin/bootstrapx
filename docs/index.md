# bootstrapx

`bootstrapx` estimates uncertainty for scalar statistics and explicit
treatment-versus-control effects. Its strongest practical use cases are
experiment comparisons, custom confidence intervals, dependent time series,
and grouped data.

## Why use it?

| Practical need | What bootstrapx provides |
|---|---|
| A/B effect interval | Independent, paired, or clustered control/treatment comparison. |
| Interval for a custom metric | Any scalar callable: quantiles, robust summaries, or a model score. |
| Dependent time series | Block and sieve bootstrap methods with explicit dependence assumptions. |
| Repeated observations per entity | Cluster resampling that keeps each user, account, or store together. |
| Stratified sample | Resampling that preserves the observed strata. |
| Reproducible reporting | Seeded runs plus `BootstrapResult.to_dict()` and `.to_frame()`. |

For a standard IID interval, SciPy can be sufficient. Choose bootstrapx when
the resampling design itself is part of the analysis and should be visible,
controlled, and reproducible.

## Start with the shape of your data

| Your observations | Start with | Required argument |
|---|---|---|
| Independent control and treatment units | `bootstrap_two_sample` | two samples |
| Matched control/treatment outcomes | `bootstrap_two_sample` | `paired=True` |
| Repeated events inside experiment arms | `bootstrap_two_sample` | cluster IDs for both arms |
| Independent rows | `bca` or `percentile` | none |
| Repeated rows per user, account, or store | `cluster` | `cluster_ids=` |
| Stationary time series | `stationary` or `mbb` | `mean_block=` or `block_length=` |
| Known sampling strata | `strata` | `strata=` |

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

Two-sample results additionally report the control estimate, treatment
estimate, named effect, experiment design, and optional cluster counts. See
[Experiment comparisons](ab-testing.md).

## Important scope boundary

bootstrapx 0.5 still expects one scalar statistic per arm and one scalar
effect. It does not provide p-values, vector-valued simultaneous intervals,
automatic missing-value handling, CUPED, or sequential-testing guarantees.
Two separate one-sample intervals remain different from a direct interval for
their effect.

See [Current limitations](limitations.md) before using the result for a
decision-critical analysis.

## Read next

1. [First analysis](getting-started.md) — installation, result interpretation,
   and basic checks.
2. [Choose a method](methods.md) — decision table and assumptions.
3. [Grouped and experiment data](ab-testing.md) or
   [time series](time-series.md) — complete workflow examples.
4. [Product A/B reference](product-ab.md) — a controlled, reproducible
   user-randomized workflow with a known effect and a decision threshold.
5. [Real-data A/B case study](real-world-ab.md) — a public email experiment
   with sparse conversion and spend outcomes.
6. [Integrations and performance](integrations.md) — pandas, sklearn, and when
   the optional Numba extra is useful.

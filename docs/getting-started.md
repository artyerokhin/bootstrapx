# First Analysis

## Install only what you use

```bash
pip install bootstrapx-lib
```

The core install contains NumPy/SciPy bootstrap methods. Extras are independent:

| Extra | Install command | Use it when |
|---|---|---|
| pandas | `pip install "bootstrapx-lib[pandas]"` | using `.bootstrap` or `to_frame()` |
| sklearn | `pip install "bootstrapx-lib[sklearn]"` | using `BootstrapCV` |
| numba | `pip install "bootstrapx-lib[numba]"` | repeatedly running block-bootstrap methods |

Numba is a performance option, not a correctness requirement. See
[When Numba helps](integrations.md#when-numba-helps).

## Estimate and inspect an interval

```python
import numpy as np
from bootstrapx import bootstrap

rng = np.random.default_rng(0)
data = rng.lognormal(mean=1.0, sigma=0.8, size=300)

result = bootstrap(
    data,
    np.median,
    method="bca",
    confidence_level=0.95,
    n_resamples=4999,
    random_state=42,
)

print(f"estimate: {result.theta_hat:.3f}")
print(
    f"95% {result.confidence_interval.method} interval: "
    f"[{result.confidence_interval.low:.3f}, "
    f"{result.confidence_interval.high:.3f}]"
)
print(f"bootstrap SE: {result.standard_error:.3f}")
```

`confidence_level=0.95` describes the requested procedure; it does not mean
there is a 95% probability that this already-computed frequentist interval
contains the parameter.

## Export a compact result

```python
record = result.to_dict()
# Includes the estimate, interval, SE, method, and metadata.
# The potentially large bootstrap_distribution is omitted by default.

complete = result.to_dict(include_distribution=True)
frame = result.to_frame()  # requires the pandas extra
```

## Compare experiment arms

Use the dedicated API when the target is a treatment-versus-control effect:

```python
from bootstrapx import bootstrap_two_sample

control = rng.normal(10.0, 2.0, size=300)
treatment = rng.normal(10.5, 2.0, size=350)

comparison = bootstrap_two_sample(
    control,
    treatment,
    np.mean,
    effect="difference",
    method="bca",
    n_resamples=4_999,
    random_state=42,
)

print(comparison.estimate)
print(comparison.confidence_interval)
```

The default resamples both arms independently. Set `paired=True` only for
genuine matched rows; provide both cluster-ID arrays when repeated events from
the same randomized unit must stay together. Continue with
[Experiment comparisons](ab-testing.md), reproduce the controlled
[product A/B reference](product-ab.md), and then inspect the limitations in
the [Hillstrom real-data case study](real-world-ab.md).

## Checks before trusting the result

1. Confirm the resampling unit matches how observations became dependent.
2. Inspect the data and statistic; NaN and infinite values are rejected.
3. Repeat with a second seed or more resamples when endpoints affect a decision.
4. For block methods, compare nearby block lengths.
5. Estimate a treatment effect directly instead of reasoning from two separate
   one-group intervals.

Use `random_state` in saved analyses and tests. Use at least a few thousand
resamples for final percentile-based endpoints; the right number still depends
on the stability you need, not on a universal constant.

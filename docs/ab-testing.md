# Experiment Comparisons

`bootstrap_two_sample()` estimates one scalar treatment-versus-control effect.
It resamples according to the experiment design, calculates the statistic in
each arm, and then applies the requested effect:

```text
effect(statistic(control), statistic(treatment))
```

The order is always control first and treatment second. Consequently,
`effect="difference"` means `treatment - control`.

## Choose the analysis unit first

| Data design | Configuration | What is resampled |
|---|---|---|
| Different units in control and treatment | default (`paired=False`) | rows independently within each arm |
| Matched or before/after observations | `paired=True` | the same row indices in both arms |
| Repeated events per user/account/store | cluster IDs for both arms | complete clusters independently within each arm |

The analysis unit should match the randomization unit whenever possible. If an
experiment randomizes users and the business metric is revenue per user,
aggregate to one value per user before comparing the arms.

## Conversion difference

Binary zeros and ones can be summarized with `np.mean`:

```python
import numpy as np
from bootstrapx import bootstrap_two_sample

rng = np.random.default_rng(42)
control = rng.binomial(1, 0.10, size=2_000)
treatment = rng.binomial(1, 0.12, size=2_200)

result = bootstrap_two_sample(
    control,
    treatment,
    np.mean,
    effect="difference",
    method="bca",
    n_resamples=4_999,
    random_state=42,
)

print(f"control:  {result.control_estimate:.2%}")
print(f"treatment:{result.treatment_estimate:.2%}")
print(f"difference: {result.estimate:+.2%}")
print(result.confidence_interval)
```

The estimate is an absolute conversion difference. An estimate of `0.02`
means two percentage points, not a two-percent relative increase.

The interval quantifies sampling uncertainty under the chosen bootstrap
design. It is not a p-value, and causal interpretation still requires valid
randomization, assignment integrity, and an analysis that was not selected
after inspecting the result.

## Ratio and relative lift

The built-in effects are:

| `effect=` | Definition | Example interpretation |
|---|---|---|
| `"difference"` | `treatment - control` | `+0.02` means +2 percentage points for conversion |
| `"ratio"` | `treatment / control` | `1.20` means treatment is 1.2 times control |
| `"relative_lift"` | `(treatment - control) / control` | `0.20` means +20% relative lift |

```python
lift = bootstrap_two_sample(
    control,
    treatment,
    np.mean,
    effect="relative_lift",
    method="bca",
    n_resamples=4_999,
    random_state=42,
)
```

Ratio effects are unstable when the control estimate is near zero. bootstrapx
rejects an observed or resampled zero denominator rather than dropping that
replicate. Prefer an absolute difference when a stable ratio estimand cannot be
defined.

## Paired or before/after outcomes

Use paired resampling only when row `i` in control and treatment represents the
same unit:

```python
rng = np.random.default_rng(7)
before = rng.normal(100, 15, size=300)
after = before + rng.normal(3, 8, size=300)

paired = bootstrap_two_sample(
    before,
    after,
    np.mean,
    effect="difference",
    paired=True,
    method="bca",
    n_resamples=4_999,
    random_state=42,
)
```

Paired resampling preserves within-unit correlation and is equivalent in
estimand to bootstrapping a row-wise difference for additive effects. It is not
appropriate for independent control and treatment users.

## Repeated events and clustered experiments

Sessions from the same user are not independent. Supply cluster IDs for both
arms to resample complete users:

```python
rng = np.random.default_rng(8)
control_user_ids = np.repeat(np.arange(150), 4)
treatment_user_ids = np.repeat(np.arange(180), 4)

control_revenue = rng.lognormal(1.0, 0.8, len(control_user_ids))
treatment_revenue = 1.05 * rng.lognormal(1.0, 0.8, len(treatment_user_ids))

clustered = bootstrap_two_sample(
    control_revenue,
    treatment_revenue,
    np.mean,
    effect="difference",
    control_cluster_ids=control_user_ids,
    treatment_cluster_ids=treatment_user_ids,
    method="percentile",
    n_resamples=4_999,
    random_state=42,
)

print(clustered.n_control_clusters, clustered.n_treatment_clusters)
print(clustered.confidence_interval)
```

This example estimates an event-weighted mean while resampling users. Users
with more observed events still contribute more values to `np.mean`. For an
equally weighted user-level metric, aggregate each user first and run the
ordinary independent comparison:

```python
# Conceptual pandas preparation:
# user_metric = events.groupby(["variant", "user_id"])["revenue"].sum()
# control = user_metric.loc["control"].to_numpy()
# treatment = user_metric.loc["treatment"].to_numpy()
```

Clustered BCa uses leave-one-cluster-out jackknife acceleration. It requires at
least three clusters in each arm; materially larger counts are needed for a
stable applied analysis.

## Custom statistics and effects

The same scalar statistic is evaluated in each arm. For example, compare a
trimmed mean and define an application-specific percent change:

```python
from scipy.stats import trim_mean

def trimmed(values):
    return trim_mean(values, 0.1)

def percent_change(control_stat, treatment_stat):
    return 100 * (treatment_stat - control_stat) / control_stat

custom = bootstrap_two_sample(
    control_revenue,
    treatment_revenue,
    trimmed,
    effect=percent_change,
    method="percentile",
    random_state=42,
)
```

Both callables must return one finite scalar. A custom effect is responsible
for its own mathematical domain; division by zero and non-finite results are
rejected.

## Result export

```python
record = result.to_dict()
frame = result.to_frame()  # requires bootstrapx-lib[pandas]
complete = result.to_dict(include_distribution=True)
```

Compact exports contain arm estimates, the effect estimate and interval,
standard error, sample sizes, method, design, and cluster counts. The full
bootstrap distribution is copied only when explicitly requested.

## Current experiment boundaries

Version 0.5 does not provide:

- p-values or multiple-testing correction;
- CUPED or regression adjustment;
- sequential-testing guarantees;
- two-sample stratified resampling;
- multiway clustering;
- automatic missing-value handling;
- vector-valued or simultaneous intervals.

See [Current limitations](limitations.md) and inspect the scenarios matching
your design in [Benchmarks](benchmarks.md) before a decision-critical analysis.

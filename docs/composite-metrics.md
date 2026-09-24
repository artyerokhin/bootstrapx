# Composite metrics: assigned users and orders

!!! warning "Unreleased development API"
    This guide describes the local 0.6.0 development branch, not the published
    0.5.1 package. Install the checkout to try it. Release coverage and
    performance evidence for this path is still pending.

## Start from assigned users, not just observed orders

The executable [offline example](https://github.com/artyerokhin/bootstrapx/blob/main/examples/assigned_users_composite_metrics.py)
generates synthetic assignments and orders, validates them, aggregates orders
by user, and left-joins the aggregates to **all assigned users**. The file is
currently local to the development branch; the main-branch link becomes valid
only when that branch is merged.

From the checkout:

```bash
python -m pip install -e ".[pandas]"
python examples/assigned_users_composite_metrics.py
```

Zero revenue/orders for a user without orders are justified only because this
example assumes complete order capture over a common fixed observation window.
Missing revenue in a recorded order is rejected, not filled with zero. Unknown
users, duplicate assignments/orders, and non-finite revenue are also rejected.
Real pipelines must establish exposure, observation-window, currency/refund,
and telemetry-completeness policies separately.

## Define the metric before choosing an interval

| Metric | Definition | Interpretation |
|---|---|---|
| Revenue per assigned user | `mean(user_revenue)` | Includes non-buyers; primary metric in this example |
| Orders per assigned user | `mean(user_orders)` | Includes non-buyers; exploratory explanation of activity |
| Revenue per order | `sum(user_revenue) / sum(user_orders)` | Order-weighted metric; inspect denominator changes |
| Mean buyer revenue/order | `mean(user_revenue / user_orders)` among buyers | Buyer-weighted description of a post-treatment subset |

The last two are not interchangeable. Conditioning on buying selects users
using post-treatment behavior; a contrast among those buyers is not the
assigned-user average treatment effect.

## Joint columns, scalar result

```python
from bootstrapx import RatioOfSums, bootstrap_two_sample

# control/treatment: one row per assigned user, including non-buyers.
result = bootstrap_two_sample(
    control[["revenue", "orders"]],
    treatment[["revenue", "orders"]],
    RatioOfSums(numerator=0, denominator=1),
    allow_2d=True,
    control_unit_ids=control["user_id"],
    treatment_unit_ids=treatment["user_id"],
    metric_name="revenue/order",
    effect_unit="currency/order",
    effect="difference",
    method="basic",
    random_state=42,
)
```

The statistic receives a float64 NumPy matrix, not a DataFrame. Columns always
share resampled row indices; separately bootstrapping revenue and orders would
lose their dependence. Each matrix DataFrame must have unique column labels;
two DataFrames must also have identical labels/order. Duplicate labels are
rejected even when only one arm is a DataFrame: selecting `['revenue', 'orders']`
does not remove duplicate `revenue` columns and could otherwise change the
metric's positional meaning.
Default `allow_2d=False` preserves existing 1-D behavior.

`RatioOfSums` defines a metric **within** each arm. `effect="ratio"` would then
divide the treatment metric by the control metric; `effect="difference"`
instead subtracts them. A zero denominator in an observed, bootstrap, or
jackknife sample raises an error. Such samples are never dropped or redrawn.
Signed denominators are mathematically allowed; business-domain checks belong
in the metric/data-preparation policy.

### Interval method is part of the analysis

`BCa` is not an automatic accuracy upgrade for a nonlinear ratio. In the 0.6
release study it tracked SciPy closely but materially undercovered known truth
with strong skew, dependent activity/price, and especially only 24/30 clusters.
The example therefore uses `basic` for revenue/order while retaining `BCa` for
the two simple assigned-user means. This is a tested example choice, not a
universal ranking: `basic` also undercovered in small-cluster cases.

For a decision-critical ratio, simulate a plausible data-generating process,
include denominator changes and zero-denominator samples, and compare `basic`,
`percentile`, and `bca`. More bootstrap resamples reduce endpoint simulation
noise; they do not fix finite-sample coverage. With few clusters, report that
limitation or use a method whose assumptions are justified for that design.

## Check identities, do not infer them

Supply both unit-ID arrays to validate one-row-per-unit input. IDs must be
unique within each arm. Independent arms must not overlap; paired arms must
have the same IDs in the same order. No automatic sorting or pandas-index
alignment occurs. IDs need consistent global meaning; passing IDs does not
prove randomization or causal validity.

Unit IDs cannot be combined with repeated-event cluster IDs. Cluster IDs remain
a separate contract, and equal arm-local numeric cluster labels are allowed.
Within each arm, cluster labels must be scalar, non-missing and mutually
comparable; infinite numeric labels are rejected. Mixed numeric/string labels
such as `1` and `'1'` raise an error instead of silently becoming one cluster.
Validate labels before converting them yourself: precision or type information
already lost in an input array cannot be recovered by the library.
Check global assignment integrity separately for repeated-event analysis.
When IDs are omitted, result metadata explicitly records that they were not
validated.

## Reporting and limits

`result.to_dict()` / `to_frame()` include copy-safe `metadata`: metric/unit
labels, resampling unit and counts, feature count, ID-validation status,
confidence level, effective batch size, seed kind/value, and package version.
Raw IDs/data and callable objects are not stored. A supplied Generator is
identified as such; its full state is not saved, and the seed is unknown.
Local development still reports package version 0.5.1 until the release bump;
that value alone does not identify uncommitted development code.

Labels do not scale values or verify units. Difference units follow the metric;
ratio/lift effects are dimensionless. Revenue/user is the example's primary
metric; other metrics are exploratory. Intervals are separate pointwise
intervals, not simultaneous inference or multiple-testing control. Reusing a
seed for separate calls is not a multi-metric covariance/result API.

Synthetic data illustrate the workflow. The displayed single experiment cannot
establish nominal coverage, validate real-world telemetry, or make a shipping
decision. Known-truth benchmark simulations are a separate release gate.

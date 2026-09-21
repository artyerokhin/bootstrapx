# Composite metric API draft (unreleased)

Implemented locally on `feature/v0.6.0-composite-metrics`. This is not an API
available in the published 0.5.1 package. The version will change only when the
candidate is ready; do not publish artifacts from this development checkout.

## Minimal example

Each row below is one assigned user. Columns are total revenue and total orders
for that user over a fixed observation window. Users with no orders remain in
the input; zero revenue/orders must be justified by the data collection policy,
not inferred from arbitrary missing values.

```python
import numpy as np
from bootstrapx import RatioOfSums, bootstrap_two_sample

control = np.array(
    [[0, 0], [30, 2], [12, 1], [40, 3], [18, 1], [25, 2], [32, 2], [15, 1]],
    dtype=float,
)
treatment = np.array(
    [[0, 0], [36, 2], [15, 1], [45, 3], [20, 1], [30, 2], [36, 2], [18, 1]],
    dtype=float,
)

result = bootstrap_two_sample(
    control,
    treatment,
    RatioOfSums(numerator=0, denominator=1),
    allow_2d=True,
    effect="difference",
    method="percentile",
    n_resamples=499,
    random_state=42,
)

print(result.control_estimate, result.treatment_estimate)
print(result.estimate, result.confidence_interval)
```

This tiny table illustrates API mechanics, not a well-powered experiment. Its
ratio is revenue per order, not revenue per assigned user and not the average
of individual users' revenue/order ratios. A between-arm `effect="ratio"`
would additionally divide the treatment arm metric by the control arm metric.
A change in orders can change the metric's interpretation; inspect numerator
and denominator as well as their ratio.

If an observed, bootstrap, or jackknife sample has zero total orders, the
computation fails explicitly. No replicates are discarded or redrawn. An
absolute difference between ratio metrics does not cure an undefined within-arm
ratio. Choose a defensible metric/data design rather than silently adding epsilon.

## Contract

- Default `allow_2d=False` keeps existing 1-D handling, including flattening
  single-column DataFrames. Opt-in matrix inputs keep their 2-D shape even if
  they contain only one column.
- Columns within an arm always share resampled row indices. Arms must have
  equal dimensionality and feature count; their row counts may differ unless
  paired. Empty feature sets and NaN/inf values are rejected.
- Two DataFrames must have matching column labels in the same order. No schema
  matching is possible for unlabeled arrays; select and order numeric features
  explicitly. Do not include IDs or variant labels in the metric matrix.
- `statistic` receives a float64 NumPy array, not a DataFrame. It returns exactly
  one finite scalar. Vector-valued and simultaneous intervals are not provided.
- `paired=True` pairs rows by position, never by pandas index. Prepare and
  validate correspondence explicitly; optionally supply matching
  `control_unit_ids` / `treatment_unit_ids` to verify row correspondence.
- Clustered input resamples all rows/features of each selected cluster, preserving
  repeated cluster selections. Arm-local numeric cluster labels may overlap.
- BCa jackknife deletes rows, corresponding pairs, or complete clusters according
  to the selected design. More columns do not create more independent units.
- `RatioOfSums` uses non-negative integer column positions. Signed denominators
  are mathematically allowed; the helper does not infer business-domain rules.
  Zero denominators, non-finite sums, and overflowed ratios are rejected.
- A custom callable may implement another composite scalar metric; it is
  responsible for the definition and domain of that metric.

## Local checks

The test runner uses source code through `pytest.ini`. When trying the new API
directly, ensure imports use the local source, not an installed 0.5.1 wheel:

```bash
PYTHONPATH=src .venv-release/bin/python -c \
  'from bootstrapx import RatioOfSums; print(RatioOfSums())'
.venv-release/bin/pytest tests/test_composite_metrics.py -q
```

Exact-reference tests verify implementation mechanics. A matched SciPy index
bootstrap checks interval agreement for an ordinary ratio scenario. Neither
constitutes the full known-truth coverage evidence required for release; that
  study is still pending. The assigned-user/order example now exists in
  `examples/assigned_users_composite_metrics.py`; it runs in ordinary CI tests
  when pandas is present and preserves assigned non-buyers.

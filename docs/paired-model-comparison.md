# Compare fixed models on repeated observations

Added in 0.7.0. Install `bootstrapx-lib>=0.7.0` or use an editable
installation from the repository to run the examples below.

Two models often predict the same records, while one patient, user, or store
contributes several records. Neither independent-arm resampling nor paired
IID rows represent that structure. `bootstrap_two_sample` can now draw whole
entities once and use exactly the same selected rows for both models.
The practical benefit is an explicit design, arm estimates, alignment checks,
and a reusable result instead of a custom cluster-delta resampling loop.
SciPy can express the same procedure with a custom statistic; the mathematics
is not unique to bootstrapx.

## Define the target first

Use independent held-out entities and **fixed** predictions. This estimates
the difference between scalar metrics for two already fitted models, over
the entity population represented by the held-out sample. It does not
estimate uncertainty from training, tuning, feature selection, or calibration.
Resampling test records does not repair leakage or an unrepresentative test set.

For a pooled Brier score, every row has equal weight in the metric, but the
whole entity is the independent resampling unit. Entities with more rows
contribute more to the pooled score. If your target is an equally weighted
entity-average score, calculate one loss summary per entity first and use a
paired row comparison. Pooled AUC is nonlinear: averaging per-entity AUCs is
a different estimand, and some entity-specific AUCs may be undefined.

## Prepare and align prediction tables

1. Check that the two tables refer to the same unique observations, labels,
   and entity membership. Apply an explicit missing-data policy.
2. Align tables by their real observation keys before extracting numeric
   arrays. Do not silently inner-join away failed or missing predictions.
3. Supply the observation keys from each table and one common entity-ID
   array. The library checks unique keys and exact positional equality.
   It does not infer alignment from pandas indices or verify labels for you.
   Keys must be scalar and nonmissing; represent compound keys explicitly
   rather than passing a two-dimensional ID array.

The metric receives one numeric matrix at a time:

```python
import numpy as np
from bootstrapx import bootstrap_two_sample

def brier(rows):
    # Column 0 is the binary outcome; column 1 is its predicted probability.
    return float(np.mean((rows[:, 1] - rows[:, 0]) ** 2))

# labels/predictions/keys are extracted from explicitly aligned tables.
result = bootstrap_two_sample(
    np.column_stack((labels, predictions_a)),
    np.column_stack((labels, predictions_b)),
    brier,
    allow_2d=True,
    paired=True,
    paired_cluster_ids=entity_ids,
    control_observation_ids=observation_ids_a,
    treatment_observation_ids=observation_ids_b,
    effect="difference",             # B - A; negative favors B for Brier
    method="percentile",             # basic is also supported; BCa is not
    n_resamples=4999,
    random_state=42,
    metric_name="Brier score",
    effect_unit="score difference",
)

print(result.control_estimate, result.treatment_estimate)
print(result.estimate, result.confidence_interval)
print(result.extra["design"]["paired"])
print(result.extra["distribution_diagnostics"])
```

Omitting observation IDs leaves correspondence validation to the caller.
Do not pass per-arm cluster arrays for this design: those implement
independent-arm resampling. Old one-row-per-unit ID arguments cannot be
combined with cluster IDs; the new observation keys identify rows, not
independent entities. No raw keys are retained in reporting metadata.

## Two executable cross-checks

```bash
python -m pip install -e ".[sklearn]"
python examples/paired_cluster_brier.py
python examples/practitioner_grouped_model_comparison.py
```

The Brier example is NumPy/SciPy-only. It generates 120 independent entities
with five rows each. An entity has outcome probability `q=0.2` or `q=0.8`,
with equal population probability; outcomes are conditionally independent.
Forecast A always predicts `0.4`. Forecast B knows `q`: it is an **oracle
teaching reference**, not a trained model or real-world improvement claim.
Expected scores are `0.26` and `0.16`, hence the population difference is
exactly `-0.10`. The fixed-seed run estimates `-0.1017`, with a 95% percentile
interval `[-0.1250, -0.0813]`. Its SciPy reference resamples entity-average
loss deltas, which is equivalent here because group sizes are equal.
With unequal sizes, an unweighted mean of entity losses would change the target.

The scikit-learn example uses a group-aware train/test split, freezes two
fitted models, and compares pooled AUC and Brier on 120 unseen entities.
It also includes a SciPy AUC reference that reconstructs complete selected
entity rows. See [Practitioner workflows](practitioner-workflows.md) for
recorded results and interpretation. Nearby endpoints with different random
draws are a cross-check, not evidence of coverage or a speed advantage.

## Reuse the draws, not the assumptions

```python
ci90 = result.interval(confidence_level=0.90, method="percentile")
ci95_basic = result.interval(confidence_level=0.95, method="basic")
```

This evaluates quantiles/reflections on the stored distribution without
calling the metric, drawing samples, or modifying the original result.
Defaults are 95% and percentile, even if the original interval differs.
The same method exists on ordinary `BootstrapResult` objects.
It does not change the design, add draws, reduce Monte Carlo uncertainty, or
justify choosing whichever interval gives a desired decision. BCa needs
jackknife information and is not reconstructed. Bayesian, studentized,
subsampling, and Bernoulli results cannot be reinterpreted this way.

## Inspect, then interpret cautiously

The design report gives entity counts and size spread, not a pass/fail safety
score. Distribution diagnostics report the exact number of distinct values
and whether all replicates are identical; a degenerate distribution can be
a genuine constant paired effect, not necessarily an error. Many distinct
replicates do not establish calibration either.

Whole-cluster bootstrap assumes independent entities and one grouping level.
It does not implement time dependence across entities, multiway clustering,
complex-survey weighting, or a guaranteed small-cluster correction. AUC may
be undefined in a one-class resample; nonfinite metrics fail explicitly,
rather than dropping draws or stratifying by the outcome after the fact.
The optional binary-risk inspector currently requires labels constant
within each cluster, so it does not apply to arbitrary repeated-outcome AUC.

Check [paired-cluster simulations](benchmarks.md#paired-cluster-development-checks)
and [limitations](limitations.md). Known-truth experiments include both
equal group sizes and informative sizes. They are evidence for those
generating models only, not universal nominal coverage.

# Practitioner workflows

These examples test whether the API can answer common questions
without writing a resampling loop. Both use synthetic data, fixed seeds, and
999 resamples so they run offline in a few seconds. They demonstrate a workflow;
one generated dataset cannot establish 95% coverage or real-world adoption.

The assigned-user example works on 0.6.0. The paired-cluster comparison and
saved-interval reuse below require 0.7.0 or later. From a repository checkout:

```bash
python -m pip install -e ".[pandas,sklearn]"
python examples/practitioner_user_orders.py
python examples/practitioner_grouped_model_comparison.py
python examples/paired_cluster_brier.py
```

The scripts run the same estimands through SciPy as a reference. Their random
draws are not identical, so nearby interval endpoints are a cross-check, not a
speed comparison or proof of correct coverage.

## Orders belong to assigned users

**Question:** How much did treatment change revenue per *assigned user*?
The [executable orders example](https://github.com/artyerokhin/bootstrapx/blob/main/examples/practitioner_user_orders.py)
starts with 520 assignments and 1,047 observed orders. It validates the IDs
and revenues, aggregates orders by user, and left-joins the aggregates onto
every assigned user. Users with no order remain in the analysis with zero
revenue, under the example's explicit assumption of complete order capture.

```python
result = bootstrap_two_sample(
    control_revenue,                 # one value per assigned control user
    treatment_revenue,               # one value per assigned treatment user
    np.mean,
    effect="difference",             # treatment - control
    method="basic",
    control_unit_ids=control["user_id"],
    treatment_unit_ids=treatment["user_id"],
    n_resamples=999,
    random_state=42,
)
```

The script prints an estimated difference of about **+3.43 currency units per
assigned user**. In its recorded run, the 95% basic intervals were
`[+1.17, +5.50]` from bootstrapx and `[+1.34, +5.69]` from SciPy. The same
orders give only `+0.56` when averaged *per observed order*. That number
answers a different question: buyers with more orders get more weight, and
non-buyers disappear. Do not compare these numbers as competing estimators of
one effect.

For the user-level comparison, SciPy needs the same prepared arrays and a
two-argument effect function. bootstrapx additionally checks that supplied
unit IDs are unique within each arm and disjoint across arms. Neither library
can infer whether a missing order means zero revenue or missing telemetry;
the example's table preparation makes that decision explicitly. The complete
[assigned-user preparation](composite-metrics.md) also shows revenue/order as
a separate ratio metric.

## Patients contribute repeated model-evaluation rows

**Question:** On previously unseen patients, what is the difference between
the AUC of two already fitted models? The
[executable model example](https://github.com/artyerokhin/bootstrapx/blob/main/examples/practitioner_grouped_model_comparison.py)
generates synthetic patients with five observations each. A group-aware split
keeps training and test patients disjoint. It then freezes both models and
compares their predictions on the same 120 held-out patients (600 rows).

```python
from bootstrapx import bootstrap_two_sample

def auc_metric(rows):
    return roc_auc_score(rows[:, 0].astype(int), rows[:, 1])

result = bootstrap_two_sample(
    np.column_stack((labels, predictions_a)),
    np.column_stack((labels, predictions_b)),
    auc_metric,
    allow_2d=True,
    paired=True,
    paired_cluster_ids=patient_ids,
    control_observation_ids=observation_ids_a,
    treatment_observation_ids=observation_ids_b,
    method="percentile",
    n_resamples=999,
    random_state=42,
)
```

The observed AUC gain in the recorded run was about **+0.047**. Its 95%
percentile intervals were `[+0.022, +0.076]` from bootstrapx and
`[+0.022, +0.074]` from SciPy. The latter required drawing patient IDs and
reconstructing all rows for each selected patient in a custom statistic.
bootstrapx uses the same complete-patient row indices on both matrices;
labels and both predictions always move together. The observation-ID arrays
come from the two prediction tables and must identify the same unique rows
in the same order; they are not generated separately to make the check pass.
The example also compares Brier score: the observed B-minus-A difference is
about `-0.017`, with a 95% percentile interval of `[-0.027, -0.008]`.
Negative is better for Brier; positive is better for AUC.

The target is the difference between **pooled window-level AUCs** for fixed
models on a population of patients represented by the held-out set. It is not
an interval for the full train/tune/calibrate/deploy procedure: the models are
not refitted inside each resample. If patients contribute unequal numbers of
windows, their windows have unequal influence on this pooled metric. AUC can
also be undefined when a resample contains only one class; the library raises
an error instead of silently dropping that resample. The example's fixed data
and seed do not encounter that case.

## What this exercise changed in the roadmap

Both questions were already answerable in 0.6.0, using a custom stacked delta
statistic for model comparison. The 0.7.0 API now removes specific
obstacles visible in those scripts:

| Workflow step | Caller responsibility | 0.7.0 API/recipe |
|---|---|---|
| Assigned-user metric | Validate and join assignment and event tables | Preparation recipe; no automatic missing-to-zero policy |
| Related model outputs | Align unique observations and identify real clusters | Two matrices, shared cluster draws, optional positional ID validation |
| Clustered result | Decide whether the independence assumptions are credible | Counts and cluster-size summaries in `extra["design"]` |
| Expensive metric | Choose interval level/method before interpreting results | `result.interval()` reuses draws for percentile/basic; it does not change a seed or add draws |

The [paired model-comparison guide](paired-model-comparison.md) includes an
offline Brier example with an analytically known population difference of
`-0.1`; model B is explicitly an oracle simulation reference, not a claimed
trained-model improvement. Its recorded estimate is `-0.1017`, with a 95%
interval `[-0.1250, -0.0813]`. One dataset is not coverage evidence.

The current evidence supports these workflows and their diagnostics.
It does not yet justify a new general `ResamplingPlan` API, automatic method
selection, or a claim that bootstrapx outperforms other libraries. SciPy
already supports [paired resampling and result reuse](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html);
[tea-tasting](https://github.com/e10v/tea-tasting) covers a much broader A/B
workflow. Our two SciPy calls are matched methodological references, not a
benchmark against tea-tasting or other specialized products.

If you have had to write a resampling loop for your own metric, please
[describe the task in an issue](https://github.com/artyerokhin/bootstrapx/issues/new/choose).
An anonymized schema, the independent unit, the target metric, and the current
workaround are more useful than a feature name. Do not attach private records.

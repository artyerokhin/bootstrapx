# Clusters within fixed strata

Use `method="cluster_strata"` when the data were collected as independent
clusters **within strata fixed before observing the outcome**. The method
draws whole clusters with replacement separately in every stratum, retaining
the observed number of sampled clusters per stratum. It does not infer the
sampling design from the data.

```python
import numpy as np
from bootstrapx import bootstrap, inspect_cluster_design

rng = np.random.default_rng(42)
cluster_ids = np.repeat(np.arange(60), 5)
strata_ids = np.repeat(np.repeat(["north", "south"], 30), 5)
values = (
    np.repeat(rng.normal(size=60), 5)
    + rng.normal(size=300)
)

design = inspect_cluster_design(cluster_ids, strata_ids=strata_ids)
print(design.clusters_per_stratum)  # (30, 30)
print(design.min_cluster_size, design.max_cluster_size)  # 5 5

result = bootstrap(
    values,
    np.mean,
    method="cluster_strata",
    cluster_ids=cluster_ids,
    strata_ids=strata_ids,
    ci_method="percentile",  # or "basic"
    n_resamples=4_999,
    random_state=42,
)
print(result.confidence_interval)
print(result.extra["design"])
```

Each cluster ID must belong to exactly one stratum. Both ID arrays must align
with the rows of `values`. The method rejects a stratum with fewer than two
clusters because its within-stratum variability cannot be estimated from one
observed cluster. `inspect_cluster_design()` reports counts without applying a
universal minimum-safe-count threshold: having two clusters is permitted, not
evidence that a 95% interval really covers 95% of the time.

## Decide what the mean means

The statistic determines the *estimand*. In the example, `np.mean` weights
each row equally. If larger clusters tend to have different outcomes, that
differs from weighting each cluster equally. Resampling complete clusters
does not change this fact. To estimate an equally weighted cluster mean,
first calculate one value per cluster and pass those values with their
cluster-level stratum IDs. Decide the population weights of the strata as
part of the study design; observed row counts are not automatically those
weights.

## Diagnose a binary score metric

For an AUC-like statistic requiring both classes, pass patient-level 0/1
labels repeated on the same rows to the **diagnostic**, not as automatically
chosen strata:

```python
patient_labels = rng.binomial(1, 0.1, size=60)
report = inspect_cluster_design(
    cluster_ids,
    strata_ids=strata_ids,
    binary_labels=np.repeat(patient_labels, 5),
    n_resamples=4_999,
)
print(report.binary_cluster_counts_by_stratum)
print(report.probability_any_one_class_resample)
```

Labels must be constant within each cluster. The reported probability is
conditional on the observed class counts and on the specified resampling
scheme. It predicts a particular failure mode—at least one draw having just
one class—not confidence-interval coverage. If the original data have only
one class, AUC itself is undefined. The library rejects non-finite statistics
instead of silently discarding invalid draws.

Do not stratify on observed outcomes merely to force every AUC replicate to
contain both classes. That changes the resampling scheme, and for
prevalence-dependent metrics such as accuracy it can remove a real source of
uncertainty. When positive independent units are scarce, more bootstrap draws
alone cannot supply the missing information.

This method is a one-stage, with-replacement cluster bootstrap for fixed
strata. It is **not** a general survey bootstrap: finite-population
corrections, unequal selection probabilities, calibration weights, and
multi-stage sampling require additional methods. It also does not resample
control and treatment arms jointly; `bootstrap_two_sample()` currently has
no stratified-cluster option. Only percentile and basic intervals are exposed
for `cluster_strata`; BCa acceleration for this design is not implemented.

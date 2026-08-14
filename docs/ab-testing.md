# Grouped and Experiment Data

## Repeated rows require the right resampling unit

Sessions from the same user are usually correlated. Resampling session rows as
IID observations treats them as more independent information than they are and
can make an interval too narrow.

This example estimates the event-level mean while resampling complete users:

```python
import numpy as np
from bootstrapx import bootstrap

rng = np.random.default_rng(42)
n_users = 200
sessions_per_user = 5
user_ids = np.repeat(np.arange(n_users), sessions_per_user)

user_effect = rng.normal(0.0, 1.5, size=n_users)
session_noise = rng.normal(0.0, 0.5, size=len(user_ids))
metric = 10.0 + np.repeat(user_effect, sessions_per_user) + session_noise

clustered = bootstrap(
    metric,
    np.mean,
    method="cluster",
    cluster_ids=user_ids,
    n_resamples=4999,
    random_state=42,
)

print(clustered.confidence_interval)
```

The cluster method resamples one grouping level and keeps all rows belonging to
a selected group. Here the estimand remains the mean over event rows. If the
business estimand is an equally weighted mean over users, aggregate to one
value per user first and bootstrap those user-level values as IID observations.

## Paired effects are supported through transformation

When treatment and control measurements form genuine pairs, bootstrap the
within-pair effect:

```python
import numpy as np
from bootstrapx import bootstrap

rng = np.random.default_rng(42)
control = rng.normal(100, 15, size=300)
treatment = control + rng.normal(3, 8, size=300)
paired_difference = treatment - control

effect = bootstrap(
    paired_difference,
    np.mean,
    method="bca",
    n_resamples=4999,
    random_state=42,
)
print(effect.theta_hat, effect.confidence_interval)
```

This is valid only when row `i` in treatment is meaningfully paired with row
`i` in control.

## Unpaired A/B effects are not a native 0.4 workflow

A usual randomized experiment has separate treatment and control users. Its
difference, ratio, or relative lift must resample both groups according to the
randomization and analysis unit. bootstrapx 0.4 does not yet expose that
two-sample API.

Do not replace the missing workflow with either of these:

- separate confidence intervals for the two groups;
- concatenating both groups into one array without preserving assignment.

Those procedures do not produce the bootstrap distribution of the treatment
effect. Explicit two-sample and clustered experiment estimators are planned for
a later release.

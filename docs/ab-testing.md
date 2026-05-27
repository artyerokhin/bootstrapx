# A/B Testing

## Why plain iid bootstrap is often wrong

In product experiments, rows are often not independent.
Common examples:

- multiple sessions per user
- multiple events per order
- repeated exposures per account

If you resample rows independently, uncertainty is often underestimated.

## Cluster bootstrap pattern

Use `method="cluster"` and pass a cluster id per observation.

```python
import numpy as np
from bootstrapx import bootstrap

rng = np.random.default_rng(42)
n_users = 200
n_sessions = 5
cluster_ids = np.repeat(np.arange(n_users), n_sessions)

user_effects = rng.normal(0, 2.0, n_users)
session_noise = rng.normal(0, 0.5, size=n_users * n_sessions)

diff = np.repeat(user_effects, n_sessions) * 0 + 0.1 + session_noise

result = bootstrap(
    diff,
    np.mean,
    method="cluster",
    cluster_ids=cluster_ids,
    n_resamples=4999,
    random_state=42,
)
print(result)
```

## Recommended workflow

1. Define the unit of randomization.
2. Bootstrap at that unit, not at the event row.
3. Keep `random_state` fixed for repeatable analyses.
4. Compare CI width between iid and cluster methods to sanity-check dependence.

## Future extensions

Natural next additions for `bootstrapx` are explicit two-sample APIs, lift estimation, and bootstrap-based p-values for experimentation workflows.

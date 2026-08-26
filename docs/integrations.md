# Integrations and Performance

## pandas

Install with `pip install "bootstrapx-lib[pandas]"`. Importing `bootstrapx`
registers the accessor.

```python
import numpy as np
import pandas as pd
import bootstrapx

s = pd.Series(np.random.default_rng(0).normal(size=300))
result = s.bootstrap.bca(np.mean, n_resamples=4999, random_state=42)
print(result.to_frame())
```

`DataFrame.bootstrap.summary()` evaluates columns independently. It is useful
for a table of separate column estimates, but it does not estimate a
difference, ratio, lift, paired effect, or p-value between columns. Use the
underlying arrays with `bootstrap_two_sample()` instead:

```python
from bootstrapx import bootstrap_two_sample

comparison = bootstrap_two_sample(
    frame.loc[frame["variant"] == "control", "metric"],
    frame.loc[frame["variant"] == "treatment", "metric"],
    np.mean,
    effect="difference",
    random_state=42,
)
print(comparison.to_frame())
```

## scikit-learn

Install with `pip install "bootstrapx-lib[sklearn]"`. `BootstrapCV` supplies
bootstrap training indices and out-of-bag test indices to scikit-learn:

```python
from bootstrapx import BootstrapCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)
cv = BootstrapCV(n_splits=200, random_state=42)
scores = cross_val_score(
    GradientBoostingClassifier(random_state=0),
    X,
    y,
    cv=cv,
    scoring="roc_auc",
)
print(scores.mean(), scores.std())
```

The score distribution is based on varying out-of-bag subsets. Its standard
deviation is not automatically a confidence interval for future deployment
performance.

## When Numba helps

The core library does not require Numba. Installing
`pip install "bootstrapx-lib[numba]"` JIT-compiles index generation for:

- moving block bootstrap (`mbb`);
- circular block bootstrap (`cbb`);
- stationary bootstrap (`stationary`);
- the block-index part of tapered bootstrap (`tapered`).

It does **not** accelerate IID/BCa methods, the user statistic, pandas,
scikit-learn, sieve bootstrap, or wild bootstrap. Results and random-state
behavior are tested with and without the optional dependency.

Use it for repeated block-bootstrap analysis or large series. Skip it when you
want the smallest environment or only run IID methods. In the audited 0.4.4
Apple Silicon run, warm end-to-end `np.mean` calls with 500 resamples were
11–34× faster for MBB, CBB, and stationary bootstrap across `n=100`–`10,000`.
The first process call took 0.007–0.121 seconds. The factor and startup cost
vary with CPU, cache state, series length, method, and statistic; benchmark your
own repeated workflow rather than treating these as cross-machine guarantees.

Reproduce the comparison on your machine:

```bash
pip install -e ".[numba]"
python benchmarks/bench_numba.py
```

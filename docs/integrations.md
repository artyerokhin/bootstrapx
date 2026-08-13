# Integrations

## pandas accessor

When pandas is installed, `bootstrapx` registers a `.bootstrap` accessor.

### Series

```python
import pandas as pd
import numpy as np
import bootstrapx

s = pd.Series(np.random.default_rng(0).normal(5, 2, 300))
r = s.bootstrap.bca(np.mean, n_resamples=4999, random_state=42)
print(r)
```

### DataFrame summary

```python
import pandas as pd
import numpy as np
import bootstrapx

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "control": rng.normal(0, 1, 200),
    "treatment": rng.normal(0.2, 1, 200),
})

summary = df.bootstrap.summary(np.mean, n_resamples=2999, random_state=42)
print(summary)
```

`DataFrame.bootstrap.summary()` estimates every column independently. It does
not estimate a difference, ratio, lift, paired effect, or p-value between
columns. For paired observations, bootstrap the row-wise effect explicitly;
for repeated observations per unit, use cluster bootstrap at that unit.

## scikit-learn

`BootstrapCV` implements a scikit-learn compatible cross-validator using bootstrap training samples and out-of-bag evaluation.

```python
from bootstrapx import BootstrapCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)
cv = BootstrapCV(n_splits=200, random_state=42)

scores = cross_val_score(
    GradientBoostingClassifier(n_estimators=100, random_state=0),
    X, y, cv=cv, scoring="roc_auc"
)
print(scores.mean(), scores.std())
```

## Backend guidance

`bootstrapx` is CPU-first by design so it works on most developer and production machines with a normal `pip install`.

Recommended deployment baseline:

- NumPy + SciPy core for correctness and portability.
- Optional pandas for tabular workflows.
- Optional scikit-learn for model evaluation.
- Optional `bootstrapx-lib[numba]` for JIT-accelerated time-series index generation.

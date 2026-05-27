<div align="center">

# bootstrapx

**Production-grade bootstrap uncertainty estimation for Python.**

[![CI](https://github.com/artyerokhin/bootstrapx/actions/workflows/ci.yml/badge.svg)](https://github.com/artyerokhin/bootstrapx/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/bootstrapx-lib)](https://pypi.org/project/bootstrapx-lib/)
[![Downloads](https://img.shields.io/pypi/dm/bootstrapx-lib)](https://pypi.org/project/bootstrapx-lib/)
[![Python](https://img.shields.io/pypi/pyversions/bootstrapx-lib)](https://pypi.org/project/bootstrapx-lib/)
[![Coverage Status](https://coveralls.io/repos/github/artyerokhin/bootstrapx/badge.svg?branch=main)](https://coveralls.io/github/artyerokhin/bootstrapx?branch=main)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://artyerokhin.github.io/bootstrapx)

*15 bootstrap methods · sklearn-compatible · pandas accessor · memory-safe batching*

</div>

---

## Why bootstrapx?

`scipy.stats.bootstrap` covers 3 CI types and only iid data.  
The R `boot` package is comprehensive but not Pythonic.  
**bootstrapx** bridges this gap.

| Feature | `scipy` | `arch` | **bootstrapx** |
|---|:---:|:---:|:---:|
| BCa interval | ✅ | ❌ | ✅ |
| Studentized (bootstrap-t) | ❌ | ❌ | ✅ |
| Bayesian bootstrap | ❌ | ❌ | ✅ |
| Poisson / Bernoulli weights | ❌ | ❌ | ✅ |
| MBB / CBB / Stationary block | ❌ | ✅ | ✅ |
| Sieve (AR-based) | ❌ | ❌ | ✅ |
| Wild bootstrap | ❌ | ✅ | ✅ |
| Cluster / Stratified | ❌ | ❌ | ✅ |
| **scikit-learn CV API** | ❌ | ❌ | ✅ |
| **pandas `.bootstrap` accessor** | ❌ | ❌ | ✅ |
| Reproducible (seeded RNG) | ✅ | partial | ✅ |
| Constant memory (batched) | ❌ | ❌ | ✅ |

---

## Installation

```bash
pip install bootstrapx-lib                  # core (numpy + scipy only)
pip install "bootstrapx-lib[pandas]"        # + pandas accessor
pip install "bootstrapx-lib[numba]"         # + Numba JIT acceleration
pip install "bootstrapx-lib[pandas,numba]"  # everything
```

---

## Quick Start

### Basic usage

```python
import numpy as np
from bootstrapx import bootstrap

data = np.random.default_rng(42).normal(5, 2, size=300)

result = bootstrap(data, np.mean)
print(result)
# BootstrapResult(method='bca', theta_hat=4.97, se=0.11, CI=[4.75, 5.19])

print(result.confidence_interval.low, result.confidence_interval.high)
print(5.0 in result.confidence_interval)  # True
```

### pandas accessor

```python
import pandas as pd
import numpy as np
import bootstrapx  # registers .bootstrap accessor

s = pd.Series(np.random.default_rng(0).exponential(scale=2, size=500))

# On a Series
r = s.bootstrap.bca(np.mean)
print(r)

# On a DataFrame — column-wise summary
df = pd.DataFrame({"control": s, "treatment": s * 1.1 + 0.3})
print(df.bootstrap.summary(np.mean))
#              theta_hat    ci_low   ci_high        se method
# column
# control       1.9973    1.8215    2.1862    0.0941    bca
# treatment     2.4970    2.3036    2.7048    0.1035    bca
```

### scikit-learn cross-validation

```python
from bootstrapx import BootstrapCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

cv = BootstrapCV(n_splits=200, random_state=42)
scores = cross_val_score(
    GradientBoostingClassifier(n_estimators=100),
    X, y, cv=cv, scoring="roc_auc"
)
print(f"AUC: {scores.mean():.4f} ± {scores.std():.4f}")
# AUC: 0.9921 ± 0.0071
```

### Time-series bootstrap

```python
import numpy as np
from bootstrapx import bootstrap

rng = np.random.default_rng(0)
y = np.zeros(500)
for t in range(1, 500):
    y[t] = 0.7 * y[t-1] + rng.normal()

# Moving Block Bootstrap — preserves serial correlation
result = bootstrap(y, np.mean, method="mbb", block_length=15, n_resamples=4999)
print(result)

# Sieve Bootstrap — fits AR(p) model to residuals
result = bootstrap(y, np.mean, method="sieve", n_resamples=9999)
print(result)
```

### A/B test with clustered data

```python
import numpy as np
from bootstrapx import bootstrap

n_clusters = 50
cluster_ids = np.repeat(np.arange(n_clusters), 20)
rng = np.random.default_rng(1)
data = rng.normal(loc=cluster_ids * 0.1, scale=1.0)

result = bootstrap(
    data, np.mean,
    method="cluster",
    cluster_ids=cluster_ids,
    n_resamples=4999,
)
print(result)
# Correctly wider CI that accounts for within-cluster correlation
```

---

## Performance

Measured on Apple M1, Python 3.12, `n_resamples=4 999`, median of 5 runs.  
Run yourself: `python benchmarks/bench_speed.py --quick`

**BCa (bias-corrected and accelerated):**

| n | scipy (ms) | bootstrapx (ms) | Speedup |
|---|---|---|---|
| 200 | 9.6 | 5.8 | **1.7×** |
| 2 000 | 69 | 58 | 1.2× |
| 5 000 | 433 | 156 | **2.8×** |
| 10 000 | 1 015 | 289 | **3.5×** |

At n < 1 000, scipy and bootstrapx are comparable; bootstrapx applies a
vectorised fast path for numpy built-ins (mean, median, std, etc.) at n < 500.
Speedup grows with sample size due to O(n) vectorised jackknife vs O(n²) in scipy.

### Coverage accuracy

BCa empirical coverage at nominal 95%, 1 000 Monte Carlo simulations
across normal, log-normal, exponential and t(3) distributions:
bootstrapx matches scipy to within simulation noise (< 0.01) for mean and median.

> **Note:** BCa coverage for `np.std` on heavy-tailed distributions (exponential)
> is ~91–93% at n = 200 — identical behaviour in both bootstrapx and scipy.
> This reflects known instability of jackknife acceleration for scale statistics,
> not a library-specific issue. Use `n_resamples ≥ 9 999` or
> `method="studentized"` for better coverage when estimating variance.

Run yourself: `python benchmarks/bench_coverage_accuracy.py --fast`

---

## Documentation

📖 **Full docs:** [artyerokhin.github.io/bootstrapx](https://artyerokhin.github.io/bootstrapx)

- [Getting Started](https://artyerokhin.github.io/bootstrapx/getting-started/)
- [Methods Guide](https://artyerokhin.github.io/bootstrapx/methods/)
- [API Reference](https://artyerokhin.github.io/bootstrapx/reference/)
- [Benchmarks](https://artyerokhin.github.io/bootstrapx/benchmarks/)

---

## All supported methods

| Method | `method=` | Use case |
|---|---|---|
| BCa | `"bca"` | General purpose, best coverage accuracy |
| Percentile | `"percentile"` | Simple, fast |
| Basic (Hall) | `"basic"` | Symmetric distributions |
| Studentized | `"studentized"` | Known variance structure |
| Bayesian | `"bayesian"` | Bayesian UQ, non-parametric posterior |
| Poisson weights | `"poisson"` | Weighted bootstrap, survey data |
| Bernoulli weights | `"bernoulli"` | Subsampling variant |
| Subsampling | `"subsampling"` | Heavy tails, no finite variance |
| Moving Block (MBB) | `"mbb"` | Stationary time series |
| Circular Block (CBB) | `"cbb"` | Stationary TS, edge-effect free |
| Stationary | `"stationary"` | Politis & Romano (1994) |
| Tapered Block | `"tapered"` | Paparoditis & Politis (2001) |
| Sieve | `"sieve"` | AR(p) time series (Bühlmann 1997) |
| Wild | `"wild"` | Heteroscedastic residuals (Wu 1986) |
| Cluster | `"cluster"` | Multi-level / panel data |
| Stratified | `"strata"` | Stratified sampling designs |

---

## Contributing

```bash
git clone https://github.com/artyerokhin/bootstrapx.git
cd bootstrapx
pip install -e ".[dev,pandas]"
pytest tests/ -v
```

---

## Citation

If you use bootstrapx in academic work:

```bibtex
@software{bootstrapx,
  author  = {Erokhin, Artem},
  title   = {bootstrapx: Production-grade bootstrap uncertainty estimation},
  url     = {https://github.com/artyerokhin/bootstrapx},
  version = {0.3.1},
  year    = {2026},
}
```

---

## License

MIT — see [LICENSE](LICENSE).
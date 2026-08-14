<div align="center">

# bootstrapx

**Practical bootstrap uncertainty estimation for Python.**

[![CI](https://github.com/artyerokhin/bootstrapx/actions/workflows/ci.yml/badge.svg)](https://github.com/artyerokhin/bootstrapx/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/bootstrapx-lib)](https://pypi.org/project/bootstrapx-lib/)
[![Downloads](https://img.shields.io/pypi/dm/bootstrapx-lib)](https://pypi.org/project/bootstrapx-lib/)
[![Python](https://img.shields.io/pypi/pyversions/bootstrapx-lib)](https://pypi.org/project/bootstrapx-lib/)
[![Coverage Status](https://coveralls.io/repos/github/artyerokhin/bootstrapx/badge.svg?branch=main)](https://coveralls.io/github/artyerokhin/bootstrapx?branch=main)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://artyerokhin.github.io/bootstrapx)

*16 bootstrap methods · sklearn-compatible · pandas accessor · bounded batched working memory*

</div>

---

## Why bootstrapx?

Use **bootstrapx** when ordinary IID resampling is not enough or when you want
one API for IID intervals, block bootstrap, clustered/stratified resampling,
Bayesian bootstrap, pandas summaries, and bootstrap cross-validation.

The library keeps resample matrices in bounded batches. The returned bootstrap
distribution and some method-specific state still grow with `n_resamples` or
sample size, so this is not a claim of constant total memory.

---

## Installation

```bash
pip install bootstrapx-lib                  # core (numpy + scipy only)
pip install "bootstrapx-lib[pandas]"        # + pandas accessor
pip install "bootstrapx-lib[sklearn]"       # + scikit-learn CV integration
pip install "bootstrapx-lib[numba]"         # + faster MBB/CBB/stationary indexing
pip install "bootstrapx-lib[pandas,sklearn]"  # pandas + scikit-learn integrations
pip install "bootstrapx-lib[pandas,sklearn,numba]"  # all optional features
```

---

## Quick Start

### Basic usage

```python
import numpy as np
from bootstrapx import bootstrap

data = np.random.default_rng(42).normal(5, 2, size=300)

result = bootstrap(data, np.mean, random_state=42)
print(result)

print(result.confidence_interval.low, result.confidence_interval.high)
print(5.0 in result.confidence_interval)  # True

# Compact exports for reports and experiment tracking
print(result.to_dict())
print(result.to_frame())  # requires bootstrapx-lib[pandas]
```

### pandas accessor

```python
import pandas as pd
import numpy as np
import bootstrapx  # registers .bootstrap accessor

s = pd.Series(np.random.default_rng(0).exponential(scale=2, size=500))

# On a Series
r = s.bootstrap.bca(np.mean, random_state=42)
print(r)

# On a DataFrame — column-wise summary
df = pd.DataFrame({"control": s, "treatment": s * 1.1 + 0.3})
print(df.bootstrap.summary(np.mean, random_state=42))
```

This DataFrame helper estimates each column separately. It does **not** test
the treatment effect or account for pairing between columns. For paired rows,
bootstrap the row-wise effect; unpaired two-sample effects are not yet a native
workflow. Use the cluster pattern below for repeated observations per unit.

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
result = bootstrap(
    y,
    np.mean,
    method="mbb",
    block_length=15,
    n_resamples=4999,
    random_state=42,
)
print(result)

# Sieve Bootstrap — fits AR(p) model to residuals
result = bootstrap(y, np.mean, method="sieve", n_resamples=9999, random_state=42)
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
    random_state=42,
)
print(result)
```

### Bayesian bootstrap with a custom statistic

Bayesian bootstrap evaluates a functional directly under Dirichlet weights.
`np.mean`, `np.nanmean`, and `np.average` work without extra configuration.
For a custom statistic, provide its weighted form explicitly:

```python
def second_moment(x):
    return np.mean(x**2)

def weighted_second_moment(x, weights):
    return np.sum(weights * x**2)

result = bootstrap(
    data,
    second_moment,
    method="bayesian",
    weighted_statistic=weighted_second_moment,
    random_state=42,
)
```

---

## Benchmarks

bootstrapx is not faster than SciPy in every regime. The audited 0.4.4 release
run on Apple Silicon/macOS 15.7.4, Python 3.11.5, NumPy 2.4.6, and SciPy 1.17.1
found:

| Workflow (`np.mean`, 4,999 resamples) | n | scipy / bootstrapx |
|---|---:|---:|
| BCa | 200 | 1.92× |
| BCa | 1,000 | 1.01× |
| Percentile | 1,000 | 0.94× |
| Percentile | 10,000 | 3.38× |

Values above 1 mean bootstrapx was faster; below 1 mean SciPy was faster. They
are local measurements, not cross-machine guarantees. The complete table,
memory-method caveats, arbitrary-callable results, and optional Numba scope are
in the [benchmark documentation](https://artyerokhin.github.io/bootstrapx/benchmarks/).
The versioned raw results and environment metadata live in
[`benchmark_runs/v0.4.4-release`](benchmark_runs/v0.4.4-release/).

A matched coverage study completed 160 cells: BCa and percentile intervals for
mean, median, and standard deviation over four sample sizes and the documented
distributions. Each cell used 300 independently generated datasets and 4,999
resamples; no trial failed or produced an invalid interval. Mean empirical
coverage was 94.2% for both libraries, and their largest cell-level difference
was 0.67 percentage points. This compares implementations rather than proving
nominal coverage in every finite-sample setting: the 95% Wilson interval for a
single 300-dataset cell is still about six percentage points wide, and both
libraries under-covered the standard deviation of exponential data at n=200.

Run the safe local suite without overwriting previous results:

```bash
pip install -e ".[dev,numba]"
python benchmarks/run_release.py --profile quick
```

For release-candidate coverage with checkpoints, use `--profile release`.
Commands and resume instructions are in the benchmark documentation.

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
| BCa | `"bca"` | General-purpose starting point for scalar statistics |
| Percentile | `"percentile"` | Simple, fast |
| Basic (Hall) | `"basic"` | Reflected bootstrap interval |
| Studentized | `"studentized"` | Bootstrap-t; expensive nested resampling |
| Bayesian | `"bayesian"` | Bayesian UQ, non-parametric posterior |
| Poisson weights | `"poisson"` | Poisson multiplier resampling |
| Bernoulli subsets | `"bernoulli"` | Calibrated random-subset inference |
| Subsampling | `"subsampling"` | Root-scaled inference from smaller samples |
| Moving Block (MBB) | `"mbb"` | Stationary time series |
| Circular Block (CBB) | `"cbb"` | Stationary series with circular blocks |
| Stationary | `"stationary"` | Politis & Romano (1994) |
| Tapered Block | `"tapered"` | Paparoditis & Politis (2001) |
| Sieve | `"sieve"` | AR(p) time series (Bühlmann 1997) |
| Wild | `"wild"` | Heteroscedastic residuals (Wu 1986) |
| Cluster | `"cluster"` | One-level grouped / panel data |
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
  title   = {bootstrapx: Practical bootstrap uncertainty estimation},
  url     = {https://github.com/artyerokhin/bootstrapx},
  version = {0.4.4},
  year    = {2026},
}
```

---

## License

MIT — see [LICENSE](LICENSE).

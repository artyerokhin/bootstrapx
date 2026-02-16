# bootstrapx

**Production-grade uncertainty estimation for Python.**

14 bootstrap methods · Numba JIT · Optional CUDA GPU · Memory-safe batching

---

## Quick Start

```python
import numpy as np
from bootstrapx import bootstrap

data = np.random.default_rng(42).normal(5, 2, size=200)
result = bootstrap(data, np.mean)
print(result)
# BootstrapResult(method='bca', theta_hat=4.94, se=0.13, CI=[4.70, 5.19])
```

## Why bootstrapx?

| Feature | `scipy.stats.bootstrap` | R `boot` | **bootstrapx** |
|---|:---:|:---:|:---:|
| BCa interval | ✅ | ✅ | ✅ |
| Studentized (bootstrap-t) | ❌ | ✅ | ✅ |
| Poisson / Bernoulli weights | ❌ | ❌ | ✅ |
| Time-series methods (6 types) | ❌ | Partial | ✅ |
| Wild bootstrap | ❌ | ❌ | ✅ |
| Cluster / Stratified | ❌ | Partial | ✅ |
| Numba JIT acceleration | ❌ | N/A | ✅ |
| GPU (CUDA) support | ❌ | ❌ | ✅ |
| Generator-based batching | ❌ | ❌ | ✅ |

## Install

```bash
pip install bootstrapx
```

<div align="center">

# bootstrapx

**Production-grade uncertainty estimation for Python.**

[![CI](https://github.com/artyerokhin/bootstrapx/actions/workflows/ci.yml/badge.svg)](https://github.com/artyerokhin/bootstrapx/actions)
[![PyPI](https://img.shields.io/pypi/v/bootstrapx)](https://pypi.org/project/bootstrapx-lib/)
[![Python](https://img.shields.io/pypi/pyversions/bootstrapx)](https://pypi.org/project/bootstrapx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://artyerokhin.github.io/bootstrapx)

*14 bootstrap methods · Numba JIT · Optional CUDA GPU · Memory-safe batching*

</div>

---

## Why bootstrapx?

`scipy.stats.bootstrap` supports only 3 CI methods and has no time-series support.
The R `boot` package is comprehensive but not accessible from Python.
**bootstrapx** bridges this gap with 14 methods, Numba acceleration, and a clean API.

| Feature | `scipy` | R `boot` | **bootstrapx** |
|---|:---:|:---:|:---:|
| BCa interval | ✅ | ✅ | ✅ |
| Studentized (bootstrap-t) | ❌ | ✅ | ✅ |
| Poisson / Bernoulli weights | ❌ | ❌ | ✅ |
| Time-series (MBB, CBB, Stationary, Sieve, Tapered, Wild) | ❌ | Partial | ✅ |
| Cluster / Stratified | ❌ | Partial | ✅ |
| Numba JIT | ❌ | N/A | ✅ |
| CUDA GPU | ❌ | ❌ | ✅ |
| Generator batching (constant memory) | ❌ | ❌ | ✅ |

---

## Installation

```bash
pip install bootstrapx-lib

# With GPU
pip install "bootstrapx-lib[cuda]"
```

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

---

## Performance

bootstrapx is significantly faster than `scipy.stats.bootstrap` for large datasets:

| N | Method | Scipy | Bootstrapx | Speedup |
|---|---|---|---|---|
| 5,000 | BCa | 0.80s | 0.27s | **3.0x** |
| 50,000 | Percentile | 7.29s | 2.01s | **3.6x** |
| 100,000 | Percentile | 54.34s | 3.99s | **13.6x** |

*Benchmark on Apple M1, Python 3.12. See [Benchmarks](https://artyerokhin.github.io/bootstrapx/benchmarks/) for details.*

---

## Coverage Accuracy

Monte Carlo simulation ($N=1000$, 500 runs) confirms correct statistical coverage:

| Distribution | BCa Coverage (Nominal 95%) |
|---|---|
| Normal | **94.8%** |
| Skewed (Exponential) | **95.0%** |
| Heavy-Tailed (t-dist) | **94.0%** |

---

## Documentation

📖 **Full docs:** [artyerokhin.github.io/bootstrapx](https://artyerokhin.github.io/bootstrapx)

- [Getting Started](https://artyerokhin.github.io/bootstrapx/getting-started/)
- [Methods Guide](https://artyerokhin.github.io/bootstrapx/methods/) — math behind each method
- [API Reference](https://artyerokhin.github.io/bootstrapx/reference/)
- [Benchmarks](https://artyerokhin.github.io/bootstrapx/benchmarks/)

---

## Contributing

```bash
git clone https://github.com/artyerokhin/bootstrapx.git
cd bootstrapx
pip install -e ".[dev,docs]"
pytest tests/ -v
mkdocs serve
```

---

## License

MIT — see [LICENSE](LICENSE).

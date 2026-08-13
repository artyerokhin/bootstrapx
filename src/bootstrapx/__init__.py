"""bootstrapx — Practical bootstrap uncertainty estimation for Python.

Provides 16 bootstrap methods: iid (BCa, percentile, basic, studentized,
Bayesian, Poisson, Bernoulli, subsampling), time-series (MBB, CBB, stationary,
tapered, sieve, wild), and hierarchical (cluster, stratified).

Also exposes:
- ``BootstrapCV`` — scikit-learn compatible cross-validator using bootstrap
- ``pd.Series.bootstrap`` / ``pd.DataFrame.bootstrap`` — pandas accessor
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .api import BootstrapResult, bootstrap
from .stats.confidence import ConfidenceInterval

# Optional integrations — only register if their deps are present
try:
    from .compat.sklearn_cv import BootstrapCV  # noqa: F401

    _sklearn_available = True
except ImportError:
    _sklearn_available = False

try:
    from .compat.pandas_accessor import _  # noqa: F401  registers accessor

    _pandas_available = True
except ImportError:
    _pandas_available = False

try:
    __version__ = version("bootstrapx-lib")
except PackageNotFoundError:
    __version__ = "0.4.4"

__all__ = [
    "bootstrap",
    "BootstrapResult",
    "ConfidenceInterval",
    "__version__",
]

if _sklearn_available:
    __all__.append("BootstrapCV")

"""bootstrapx — Production-grade uncertainty estimation for Python."""
from __future__ import annotations

from importlib.metadata import version, PackageNotFoundError

from .api import bootstrap, BootstrapResult
from .stats.confidence import ConfidenceInterval

try:
    __version__ = version("bootstrapx-lib")
except PackageNotFoundError:
    __version__ = "0.2.0"

__all__ = ["bootstrap", "BootstrapResult", "ConfidenceInterval"]

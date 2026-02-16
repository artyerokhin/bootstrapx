"""bootstrapx — Production-grade uncertainty estimation for Python."""
from __future__ import annotations

# Use relative import to work without installation
from .api import bootstrap, BootstrapResult

__version__ = "0.1.0"
__all__ = ["bootstrap", "BootstrapResult"]

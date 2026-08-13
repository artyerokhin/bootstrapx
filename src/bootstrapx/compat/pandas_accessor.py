"""Pandas accessor: ``Series.bootstrap`` and ``DataFrame.bootstrap``.

Usage
-----
>>> import pandas as pd
>>> import numpy as np
>>> from bootstrapx import bootstrap  # registers accessor on import
>>>
>>> s = pd.Series(np.random.default_rng(0).normal(5, 2, 300))
>>>
>>> # Fluent API on Series
>>> result = s.bootstrap.bca(np.mean)
>>> print(result)
>>> result = s.bootstrap.ci(np.median, method="percentile", n_resamples=4999)
>>>
>>> # DataFrame: operates column-wise
>>> df = pd.DataFrame({"a": s, "b": s * 1.2 + 1})
>>> summary = df.bootstrap.summary(np.mean)
>>> print(summary)  # DataFrame with columns: theta_hat, ci_low, ci_high, se

Notes
-----
The accessor is registered automatically when ``bootstrapx`` is imported.
No extra import is needed beyond ``import bootstrapx``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "pandas is required for the bootstrap accessor. Install with: pip install pandas"
    ) from exc

from bootstrapx.api import BootstrapResult, bootstrap


class _BootstrapSeriesAccessor:
    """Accessor registered as ``pd.Series.bootstrap``."""

    def __init__(self, obj: pd.Series):
        self._obj = obj

    def ci(
        self,
        statistic: Callable[..., float],
        *,
        method: str = "bca",
        n_resamples: int = 9999,
        confidence_level: float = 0.95,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> BootstrapResult:
        """Run bootstrap and return a :class:`~bootstrapx.BootstrapResult`."""
        return bootstrap(
            self._obj,
            statistic,
            method=method,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            random_state=random_state,
            **kwargs,
        )

    def bca(
        self,
        statistic: Callable[..., float],
        n_resamples: int = 9999,
        confidence_level: float = 0.95,
        random_state: int | None = None,
    ) -> BootstrapResult:
        """Shortcut for ``method='bca'``."""
        return self.ci(
            statistic,
            method="bca",
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            random_state=random_state,
        )

    def percentile(
        self,
        statistic: Callable[..., float],
        n_resamples: int = 9999,
        confidence_level: float = 0.95,
        random_state: int | None = None,
    ) -> BootstrapResult:
        """Shortcut for ``method='percentile'``."""
        return self.ci(
            statistic,
            method="percentile",
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            random_state=random_state,
        )


class _BootstrapDataFrameAccessor:
    """Accessor registered as ``pd.DataFrame.bootstrap``.

    Applies bootstrap column-wise and returns a summary DataFrame.
    """

    def __init__(self, obj: pd.DataFrame):
        self._obj = obj

    def summary(
        self,
        statistic: Callable[..., float],
        *,
        method: str = "bca",
        n_resamples: int = 9999,
        confidence_level: float = 0.95,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Return a DataFrame with bootstrap CI summary for each column.

        Returns
        -------
        pd.DataFrame
            Index: column names of the original DataFrame.
            Columns: ``theta_hat``, ``ci_low``, ``ci_high``, ``se``, ``method``.
        """
        rows = []
        for col in self._obj.columns:
            r = bootstrap(
                self._obj[col],
                statistic,
                method=method,
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                random_state=random_state,
                **kwargs,
            )
            rows.append(
                {
                    "column": col,
                    "theta_hat": r.theta_hat,
                    "ci_low": r.confidence_interval.low,
                    "ci_high": r.confidence_interval.high,
                    "se": r.standard_error,
                    "method": r.method,
                }
            )
        return pd.DataFrame(rows).set_index("column")

    def ci(
        self,
        statistic: Callable[..., float],
        *,
        method: str = "bca",
        n_resamples: int = 9999,
        confidence_level: float = 0.95,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Alias for :meth:`summary`."""
        return self.summary(
            statistic,
            method=method,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            random_state=random_state,
            **kwargs,
        )


# Register accessors
pd.api.extensions.register_series_accessor("bootstrap")(_BootstrapSeriesAccessor)
pd.api.extensions.register_dataframe_accessor("bootstrap")(_BootstrapDataFrameAccessor)

# Sentinel for __init__.py import check
_ = True

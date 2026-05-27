"""Validation helpers and heuristics.

Changes vs 0.2.0:
- auto_batch_size: previous code divided target_elements (65 536) by n
  *without accounting for itemsize*, so for float64 the actual batch was
  65 536 × 8 = 512 KB — 8× larger than the intended L2 target.
  Fixed: divide target_bytes by (n × itemsize).
"""
from __future__ import annotations

from typing import Any

import numpy as np


def validate_data(data: Any, *, allow_2d: bool = False) -> np.ndarray:
    """Validate and convert input data to a C-contiguous float64 array.

    Accepts numpy arrays, lists, pandas Series and DataFrames.
    """
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            if not allow_2d and data.shape[1] != 1:
                raise ValueError(
                    f"DataFrame with {data.shape[1]} columns passed. "
                    "Use a single column or pass allow_2d=True."
                )
            arr = data.to_numpy(dtype=np.float64, na_value=np.nan)
            if not allow_2d and arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.ravel()
        elif isinstance(data, pd.Series):
            arr = data.to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            arr = np.asarray(data, dtype=np.float64)
    except ImportError:
        arr = np.asarray(data, dtype=np.float64)

    if arr.ndim == 0:
        raise ValueError("Scalar data is not supported.")
    if arr.ndim > 2 or (arr.ndim == 2 and not allow_2d):
        raise ValueError(
            f"Expected 1-D array, got shape {arr.shape}. "
            "Pass allow_2d=True for matrix data."
        )
    if np.any(np.isnan(arr)):
        raise ValueError("Data contains NaN values. Remove or impute them first.")
    if arr.shape[0] < 2:
        raise ValueError("Data must have at least 2 observations.")

    # Ensure C-contiguous layout for downstream index operations
    return np.ascontiguousarray(arr)


def auto_batch_size(n: int, n_resamples: int, itemsize: int = 8) -> int:
    """Heuristic batch sizing targeting ~64 KiB per batch to fit in L2 cache.

    A batch of shape ``(batch_size, n)`` occupies
    ``batch_size × n × itemsize`` bytes.  The previous implementation divided
    a *element count* target by ``n``, ignoring itemsize, so float64 batches
    were 8× too large.

    Parameters
    ----------
    n : int
        Sample size (number of observations).
    n_resamples : int
        Total number of resamples requested.
    itemsize : int
        Bytes per element (default 8 for float64).
    """
    target_bytes = 65_536          # 64 KiB — fits comfortably in most L2 caches
    bs = max(1, target_bytes // (max(n, 1) * itemsize))
    return min(bs, n_resamples)

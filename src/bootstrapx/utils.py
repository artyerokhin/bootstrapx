"""Validation helpers and heuristics."""
from __future__ import annotations

from typing import Any

import numpy as np


def validate_data(data: Any, *, allow_2d: bool = False) -> np.ndarray:
    """Validate and convert input data to a numpy array.

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
    return arr


def auto_batch_size(n: int, n_resamples: int) -> int:
    """Heuristic batch sizing targeting ~64 KiB per batch for L2 cache fit."""
    target_elements = 65_536
    bs = max(1, target_elements // max(n, 1))
    return min(bs, n_resamples)

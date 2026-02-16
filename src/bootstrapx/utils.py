from __future__ import annotations

import numpy as np


def validate_data(data, *, allow_2d: bool = False) -> np.ndarray:
    arr = np.asarray(data)
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
    # Heuristic: aim for chunks that fit in L2 cache but huge enough for vectorization
    # 32k elements per batch is usually a sweet spot for numpy
    target_elements = 32_768
    bs = max(1, target_elements // n)
    return min(bs, n_resamples)

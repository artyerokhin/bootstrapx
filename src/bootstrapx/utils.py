"""Validation helpers and heuristics.

Changes vs 0.2.0:
- auto_batch_size: previous code divided target_elements (65 536) by n
  *without accounting for itemsize*, so for float64 the actual batch was
  65 536 × 8 = 512 KB — 8× larger than the intended L2 target.
  Fixed: divide target_bytes by (n × itemsize).
"""

from __future__ import annotations

from collections.abc import Mapping
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
            f"Expected 1-D array, got shape {arr.shape}. Pass allow_2d=True for matrix data."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Data must contain only finite values. Remove or impute NaN/inf values.")
    if arr.shape[0] < 2:
        raise ValueError("Data must have at least 2 observations.")

    # Ensure C-contiguous layout for downstream index operations
    return np.ascontiguousarray(arr)


def validate_bootstrap_params(
    *,
    method: str,
    n_observations: int,
    n_resamples: int,
    batch_size: int | None,
    confidence_level: float,
    ci_method: str | None,
    n_jobs: int,
    kwargs: Mapping[str, Any],
) -> None:
    """Validate public ``bootstrap`` parameters before starting computation.

    Several generators advance with ``done += batch_size``.  Validating these
    values at the API boundary prevents invalid input from causing a hang or a
    late, implementation-specific exception.
    """
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int | np.integer):
        raise TypeError("n_resamples must be an integer.")
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2.")
    if batch_size is not None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int | np.integer):
            raise TypeError("batch_size must be an integer or None.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
    if not isinstance(confidence_level, int | float | np.number) or not np.isfinite(
        confidence_level
    ):
        raise ValueError("confidence_level must be a finite number between 0 and 1.")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1.")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int | np.integer) or n_jobs == 0:
        raise ValueError("n_jobs must be a non-zero integer.")

    ci_capable = {"percentile", "basic", "bca", "studentized"}
    if method in ci_capable and ci_method is not None:
        raise ValueError("ci_method is only supported for generator-based bootstrap methods.")
    if method not in ci_capable and ci_method not in (None, "percentile", "basic"):
        raise ValueError("ci_method must be 'percentile', 'basic', or None.")

    allowed_kwargs = {
        "studentized": {"n_inner"},
        "bernoulli": {"prob"},
        "subsampling": {"subsample_size"},
        "mbb": {"block_length"},
        "cbb": {"block_length"},
        "stationary": {"mean_block"},
        "tapered": {"block_length", "taper"},
        "sieve": {"ar_order"},
        "wild": {"fitted", "distribution"},
        "cluster": {"cluster_ids"},
        "strata": {"strata_ids"},
    }
    unknown = set(kwargs) - allowed_kwargs.get(method, set())
    if unknown:
        raise TypeError(
            f"Unsupported keyword argument(s) for method={method!r}: {sorted(unknown)}."
        )

    def positive_int(name: str, value: Any, *, maximum: int | None = None) -> None:
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise TypeError(f"{name} must be an integer.")
        if value < 1 or (maximum is not None and value > maximum):
            upper = f" and at most {maximum}" if maximum is not None else ""
            raise ValueError(f"{name} must be at least 1{upper}.")

    if method == "studentized":
        positive_int("n_inner", kwargs.get("n_inner", 50))
        if kwargs.get("n_inner", 50) < 2:
            raise ValueError("n_inner must be at least 2.")
    if method == "bernoulli":
        prob = kwargs.get("prob", 0.5)
        if (
            not isinstance(prob, int | float | np.number)
            or not np.isfinite(prob)
            or not 0 < prob <= 1
        ):
            raise ValueError("prob must be a finite number in (0, 1].")
    if method == "subsampling" and kwargs.get("subsample_size") is not None:
        positive_int("subsample_size", kwargs["subsample_size"], maximum=n_observations - 1)
    if method in {"mbb", "cbb", "tapered"}:
        positive_int("block_length", kwargs.get("block_length", 10), maximum=n_observations - 1)
    if method == "stationary":
        mean_block = kwargs.get("mean_block", 10.0)
        if (
            not isinstance(mean_block, int | float | np.number)
            or not np.isfinite(mean_block)
            or mean_block <= 1
        ):
            raise ValueError("mean_block must be a finite number greater than 1.")
    if method == "sieve" and kwargs.get("ar_order") is not None:
        positive_int("ar_order", kwargs["ar_order"], maximum=n_observations - 2)
    if method == "wild":
        if kwargs.get("distribution", "rademacher") not in {"rademacher", "mammen"}:
            raise ValueError("distribution must be 'rademacher' or 'mammen'.")
        fitted = kwargs.get("fitted")
        if fitted is not None:
            fitted_arr = np.asarray(fitted, dtype=np.float64)
            if fitted_arr.shape != (n_observations,) or not np.all(np.isfinite(fitted_arr)):
                raise ValueError(
                    "fitted must be a finite one-dimensional array matching data length."
                )
    if method in {"cluster", "strata"}:
        name = "cluster_ids" if method == "cluster" else "strata_ids"
        identifiers = kwargs.get(name)
        if identifiers is None:
            raise ValueError(f"{method} method requires `{name}` kwarg.")
        ids = np.asarray(identifiers)
        if ids.ndim != 1 or len(ids) != n_observations:
            raise ValueError(f"{name} must be one-dimensional and match data length.")
        if len(np.unique(ids)) < 2:
            raise ValueError(f"{name} must contain at least two distinct groups.")


def validate_random_state(random_state: Any) -> None:
    """Validate the documented random-state inputs before NumPy is invoked."""
    if random_state is None or isinstance(random_state, np.random.Generator):
        return
    if isinstance(random_state, bool) or not isinstance(random_state, int | np.integer):
        raise TypeError("random_state must be an integer, numpy Generator, or None.")


def validate_bootstrap_distribution(distribution: Any, n_resamples: int) -> np.ndarray:
    """Return a finite, correctly shaped bootstrap distribution.

    A custom vectorized statistic can otherwise silently return a different
    number of values than resamples requested, producing an invalid interval.
    """
    values = np.asarray(distribution, dtype=np.float64)
    if values.ndim != 1 or values.size != n_resamples:
        raise ValueError(
            "statistic must return exactly one scalar per resample "
            f"(expected {n_resamples}, received shape {values.shape})."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("statistic returned NaN or inf for at least one bootstrap resample.")
    return values


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
    target_bytes = 65_536  # 64 KiB — fits comfortably in most L2 caches
    bs = max(1, target_bytes // (max(n, 1) * itemsize))
    return min(bs, n_resamples)

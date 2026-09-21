"""Scalar metrics for jointly observed numeric columns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RatioOfSums:
    """Compute ``sum(sample[:, numerator]) / sum(sample[:, denominator])``.

    Column positions are non-negative integers. This is not the mean of
    row-wise ratios, and it does not compare experiment arms: use it as the
    ``statistic`` in ``bootstrap_two_sample(..., allow_2d=True)``. The callable
    receives a numeric NumPy matrix, including when the original input is a
    DataFrame. A zero total denominator or non-finite value raises an error;
    no observations or replicates are discarded or stabilized.
    """

    numerator: int = 0
    denominator: int = 1

    def __post_init__(self) -> None:
        for name in ("numerator", "denominator"):
            index = getattr(self, name)
            if isinstance(index, bool) or not isinstance(index, int | np.integer):
                raise TypeError(f"{name} must be an integer column position.")
            if index < 0:
                raise ValueError(f"{name} must be a non-negative column position.")

    def __call__(self, sample: NDArray[np.float64]) -> float:
        values = np.asarray(sample, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("RatioOfSums requires a two-dimensional numeric sample.")
        if values.shape[0] == 0:
            raise ValueError("RatioOfSums requires at least one observation.")
        if max(self.numerator, self.denominator) >= values.shape[1]:
            raise ValueError("RatioOfSums column positions exceed the sample's feature count.")
        if not np.isfinite(values).all():
            raise ValueError("RatioOfSums requires finite values.")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            numerator = float(np.sum(values[:, self.numerator]))
            denominator = float(np.sum(values[:, self.denominator]))
            if not np.isfinite(numerator) or not np.isfinite(denominator):
                raise ValueError("RatioOfSums requires finite column sums.")
            if denominator == 0.0:
                raise ValueError("RatioOfSums is undefined because the denominator sum is zero.")
            result = numerator / denominator
        if not np.isfinite(result):
            raise ValueError("RatioOfSums must return a finite value.")
        return result

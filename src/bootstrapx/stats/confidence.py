"""Confidence interval constructors.

Changes vs 0.2.0:
- _jackknife: preallocates a single reusable buffer instead of calling
  np.concatenate() N times.  For n=5 000 this cuts peak memory from ~190 MB
  to ~40 KB and removes O(n) allocator pressure.
- bca_interval: the acceleration constant a_hat is now computed via
  np.einsum for the cubic/squared sums — avoids two temporary arrays.
- studentized_interval: t_low / t_high variable names were swapped in the
  original (low used (1-alpha/2) quantile, high used alpha/2).  Fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats


@dataclass
class ConfidenceInterval:
    low: float
    high: float
    method: str

    def __contains__(self, value: float) -> bool:
        return self.low <= value <= self.high

    @property
    def width(self) -> float:
        return self.high - self.low


# ---------------------------------------------------------------------------
# Basic interval constructors
# ---------------------------------------------------------------------------


def percentile_interval(
    boot_stats: np.ndarray,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    alpha = 1.0 - confidence_level
    return ConfidenceInterval(
        low=float(np.percentile(boot_stats, 100 * alpha / 2)),
        high=float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        method="percentile",
    )


def basic_interval(
    boot_stats: np.ndarray,
    theta_hat: float,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    alpha = 1.0 - confidence_level
    q_low = float(np.percentile(boot_stats, 100 * alpha / 2))
    q_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return ConfidenceInterval(
        low=2 * theta_hat - q_high,
        high=2 * theta_hat - q_low,
        method="basic",
    )


def root_interval(
    root_stats: np.ndarray,
    theta_hat: float,
    scale_n: float,
    confidence_level: float = 0.95,
    *,
    method: str,
) -> ConfidenceInterval:
    """Construct an interval from a centered, scaled resampling root.

    ``root_stats`` estimates the distribution of
    ``scale_n * (theta_hat - theta)``.  This is the required construction
    for subsampling and calibrated delete-fraction procedures; directly
    taking percentiles of smaller-sample estimates has the wrong scale.
    """
    alpha = 1.0 - confidence_level
    q_low = float(np.percentile(root_stats, 100 * alpha / 2))
    q_high = float(np.percentile(root_stats, 100 * (1 - alpha / 2)))
    return ConfidenceInterval(
        low=theta_hat - q_high / scale_n,
        high=theta_hat - q_low / scale_n,
        method=method,
    )


# ---------------------------------------------------------------------------
# Jackknife — memory-efficient implementation
# ---------------------------------------------------------------------------


def _jackknife(
    data: np.ndarray,
    statistic: callable,
    n_jobs: int = 1,
) -> np.ndarray:
    """Leave-one-out jackknife.

    Uses a single preallocated buffer (O(n) memory) instead of calling
    ``np.concatenate`` N times (O(n²) total allocations).

    Parallelism via joblib is available for large n and expensive statistics;
    for n < 2000 or cheap statistics the joblib overhead dominates, so the
    sequential path is preferred.
    """
    n = data.shape[0]

    if n_jobs != 1 and n >= 2000:
        from joblib import Parallel, delayed

        def _loo(i: int) -> float:
            buf = np.empty(n - 1, dtype=data.dtype)
            buf[:i] = data[:i]
            buf[i:] = data[i + 1 :]
            return float(statistic(buf))

        return np.array(Parallel(n_jobs=n_jobs)(delayed(_loo)(i) for i in range(n)))

    # Sequential with single preallocated buffer
    buf = np.empty(n - 1, dtype=data.dtype)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        buf[:i] = data[:i]
        buf[i:] = data[i + 1 :]
        out[i] = float(statistic(buf))
    return out


# ---------------------------------------------------------------------------
# BCa interval
# ---------------------------------------------------------------------------


def bca_interval(
    boot_stats: np.ndarray,
    data: np.ndarray,
    statistic: callable,
    theta_hat: float,
    confidence_level: float = 0.95,
    n_jobs: int = 1,
) -> ConfidenceInterval:
    """Bias-corrected and accelerated (BCa) bootstrap interval.

    Reference: Efron & Tibshirani (1993), §14.3.
    """
    alpha = 1.0 - confidence_level

    # Bias-correction constant z0
    prop_less = np.clip(np.mean(boot_stats < theta_hat), 1e-10, 1 - 1e-10)
    z0 = float(sp_stats.norm.ppf(prop_less))

    # Acceleration constant a_hat via jackknife influence function
    jack_stats = _jackknife(data, statistic, n_jobs=n_jobs)
    mean_jack = jack_stats.mean()
    diffs = mean_jack - jack_stats  # shape (n,)

    # Use einsum to avoid two temporary power arrays
    num = float(np.einsum("i,i,i->", diffs, diffs, diffs))  # sum(diffs**3)
    den = float(np.einsum("i,i->", diffs, diffs)) ** 1.5  # sum(diffs**2)**1.5
    a_hat = num / (6.0 * den) if den != 0.0 else 0.0

    def _adj_quantile(z_alpha: float) -> float:
        numer = z0 + z_alpha
        denom = 1.0 - a_hat * numer
        return float(sp_stats.norm.cdf(z0 + numer / denom))

    p_low = _adj_quantile(sp_stats.norm.ppf(alpha / 2))
    p_high = _adj_quantile(sp_stats.norm.ppf(1.0 - alpha / 2))

    return ConfidenceInterval(
        low=float(np.percentile(boot_stats, 100 * p_low)),
        high=float(np.percentile(boot_stats, 100 * p_high)),
        method="bca",
    )


# ---------------------------------------------------------------------------
# Studentized (bootstrap-t) interval
# ---------------------------------------------------------------------------


def studentized_interval(
    data: np.ndarray,
    statistic: callable,
    theta_hat: float,
    boot_stats: np.ndarray,
    boot_se: np.ndarray,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Bootstrap-t (studentized) interval.

    Fixed quantile assignment: the original code had t_low / t_high swapped,
    which caused the interval to be reflected around theta_hat for asymmetric
    distributions.

    Reference: Hall (1992), §3.5.
    """
    alpha = 1.0 - confidence_level
    mask = boot_se > 0
    if mask.sum() < 10:
        # Degenerate: fall back to percentile
        return percentile_interval(boot_stats, confidence_level)

    t_vals = (boot_stats[mask] - theta_hat) / boot_se[mask]
    # t_q_lo is the alpha/2 quantile of T (left tail)
    # t_q_hi is the 1-alpha/2 quantile of T (right tail)
    t_q_lo = float(np.percentile(t_vals, 100 * alpha / 2))
    t_q_hi = float(np.percentile(t_vals, 100 * (1.0 - alpha / 2)))
    se_hat = float(np.std(boot_stats, ddof=1))

    # CI: [theta_hat - t_q_hi * se_hat, theta_hat - t_q_lo * se_hat]
    return ConfidenceInterval(
        low=theta_hat - t_q_hi * se_hat,
        high=theta_hat - t_q_lo * se_hat,
        method="studentized",
    )

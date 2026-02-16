from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats


@dataclass
class ConfidenceInterval:
    low: float
    high: float
    method: str


def percentile_interval(boot_stats, confidence_level=0.95):
    alpha = 1.0 - confidence_level
    return ConfidenceInterval(
        low=float(np.percentile(boot_stats, 100 * alpha / 2)),
        high=float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        method="percentile",
    )


def basic_interval(boot_stats, theta_hat, confidence_level=0.95):
    alpha = 1.0 - confidence_level
    q_low = np.percentile(boot_stats, 100 * alpha / 2)
    q_high = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return ConfidenceInterval(
        low=float(2 * theta_hat - q_high),
        high=float(2 * theta_hat - q_low),
        method="basic",
    )


def _jackknife(data, statistic):
    n = data.shape[0]
    # Simple leave-one-out
    return np.array(
        [statistic(np.concatenate([data[:i], data[i + 1 :]])) for i in range(n)]
    )


def bca_interval(boot_stats, data, statistic, theta_hat, confidence_level=0.95):
    alpha = 1.0 - confidence_level

    # Bias correction z0
    prop_less = np.mean(boot_stats < theta_hat)
    # Clip to avoid infinity
    prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
    z0 = float(sp_stats.norm.ppf(prop_less))

    # Acceleration a
    jack_stats = _jackknife(data, statistic)
    mean_jack = jack_stats.mean()
    diffs = mean_jack - jack_stats

    num = (diffs**3).sum()
    den = ((diffs**2).sum()) ** 1.5

    a_hat = num / (6 * den) if den != 0 else 0.0

    # Adjusted percentiles
    def adjust_percentile(z_alpha):
        num_adj = z0 + z_alpha
        denom_adj = 1 - a_hat * num_adj
        return float(sp_stats.norm.cdf(z0 + num_adj / denom_adj))

    p_low = adjust_percentile(sp_stats.norm.ppf(alpha / 2))
    p_high = adjust_percentile(sp_stats.norm.ppf(1 - alpha / 2))

    return ConfidenceInterval(
        low=float(np.percentile(boot_stats, 100 * p_low)),
        high=float(np.percentile(boot_stats, 100 * p_high)),
        method="bca",
    )


def studentized_interval(
    data, statistic, theta_hat, boot_stats, boot_se, confidence_level=0.95
):
    alpha = 1.0 - confidence_level
    mask = boot_se > 0
    t_vals = (boot_stats[mask] - theta_hat) / boot_se[mask]

    t_low = np.percentile(t_vals, 100 * (1 - alpha / 2))
    t_high = np.percentile(t_vals, 100 * alpha / 2)

    se_hat = np.std(boot_stats, ddof=1)

    return ConfidenceInterval(
        low=float(theta_hat - t_low * se_hat),
        high=float(theta_hat - t_high * se_hat),
        method="studentized",
    )

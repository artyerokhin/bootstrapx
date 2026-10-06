"""Offline paired-cluster Brier comparison with an analytically known target.

This teaching dataset uses fixed forecasts, not fitted models or real patients.
Forecast B knows the simulated group probability; it is an oracle reference.
Run with PYTHONPATH=src python examples/paired_cluster_brier.py.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import bootstrap as scipy_bootstrap

from bootstrapx import bootstrap_two_sample


def brier(rows: np.ndarray) -> float:
    return float(np.mean((rows[:, 1] - rows[:, 0]) ** 2))


def run(n_resamples: int = 999) -> dict[str, object]:
    rng = np.random.default_rng(707)
    groups, rows_per_group = 120, 5
    ids = np.repeat(np.arange(groups), rows_per_group)
    q = np.repeat(rng.choice([0.2, 0.8], size=groups), rows_per_group)
    labels = rng.binomial(1, q)
    a = np.column_stack((labels, np.full(len(ids), 0.4)))
    b = np.column_stack((labels, q))
    observations = np.arange(len(ids))
    result = bootstrap_two_sample(
        a,
        b,
        brier,
        allow_2d=True,
        paired=True,
        paired_cluster_ids=ids,
        control_observation_ids=observations,
        treatment_observation_ids=observations,
        method="percentile",
        metric_name="Brier score",
        effect_unit="score difference",
        n_resamples=n_resamples,
        random_state=42,
    )
    # Equal group sizes make the effect exactly the mean of group loss deltas.
    group_deltas = (
        (((b[:, 1] - labels) ** 2) - ((a[:, 1] - labels) ** 2))
        .reshape(groups, rows_per_group)
        .mean(axis=1)
    )
    reference = scipy_bootstrap(
        (group_deltas,),
        np.mean,
        method="percentile",
        vectorized=False,
        n_resamples=n_resamples,
        random_state=42,
    )
    # E[Brier A]=.26; E[Brier B]=.16 under q=.2/.8 with equal probability.
    return {"bootstrapx": result, "scipy": reference, "true_difference": -0.1}


if __name__ == "__main__":
    analysis = run()
    result = analysis["bootstrapx"]
    print(result)
    print(f"Analytical population difference B - A: {analysis['true_difference']:+.3f}")
    print(f"SciPy reference interval: {analysis['scipy'].confidence_interval}")
    print(f"90% interval from saved replicates: {result.interval(confidence_level=0.90)}")
    print("Negative difference favors B. One dataset does not establish interval coverage.")

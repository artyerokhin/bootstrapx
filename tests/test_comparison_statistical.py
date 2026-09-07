"""Small statistical regression checks for experiment comparisons."""

import numpy as np

from bootstrapx import bootstrap_two_sample


def test_independent_bca_mean_has_reasonable_coverage() -> None:
    rng = np.random.default_rng(104)
    covered = 0
    n_simulations = 60
    true_effect = 0.35
    for _ in range(n_simulations):
        control = rng.normal(0.0, 1.0, 80)
        treatment = rng.normal(true_effect, 1.2, 100)
        result = bootstrap_two_sample(
            control,
            treatment,
            np.mean,
            method="bca",
            n_resamples=299,
            random_state=int(rng.integers(0, 2**31)),
        )
        covered += true_effect in result.confidence_interval

    assert covered / n_simulations >= 0.85


def test_paired_design_uses_correlation_to_reduce_uncertainty() -> None:
    rng = np.random.default_rng(105)
    control = rng.normal(10.0, 3.0, 200)
    treatment = control + rng.normal(0.5, 0.8, 200)
    paired = bootstrap_two_sample(
        control,
        treatment,
        np.mean,
        paired=True,
        method="percentile",
        n_resamples=999,
        random_state=8,
    )
    independent = bootstrap_two_sample(
        control,
        treatment,
        np.mean,
        method="percentile",
        n_resamples=999,
        random_state=8,
    )

    assert paired.standard_error < independent.standard_error * 0.5


def test_cluster_design_restores_experiment_uncertainty() -> None:
    rng = np.random.default_rng(106)
    n_simulations = 50
    true_effect = 0.4
    clustered_covered = 0
    iid_covered = 0

    for _ in range(n_simulations):
        n_clusters = 24
        rows_per_cluster = 6
        control_ids = np.repeat(np.arange(n_clusters), rows_per_cluster)
        treatment_ids = np.repeat(np.arange(n_clusters), rows_per_cluster)
        control_effects = rng.normal(0.0, 1.2, n_clusters)
        treatment_effects = rng.normal(true_effect, 1.2, n_clusters)
        control = np.repeat(control_effects, rows_per_cluster) + rng.normal(
            0.0, 0.35, len(control_ids)
        )
        treatment = np.repeat(treatment_effects, rows_per_cluster) + rng.normal(
            0.0, 0.35, len(treatment_ids)
        )
        seed = int(rng.integers(0, 2**31))
        clustered = bootstrap_two_sample(
            control,
            treatment,
            np.mean,
            control_cluster_ids=control_ids,
            treatment_cluster_ids=treatment_ids,
            method="percentile",
            n_resamples=299,
            random_state=seed,
        )
        iid = bootstrap_two_sample(
            control,
            treatment,
            np.mean,
            method="percentile",
            n_resamples=299,
            random_state=seed,
        )
        clustered_covered += true_effect in clustered.confidence_interval
        iid_covered += true_effect in iid.confidence_interval

    assert clustered_covered / n_simulations >= 0.82
    assert clustered_covered - iid_covered >= 8

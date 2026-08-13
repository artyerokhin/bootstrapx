"""Regression tests for the statistical construction of resampling methods."""

import numpy as np
import pytest

from bootstrapx import bootstrap


def test_bayesian_mean_is_direct_dirichlet_functional() -> None:
    data = np.linspace(-2.0, 3.0, 25)
    n_resamples = 200
    seed = 314

    result = bootstrap(
        data,
        np.mean,
        method="bayesian",
        n_resamples=n_resamples,
        batch_size=n_resamples,
        random_state=seed,
    )

    weights = np.random.default_rng(seed).dirichlet(np.ones(len(data)), size=n_resamples)
    expected = weights @ data
    np.testing.assert_allclose(result.bootstrap_distribution, expected, rtol=1e-13, atol=1e-13)
    assert result.confidence_interval.method == "bayesian"
    assert result.extra["interval_type"] == "credible"


def test_bayesian_custom_statistic_requires_weighted_functional() -> None:
    data = np.arange(10.0)
    with pytest.raises(ValueError, match="weighted_statistic"):
        bootstrap(data, np.median, method="bayesian", n_resamples=20, random_state=0)


def test_bayesian_custom_weighted_functional() -> None:
    data = np.linspace(-1.0, 1.0, 20)

    def second_moment(sample: np.ndarray) -> float:
        return float(np.mean(sample**2))

    def weighted_second_moment(sample: np.ndarray, weights: np.ndarray) -> float:
        return float(np.sum(weights * sample**2))

    result = bootstrap(
        data,
        second_moment,
        method="bayesian",
        weighted_statistic=weighted_second_moment,
        n_resamples=100,
        random_state=4,
    )
    assert result.standard_error > 0


def test_poisson_resamples_are_never_replaced_with_original_small_sample() -> None:
    data = np.array([1.0, 4.0])
    result = bootstrap(
        data,
        np.sum,
        method="poisson",
        n_resamples=500,
        batch_size=500,
        random_state=8,
    )
    assert np.all(result.bootstrap_distribution > 0)


def test_subsampling_uses_full_sample_scale() -> None:
    data = np.random.default_rng(20).normal(size=100)
    result = bootstrap(
        data,
        np.mean,
        method="subsampling",
        subsample_size=10,
        rate=0.5,
        n_resamples=1000,
        random_state=9,
    )

    expected_root = np.sqrt(10) * (result.bootstrap_distribution - data.mean())
    np.testing.assert_allclose(result.extra["root_distribution"], expected_root)
    assert result.standard_error == pytest.approx(
        np.std(expected_root, ddof=1) / np.sqrt(len(data))
    )
    assert result.confidence_interval.method == "subsampling"


@pytest.mark.parametrize("prob", [0.25, 0.5, 0.75])
def test_bernoulli_root_has_finite_population_correction(prob: float) -> None:
    data = np.random.default_rng(21).normal(size=80)
    result = bootstrap(
        data,
        np.mean,
        method="bernoulli",
        prob=prob,
        n_resamples=1000,
        random_state=11,
    )

    sizes = result.extra["subset_sizes"]
    expected_root = np.sqrt(sizes / (1.0 - sizes / len(data))) * (
        result.bootstrap_distribution - data.mean()
    )
    np.testing.assert_allclose(result.extra["root_distribution"], expected_root)
    assert np.all((sizes > 0) & (sizes < len(data)))
    assert result.standard_error == pytest.approx(
        np.std(expected_root, ddof=1) / np.sqrt(len(data))
    )


def test_cluster_bootstrap_restores_cluster_level_uncertainty() -> None:
    rng = np.random.default_rng(222)
    n_simulations = 80
    cluster_covered = 0
    iid_covered = 0

    for _ in range(n_simulations):
        n_clusters = 25
        cluster_size = 6
        cluster_effect = rng.normal(size=n_clusters)
        data = np.repeat(cluster_effect, cluster_size) + rng.normal(size=n_clusters * cluster_size)
        cluster_ids = np.repeat(np.arange(n_clusters), cluster_size)
        seed = int(rng.integers(0, 2**31))
        clustered = bootstrap(
            data,
            np.mean,
            method="cluster",
            cluster_ids=cluster_ids,
            n_resamples=299,
            random_state=seed,
        )
        iid = bootstrap(
            data,
            np.mean,
            method="percentile",
            n_resamples=299,
            random_state=seed,
        )
        cluster_covered += 0.0 in clustered.confidence_interval
        iid_covered += 0.0 in iid.confidence_interval

    assert cluster_covered / n_simulations >= 0.85
    assert cluster_covered - iid_covered >= 12

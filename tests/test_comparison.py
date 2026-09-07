"""Tests for two-sample experiment comparisons."""

from __future__ import annotations

import builtins

import numpy as np
import pytest
from scipy.stats import bootstrap as scipy_bootstrap

from bootstrapx import TwoSampleBootstrapResult, bootstrap_two_sample


@pytest.fixture
def experiment_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026)
    return rng.normal(10.0, 2.0, 80), rng.normal(10.7, 2.5, 120)


@pytest.mark.parametrize(
    ("effect", "expected"),
    [
        ("difference", 3.0),
        ("ratio", 1.5),
        ("relative_lift", 0.5),
    ],
)
def test_builtin_effect_estimates(effect: str, expected: float) -> None:
    result = bootstrap_two_sample(
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 3.0, 4.0]),
        np.sum,
        effect=effect,
        method="percentile",
        n_resamples=30,
        random_state=1,
    )
    assert result.estimate == pytest.approx(expected)


def test_custom_effect_and_result_metadata(experiment_data) -> None:
    control, treatment = experiment_data

    def squared_gap(control_stat: float, treatment_stat: float) -> float:
        return (treatment_stat - control_stat) ** 2

    result = bootstrap_two_sample(
        control,
        treatment,
        np.mean,
        effect=squared_gap,
        method="basic",
        n_resamples=100,
        random_state=2,
    )

    assert isinstance(result, TwoSampleBootstrapResult)
    assert result.effect == "squared_gap"
    assert result.method == "basic"
    assert result.resampling == "iid"
    assert result.n_control == 80
    assert result.n_treatment == 120
    assert result.n_control_clusters is None
    assert result.n_treatment_clusters is None
    assert result.theta_hat == result.estimate
    assert "TwoSampleBootstrapResult" in repr(result)


def test_independent_distribution_matches_seeded_reference() -> None:
    control = np.array([1.0, 2.0, 4.0])
    treatment = np.array([5.0, 7.0, 9.0, 11.0])
    seed = 14
    n_resamples = 25
    result = bootstrap_two_sample(
        control,
        treatment,
        np.mean,
        method="percentile",
        n_resamples=n_resamples,
        batch_size=7,
        random_state=seed,
    )

    root = np.random.default_rng(seed)
    seeds = root.integers(0, np.iinfo(np.uint64).max, size=2, dtype=np.uint64)
    control_rng, treatment_rng = (np.random.default_rng(value) for value in seeds)
    control_indices = control_rng.integers(0, len(control), size=(n_resamples, len(control)))
    treatment_indices = treatment_rng.integers(
        0, len(treatment), size=(n_resamples, len(treatment))
    )
    expected = treatment[treatment_indices].mean(axis=1) - control[control_indices].mean(axis=1)
    np.testing.assert_allclose(result.bootstrap_distribution, expected)


def test_paired_distribution_uses_shared_indices() -> None:
    control = np.array([1.0, 2.0, 5.0, 8.0])
    treatment = np.array([2.0, 5.0, 7.0, 13.0])
    seed = 99
    n_resamples = 30
    result = bootstrap_two_sample(
        control,
        treatment,
        np.mean,
        paired=True,
        method="percentile",
        n_resamples=n_resamples,
        random_state=seed,
    )

    root = np.random.default_rng(seed)
    child_seed = root.integers(0, np.iinfo(np.uint64).max, size=1, dtype=np.uint64)[0]
    pair_rng = np.random.default_rng(child_seed)
    indices = pair_rng.integers(0, len(control), size=(n_resamples, len(control)))
    expected = (treatment - control)[indices].mean(axis=1)
    np.testing.assert_allclose(result.bootstrap_distribution, expected)
    assert result.paired is True


@pytest.mark.parametrize("paired", [False, True])
def test_distribution_is_independent_of_batch_size(experiment_data, paired: bool) -> None:
    control, treatment = experiment_data
    if paired:
        treatment = treatment[: len(control)]
    first = bootstrap_two_sample(
        control,
        treatment,
        np.mean,
        paired=paired,
        method="percentile",
        batch_size=1,
        n_resamples=101,
        random_state=73,
    )
    second = bootstrap_two_sample(
        control,
        treatment,
        np.mean,
        paired=paired,
        method="percentile",
        batch_size=37,
        n_resamples=101,
        random_state=73,
    )
    np.testing.assert_array_equal(first.bootstrap_distribution, second.bootstrap_distribution)


@pytest.mark.parametrize("method", ["percentile", "basic", "bca"])
def test_interval_methods_are_finite(experiment_data, method: str) -> None:
    control, treatment = experiment_data
    result = bootstrap_two_sample(
        control,
        treatment,
        np.median,
        method=method,
        n_resamples=499,
        random_state=8,
    )
    assert result.confidence_interval.method == method
    assert np.isfinite(result.confidence_interval.low)
    assert np.isfinite(result.confidence_interval.high)
    assert result.confidence_interval.low <= result.confidence_interval.high


@pytest.mark.parametrize("method", ["percentile", "basic", "bca"])
def test_agrees_with_scipy_for_difference_of_means(experiment_data, method: str) -> None:
    control, treatment = experiment_data
    result = bootstrap_two_sample(
        control,
        treatment,
        np.mean,
        method=method,
        n_resamples=1999,
        random_state=123,
    )
    reference = scipy_bootstrap(
        (control, treatment),
        lambda control_sample, treatment_sample: (
            np.mean(treatment_sample) - np.mean(control_sample)
        ),
        vectorized=False,
        paired=False,
        method=method,
        n_resamples=1999,
        random_state=np.random.default_rng(123),
    )

    assert result.estimate == pytest.approx(treatment.mean() - control.mean())
    assert result.confidence_interval.low == pytest.approx(
        reference.confidence_interval.low, abs=0.12
    )
    assert result.confidence_interval.high == pytest.approx(
        reference.confidence_interval.high, abs=0.12
    )


def test_cluster_distribution_resamples_complete_units() -> None:
    control = np.repeat([1.0, 4.0, 9.0], [2, 3, 1])
    treatment = np.repeat([2.0, 6.0, 12.0, 15.0], [1, 2, 2, 1])
    control_ids = np.repeat(["c1", "c2", "c3"], [2, 3, 1])
    treatment_ids = np.repeat(["t1", "t2", "t3", "t4"], [1, 2, 2, 1])
    result = bootstrap_two_sample(
        control,
        treatment,
        np.mean,
        control_cluster_ids=control_ids,
        treatment_cluster_ids=treatment_ids,
        method="percentile",
        n_resamples=80,
        batch_size=11,
        random_state=5,
    )

    assert result.resampling == "cluster"
    assert result.n_control_clusters == 3
    assert result.n_treatment_clusters == 4
    assert result.standard_error > 0


def test_cluster_distribution_is_independent_of_batch_size() -> None:
    rng = np.random.default_rng(4)
    control_ids = np.repeat(np.arange(8), [2, 3, 4, 2, 5, 3, 4, 2])
    treatment_ids = np.repeat(np.arange(10), [3, 2, 4, 5, 2, 3, 4, 2, 3, 5])
    control = rng.normal(size=len(control_ids))
    treatment = rng.normal(0.5, size=len(treatment_ids))
    kwargs = {
        "control_cluster_ids": control_ids,
        "treatment_cluster_ids": treatment_ids,
        "method": "percentile",
        "n_resamples": 100,
        "random_state": 51,
    }
    first = bootstrap_two_sample(control, treatment, np.mean, batch_size=1, **kwargs)
    second = bootstrap_two_sample(control, treatment, np.mean, batch_size=29, **kwargs)
    np.testing.assert_array_equal(first.bootstrap_distribution, second.bootstrap_distribution)


def test_cluster_bca_uses_cluster_deletions() -> None:
    rng = np.random.default_rng(6)
    control_ids = np.repeat(np.arange(12), 3)
    treatment_ids = np.repeat(np.arange(15), 4)
    control = rng.normal(size=len(control_ids))
    treatment = rng.normal(0.4, size=len(treatment_ids))
    result = bootstrap_two_sample(
        control,
        treatment,
        np.mean,
        control_cluster_ids=control_ids,
        treatment_cluster_ids=treatment_ids,
        method="bca",
        n_resamples=499,
        random_state=3,
    )
    assert result.confidence_interval.method == "bca"
    assert result.confidence_interval.low <= result.confidence_interval.high


def test_result_exports_are_compact_and_independent(experiment_data) -> None:
    control, treatment = experiment_data
    result = bootstrap_two_sample(
        control,
        treatment,
        np.mean,
        method="percentile",
        n_resamples=50,
        random_state=10,
    )
    compact = result.to_dict()
    complete = result.to_dict(include_distribution=True)

    assert "bootstrap_distribution" not in compact
    assert compact["estimate"] == result.estimate
    assert compact["resampling"] == "iid"
    distribution = complete["bootstrap_distribution"]
    np.testing.assert_array_equal(distribution, result.bootstrap_distribution)
    assert not np.shares_memory(distribution, result.bootstrap_distribution)


def test_result_to_frame_is_compact(experiment_data) -> None:
    pytest.importorskip("pandas")
    control, treatment = experiment_data
    result = bootstrap_two_sample(
        control,
        treatment,
        np.mean,
        method="percentile",
        n_resamples=50,
        random_state=10,
    )
    compact = result.to_dict()

    assert result.to_frame().shape == (1, len(compact))


def test_result_to_frame_explains_missing_pandas(experiment_data, monkeypatch) -> None:
    control, treatment = experiment_data
    result = bootstrap_two_sample(
        control,
        treatment,
        np.mean,
        method="percentile",
        n_resamples=20,
        random_state=1,
    )
    original_import = builtins.__import__

    def import_without_pandas(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("pandas is intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_pandas)
    with pytest.raises(ImportError, match=r"bootstrapx-lib\[pandas\]"):
        result.to_frame()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "studentized"}, "Unknown method"),
        ({"paired": "yes"}, "paired must be a boolean"),
        ({"paired": True}, "same number"),
        ({"control_cluster_ids": [1, 2, 3]}, "provided together"),
        (
            {
                "paired": True,
                "control_cluster_ids": [1, 2, 3],
                "treatment_cluster_ids": [4, 5, 6, 7],
            },
            "cannot be combined",
        ),
    ],
)
def test_rejects_invalid_designs(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        bootstrap_two_sample(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.mean,
            n_resamples=20,
            **kwargs,
        )


@pytest.mark.parametrize("effect", ["ratio", "relative_lift"])
def test_rejects_zero_control_for_relative_effects(effect: str) -> None:
    with pytest.raises(ValueError, match="control estimate is zero"):
        bootstrap_two_sample(
            np.array([-1.0, 0.0, 1.0]),
            np.array([1.0, 2.0, 3.0]),
            np.mean,
            effect=effect,
            method="percentile",
            n_resamples=20,
            random_state=0,
        )


def test_rejects_zero_control_created_by_a_resample() -> None:
    with pytest.raises(ValueError, match="control estimate is zero"):
        bootstrap_two_sample(
            np.array([0.0, 0.0, 1.0]),
            np.ones(3),
            np.mean,
            effect="ratio",
            method="percentile",
            n_resamples=20,
            random_state=0,
        )


@pytest.mark.parametrize("effect", ["ratio", "relative_lift"])
@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_relative_effects_are_invariant_to_common_scale(effect: str, scale: float) -> None:
    control = scale * np.array([1.0, 2.0, 4.0, 8.0])
    treatment = scale * np.array([2.0, 5.0, 7.0, 11.0])
    baseline = bootstrap_two_sample(
        control / scale,
        treatment / scale,
        np.mean,
        effect=effect,
        method="percentile",
        n_resamples=100,
        random_state=13,
    )
    scaled = bootstrap_two_sample(
        control,
        treatment,
        np.mean,
        effect=effect,
        method="percentile",
        n_resamples=100,
        random_state=13,
    )

    assert scaled.estimate == pytest.approx(baseline.estimate)
    np.testing.assert_allclose(
        scaled.bootstrap_distribution,
        baseline.bootstrap_distribution,
        rtol=1e-14,
        atol=0.0,
    )


def test_ratio_guard_does_not_depend_on_treatment_scale() -> None:
    result = bootstrap_two_sample(
        np.ones(10),
        np.full(10, 1e20),
        np.mean,
        effect="ratio",
        method="percentile",
        n_resamples=20,
        random_state=2,
    )

    assert result.estimate == pytest.approx(1e20)


def test_rejects_invalid_custom_outputs() -> None:
    with pytest.raises(ValueError, match="statistic must return exactly one scalar"):
        bootstrap_two_sample(
            [1, 2, 3],
            [4, 5, 6],
            lambda sample: np.array([sample.mean(), sample.std()]),
            n_resamples=20,
        )
    with pytest.raises(ValueError, match="effect must return exactly one scalar"):
        bootstrap_two_sample(
            [1, 2, 3],
            [4, 5, 6],
            np.mean,
            effect=lambda control, treatment: [control, treatment],
            n_resamples=20,
        )


@pytest.mark.parametrize(
    "ids",
    [
        [1, 2],
        [1, 1, 1],
        [1, 2, None],
        [1, 2, np.nan],
    ],
)
def test_rejects_invalid_cluster_ids(ids) -> None:
    with pytest.raises(ValueError, match="cluster"):
        bootstrap_two_sample(
            [1, 2, 3],
            [4, 5, 6],
            np.mean,
            control_cluster_ids=ids,
            treatment_cluster_ids=[4, 5, 6],
            method="percentile",
            n_resamples=20,
        )


def test_bca_requires_three_resampling_units() -> None:
    with pytest.raises(ValueError, match="at least three observations"):
        bootstrap_two_sample(
            [1, 2],
            [3, 4, 5],
            np.mean,
            method="bca",
            n_resamples=20,
        )
    with pytest.raises(ValueError, match="at least three clusters"):
        bootstrap_two_sample(
            [1, 2, 3, 4],
            [5, 6, 7, 8, 9, 10],
            np.mean,
            control_cluster_ids=[1, 1, 2, 2],
            treatment_cluster_ids=[3, 3, 4, 4, 5, 5],
            method="bca",
            n_resamples=20,
        )

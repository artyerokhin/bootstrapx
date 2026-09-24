"""Joint-feature resampling and scalar ratio-metric regression checks."""

from __future__ import annotations

import builtins

import numpy as np
import pytest
from scipy.stats import bootstrap as scipy_bootstrap

from bootstrapx import RatioOfSums, bootstrap_two_sample
from bootstrapx.comparison import _loo_effects
from bootstrapx.stats.confidence import bca_interval_from_jackknife


@pytest.fixture
def matrix_samples() -> tuple[np.ndarray, np.ndarray]:
    control = np.array([[5, 1], [12, 3], [18, 2], [8, 1], [24, 4], [14, 2]], dtype=float)
    treatment = np.array([[9, 1], [21, 3], [12, 2], [30, 4], [18, 2], [8, 1]], dtype=float)
    return control, treatment


def test_ratio_of_sums_is_not_mean_of_ratios(matrix_samples) -> None:
    control, treatment = matrix_samples
    metric = RatioOfSums()
    assert metric(control) == pytest.approx(81 / 13)
    assert metric(control) != pytest.approx(np.mean(control[:, 0] / control[:, 1]))
    result = bootstrap_two_sample(
        control, treatment, metric, allow_2d=True, n_resamples=50, random_state=6
    )
    assert result.control_estimate == pytest.approx(metric(control))
    assert result.treatment_estimate == pytest.approx(metric(treatment))
    assert result.estimate == pytest.approx(metric(treatment) - metric(control))


@pytest.mark.parametrize("paired", [False, True])
def test_joint_distribution_matches_independent_row_reference(matrix_samples, paired) -> None:
    control, treatment = matrix_samples
    seed, count = 19, 101
    metric = RatioOfSums()
    result = bootstrap_two_sample(
        control,
        treatment,
        metric,
        allow_2d=True,
        paired=paired,
        method="percentile",
        n_resamples=count,
        batch_size=7,
        random_state=seed,
    )
    root = np.random.default_rng(seed)
    seeds = root.integers(0, np.iinfo(np.uint64).max, size=1 if paired else 2, dtype=np.uint64)
    control_indices = np.random.default_rng(seeds[0]).integers(
        0, len(control), size=(count, len(control))
    )
    treatment_indices = (
        control_indices
        if paired
        else np.random.default_rng(seeds[1]).integers(
            0, len(treatment), size=(count, len(treatment))
        )
    )
    expected = np.array(
        [
            metric(treatment[treatment_indices[i]]) - metric(control[control_indices[i]])
            for i in range(count)
        ]
    )
    np.testing.assert_array_equal(result.bootstrap_distribution, expected)


def test_clustered_matrix_distribution_matches_complete_cluster_reference(matrix_samples) -> None:
    control, treatment = matrix_samples
    control_ids = np.array([0, 0, 1, 2, 2, 2])
    treatment_ids = np.array([0, 1, 1, 2, 2, 2])
    metric = RatioOfSums()
    seed, count = 4, 83
    result = bootstrap_two_sample(
        control,
        treatment,
        metric,
        allow_2d=True,
        control_cluster_ids=control_ids,
        treatment_cluster_ids=treatment_ids,
        method="bca",
        n_resamples=count,
        batch_size=9,
        random_state=seed,
    )
    seeds = np.random.default_rng(seed).integers(
        0, np.iinfo(np.uint64).max, size=2, dtype=np.uint64
    )
    choices = [
        np.random.default_rng(child).choice(np.unique(ids), size=(count, 3), replace=True)
        for child, ids in zip(seeds, (control_ids, treatment_ids), strict=True)
    ]
    expected = []
    for control_choice, treatment_choice in zip(*choices, strict=True):
        c = np.concatenate([control[control_ids == cluster] for cluster in control_choice])
        t = np.concatenate([treatment[treatment_ids == cluster] for cluster in treatment_choice])
        expected.append(metric(t) - metric(c))
    np.testing.assert_array_equal(result.bootstrap_distribution, expected)
    assert result.n_control_clusters == result.n_treatment_clusters == 3
    assert result.n_control == result.n_treatment == 6


@pytest.mark.parametrize("design", ["independent", "paired", "cluster"])
def test_matrix_bca_jackknife_deletes_correct_units(matrix_samples, design) -> None:
    control, treatment = matrix_samples
    metric = RatioOfSums()

    def effect(c, t):
        return t - c

    ids = np.array([0, 0, 1, 2, 2, 2]) if design == "cluster" else None
    actual = _loo_effects(
        control,
        treatment,
        metric,
        effect,
        paired=design == "paired",
        control_cluster_ids=ids,
        treatment_cluster_ids=ids,
    )
    if design == "paired":
        expected = [
            np.array(
                [
                    metric(np.delete(treatment, i, axis=0)) - metric(np.delete(control, i, axis=0))
                    for i in range(len(control))
                ]
            )
        ]
    else:
        c_deleted = (
            [control[ids != cluster] for cluster in np.unique(ids)]
            if ids is not None
            else [np.delete(control, i, axis=0) for i in range(len(control))]
        )
        t_deleted = (
            [treatment[ids != cluster] for cluster in np.unique(ids)]
            if ids is not None
            else [np.delete(treatment, i, axis=0) for i in range(len(treatment))]
        )
        expected = [
            np.array([metric(treatment) - metric(c) for c in c_deleted]),
            np.array([metric(t) - metric(control) for t in t_deleted]),
        ]
    for actual_group, expected_group in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_group, expected_group)
    result = bootstrap_two_sample(
        control,
        treatment,
        metric,
        allow_2d=True,
        paired=design == "paired",
        control_cluster_ids=ids,
        treatment_cluster_ids=ids,
        n_resamples=199,
        random_state=12,
    )
    reference = bca_interval_from_jackknife(
        result.bootstrap_distribution, result.estimate, expected, 0.95
    )
    assert result.confidence_interval == reference


@pytest.mark.parametrize("design", ["independent", "paired", "cluster"])
@pytest.mark.parametrize("method", ["percentile", "basic", "bca"])
def test_scalar_projection_preserves_legacy_seeded_result(matrix_samples, design, method) -> None:
    control, treatment = matrix_samples
    kwargs = {
        "method": method,
        "paired": design == "paired",
        "n_resamples": 101,
        "random_state": 42,
    }
    if design == "cluster":
        kwargs.update(
            control_cluster_ids=[0, 0, 1, 2, 2, 2],
            treatment_cluster_ids=[0, 1, 1, 2, 2, 2],
        )
    scalar = bootstrap_two_sample(control[:, 0], treatment[:, 0], np.mean, **kwargs)
    matrix = bootstrap_two_sample(
        control, treatment, lambda sample: np.mean(sample[:, 0]), allow_2d=True, **kwargs
    )
    np.testing.assert_array_equal(scalar.bootstrap_distribution, matrix.bootstrap_distribution)
    assert scalar.confidence_interval == matrix.confidence_interval
    assert scalar.estimate == matrix.estimate


@pytest.mark.parametrize("design", ["independent", "paired", "cluster"])
def test_matrix_results_are_batch_invariant(matrix_samples, design) -> None:
    control, treatment = matrix_samples
    kwargs = {
        "allow_2d": True,
        "paired": design == "paired",
        "n_resamples": 101,
        "random_state": 13,
    }
    if design == "cluster":
        kwargs.update(
            control_cluster_ids=[0, 0, 1, 2, 2, 2],
            treatment_cluster_ids=[0, 1, 1, 2, 2, 2],
        )
    first = bootstrap_two_sample(control, treatment, RatioOfSums(), batch_size=1, **kwargs)
    second = bootstrap_two_sample(control, treatment, RatioOfSums(), batch_size=37, **kwargs)
    np.testing.assert_array_equal(first.bootstrap_distribution, second.bootstrap_distribution)
    assert first.confidence_interval == second.confidence_interval


@pytest.mark.parametrize("paired", [False, True])
def test_proportional_features_cannot_be_resampled_independently(paired) -> None:
    orders = np.arange(1, 11, dtype=float)
    control = np.column_stack((2 * orders, orders))
    treatment = np.column_stack((3 * orders, orders))
    result = bootstrap_two_sample(
        control,
        treatment,
        RatioOfSums(),
        allow_2d=True,
        paired=paired,
        method="percentile",
        n_resamples=99,
        random_state=0,
    )
    np.testing.assert_array_equal(result.bootstrap_distribution, np.ones(99))
    assert result.standard_error == 0


@pytest.mark.parametrize("method", ["percentile", "basic", "bca"])
def test_matrix_ratio_interval_agrees_with_scipy_index_bootstrap(method) -> None:
    rng = np.random.default_rng(7)
    control_orders = rng.integers(1, 8, size=250)
    treatment_orders = rng.integers(1, 8, size=300)
    control = np.column_stack((control_orders * rng.lognormal(2, 0.3, 250), control_orders))
    treatment = np.column_stack((treatment_orders * rng.lognormal(2.1, 0.3, 300), treatment_orders))
    metric = RatioOfSums()
    result = bootstrap_two_sample(
        control,
        treatment,
        metric,
        allow_2d=True,
        method=method,
        n_resamples=2999,
        random_state=8,
    )

    def reference_statistic(c_indices, t_indices):
        c, t = control[c_indices.astype(int)], treatment[t_indices.astype(int)]
        return t[:, 0].sum() / t[:, 1].sum() - c[:, 0].sum() / c[:, 1].sum()

    reference = scipy_bootstrap(
        (np.arange(len(control)), np.arange(len(treatment))),
        reference_statistic,
        vectorized=False,
        paired=False,
        method=method,
        n_resamples=2999,
        random_state=np.random.default_rng(8),
    )
    assert result.confidence_interval.low == pytest.approx(
        reference.confidence_interval.low, abs=0.1
    )
    assert result.confidence_interval.high == pytest.approx(
        reference.confidence_interval.high, abs=0.1
    )


def test_matrix_input_requires_explicit_opt_in(matrix_samples) -> None:
    with pytest.raises(ValueError, match="Expected 1-D"):
        bootstrap_two_sample(*matrix_samples, RatioOfSums(), n_resamples=10)


@pytest.mark.parametrize("value", [1, "yes", None])
def test_rejects_non_boolean_matrix_opt_in(value) -> None:
    with pytest.raises(TypeError, match="allow_2d must be a boolean"):
        bootstrap_two_sample([1, 2, 3], [4, 5, 6], np.mean, allow_2d=value)


@pytest.mark.parametrize(
    ("control", "treatment", "message"),
    [
        (np.ones((4, 0)), np.ones((4, 0)), "at least one feature"),
        (np.ones((4, 2)), np.ones((5, 3)), "same number of feature"),
        (np.ones(4), np.ones((4, 1)), "same number of dimensions"),
        (np.ones((4, 2, 2)), np.ones((4, 2, 2)), "Expected 1-D"),
        (np.array([[1, 2], [3, np.nan]]), np.ones((4, 2)), "finite"),
        (np.ones((1, 2)), np.ones((4, 2)), "at least 2 observations"),
    ],
)
def test_rejects_invalid_matrix_inputs_before_statistic(control, treatment, message) -> None:
    def must_not_run(sample):
        raise AssertionError("Validation must precede statistic evaluation")

    with pytest.raises(ValueError, match=message):
        bootstrap_two_sample(control, treatment, must_not_run, allow_2d=True)


def test_matrix_statistic_still_must_return_scalar(matrix_samples) -> None:
    with pytest.raises(ValueError, match="exactly one scalar"):
        bootstrap_two_sample(*matrix_samples, lambda sample: sample.mean(axis=0), allow_2d=True)


def test_dataframes_require_identical_column_labels_and_order(matrix_samples) -> None:
    pd = pytest.importorskip("pandas")
    control, treatment = (
        pd.DataFrame(sample, columns=["revenue", "orders"]) for sample in matrix_samples
    )
    for invalid in (
        treatment[["orders", "revenue"]],
        treatment.rename(columns={"orders": "sessions"}),
    ):
        with pytest.raises(ValueError, match="identical column labels"):
            bootstrap_two_sample(control, invalid, RatioOfSums(), allow_2d=True)
    matrix = bootstrap_two_sample(
        *matrix_samples, RatioOfSums(), allow_2d=True, n_resamples=101, random_state=2
    )
    frame = bootstrap_two_sample(
        control, treatment, RatioOfSums(), allow_2d=True, n_resamples=101, random_state=2
    )
    np.testing.assert_array_equal(matrix.bootstrap_distribution, frame.bootstrap_distribution)


@pytest.mark.parametrize("duplicate_arm", ["control", "treatment", "both"])
def test_duplicate_dataframe_columns_rejected_before_statistic(duplicate_arm) -> None:
    pd = pytest.importorskip("pandas")
    data = np.array([[10, 100, 1], [20, 200, 2], [30, 300, 3]], dtype=float)
    ambiguous = pd.DataFrame(data, columns=["revenue", "revenue", "orders"])
    # A seemingly explicit column selection still contains duplicate features.
    selected = ambiguous[["revenue", "orders"]]
    control = selected if duplicate_arm in {"control", "both"} else data
    treatment = selected if duplicate_arm in {"treatment", "both"} else data

    def must_not_run(sample):
        pytest.fail("Ambiguous column labels must be rejected before evaluation.")

    with pytest.raises(ValueError, match="unique column labels"):
        bootstrap_two_sample(control, treatment, must_not_run, allow_2d=True)


def test_single_column_dataframe_keeps_legacy_scalar_contract() -> None:
    pd = pytest.importorskip("pandas")
    control = pd.DataFrame({"outcome": [1, 2, 4, 8]})
    treatment = pd.DataFrame({"outcome": [2, 3, 6, 9]})
    result = bootstrap_two_sample(
        control,
        treatment,
        lambda sample: sample.mean() if sample.ndim == 1 else np.nan,
        n_resamples=50,
        random_state=2,
    )
    assert result.estimate == 1.25


def test_numpy_matrix_needs_no_pandas(matrix_samples, monkeypatch) -> None:
    original_import = builtins.__import__

    def without_pandas(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("pandas intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pandas)
    result = bootstrap_two_sample(
        *matrix_samples, RatioOfSums(), allow_2d=True, n_resamples=50, random_state=2
    )
    assert np.isfinite(result.estimate)


@pytest.mark.parametrize("field", ["numerator", "denominator"])
@pytest.mark.parametrize("position", [True, 0.5, "0", -1])
def test_ratio_of_sums_validates_column_positions(field, position) -> None:
    with pytest.raises((TypeError, ValueError), match="column position"):
        RatioOfSums(**{field: position})


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        (np.ones(3), "two-dimensional"),
        (np.ones((0, 2)), "at least one observation"),
        (np.ones((3, 1)), "feature count"),
        (np.array([[1, 0], [3, 0]]), "denominator sum is zero"),
        (np.array([[1, np.inf], [3, 1]]), "finite values"),
        (np.array([[1e308, 1], [1e308, 1]]), "finite column sums"),
        (np.array([[1e308, 1e-308]]), "finite value"),
    ],
)
def test_ratio_of_sums_rejects_undefined_values(sample, message) -> None:
    with pytest.raises(ValueError, match=message):
        RatioOfSums()(sample)


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_ratio_of_sums_is_invariant_to_common_units(matrix_samples, scale) -> None:
    control, _ = matrix_samples
    assert RatioOfSums()(control * scale) == pytest.approx(RatioOfSums()(control))
    assert RatioOfSums(numerator=np.int64(1), denominator=0)(control) == pytest.approx(13 / 81)


def test_zero_denominator_replicates_are_not_dropped_or_redrawn() -> None:
    control = np.array([[0, 0], [0, 0], [10, 1]], dtype=float)
    treatment = np.array([[10, 1], [12, 1], [14, 1]], dtype=float)
    with pytest.raises(ValueError, match="denominator sum is zero"):
        bootstrap_two_sample(
            control,
            treatment,
            RatioOfSums(),
            allow_2d=True,
            method="percentile",
            n_resamples=100,
            random_state=0,
        )


def test_zero_denominator_jackknife_is_not_dropped() -> None:
    control = np.array([[0, 0], [0, 0], [10, 1]], dtype=float)
    treatment = np.array([[10, 1], [12, 1], [14, 1]], dtype=float)
    with pytest.raises(ValueError, match="denominator sum is zero"):
        _loo_effects(
            control,
            treatment,
            RatioOfSums(),
            lambda c, t: t - c,
            paired=False,
            control_cluster_ids=None,
            treatment_cluster_ids=None,
        )


def test_custom_statistic_zero_division_has_actionable_error(matrix_samples) -> None:
    def invalid_statistic(sample):
        return float(sample[:, 0].sum()) / 0.0

    with pytest.raises(ValueError, match="statistic is undefined"):
        bootstrap_two_sample(*matrix_samples, invalid_statistic, allow_2d=True)


def test_draft_example_retains_assigned_non_buyers() -> None:
    control = np.array(
        [[0, 0], [30, 2], [12, 1], [40, 3], [18, 1], [25, 2], [32, 2], [15, 1]],
        dtype=float,
    )
    treatment = np.array(
        [[0, 0], [36, 2], [15, 1], [45, 3], [20, 1], [30, 2], [36, 2], [18, 1]],
        dtype=float,
    )
    result = bootstrap_two_sample(
        control,
        treatment,
        RatioOfSums(),
        allow_2d=True,
        effect="difference",
        method="percentile",
        n_resamples=499,
        random_state=42,
    )
    assert result.n_control == result.n_treatment == 8
    assert result.control_estimate == pytest.approx(172 / 12)
    assert result.treatment_estimate == pytest.approx(200 / 12)

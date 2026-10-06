"""Paired cluster identity, alignment, reporting and interval reuse."""

import numpy as np
import pytest

from bootstrapx import bootstrap, bootstrap_two_sample


def test_shared_clusters_match_independent_reference_exactly():
    ids = np.array([2, 0, 2, 1, 0, 2])  # unequal, noncontiguous groups
    a = np.array([2.0, 0.0, 4.0, 10.0, 1.0, 5.0])
    b = a + np.array([1.0, 4.0, 2.0, -3.0, 5.0, 3.0])
    root = np.random.default_rng(12)
    seed = root.integers(0, np.iinfo(np.uint64).max, size=1, dtype=np.uint64)[0]
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(ids == group) for group in np.unique(ids)]
    expected = []
    for choices in rng.integers(0, 3, size=(199, 3)):
        rows = np.concatenate([groups[index] for index in choices])
        expected.append(b[rows].mean() - a[rows].mean())
    for batch_size in (1, 17, 199):
        result = bootstrap_two_sample(
            a,
            b,
            np.mean,
            paired=True,
            paired_cluster_ids=ids,
            method="percentile",
            random_state=12,
            n_resamples=199,
            batch_size=batch_size,
        )
        np.testing.assert_allclose(result.bootstrap_distribution, expected, rtol=0, atol=1e-14)
        assert result.n_control_clusters == result.n_treatment_clusters == 3
        assert result.resampling == result.metadata["resampling_unit"] == "paired_cluster"
        assert result.extra["design"]["paired"]["max_cluster_size"] == 3


def test_matrix_rows_move_together_and_constant_gain_is_degenerate():
    a = np.column_stack((np.arange(12, dtype=float), np.arange(12)))
    b = a.copy()
    b[:, 0] += 2
    calls = []

    def metric(rows):
        calls.append(rows[:, 1].copy())
        return float(rows[:, 0].mean())

    result = bootstrap_two_sample(
        a,
        b,
        metric,
        allow_2d=True,
        paired=True,
        paired_cluster_ids=np.repeat(np.arange(4), 3),
        control_observation_ids=np.arange(12),
        treatment_observation_ids=np.arange(12),
        method="basic",
        n_resamples=99,
        random_state=0,
    )
    for index in range(0, len(calls), 2):
        np.testing.assert_array_equal(calls[index], calls[index + 1])
    np.testing.assert_allclose(result.bootstrap_distribution, 2)
    assert result.extra["distribution_diagnostics"]["is_degenerate"]
    assert result.metadata["observation_ids_validated"]
    assert "control_observation_ids" not in result.metadata
    assert "treatment_observation_ids" not in result.metadata


@pytest.mark.parametrize(
    "options, message",
    [
        ({"paired": False, "paired_cluster_ids": [0, 0, 1, 1]}, "requires paired=True"),
        ({"paired": True, "paired_cluster_ids": [0, 0, 1, 1], "method": "bca"}, "not BCa"),
        ({"paired": True, "paired_cluster_ids": [0, 0, 0, 0]}, "two distinct clusters"),
        ({"paired": True, "paired_cluster_ids": [0, 1]}, "sample length"),
        (
            {
                "paired": True,
                "paired_cluster_ids": [0, 0, 1, 1],
                "control_cluster_ids": [0, 0, 1, 1],
                "treatment_cluster_ids": [0, 0, 1, 1],
            },
            "cannot be combined",
        ),
    ],
)
def test_rejects_ambiguous_cluster_designs(options, message):
    settings = {"method": "percentile", "n_resamples": 20} | options
    with pytest.raises(ValueError, match=message):
        bootstrap_two_sample(np.arange(4), np.arange(4), np.mean, **settings)


@pytest.mark.parametrize(
    "ids, message",
    [
        ([1, 0, 2, 3], "same row order"),
        ([0, 0, 2, 3], "unique"),
        ([0, 1], "sample length"),
        ([0, 1, 2, None], "missing"),
    ],
)
def test_observation_alignment_is_explicit(ids, message):
    with pytest.raises(ValueError, match=message):
        bootstrap_two_sample(
            np.arange(4),
            np.arange(4),
            np.mean,
            paired=True,
            paired_cluster_ids=[0, 0, 1, 1],
            method="percentile",
            n_resamples=20,
            control_observation_ids=np.arange(4),
            treatment_observation_ids=ids,
        )


def test_observation_ids_require_both_arrays_and_paired_design():
    for options, match in [
        ({"paired": True, "control_observation_ids": np.arange(4)}, "required together"),
        (
            {"control_observation_ids": np.arange(4), "treatment_observation_ids": np.arange(4)},
            "only supported for paired",
        ),
    ]:
        with pytest.raises(ValueError, match=match):
            bootstrap_two_sample(np.arange(4), np.arange(4), np.mean, n_resamples=20, **options)


def test_existing_cluster_paths_report_counts_without_storing_ids():
    ids = np.repeat(["alice", "bob", "carol"], [2, 3, 4])
    single = bootstrap(
        np.arange(9), np.mean, method="cluster", cluster_ids=ids, n_resamples=99, random_state=2
    )
    pair = bootstrap_two_sample(
        np.arange(9),
        np.arange(9),
        np.mean,
        control_cluster_ids=ids,
        treatment_cluster_ids=ids,
        method="percentile",
        n_resamples=99,
        random_state=2,
    )
    assert single.extra["design"]["n_clusters"] == 3
    assert pair.extra["design"]["control"]["max_cluster_size"] == 4
    assert "alice" not in str(single.to_dict()) + str(pair.to_dict())
    exported = pair.to_dict()
    exported["extra"]["design"]["control"]["n_clusters"] = 100
    assert pair.extra["design"]["control"]["n_clusters"] == 3


@pytest.mark.parametrize("two_sample", [False, True])
def test_interval_reuse_never_calls_metric_or_mutates_original(two_sample):
    calls = 0

    def metric(rows):
        nonlocal calls
        calls += 1
        return float(np.mean(rows))

    if two_sample:
        result = bootstrap_two_sample(
            np.arange(10), np.arange(10) + 1, metric, method="basic", n_resamples=99, random_state=3
        )
    else:
        result = bootstrap(np.arange(10), metric, method="basic", n_resamples=99, random_state=3)
    before = calls
    original = result.confidence_interval
    quantiles = np.quantile(result.bootstrap_distribution, [0.05, 0.95])
    percentile = result.interval(confidence_level=0.9)
    basic = result.interval(confidence_level=0.9, method="basic")
    assert [percentile.low, percentile.high] == pytest.approx(quantiles)
    assert [basic.low, basic.high] == pytest.approx(2 * result.theta_hat - quantiles[::-1])
    assert calls == before
    assert result.confidence_interval is original
    with pytest.raises(ValueError, match="jackknife"):
        result.interval(method="bca")
    for level in (0, 1, np.nan, True):
        with pytest.raises(ValueError, match="confidence_level"):
            result.interval(confidence_level=level)


def test_specialized_interval_cannot_be_reinterpreted_as_confidence():
    result = bootstrap(np.arange(10), np.mean, method="bayesian", n_resamples=99, random_state=0)
    with pytest.raises(ValueError, match="specialized"):
        result.interval()


def test_cluster_strata_checks_ids_against_actual_data_length():
    with pytest.raises(ValueError, match="data length"):
        bootstrap(
            np.arange(5),
            np.mean,
            method="cluster_strata",
            n_resamples=20,
            cluster_ids=[0, 1, 2, 3],
            strata_ids=[0, 0, 1, 1],
        )


def test_large_numeric_cluster_and_observation_ids_keep_their_identity():
    # Inferred float64 can silently merge neighboring uint64/int identifiers.
    groups = [2**63, np.uint64(2**63 + 1), 2**63, np.uint64(2**63 + 1)]
    observations = [2**63 + i for i in range(4)]
    result = bootstrap_two_sample(
        np.arange(4),
        [0, 2, 4, 6],
        np.mean,
        paired=True,
        paired_cluster_ids=groups,
        control_observation_ids=observations,
        treatment_observation_ids=observations,
        method="percentile",
        n_resamples=99,
        random_state=4,
    )
    reference = bootstrap_two_sample(
        np.arange(4),
        [0, 2, 4, 6],
        np.mean,
        paired=True,
        paired_cluster_ids=[0, 1, 0, 1],
        method="percentile",
        n_resamples=99,
        random_state=4,
    )
    assert result.n_control_clusters == 2
    np.testing.assert_array_equal(result.bootstrap_distribution, reference.bootstrap_distribution)


def test_dataframe_pairing_does_not_silently_align_indices():
    pd = pytest.importorskip("pandas")
    a = pd.DataFrame({"score": [0, 1, 2, 3]}, index=["a", "b", "c", "d"])
    b = a.iloc[::-1]
    with pytest.raises(ValueError, match="same row order"):
        bootstrap_two_sample(
            a,
            b,
            lambda rows: float(rows.mean()),
            allow_2d=True,
            paired=True,
            paired_cluster_ids=[0, 0, 1, 1],
            control_observation_ids=a.index,
            treatment_observation_ids=b.index,
            method="percentile",
            n_resamples=20,
        )


def test_nonfinite_paired_metric_is_not_silently_dropped():
    calls = 0

    def metric(rows):
        nonlocal calls
        calls += 1
        return float(rows.mean()) if calls <= 2 else np.nan

    with pytest.raises(ValueError, match="finite scalar"):
        bootstrap_two_sample(
            np.arange(4),
            np.arange(4),
            metric,
            paired=True,
            paired_cluster_ids=[0, 0, 1, 1],
            method="percentile",
            n_resamples=20,
        )
    assert calls == 3


@pytest.mark.parametrize("method", ["studentized", "subsampling", "bernoulli"])
def test_reuse_rejects_other_specialized_intervals(method):
    result = bootstrap(np.arange(20), np.mean, method=method, n_resamples=20, random_state=1)
    with pytest.raises(ValueError, match="specialized"):
        result.interval()


def test_reuse_validates_saved_distribution_and_method():
    result = bootstrap(np.arange(10), np.mean, n_resamples=20, random_state=1)
    with pytest.raises(TypeError, match="method"):
        result.interval(method=None)
    result.bootstrap_distribution[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        result.interval()

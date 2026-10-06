from collections import Counter

import numpy as np
import pytest

from bootstrapx import bootstrap, inspect_cluster_design
from bootstrapx.generators.hierarchical import cluster_strata_resample


@pytest.fixture
def nested_design():
    # Four complete clusters in two fixed strata, with unequal row counts.
    data = np.array([0.0, 1.0, 10.0, 20.0, 21.0, 30.0, 31.0, 32.0])
    clusters = np.array(["a", "a", "b", "c", "c", "d", "d", "d"])
    strata = np.array(["north", "north", "north", "south", "south", "south", "south", "south"])
    return data, clusters, strata


@pytest.mark.parametrize("shuffle", [False, True])
def test_complete_clusters_are_drawn_within_each_stratum(nested_design, shuffle):
    data, clusters, strata = nested_design
    if shuffle:
        order = np.array([7, 0, 3, 5, 2, 6, 1, 4])
        data, clusters, strata = data[order], clusters[order], strata[order]
    generated = cluster_strata_resample(data, clusters, strata, 40, 7, np.random.default_rng(3))
    samples = [sample for batch in generated for sample in batch]
    assert len(samples) == 40
    for sample in samples:
        counts = Counter(sample)
        # Every row of a selected cluster appears the same number of times.
        assert counts[0.0] == counts[1.0]
        assert counts[20.0] == counts[21.0]
        assert counts[30.0] == counts[31.0] == counts[32.0]
        # Every stratum contributes exactly two sampled clusters.
        assert counts[0.0] + counts[10.0] == 2
        assert counts[20.0] + counts[30.0] == 2


def test_seeded_results_do_not_depend_on_batch_size(nested_design):
    data, clusters, strata = nested_design
    shared = dict(
        data=data,
        statistic=np.mean,
        method="cluster_strata",
        cluster_ids=clusters,
        strata_ids=strata,
        n_resamples=101,
        random_state=9,
    )
    first = bootstrap(batch_size=1, **shared)
    second = bootstrap(batch_size=19, **shared)
    np.testing.assert_array_equal(first.bootstrap_distribution, second.bootstrap_distribution)
    assert first.extra["design"]["clusters_per_stratum"] == (2, 2)
    assert first.extra["design"]["max_cluster_size"] == 3
    assert first.confidence_interval.method == "percentile"


def test_matrix_statistic_and_basic_interval(nested_design):
    data, clusters, strata = nested_design
    matrix = np.column_stack((data, data**2))
    result = bootstrap(
        matrix,
        lambda rows: float(np.mean(rows[:, 1])),
        method="cluster_strata",
        ci_method="basic",
        cluster_ids=clusters,
        strata_ids=strata,
        n_resamples=99,
        random_state=4,
    )
    assert result.theta_hat == np.mean(data**2)
    assert result.confidence_interval.method == "basic"
    assert result.n_resamples == 99


@pytest.mark.parametrize("which", ["cluster_ids", "strata_ids"])
def test_both_identifiers_are_required(nested_design, which):
    data, clusters, strata = nested_design
    kwargs = {"cluster_ids": clusters, "strata_ids": strata}
    del kwargs[which]
    with pytest.raises(ValueError, match=which):
        bootstrap(data, np.mean, method="cluster_strata", n_resamples=20, **kwargs)


def test_cluster_cannot_cross_strata(nested_design):
    data, clusters, strata = nested_design
    bad = strata.copy()
    bad[1] = "south"
    with pytest.raises(ValueError, match="exactly one stratum"):
        bootstrap(
            data,
            np.mean,
            method="cluster_strata",
            cluster_ids=clusters,
            strata_ids=bad,
            n_resamples=20,
        )


def test_identifier_lengths_are_validated(nested_design):
    data, clusters, strata = nested_design
    with pytest.raises(ValueError, match="strata_ids must match data length"):
        bootstrap(
            data,
            np.mean,
            method="cluster_strata",
            cluster_ids=clusters,
            strata_ids=strata[:-1],
            n_resamples=20,
        )


def test_cluster_strata_rejects_bca(nested_design):
    data, clusters, strata = nested_design
    with pytest.raises(ValueError, match="ci_method"):
        bootstrap(
            data,
            np.mean,
            method="cluster_strata",
            cluster_ids=clusters,
            strata_ids=strata,
            ci_method="bca",
            n_resamples=20,
        )


def test_one_cluster_in_stratum_is_rejected_before_statistic(nested_design):
    data, clusters, strata = nested_design
    changed = strata.copy()
    changed[3:] = "north"
    changed[5:] = "south"
    calls = 0

    def statistic(values):
        nonlocal calls
        calls += 1
        return float(np.mean(values))

    with pytest.raises(ValueError, match="at least two clusters in every stratum"):
        bootstrap(
            data,
            statistic,
            method="cluster_strata",
            cluster_ids=clusters,
            strata_ids=changed,
            n_resamples=20,
        )
    assert calls == 0


def test_report_exact_one_class_probability():
    clusters = np.repeat(["a", "b", "c", "d"], 2)
    strata = np.repeat(["first", "second"], 4)
    labels = np.repeat([0, 1, 0, 1], 2)
    report = inspect_cluster_design(
        clusters, strata_ids=strata, binary_labels=labels, n_resamples=10
    )
    assert report.n_rows == 8
    assert report.n_clusters == 4
    assert report.clusters_per_stratum == (2, 2)
    assert report.binary_cluster_counts_by_stratum == ((1, 1), (1, 1))
    assert report.one_class_resample_probability == pytest.approx(0.125)
    assert report.probability_any_one_class_resample == pytest.approx(1 - 0.875**10)
    assert report.cluster_size_cv == 0


def test_report_can_show_zero_one_class_risk_for_fixed_class_strata():
    clusters = np.repeat(np.arange(4), 2)
    strata = np.repeat([0, 0, 1, 1], 2)
    labels = np.repeat([0, 0, 1, 1], 2)
    report = inspect_cluster_design(clusters, strata_ids=strata, binary_labels=labels)
    assert report.one_class_resample_probability == 0
    assert report.probability_any_one_class_resample == 0


def test_invalid_class_draws_are_not_silently_dropped():
    clusters = np.repeat(np.arange(4), 2)
    strata = np.repeat([0, 0, 1, 1], 2)
    labels = np.repeat([0, 1, 0, 1], 2)
    data = np.column_stack((labels, np.arange(8, dtype=float)))

    def two_class_statistic(rows):
        if len(np.unique(rows[:, 0])) < 2:
            return np.nan
        return float(np.mean(rows[:, 1]))

    with pytest.raises(ValueError, match="NaN or inf"):
        bootstrap(
            data,
            two_class_statistic,
            method="cluster_strata",
            cluster_ids=clusters,
            strata_ids=strata,
            n_resamples=99,
            random_state=3,
        )


@pytest.mark.parametrize(
    ("labels", "message"),
    [([0, 1, 1, 1], "constant within each cluster"), ([0, 1, 2, 2], "0/1")],
)
def test_report_rejects_invalid_binary_labels(labels, message):
    with pytest.raises(ValueError, match=message):
        inspect_cluster_design(["a", "a", "b", "b"], binary_labels=labels)


def test_report_rejects_missing_identifiers():
    with pytest.raises(ValueError, match="missing"):
        inspect_cluster_design(["a", None, "b"])
    with pytest.raises(ValueError, match="missing"):
        inspect_cluster_design(["a", np.nan, "b"])


def test_report_does_not_conflate_mixed_identifier_types():
    with pytest.raises(ValueError, match="comparable"):
        inspect_cluster_design([1, "1", 2, "2"])


def test_large_mixed_integer_ids_keep_distinct_clusters():
    clusters = (
        [np.uint64(2**63 + 1)] * 2
        + [np.uint64(2**63 + 2)] * 2
        + [np.int64(0)] * 2
        + [np.int64(1)] * 2
    )
    data = np.repeat([0.0, 1.0, 2.0, 3.0], 2)
    strata = np.repeat([0, 0, 1, 1], 2)
    result = bootstrap(
        data,
        lambda values: float(np.mean(values[values < 2])),
        method="cluster_strata",
        cluster_ids=clusters,
        strata_ids=strata,
        n_resamples=100,
        random_state=0,
    )
    assert result.extra["design"]["n_clusters"] == 4
    assert result.bootstrap_distribution.min() == 0
    assert result.bootstrap_distribution.max() == 1


@pytest.mark.parametrize("method", ["cluster", "strata"])
def test_existing_hierarchical_methods_preserve_large_mixed_integer_ids(method):
    identifiers = (
        [np.uint64(2**63 + 1)] * 2
        + [np.uint64(2**63 + 2)] * 2
        + [np.int64(0)] * 2
        + [np.int64(1)] * 2
    )
    name = "cluster_ids" if method == "cluster" else "strata_ids"
    options = dict(method=method, n_resamples=99, random_state=3)
    result = bootstrap(
        np.repeat([0.0, 1.0, 2.0, 3.0], 2), np.mean, **options, **{name: identifiers}
    )
    reference = bootstrap(
        np.repeat([0.0, 1.0, 2.0, 3.0], 2),
        np.mean,
        **options,
        **{name: [2, 2, 3, 3, 0, 0, 1, 1]},
    )
    np.testing.assert_array_equal(result.bootstrap_distribution, reference.bootstrap_distribution)


def test_report_handles_original_one_class_and_rejects_invalid_repeat_count():
    report = inspect_cluster_design([0, 0, 1, 1], binary_labels=[1, 1, 1, 1])
    assert report.one_class_resample_probability == 1
    assert report.probability_any_one_class_resample == 1
    with pytest.raises(ValueError, match="positive integer"):
        inspect_cluster_design([0, 1], n_resamples=0)
    with pytest.raises(TypeError, match="positive integer"):
        inspect_cluster_design([0, 1], n_resamples=True)


def test_report_rejects_non_numeric_binary_labels():
    with pytest.raises(ValueError, match="numeric 0/1"):
        inspect_cluster_design([0, 1], binary_labels=["no", "yes"])

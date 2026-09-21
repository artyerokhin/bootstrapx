"""Optional analysis-unit validation and privacy-safe reporting metadata."""

import json

import numpy as np
import pytest

from bootstrapx import bootstrap_two_sample


def compare(**kwargs):
    return bootstrap_two_sample(
        [1, 2, 4, 8],
        [2, 3, 5, 9],
        np.mean,
        n_resamples=30,
        random_state=2,
        **kwargs,
    )


def test_independent_ids_validate_without_changing_results():
    original = compare()
    checked = compare(control_unit_ids=[1, 2, 3, 4], treatment_unit_ids=[5, 6, 7, 8])
    np.testing.assert_array_equal(original.bootstrap_distribution, checked.bootstrap_distribution)
    assert original.confidence_interval == checked.confidence_interval
    assert checked.metadata["unit_ids_validated"]
    assert not original.metadata["unit_ids_validated"]


def test_paired_ids_validate_without_reordering_or_changing_results():
    original = compare(paired=True)
    checked = compare(paired=True, control_unit_ids=list("abcd"), treatment_unit_ids=list("abcd"))
    np.testing.assert_array_equal(original.bootstrap_distribution, checked.bootstrap_distribution)
    assert checked.metadata["resampling_unit"] == "pair"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"control_unit_ids": [1, 2, 3, 4]}, "provided together"),
        ({"control_unit_ids": [1, 2], "treatment_unit_ids": [5, 6, 7, 8]}, "match its sample"),
        ({"control_unit_ids": [1, 1, 3, 4], "treatment_unit_ids": [5, 6, 7, 8]}, "must be unique"),
        ({"control_unit_ids": [1, 2, 3, 4], "treatment_unit_ids": [4, 5, 6, 7]}, "must not share"),
        (
            {"paired": True, "control_unit_ids": list("abcd"), "treatment_unit_ids": list("bacd")},
            "same row order",
        ),
        (
            {
                "control_cluster_ids": [0, 0, 1, 2],
                "treatment_cluster_ids": [0, 1, 2, 2],
                "control_unit_ids": [1, 2, 3, 4],
                "treatment_unit_ids": [5, 6, 7, 8],
            },
            "cannot be combined",
        ),
    ],
)
def test_invalid_unit_designs_fail_before_statistic(kwargs, message):
    def must_not_run(sample):
        raise AssertionError("Design validation must precede statistic evaluation")

    with pytest.raises(ValueError, match=message):
        bootstrap_two_sample([1, 2, 4, 8], [2, 3, 5, 9], must_not_run, **kwargs)


@pytest.mark.parametrize("identifier", [None, np.nan, np.inf, -np.inf, {"x": 1}, [1, 2]])
def test_invalid_identifiers_are_rejected(identifier):
    with pytest.raises(ValueError, match="identifiers"):
        compare(control_unit_ids=[identifier, 2, 3, 4], treatment_unit_ids=[5, 6, 7, 8])


def test_nullable_pandas_identifier_is_rejected():
    pd = pytest.importorskip("pandas")
    with pytest.raises(ValueError, match="identifiers"):
        compare(control_unit_ids=[pd.NA, 2, 3, 4], treatment_unit_ids=[5, 6, 7, 8])


def test_mixed_scalar_identifiers_are_not_coerced_to_strings():
    result = compare(control_unit_ids=[1, "1", 2, "2"], treatment_unit_ids=[3, "3", 4, "4"])
    assert result.metadata["unit_ids_validated"]


def test_arm_local_cluster_labels_may_overlap():
    result = compare(control_cluster_ids=[0, 0, 1, 2], treatment_cluster_ids=[0, 1, 2, 2])
    assert result.metadata["resampling_unit"] == "cluster"
    assert result.metadata["n_control_units"] == result.metadata["n_treatment_units"] == 3
    assert not result.metadata["unit_ids_validated"]


def test_metadata_is_compact_copy_safe_and_contains_no_identifiers():
    ids = [f"private-user-{i}" for i in range(8)]
    result = compare(
        control_unit_ids=ids[:4],
        treatment_unit_ids=ids[4:],
        metric_name="revenue/user",
        effect_unit="USD/user",
    )
    summary = result.to_dict()
    assert summary["metadata"]["metric_name"] == "revenue/user"
    assert summary["metadata"]["effect_unit"] == "USD/user"
    assert summary["metadata"]["confidence_level"] == 0.95
    assert summary["metadata"]["seed"] == 2
    assert summary["metadata"]["package_version"]
    serialized = json.dumps(summary)
    assert "private-user" not in serialized
    assert "bootstrap_distribution" not in summary
    summary["metadata"]["metric_name"] = "changed"
    assert result.metadata["metric_name"] == "revenue/user"


@pytest.mark.parametrize("label", ["metric_name", "effect_unit"])
@pytest.mark.parametrize("value", [1, "", "   "])
def test_invalid_reporting_labels_are_rejected(label, value):
    with pytest.raises((TypeError, ValueError), match=label):
        compare(**{label: value})

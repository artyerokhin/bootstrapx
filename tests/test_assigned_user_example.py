"""Execute the assigned-user workflow and reject unsafe table preparation."""

import runpy
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def example():
    pytest.importorskip("pandas")
    return runpy.run_path(
        str(
            Path(__file__).resolve().parents[1] / "examples" / "assigned_users_composite_metrics.py"
        )
    )


def test_example_preserves_all_assigned_users_and_reports_correct_estimands(example):
    analysis = example["analyze_example"](n_resamples=99)
    users, orders = analysis["users"], analysis["orders"]
    assert len(users) == len(analysis["assignments"]) == 520
    assert users["user_id"].is_unique
    assert (users["orders"] == 0).any()
    assert (users.loc[users["orders"] == 0, "revenue"] == 0).all()
    assert users["revenue"].sum() == pytest.approx(orders["revenue"].sum())
    assert users["orders"].sum() == len(orders)
    for name, result in analysis["results"].items():
        c = users.loc[users["variant"] == "control"]
        t = users.loc[users["variant"] == "treatment"]
        statistic = (
            (lambda arm: arm["revenue"].sum() / arm["orders"].sum())
            if name == "revenue/order"
            else (lambda arm: arm["revenue"].mean())
            if name == "revenue/user"
            else (lambda arm: arm["orders"].mean())
        )
        assert result.control_estimate == pytest.approx(statistic(c))
        assert result.treatment_estimate == pytest.approx(statistic(t))
        assert result.estimate == pytest.approx(statistic(t) - statistic(c))
        assert result.metadata["unit_ids_validated"]
        assert result.metadata["n_features"] == 2
        assert (result.n_control, result.n_treatment) == (240, 280)
        assert np.isfinite(result.bootstrap_distribution).all()
        assert result.method == ("basic" if name == "revenue/order" else "bca")
    for _, record in analysis["buyer_descriptions"].iterrows():
        arm = users.loc[(users["variant"] == record["variant"]) & (users["orders"] > 0)]
        assert record["mean_buyer_revenue_per_order"] == pytest.approx(
            (arm["revenue"] / arm["orders"]).mean()
        )


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate_assignment",
        "duplicate_order",
        "unknown_user",
        "missing_revenue",
        "missing_variant",
        "infinite_revenue",
        "string_revenue",
        "conflicting_assignment_column",
    ],
)
def test_example_rejects_ambiguous_or_incomplete_data(example, problem):
    assignments, orders = example["build_tables"]()
    if problem == "duplicate_assignment":
        assignments.loc[1, "user_id"] = assignments.loc[0, "user_id"]
    elif problem == "duplicate_order":
        orders.loc[1, "order_id"] = orders.loc[0, "order_id"]
    elif problem == "unknown_user":
        orders.loc[0, "user_id"] = 9999
    elif problem == "missing_revenue":
        orders.loc[0, "revenue"] = np.nan
    elif problem == "missing_variant":
        assignments.loc[0, "variant"] = None
    elif problem == "infinite_revenue":
        orders.loc[0, "revenue"] = np.inf
    elif problem == "string_revenue":
        orders["revenue"] = orders["revenue"].astype(str)
    else:
        assignments["revenue"] = 0
    with pytest.raises(ValueError):
        example["prepare_user_metrics"](assignments, orders)

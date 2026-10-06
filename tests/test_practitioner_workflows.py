"""Keep practitioner workflows executable and honest about their units."""

import runpy
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_assigned_user_orders_workflow(monkeypatch):
    pytest.importorskip("pandas")
    monkeypatch.syspath_prepend(str(EXAMPLES))
    example = runpy.run_path(str(EXAMPLES / "practitioner_user_orders.py"))
    analysis = example["run"](n_resamples=99)
    users = analysis["users"]
    orders = analysis["orders"]
    result = analysis["bootstrapx"]

    assert len(users) == len(analysis["assignments"]) == 520
    assert users["user_id"].is_unique
    assert (users.loc[users["orders"] == 0, "revenue"] == 0).all()
    assert users["revenue"].sum() == pytest.approx(orders["revenue"].sum())
    control = users.loc[users["variant"] == "control", "revenue"]
    treatment = users.loc[users["variant"] == "treatment", "revenue"]
    assert result.estimate == pytest.approx(treatment.mean() - control.mean())
    assert result.metadata["unit_ids_validated"]
    assert abs(analysis["order_weighted_difference"] - result.estimate) > 1
    assert np.isfinite(result.bootstrap_distribution).all()
    assert np.isfinite(analysis["scipy"].bootstrap_distribution).all()


def test_grouped_model_comparison_workflow(monkeypatch):
    pytest.importorskip("sklearn")
    monkeypatch.syspath_prepend(str(EXAMPLES))
    example = runpy.run_path(str(EXAMPLES / "practitioner_grouped_model_comparison.py"))
    analysis = example["run"](n_resamples=99)
    held_out = analysis["held_out"]
    test_patients = analysis["test_patients"]
    result = analysis["bootstrapx"]

    assert held_out.shape == (600, 3)
    assert len(np.unique(test_patients)) == 120
    assert not np.intersect1d(analysis["train_patients"], test_patients).size
    assert np.unique(test_patients, return_counts=True)[1].tolist() == [5] * 120
    assert result.theta_hat == pytest.approx(example["auc_gain"](held_out))
    assert result.resampling == "paired_cluster"
    assert result.metadata["observation_ids_validated"]
    assert result.extra["design"]["paired"]["n_clusters"] == 120
    brier = analysis["brier"]
    expected_brier = np.mean((held_out[:, 2] - held_out[:, 0]) ** 2) - np.mean(
        (held_out[:, 1] - held_out[:, 0]) ** 2
    )
    assert brier.estimate == pytest.approx(expected_brier)
    assert np.isfinite(brier.bootstrap_distribution).all()
    assert np.isfinite(result.bootstrap_distribution).all()
    assert np.isfinite(analysis["scipy"].bootstrap_distribution).all()


def test_known_truth_paired_brier_example():
    example = runpy.run_path(str(EXAMPLES / "paired_cluster_brier.py"))
    analysis = example["run"](n_resamples=99)
    result = analysis["bootstrapx"]
    assert analysis["true_difference"] == -0.1
    assert result.estimate == pytest.approx(result.treatment_estimate - result.control_estimate)
    assert result.n_control_clusters == result.n_treatment_clusters == 120
    assert result.metadata["observation_ids_validated"]
    assert result.estimate < 0
    assert np.isfinite(result.bootstrap_distribution).all()
    assert np.isfinite(analysis["scipy"].bootstrap_distribution).all()

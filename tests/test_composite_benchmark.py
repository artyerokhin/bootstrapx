"""Check known-truth generation, joint reference units and resume accounting."""

import csv
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "bench_composite_metrics.py"


@pytest.fixture(scope="module")
def benchmark():
    return runpy.run_path(str(SCRIPT))


def test_denominator_change_has_zero_population_ratio_effect(benchmark):
    c, t, _, _, truth = benchmark["make_data"]("denominator_change", np.random.default_rng(8))
    assert truth == 0.0
    assert t[:, 1].mean() > c[:, 1].mean()


def test_cluster_reference_preserves_repeated_selections_and_joint_columns(benchmark):
    c = np.array([[1, 1], [4, 2], [12, 3], [8, 2]], dtype=float)
    t = np.array([[3, 1], [10, 2], [9, 1], [20, 4]], dtype=float)
    c_ids, t_ids = np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2])
    _, statistic = benchmark["reference_problem"](c, t, c_ids, t_ids)
    c_rows = np.concatenate([c[c_ids == i] for i in [0, 0, 2]])
    t_rows = np.concatenate([t[t_ids == i] for i in [1, 2, 1]])
    expected = t_rows[:, 0].sum() / t_rows[:, 1].sum() - c_rows[:, 0].sum() / c_rows[:, 1].sum()
    assert statistic(np.array([0, 0, 2]), np.array([1, 2, 1])) == expected


@pytest.mark.parametrize("name", ["activity_price", "activity_price_covariance"])
def test_activity_price_truth_matches_large_population_sample(benchmark, name):
    c, t, c_ids, t_ids, truth = benchmark["make_data"](
        name, np.random.default_rng(721), 200_000, 250_000
    )
    assert c_ids is t_ids is None
    b_control, b_treatment = 0.35, 0.65 if name.endswith("covariance") else 0.35
    m_treatment = 2.0 if name.endswith("covariance") else 2.08
    expected_c = np.exp(2.0 + b_control * 0.45 + (0.45**2 + 0.25**2) / 2)
    expected_t = np.exp(m_treatment + b_treatment * 0.45 + (0.45**2 + 0.25**2) / 2)
    assert truth == pytest.approx(expected_t - expected_c)
    assert benchmark["METRIC"](c) == pytest.approx(expected_c, rel=0.01)
    assert benchmark["METRIC"](t) == pytest.approx(expected_t, rel=0.01)
    assert np.corrcoef(c[c[:, 1] > 0, 1], c[c[:, 1] > 0, 0] / c[c[:, 1] > 0, 1])[0, 1] > 0.1
    if name.endswith("covariance"):
        assert truth > 0  # Not E[price_t] - E[price_c], which would be zero.
        assert c[:, 1].mean() == pytest.approx(1.8, rel=0.01)
        assert t[:, 1].mean() == pytest.approx(1.8, rel=0.01)


def test_cluster_activity_price_truth_and_shared_latent_prices(benchmark):
    c, t, c_ids, t_ids, truth = benchmark["make_data"](
        "cluster_activity_price", np.random.default_rng(722), cluster_counts=(30_000, 40_000)
    )
    expected_c = np.exp(2.0 + 0.35 * 0.45 + (0.45**2 + 0.25**2) / 2)
    expected_t = np.exp(2.08 + 0.35 * 0.45 + (0.45**2 + 0.25**2) / 2)
    assert truth == pytest.approx(expected_t - expected_c)
    assert benchmark["METRIC"](c) == pytest.approx(expected_c, rel=0.025)
    assert benchmark["METRIC"](t) == pytest.approx(expected_t, rel=0.025)
    assert len(np.unique(c_ids)) == 30_000
    assert len(np.unique(t_ids)) == 40_000
    for label in range(20):
        rows = c[(c_ids == label) & (c[:, 1] > 0)]
        if len(rows):
            np.testing.assert_allclose(rows[:, 0] / rows[:, 1], rows[0, 0] / rows[0, 1])


@pytest.mark.parametrize("hits", [0, 15, 30])
def test_wilson_bounds_are_probabilities(benchmark, hits):
    low, high = benchmark["wilson"](hits, 30)
    assert 0 <= low <= hits / 30 <= high <= 1


def test_large_cluster_generator_uses_more_independent_units(benchmark):
    data = benchmark["make_data"]("cluster_large", np.random.default_rng(1))
    assert len(np.unique(data[2])) == 100
    assert len(np.unique(data[3])) == 120
    with pytest.raises(ValueError, match="not clustered"):
        benchmark["interval"](
            "scipy_vectorized", "percentile", data, 99, np.random.default_rng(2), False
        )


@pytest.mark.parametrize("paired", [False, True])
def test_delta_reduces_to_mean_difference_with_constant_denominator(benchmark, paired):
    c_values, t_values = np.array([1, 2, 4, 8]), np.array([2, 4, 5, 7])
    c, t = np.column_stack((c_values, np.ones(4))), np.column_stack((t_values, np.ones(4)))
    low, high = benchmark["delta_interval"](c, t, None, None, paired)
    variance = (
        np.var(t_values - c_values, ddof=1) / 4
        if paired
        else (np.var(c_values, ddof=1) + np.var(t_values, ddof=1)) / 4
    )
    half = stats.norm.ppf(0.975) * np.sqrt(variance)
    estimate = t_values.mean() - c_values.mean()
    assert low == pytest.approx(estimate - half)
    assert high == pytest.approx(estimate + half)


@pytest.mark.parametrize("method", ["percentile", "basic", "bca"])
def test_scalar_and_vectorized_scipy_reference_agree(benchmark, method):
    data = benchmark["make_data"]("independent", np.random.default_rng(1), 30, 40)
    scalar = benchmark["interval"]("scipy", method, data, 199, np.random.default_rng(2), False)
    vector = benchmark["interval"](
        "scipy_vectorized", method, data, 199, np.random.default_rng(2), False
    )
    np.testing.assert_allclose(scalar, vector, atol=0.5)


def test_quick_runner_accounts_for_failures_and_resumes(tmp_path):
    repository = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True
    )
    if repository.returncode != 0 or Path(repository.stdout.strip()).resolve() != ROOT:
        pytest.skip("Benchmark provenance/resume integration requires a repository checkout.")
    command = [sys.executable, str(SCRIPT), "--profile", "quick", "--output-dir", str(tmp_path)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert not metadata["release_evidence"]
    assert metadata["completed_cells"] == metadata["total_cells"] == 70
    with (tmp_path / "results.csv").open() as input_file:
        rows = list(csv.DictReader(input_file))
    for row in rows:
        valid, failed, invalid, hits = (
            int(row[k])
            for k in ("valid_trials", "failed_trials", "invalid_trials", "covered_trials")
        )
        assert valid + failed + invalid == int(row["n_simulations"])
        assert float(row["coverage_all_trials"]) == hits / int(row["n_simulations"])
        if row["scenario"] != "sparse":
            assert failed == invalid == 0
    assert any(int(row["failed_trials"]) > 0 for row in rows if row["scenario"] == "sparse")
    with (tmp_path / "runtime.csv").open() as input_file:
        timing = list(csv.DictReader(input_file))
    assert len(timing) == 10
    assert {row["design"] for row in timing} == {"independent", "cluster"}
    assert any(row["library"] == "scipy_vectorized" for row in timing)
    saved = (tmp_path / "results.csv").read_bytes()
    result = subprocess.run(command + ["--resume"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "results.csv").read_bytes() == saved
    metadata["source_sha256"] = "changed-source"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    result = subprocess.run(command + ["--resume"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "cannot resume" in result.stderr

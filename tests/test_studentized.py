"""Tests for studentized (bootstrap-t) method.

Covers:
- basic sanity: runs without error, returns valid BootstrapResult
- independence: boot_se values must not be all identical (shared inner_idx bug)
- coverage rate: empirical 95% CI coverage >= 0.88 over 200 trials
- reproducibility: same random_state => identical results
"""

import numpy as np
import pytest

from bootstrapx import api as api_module
from bootstrapx import bootstrap

RNG_SEED = 42
N = 100
N_RESAMPLES = 999
N_INNER = 50
TRUE_MEAN = 5.0


@pytest.fixture
def normal_sample() -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED)
    return rng.normal(loc=TRUE_MEAN, scale=1.0, size=N)


class TestStudentized:
    def test_se_is_computed_from_matching_outer_sample(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = np.arange(1.0, 9.0)
        n_resamples = 6
        n_inner = 5
        seed = 17
        captured: dict[str, np.ndarray] = {}
        original = api_module.studentized_interval

        def capture_interval(*args, **kwargs):
            captured["boot_stats"] = args[3].copy()
            captured["boot_se"] = args[4].copy()
            return original(*args, **kwargs)

        monkeypatch.setattr(api_module, "studentized_interval", capture_interval)
        result = bootstrap(
            data,
            np.mean,
            method="studentized",
            n_resamples=n_resamples,
            n_inner=n_inner,
            batch_size=n_resamples,
            random_state=seed,
        )

        rng = np.random.default_rng(seed)
        expected_stats = np.empty(n_resamples)
        expected_se = np.empty(n_resamples)
        for b in range(n_resamples):
            indices = rng.integers(0, len(data), size=len(data))
            sample = data[indices]
            expected_stats[b] = np.mean(sample)
            inner_idx = rng.integers(0, len(data), size=(n_inner, len(data)))
            expected_se[b] = np.std(np.mean(sample[inner_idx], axis=1), ddof=1)

        np.testing.assert_array_equal(result.bootstrap_distribution, expected_stats)
        np.testing.assert_array_equal(captured["boot_stats"], expected_stats)
        np.testing.assert_allclose(captured["boot_se"], expected_se)

    def test_batch_size_does_not_change_result(self, normal_sample: np.ndarray) -> None:
        common = dict(
            method="studentized",
            n_resamples=40,
            n_inner=10,
            random_state=19,
        )
        small = bootstrap(normal_sample, np.mean, batch_size=1, **common)
        large = bootstrap(normal_sample, np.mean, batch_size=40, **common)
        np.testing.assert_array_equal(
            small.bootstrap_distribution,
            large.bootstrap_distribution,
        )
        assert small.confidence_interval == large.confidence_interval

    def test_basic_run(self, normal_sample: np.ndarray) -> None:
        r = bootstrap(
            normal_sample,
            np.mean,
            method="studentized",
            n_resamples=N_RESAMPLES,
            n_inner=N_INNER,
            random_state=RNG_SEED,
            backend="vanilla",
        )
        assert r.method == "studentized"
        assert r.n_resamples == N_RESAMPLES
        assert r.confidence_interval.low < r.theta_hat < r.confidence_interval.high
        assert r.standard_error > 0

    def test_inner_idx_independence(self, normal_sample: np.ndarray) -> None:
        """Bootstrap distribution must not be suspiciously flat.

        With shared inner_idx (bug), t-statistics within a batch collapse
        to nearly identical values => std of distribution drops sharply.
        """
        r = bootstrap(
            normal_sample,
            np.mean,
            method="studentized",
            n_resamples=N_RESAMPLES,
            n_inner=N_INNER,
            random_state=RNG_SEED,
            backend="vanilla",
        )
        dist = r.bootstrap_distribution
        assert np.std(dist) > 0.01, (
            f"Bootstrap distribution is suspiciously flat (std={np.std(dist):.4f}). "
            "Possible shared inner_idx bug."
        )
        ci_width = r.confidence_interval.high - r.confidence_interval.low
        assert ci_width > 0.05, f"CI width too narrow: {ci_width:.4f}"

    def test_ci_coverage_rate(self) -> None:
        """Empirical 95% CI coverage must be >= 0.88 over 200 trials.

        Correct implementation achieves ~0.93-0.95.
        Buggy shared inner_idx typically achieves ~0.82-0.87.
        Threshold 0.88 reliably distinguishes the two.
        """
        rng = np.random.default_rng(0)
        n_trials = 200
        covered = 0

        for _ in range(n_trials):
            sample = rng.normal(loc=TRUE_MEAN, scale=1.0, size=N)
            r = bootstrap(
                sample,
                np.mean,
                method="studentized",
                n_resamples=499,
                n_inner=25,
                random_state=int(rng.integers(0, 2**31)),
                backend="vanilla",
            )
            if r.confidence_interval.low <= TRUE_MEAN <= r.confidence_interval.high:
                covered += 1

        coverage = covered / n_trials
        assert coverage >= 0.88, (
            f"Empirical coverage {coverage:.3f} < 0.88. "
            "Likely correlated inner resamples (shared inner_idx bug)."
        )

    def test_reproducibility(self, normal_sample: np.ndarray) -> None:
        kwargs = dict(
            method="studentized",
            n_resamples=N_RESAMPLES,
            n_inner=N_INNER,
            random_state=7,
            backend="vanilla",
        )
        r1 = bootstrap(normal_sample, np.mean, **kwargs)
        r2 = bootstrap(normal_sample, np.mean, **kwargs)
        np.testing.assert_array_equal(r1.bootstrap_distribution, r2.bootstrap_distribution)
        assert r1.confidence_interval.low == r2.confidence_interval.low
        assert r1.confidence_interval.high == r2.confidence_interval.high

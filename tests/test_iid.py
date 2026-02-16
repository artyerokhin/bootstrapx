import numpy as np
import pytest
from bootstrapx import bootstrap

class TestBCa:
    def test_ci_covers(self, normal_data):
        r = bootstrap(normal_data, np.mean, method="bca", n_resamples=5000, random_state=42, backend="vanilla")
        assert r.confidence_interval.low < 5.5 and r.confidence_interval.high > 4.5
    def test_reproducibility(self, normal_data):
        r1 = bootstrap(normal_data, np.mean, method="bca", n_resamples=999, random_state=123, backend="vanilla")
        r2 = bootstrap(normal_data, np.mean, method="bca", n_resamples=999, random_state=123, backend="vanilla")
        np.testing.assert_array_equal(r1.bootstrap_distribution, r2.bootstrap_distribution)
    def test_width_shrinks(self, rng):
        sm = rng.normal(5,2,30); lg = rng.normal(5,2,500)
        ws = bootstrap(sm, np.mean, method="bca", n_resamples=2000, random_state=1, backend="vanilla")
        wl = bootstrap(lg, np.mean, method="bca", n_resamples=2000, random_state=1, backend="vanilla")
        assert (wl.confidence_interval.high - wl.confidence_interval.low) < (ws.confidence_interval.high - ws.confidence_interval.low)

class TestPercentile:
    def test_run(self, normal_data):
        r = bootstrap(normal_data, np.median, method="percentile", n_resamples=3000, random_state=0, backend="vanilla")
        assert r.confidence_interval.low < r.confidence_interval.high
    def test_confidence_levels(self, normal_data):
        r90 = bootstrap(normal_data, np.mean, method="percentile", n_resamples=3000, confidence_level=0.90, random_state=7, backend="vanilla")
        r99 = bootstrap(normal_data, np.mean, method="percentile", n_resamples=3000, confidence_level=0.99, random_state=7, backend="vanilla")
        assert (r99.confidence_interval.high - r99.confidence_interval.low) > (r90.confidence_interval.high - r90.confidence_interval.low)

class TestBasicInterval:
    def test_run(self, normal_data):
        r = bootstrap(normal_data, np.std, method="basic", n_resamples=2000, random_state=5, backend="vanilla")
        assert r.standard_error > 0

class TestStudentized:
    def test_run(self, normal_data):
        r = bootstrap(normal_data, np.mean, method="studentized", n_resamples=500, random_state=3, backend="vanilla", n_inner=20)
        assert r.confidence_interval.low < r.confidence_interval.high

class TestPoisson:
    def test_run(self, normal_data):
        r = bootstrap(normal_data, np.mean, method="poisson", n_resamples=2000, random_state=10, backend="vanilla")
        assert r.n_resamples == 2000

class TestBernoulli:
    def test_run(self, normal_data):
        assert bootstrap(normal_data, np.mean, method="bernoulli", n_resamples=2000, random_state=11, backend="vanilla", prob=0.5).n_resamples == 2000

class TestSubsampling:
    def test_run(self, normal_data):
        assert bootstrap(normal_data, np.max, method="subsampling", n_resamples=2000, random_state=12, backend="vanilla", subsample_size=50).method == "subsampling"
    def test_rejects_large(self, normal_data):
        with pytest.raises(ValueError, match="must be < n"):
            bootstrap(normal_data, np.max, method="subsampling", n_resamples=100, backend="vanilla", subsample_size=len(normal_data)+1)

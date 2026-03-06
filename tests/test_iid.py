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

class TestPercentile:
    def test_vectorized(self, normal_data):
        r = bootstrap(normal_data, np.mean, method="percentile", n_resamples=2000, random_state=0, backend="vanilla", vectorized=True)
        assert r.n_resamples == 2000

class TestBayesian:
    def test_run(self, normal_data):
        r = bootstrap(normal_data, np.mean, method="bayesian", n_resamples=2000, random_state=42, backend="vanilla")
        assert r.n_resamples == 2000 and r.standard_error > 0

class TestPoisson:
    def test_run(self, normal_data):
        r = bootstrap(normal_data, np.mean, method="poisson", n_resamples=2000, random_state=10, backend="vanilla")
        assert r.n_resamples == 2000

class TestBernoulli:
    def test_run(self, normal_data):
        r = bootstrap(normal_data, np.mean, method="bernoulli", n_resamples=2000, random_state=11, backend="vanilla", prob=0.5)
        assert r.n_resamples == 2000

class TestSubsampling:
    def test_run(self, normal_data):
        r = bootstrap(normal_data, np.max, method="subsampling", n_resamples=2000, random_state=12, backend="vanilla", subsample_size=50)
        assert r.method == "subsampling"

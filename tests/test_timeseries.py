import numpy as np
from bootstrapx import bootstrap

class TestMBB:
    def test_run(self, timeseries_data):
        r = bootstrap(timeseries_data, np.mean, method="mbb", n_resamples=1000, random_state=1, backend="vanilla", block_length=15)
        assert r.standard_error > 0

    def test_ci_method_basic(self, timeseries_data):
        r = bootstrap(timeseries_data, np.mean, method="mbb", n_resamples=1000, random_state=1, backend="vanilla", block_length=15, ci_method="basic")
        assert r.confidence_interval.method == "basic"

class TestCBB:
    def test_run(self, timeseries_data):
        r = bootstrap(timeseries_data, np.mean, method="cbb", n_resamples=1000, random_state=2, backend="vanilla", block_length=15)
        assert r.method == "cbb"

class TestStationary:
    def test_run(self, timeseries_data):
        r = bootstrap(timeseries_data, np.mean, method="stationary", n_resamples=1000, random_state=3, backend="vanilla", mean_block=12.0)
        assert r.method == "stationary"

class TestWild:
    def test_rademacher(self, normal_data):
        r = bootstrap(normal_data, np.mean, method="wild", n_resamples=1000, random_state=7, backend="vanilla")
        assert r.method == "wild"

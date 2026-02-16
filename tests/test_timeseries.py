import numpy as np
import pytest
from bootstrapx import bootstrap

class TestMBB:
    def test_run(self, timeseries_data):
        assert bootstrap(timeseries_data, np.mean, method="mbb", n_resamples=1000, random_state=1, backend="vanilla", block_length=15).standard_error > 0
    def test_rejects_long_block(self, timeseries_data):
        with pytest.raises(ValueError): bootstrap(timeseries_data, np.mean, method="mbb", n_resamples=100, backend="vanilla", block_length=len(timeseries_data)+1)

class TestCBB:
    def test_run(self, timeseries_data):
        assert bootstrap(timeseries_data, np.mean, method="cbb", n_resamples=1000, random_state=2, backend="vanilla", block_length=15).method == "cbb"

class TestStationary:
    def test_run(self, timeseries_data):
        assert bootstrap(timeseries_data, np.mean, method="stationary", n_resamples=1000, random_state=3, backend="vanilla", mean_block=12.0).method == "stationary"
    def test_blocks(self, timeseries_data):
        r5 = bootstrap(timeseries_data, np.mean, method="stationary", n_resamples=1000, random_state=0, backend="vanilla", mean_block=5.0)
        r50 = bootstrap(timeseries_data, np.mean, method="stationary", n_resamples=1000, random_state=0, backend="vanilla", mean_block=50.0)
        assert r5.standard_error > 0 and r50.standard_error > 0

class TestTapered:
    def test_run(self, timeseries_data):
        assert bootstrap(timeseries_data, np.mean, method="tapered", n_resamples=500, random_state=4, backend="vanilla", block_length=10).method == "tapered"

class TestSieve:
    def test_run(self, ar1_data):
        assert bootstrap(ar1_data, np.mean, method="sieve", n_resamples=1000, random_state=5, backend="vanilla", ar_order=2).n_resamples == 1000
    def test_auto_order(self, ar1_data):
        assert bootstrap(ar1_data, np.mean, method="sieve", n_resamples=500, random_state=6, backend="vanilla").method == "sieve"

class TestWild:
    def test_rademacher(self, normal_data):
        assert bootstrap(normal_data, np.mean, method="wild", n_resamples=1000, random_state=7, backend="vanilla", distribution="rademacher").method == "wild"
    def test_mammen(self, normal_data):
        assert bootstrap(normal_data, np.mean, method="wild", n_resamples=1000, random_state=8, backend="vanilla", distribution="mammen").method == "wild"
    def test_unknown(self, normal_data):
        with pytest.raises(ValueError, match="Unknown distribution"):
            bootstrap(normal_data, np.mean, method="wild", n_resamples=100, backend="vanilla", distribution="bad")

import numpy as np
import pytest

from bootstrapx import bootstrap


class TestMBB:
    def test_run(self, timeseries_data):
        r = bootstrap(
            timeseries_data,
            np.mean,
            method="mbb",
            n_resamples=1000,
            random_state=1,
            backend="vanilla",
            block_length=15,
        )
        assert r.standard_error > 0

    def test_ci_method_basic(self, timeseries_data):
        r = bootstrap(
            timeseries_data,
            np.mean,
            method="mbb",
            n_resamples=1000,
            random_state=1,
            backend="vanilla",
            block_length=15,
            ci_method="basic",
        )
        assert r.confidence_interval.method == "basic"


class TestCBB:
    def test_run(self, timeseries_data):
        r = bootstrap(
            timeseries_data,
            np.mean,
            method="cbb",
            n_resamples=1000,
            random_state=2,
            backend="vanilla",
            block_length=15,
        )
        assert r.method == "cbb"


class TestStationary:
    def test_run(self, timeseries_data):
        r = bootstrap(
            timeseries_data,
            np.mean,
            method="stationary",
            n_resamples=1000,
            random_state=3,
            backend="vanilla",
            mean_block=12.0,
        )
        assert r.method == "stationary"


class TestTapered:
    def test_constant_series_is_preserved(self):
        data = np.full(23, 5.0)
        for statistic in (np.mean, np.median):
            result = bootstrap(
                data,
                statistic,
                method="tapered",
                block_length=10,
                n_resamples=50,
                random_state=4,
            )
            np.testing.assert_allclose(result.bootstrap_distribution, 5.0)
            assert result.confidence_interval.low == pytest.approx(5.0)
            assert result.confidence_interval.high == pytest.approx(5.0)

    def test_location_invariance(self, timeseries_data):
        kwargs = dict(
            method="tapered",
            block_length=15,
            n_resamples=100,
            random_state=5,
        )
        original = bootstrap(timeseries_data, np.mean, **kwargs)
        shifted = bootstrap(timeseries_data + 100.0, np.mean, **kwargs)
        np.testing.assert_allclose(
            shifted.bootstrap_distribution - original.bootstrap_distribution,
            100.0,
        )

    def test_preserves_variance_scale(self, timeseries_data):
        result = bootstrap(
            timeseries_data,
            np.var,
            method="tapered",
            block_length=15,
            n_resamples=400,
            random_state=6,
        )
        ratio = result.bootstrap_distribution.mean() / np.var(timeseries_data)
        assert 0.75 <= ratio <= 1.3


class TestSieve:
    def test_constant_series_is_preserved(self):
        data = np.full(25, 7.0)
        result = bootstrap(data, np.mean, method="sieve", n_resamples=50, random_state=7)
        np.testing.assert_array_equal(result.bootstrap_distribution, np.full(50, 7.0))


class TestWild:
    def test_rademacher(self, normal_data):
        r = bootstrap(
            normal_data, np.mean, method="wild", n_resamples=1000, random_state=7, backend="vanilla"
        )
        assert r.method == "wild"

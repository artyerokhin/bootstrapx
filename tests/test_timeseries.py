import numpy as np
import pytest

from bootstrapx import bootstrap
from bootstrapx.generators import timeseries as ts_generators


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
    @pytest.mark.parametrize("taper", ["hann", "hamming", "bartlett", "boxcar"])
    def test_supported_scipy_windows(self, timeseries_data, taper):
        result = bootstrap(
            timeseries_data,
            np.mean,
            method="tapered",
            block_length=12,
            taper=taper,
            n_resamples=25,
            random_state=4,
        )
        assert result.bootstrap_distribution.shape == (25,)
        assert np.all(np.isfinite(result.bootstrap_distribution))

    def test_invalid_window_is_rejected(self, timeseries_data):
        with pytest.raises(ValueError, match="Unknown taper window 'not-a-window'"):
            bootstrap(
                timeseries_data,
                np.mean,
                method="tapered",
                block_length=12,
                taper="not-a-window",
                n_resamples=25,
                random_state=4,
            )

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

    def test_nonconstant_series(self, timeseries_data):
        result = bootstrap(
            timeseries_data,
            np.mean,
            method="sieve",
            ar_order=3,
            n_resamples=40,
            random_state=8,
        )
        assert result.bootstrap_distribution.std() > 0

    def test_singular_fit_has_actionable_error(self, monkeypatch):
        def singular(*args, **kwargs):
            raise np.linalg.LinAlgError("singular")

        monkeypatch.setattr(np.linalg, "solve", singular)
        with pytest.raises(ValueError, match="smaller ar_order"):
            bootstrap(
                np.arange(12.0),
                np.mean,
                method="sieve",
                ar_order=3,
                n_resamples=10,
                random_state=8,
            )


class TestWild:
    def test_rademacher(self, normal_data):
        r = bootstrap(
            normal_data, np.mean, method="wild", n_resamples=1000, random_state=7, backend="vanilla"
        )
        assert r.method == "wild"

    def test_mammen_with_fitted_values(self, normal_data):
        fitted = np.linspace(-0.5, 0.5, len(normal_data))
        result = bootstrap(
            normal_data,
            np.mean,
            method="wild",
            distribution="mammen",
            fitted=fitted,
            n_resamples=100,
            random_state=9,
        )
        assert np.all(np.isfinite(result.bootstrap_distribution))
        assert result.bootstrap_distribution.std() > 0


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("mbb", {"block_length": 7}),
        ("cbb", {"block_length": 7}),
        ("stationary", {"mean_block": 7.0}),
        ("tapered", {"block_length": 7, "taper": "hann"}),
        ("sieve", {"ar_order": 2}),
        ("wild", {"distribution": "mammen"}),
    ],
)
def test_time_series_methods_are_reproducible(timeseries_data, method, kwargs):
    first = bootstrap(
        timeseries_data,
        np.mean,
        method=method,
        n_resamples=30,
        batch_size=8,
        random_state=123,
        **kwargs,
    )
    second = bootstrap(
        timeseries_data,
        np.mean,
        method=method,
        n_resamples=30,
        batch_size=11,
        random_state=123,
        **kwargs,
    )
    np.testing.assert_array_equal(first.bootstrap_distribution, second.bootstrap_distribution)


def test_python_fallback_index_generators(monkeypatch):
    monkeypatch.setattr(ts_generators, "_mbb_idx", ts_generators._mbb_idx_python)
    monkeypatch.setattr(ts_generators, "_cbb_idx", ts_generators._cbb_idx_python)
    monkeypatch.setattr(ts_generators, "_stat_idx", ts_generators._stat_idx_python)

    data = np.arange(20.0)
    for method, kwargs in (
        ("mbb", {"block_length": 4}),
        ("cbb", {"block_length": 4}),
        ("stationary", {"mean_block": 4.0}),
    ):
        result = bootstrap(
            data,
            np.mean,
            method=method,
            n_resamples=12,
            random_state=10,
            **kwargs,
        )
        assert result.bootstrap_distribution.shape == (12,)


@pytest.mark.parametrize("method", ["mbb", "cbb", "tapered"])
def test_block_length_at_series_length_is_rejected(method):
    with pytest.raises(ValueError, match="block_length"):
        bootstrap(
            np.arange(5.0),
            np.mean,
            method=method,
            block_length=5,
            n_resamples=10,
        )

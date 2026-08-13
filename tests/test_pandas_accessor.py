"""Tests for pandas .bootstrap accessor."""

import numpy as np
import pytest

try:
    import pandas as pd

    import bootstrapx  # noqa: F401  — registers accessor

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


@pytest.mark.skipif(not _HAS_PANDAS, reason="pandas not installed")
class TestSeriesAccessor:
    @pytest.fixture
    def series(self):
        return pd.Series(np.random.default_rng(42).normal(5, 2, 200))

    def test_bca(self, series):
        r = series.bootstrap.bca(np.mean, n_resamples=999, random_state=0)
        assert r.confidence_interval.low < 5.5
        assert r.confidence_interval.high > 4.5

    def test_percentile(self, series):
        r = series.bootstrap.percentile(np.mean, n_resamples=999, random_state=0)
        assert r.n_resamples == 999

    def test_ci_method(self, series):
        r = series.bootstrap.ci(np.median, method="basic", n_resamples=999, random_state=1)
        assert r.method == "basic"

    def test_reproducible(self, series):
        r1 = series.bootstrap.bca(np.mean, n_resamples=500, random_state=99)
        r2 = series.bootstrap.bca(np.mean, n_resamples=500, random_state=99)
        np.testing.assert_array_equal(r1.bootstrap_distribution, r2.bootstrap_distribution)


@pytest.mark.skipif(not _HAS_PANDAS, reason="pandas not installed")
class TestDataFrameAccessor:
    @pytest.fixture
    def df(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "a": rng.normal(0, 1, 150),
                "b": rng.normal(3, 2, 150),
            }
        )

    def test_summary_shape(self, df):
        out = df.bootstrap.summary(np.mean, n_resamples=500, random_state=0)
        assert list(out.index) == ["a", "b"]
        assert set(out.columns) == {"theta_hat", "ci_low", "ci_high", "se", "method"}

    def test_summary_values(self, df):
        out = df.bootstrap.summary(np.mean, n_resamples=999, random_state=1)
        assert out.loc["a", "ci_low"] < 0.5
        assert out.loc["b", "ci_low"] > 1.0

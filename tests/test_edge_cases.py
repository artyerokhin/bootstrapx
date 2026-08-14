import builtins

import numpy as np
import pytest

from bootstrapx import bootstrap


class TestEdgeCases:
    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            bootstrap(np.array([1.0, 2.0]), np.mean, method="magic")

    def test_constant(self):
        r = bootstrap(
            np.ones(50),
            np.mean,
            method="percentile",
            n_resamples=500,
            random_state=0,
            backend="vanilla",
        )
        assert r.standard_error < 1e-12

    def test_repr(self):
        r = bootstrap(
            np.random.default_rng(0).normal(0, 1, 50),
            np.mean,
            method="percentile",
            n_resamples=500,
            random_state=0,
            backend="vanilla",
        )
        assert "BootstrapResult" in repr(r)

    def test_result_to_dict_is_compact_and_independent(self):
        r = bootstrap(
            np.arange(10.0),
            np.mean,
            method="percentile",
            n_resamples=25,
            random_state=0,
        )
        summary = r.to_dict()

        assert summary == {
            "theta_hat": r.theta_hat,
            "standard_error": r.standard_error,
            "ci_low": r.confidence_interval.low,
            "ci_high": r.confidence_interval.high,
            "ci_method": "percentile",
            "method": "percentile",
            "n_resamples": 25,
            "extra": {},
        }
        assert "bootstrap_distribution" not in summary

    def test_result_to_dict_can_include_distribution_copy(self):
        r = bootstrap(
            np.arange(10.0),
            np.mean,
            method="percentile",
            n_resamples=25,
            random_state=0,
        )
        summary = r.to_dict(include_distribution=True)
        distribution = summary["bootstrap_distribution"]

        np.testing.assert_array_equal(distribution, r.bootstrap_distribution)
        assert not np.shares_memory(distribution, r.bootstrap_distribution)

    def test_result_to_dict_copies_array_metadata(self):
        r = bootstrap(
            np.arange(20.0),
            np.mean,
            method="subsampling",
            subsample_size=5,
            n_resamples=25,
            random_state=0,
        )
        root_distribution = r.to_dict()["extra"]["root_distribution"]

        np.testing.assert_array_equal(root_distribution, r.extra["root_distribution"])
        assert not np.shares_memory(root_distribution, r.extra["root_distribution"])

    def test_result_to_frame_explains_missing_pandas(self, monkeypatch):
        result = bootstrap(
            np.arange(10.0),
            np.mean,
            method="percentile",
            n_resamples=25,
            random_state=0,
        )
        original_import = builtins.__import__

        def import_without_pandas(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("pandas is intentionally unavailable")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", import_without_pandas)

        with pytest.raises(ImportError, match=r"bootstrapx-lib\[pandas\]"):
            result.to_frame()

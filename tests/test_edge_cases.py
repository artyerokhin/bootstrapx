import numpy as np
import pytest
from bootstrapx import bootstrap

class TestEdgeCases:
    def test_min_data(self):
        assert bootstrap(np.array([1.0, 2.0]), np.mean, method="percentile", n_resamples=500, random_state=0, backend="vanilla").n_resamples == 500
    def test_constant(self):
        assert bootstrap(np.ones(50), np.mean, method="percentile", n_resamples=500, random_state=0, backend="vanilla").standard_error < 1e-12
    def test_large_B(self):
        d = np.random.default_rng(0).normal(0,1,50)
        assert bootstrap(d, np.mean, method="percentile", n_resamples=50000, random_state=0, backend="vanilla").n_resamples == 50000
    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"): bootstrap(np.array([1.0,2.0]), np.mean, method="magic")
    def test_custom_stat(self):
        def iqr(x): return float(np.percentile(x,75) - np.percentile(x,25))
        assert bootstrap(np.random.default_rng(42).normal(0,1,100), iqr, method="percentile", n_resamples=2000, random_state=0, backend="vanilla").standard_error > 0
    def test_repr(self):
        r = bootstrap(np.random.default_rng(0).normal(0,1,50), np.mean, method="percentile", n_resamples=500, random_state=0, backend="vanilla")
        assert "BootstrapResult" in repr(r)

import numpy as np
import pytest
from bootstrapx import bootstrap

class TestEdgeCases:
    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            bootstrap(np.array([1.0, 2.0]), np.mean, method="magic")
    def test_constant(self):
        r = bootstrap(np.ones(50), np.mean, method="percentile", n_resamples=500, random_state=0, backend="vanilla")
        assert r.standard_error < 1e-12
    def test_repr(self):
        r = bootstrap(np.random.default_rng(0).normal(0,1,50), np.mean, method="percentile", n_resamples=500, random_state=0, backend="vanilla")
        assert "BootstrapResult" in repr(r)

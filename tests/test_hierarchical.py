import numpy as np
import pytest
from bootstrapx import bootstrap

class TestCluster:
    def test_run(self, cluster_setup):
        data, ids = cluster_setup
        assert bootstrap(data, np.mean, method="cluster", n_resamples=1000, random_state=1, backend="vanilla", cluster_ids=ids).n_resamples == 1000
    def test_requires_ids(self, normal_data):
        with pytest.raises(ValueError, match="cluster_ids"):
            bootstrap(normal_data, np.mean, method="cluster", n_resamples=100, backend="vanilla")

class TestStrata:
    def test_run(self, strata_setup):
        data, ids = strata_setup
        assert bootstrap(data, np.mean, method="strata", n_resamples=1000, random_state=2, backend="vanilla", strata_ids=ids).n_resamples == 1000
    def test_requires_ids(self, normal_data):
        with pytest.raises(ValueError, match="strata_ids"):
            bootstrap(normal_data, np.mean, method="strata", n_resamples=100, backend="vanilla")

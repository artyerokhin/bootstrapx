import numpy as np
import pytest

@pytest.fixture
def rng():
    return np.random.default_rng(42)

@pytest.fixture
def normal_data(rng):
    return rng.normal(loc=5.0, scale=2.0, size=200)

@pytest.fixture
def timeseries_data(rng):
    return np.cumsum(rng.standard_normal(300))

@pytest.fixture
def ar1_data(rng):
    y = np.zeros(200); y[0] = rng.standard_normal()
    for t in range(1, 200): y[t] = 0.7 * y[t-1] + rng.standard_normal()
    return y

@pytest.fixture
def cluster_setup(rng):
    ids = np.repeat(np.arange(10), 20)
    return rng.normal(loc=ids * 0.5, scale=1.0), ids

@pytest.fixture
def strata_setup(rng):
    ids = np.array([0]*80 + [1]*70 + [2]*50)
    return np.concatenate([rng.normal(0,1,80), rng.normal(3,1,70), rng.normal(6,1,50)]), ids

import numpy as np
import pytest
from bootstrapx.stats.confidence import percentile_interval, basic_interval, bca_interval, studentized_interval

class TestPercentileInterval:
    def test_symmetric(self):
        b = np.random.default_rng(0).normal(0, 1, 10000)
        ci = percentile_interval(b, 0.95)
        assert -2.1 < ci.low < -1.8 and 1.8 < ci.high < 2.1
    def test_99_wider(self):
        b = np.random.default_rng(0).normal(0, 1, 10000)
        assert (percentile_interval(b, 0.99).high - percentile_interval(b, 0.99).low) > (percentile_interval(b, 0.95).high - percentile_interval(b, 0.95).low)

class TestBCa:
    def test_skewed(self):
        rng = np.random.default_rng(42); d = rng.exponential(2, 100)
        b = np.array([np.mean(rng.choice(d, 100)) for _ in range(3000)])
        ci = bca_interval(b, d, np.mean, np.mean(d))
        assert ci.low < ci.high

class TestStudentized:
    def test_shape(self):
        rng = np.random.default_rng(0)
        ci = studentized_interval(rng.normal(5,2,100), np.mean, 5.0, rng.normal(5,0.5,2000), np.abs(rng.normal(0.5,0.1,2000)))
        assert ci.low < ci.high

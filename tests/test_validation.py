import numpy as np
import pytest
from bootstrapx.utils import validate_data, auto_batch_size

class TestValidateData:
    def test_converts_list(self):
        assert isinstance(validate_data([1.0, 2.0, 3.0]), np.ndarray)
    def test_rejects_scalar(self):
        with pytest.raises(ValueError, match="Scalar"): validate_data(5.0)
    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="NaN"): validate_data([1.0, np.nan])
    def test_rejects_single_obs(self):
        with pytest.raises(ValueError, match="at least 2"): validate_data([1.0])
    def test_rejects_3d(self):
        with pytest.raises(ValueError): validate_data(np.zeros((2,3,4)))
    def test_allows_2d(self):
        assert validate_data(np.zeros((5,3)), allow_2d=True).shape == (5,3)

class TestAutoBatchSize:
    def test_small(self): assert 1 <= auto_batch_size(100, 10000) <= 10000
    def test_capped(self): assert auto_batch_size(10, 50) <= 50

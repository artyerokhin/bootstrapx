import numpy as np
import pytest
from bootstrapx.utils import validate_data

class TestValidateData:
    def test_list(self):
        assert isinstance(validate_data([1.0, 2.0, 3.0]), np.ndarray)
    def test_rejects_scalar(self):
        with pytest.raises(ValueError): validate_data(5.0)
    def test_rejects_nan(self):
        with pytest.raises(ValueError): validate_data([1.0, np.nan])
    def test_pandas_series(self):
        pd = pytest.importorskip("pandas")
        arr = validate_data(pd.Series([1.0, 2.0, 3.0]))
        assert arr.shape == (3,)
    def test_pandas_df(self):
        pd = pytest.importorskip("pandas")
        arr = validate_data(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
        assert arr.ndim == 1

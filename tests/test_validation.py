import numpy as np
import pytest

from bootstrapx import bootstrap
from bootstrapx.utils import validate_data


class TestValidateData:
    def test_list(self):
        assert isinstance(validate_data([1.0, 2.0, 3.0]), np.ndarray)

    def test_rejects_scalar(self):
        with pytest.raises(ValueError):
            validate_data(5.0)

    def test_rejects_nan(self):
        with pytest.raises(ValueError):
            validate_data([1.0, np.nan])

    def test_pandas_series(self):
        pd = pytest.importorskip("pandas")
        arr = validate_data(pd.Series([1.0, 2.0, 3.0]))
        assert arr.shape == (3,)

    def test_pandas_df(self):
        pd = pytest.importorskip("pandas")
        arr = validate_data(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
        assert arr.ndim == 1

    @pytest.mark.parametrize("data", [[1.0, np.inf], [1.0, -np.inf]])
    def test_rejects_infinite_data(self, data):
        with pytest.raises(ValueError, match="finite"):
            validate_data(data)


class TestBootstrapParameterValidation:
    @pytest.fixture
    def data(self):
        return np.arange(20, dtype=float)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"n_resamples": 0}, "n_resamples"),
            ({"batch_size": 0}, "batch_size"),
            ({"confidence_level": 1.0}, "confidence_level"),
            ({"confidence_level": 0.0}, "confidence_level"),
            ({"n_jobs": 0}, "n_jobs"),
        ],
    )
    def test_rejects_invalid_common_parameters(self, data, kwargs, message):
        with pytest.raises((TypeError, ValueError), match=message):
            bootstrap(data, np.mean, **kwargs)

    @pytest.mark.parametrize(
        ("method", "kwargs", "message"),
        [
            ("mbb", {"block_length": 0}, "block_length"),
            ("cbb", {"block_length": 0}, "block_length"),
            ("tapered", {"block_length": 0}, "block_length"),
            ("stationary", {"mean_block": 1}, "mean_block"),
            ("studentized", {"n_inner": 1}, "n_inner"),
            ("bernoulli", {"prob": 0}, "prob"),
            ("wild", {"distribution": "normal"}, "distribution"),
        ],
    )
    def test_rejects_invalid_method_parameters(self, data, method, kwargs, message):
        with pytest.raises((TypeError, ValueError), match=message):
            bootstrap(data, np.mean, method=method, **kwargs)

    def test_rejects_unknown_method_keyword(self, data):
        with pytest.raises(TypeError, match="Unsupported keyword"):
            bootstrap(data, np.mean, unknown_parameter=True)

    def test_rejects_ci_method_for_native_ci_method(self, data):
        with pytest.raises(ValueError, match="ci_method"):
            bootstrap(data, np.mean, method="bca", ci_method="percentile")

    def test_rejects_non_finite_statistic(self, data):
        with pytest.raises(ValueError, match="finite scalar"):
            bootstrap(data, lambda _: np.nan)

    def test_rejects_bad_cluster_ids_before_resampling(self, data):
        with pytest.raises(ValueError, match="match data length"):
            bootstrap(data, np.mean, method="cluster", cluster_ids=[1, 2])

    def test_cluster_accepts_string_identifiers(self, data):
        result = bootstrap(
            data,
            np.mean,
            method="cluster",
            cluster_ids=np.repeat(["control", "treatment"], 10),
            n_resamples=20,
            random_state=0,
        )
        assert result.n_resamples == 20

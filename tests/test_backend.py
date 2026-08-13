import numpy as np
import pytest

from bootstrapx import bootstrap
from bootstrapx.engine import backend as backend_module
from bootstrapx.engine.backend import BackendKind, resolve_backend


class UnhashableMean:
    __hash__ = None

    def __call__(self, sample):
        return float(np.mean(sample))


class TestResolveBackend:
    def test_auto(self):
        assert resolve_backend("auto") is BackendKind.VANILLA

    def test_vanilla_explicit(self):
        assert resolve_backend("vanilla") is BackendKind.VANILLA

    def test_numba_cpu_raises(self):
        with pytest.raises(ValueError, match="not implemented"):
            resolve_backend("numba_cpu")

    def test_numba_cuda_raises(self):
        with pytest.raises(ValueError, match="not implemented"):
            resolve_backend("numba_cuda")

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            resolve_backend("turbo_mode")


def test_numpy_fast_path_respects_batch_size(monkeypatch):
    batch_sizes = []
    original = backend_module._resample_batch

    def capture_batch(data, batch_size, rng):
        batch_sizes.append(batch_size)
        return original(data, batch_size, rng)

    monkeypatch.setattr(backend_module, "_resample_batch", capture_batch)
    result = bootstrap(
        np.arange(100.0),
        np.mean,
        method="percentile",
        n_resamples=25,
        batch_size=7,
        random_state=1,
    )
    assert result.n_resamples == 25
    assert batch_sizes == [7, 7, 7, 4]


def test_unhashable_statistic_uses_regular_path():
    result = bootstrap(
        np.arange(20.0),
        UnhashableMean(),
        method="percentile",
        n_resamples=20,
        random_state=2,
    )
    assert result.n_resamples == 20

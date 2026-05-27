from bootstrapx.engine.backend import resolve_backend, BackendKind
import pytest


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
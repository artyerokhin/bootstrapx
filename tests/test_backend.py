import pytest
from bootstrapx.engine.backend import resolve_backend, BackendKind

class TestResolveBackend:
    def test_auto(self): assert resolve_backend("auto") in (BackendKind.NUMBA_CPU, BackendKind.NUMBA_CUDA, BackendKind.VANILLA)
    def test_vanilla(self): assert resolve_backend("vanilla") is BackendKind.VANILLA
    def test_numba_cpu(self): assert resolve_backend("numba_cpu") is BackendKind.NUMBA_CPU
    def test_unknown(self):
        with pytest.raises(ValueError, match="Unknown backend"): resolve_backend("tensorflow")

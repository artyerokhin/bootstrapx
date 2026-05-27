from bootstrapx.engine.backend import resolve_backend, BackendKind


class TestResolveBackend:
    def test_auto(self):
        assert resolve_backend("auto") in (BackendKind.NUMBA_CPU, BackendKind.NUMBA_CUDA, BackendKind.VANILLA)

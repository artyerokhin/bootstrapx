"""Tests for BootstrapCV sklearn compatibility."""

import numpy as np
import pytest

try:
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    from bootstrapx.compat.sklearn_cv import BootstrapCV

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


@pytest.mark.skipif(not _HAS_SKLEARN, reason="scikit-learn not installed")
class TestBootstrapCV:
    def test_split_shapes(self):
        X = np.random.default_rng(0).normal(size=(100, 4))
        y = np.zeros(100)
        cv = BootstrapCV(n_splits=50, random_state=42)
        splits = list(cv.split(X, y))
        assert len(splits) == 50
        for train, test in splits:
            assert len(train) == 100
            assert len(test) > 0
            # No overlap
            assert len(np.intersect1d(train, test)) == 0

    def test_reproducibility(self):
        X = np.random.default_rng(0).normal(size=(80, 3))
        y = np.zeros(80)
        s1 = [t for _, t in BootstrapCV(30, random_state=7).split(X, y)]
        s2 = [t for _, t in BootstrapCV(30, random_state=7).split(X, y)]
        for a, b in zip(s1, s2, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_cross_val_score(self):
        X, y = load_iris(return_X_y=True)
        cv = BootstrapCV(n_splits=100, random_state=0)
        scores = cross_val_score(LogisticRegression(max_iter=200), X, y, cv=cv, scoring="accuracy")
        assert scores.mean() > 0.85
        assert len(scores) == 100

    def test_oob_fraction(self):
        """OOB fraction should be ~0.368 (Poisson approximation)."""
        X = np.arange(1000).reshape(-1, 1)
        cv = BootstrapCV(n_splits=500, random_state=0)
        oob_fracs = [len(t) / len(X) for _, t in cv.split(X)]
        mean_oob = np.mean(oob_fracs)
        assert 0.30 < mean_oob < 0.42, f"OOB fraction {mean_oob:.3f} out of expected range"

    def test_small_sample_still_yields_requested_splits(self):
        X = np.arange(2).reshape(-1, 1)
        cv = BootstrapCV(n_splits=50, random_state=0)
        splits = list(cv.split(X))
        assert len(splits) == cv.get_n_splits() == 50
        assert all(len(test) > 0 for _, test in splits)

    def test_rejects_single_observation(self):
        cv = BootstrapCV(n_splits=2, random_state=0)
        with pytest.raises(ValueError, match="at least 2"):
            list(cv.split(np.ones((1, 2))))

    @pytest.mark.parametrize("n_splits", [0, -1, 1.5, True])
    def test_rejects_invalid_n_splits(self, n_splits):
        error = TypeError if isinstance(n_splits, float | bool) else ValueError
        with pytest.raises(error, match="n_splits"):
            BootstrapCV(n_splits=n_splits)

    @pytest.mark.parametrize("random_state", [True, "seed", object()])
    def test_rejects_invalid_random_state(self, random_state):
        with pytest.raises(TypeError, match="random_state"):
            BootstrapCV(random_state=random_state)

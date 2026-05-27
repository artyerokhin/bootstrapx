"""Tests for BootstrapCV sklearn compatibility."""
import numpy as np
import pytest

try:
    from bootstrapx.compat.sklearn_cv import BootstrapCV
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
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
        for a, b in zip(s1, s2):
            np.testing.assert_array_equal(a, b)

    def test_cross_val_score(self):
        X, y = load_iris(return_X_y=True)
        cv = BootstrapCV(n_splits=100, random_state=0)
        scores = cross_val_score(
            LogisticRegression(max_iter=200), X, y, cv=cv, scoring="accuracy"
        )
        assert scores.mean() > 0.85
        assert len(scores) == 100

    def test_oob_fraction(self):
        """OOB fraction should be ~0.368 (Poisson approximation)."""
        X = np.arange(1000).reshape(-1, 1)
        cv = BootstrapCV(n_splits=500, random_state=0)
        oob_fracs = [len(t) / len(X) for _, t in cv.split(X)]
        mean_oob = np.mean(oob_fracs)
        assert 0.30 < mean_oob < 0.42, f"OOB fraction {mean_oob:.3f} out of expected range"

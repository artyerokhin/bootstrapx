"""scikit-learn compatible Bootstrap cross-validator.

Usage
-----
>>> from bootstrapx import BootstrapCV
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.datasets import load_iris
>>> from sklearn.metrics import accuracy_score
>>> import numpy as np
>>>
>>> X, y = load_iris(return_X_y=True)
>>> cv = BootstrapCV(n_splits=200, random_state=42)
>>>
>>> scores = []
>>> for train_idx, test_idx in cv.split(X, y):
...     model = LogisticRegression(max_iter=200)
...     _ = model.fit(X[train_idx], y[train_idx])
...     scores.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
>>>
>>> scores = np.array(scores)
>>> _ = f"Accuracy: {scores.mean():.3f} ± {scores.std():.3f}"

Notes
-----
Unlike k-fold, bootstrap splits allow an observation to appear multiple times
in the training set and guarantees ~63.2% unique training samples per split
(the "0.632 bootstrap estimator").
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]

try:
    from sklearn.model_selection._split import BaseCrossValidator
    from sklearn.utils.validation import indexable

    _SKLEARN = True
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for BootstrapCV. "
        "Install with: pip install 'bootstrapx-lib[sklearn]'"
    ) from exc


class BootstrapCV(BaseCrossValidator):  # type: ignore[misc]
    """Bootstrap cross-validator compatible with scikit-learn's CV API.

    Generates ``n_splits`` bootstrap train/test splits. Each training set
    is a bootstrap resample of size ``n`` (with replacement); the test set
    contains the out-of-bag (OOB) observations not selected for training.

    Parameters
    ----------
    n_splits : int, default=200
        Number of bootstrap iterations.
    random_state : int or np.random.Generator or None
        Seed for reproducibility.

    Notes
    -----
    - Usable with ``cross_val_score``, ``cross_validate``, ``GridSearchCV``.
    - OOB test set size ≈ 0.368 × n per split (Poisson approximation).
    - For the 0.632 bootstrap estimator, average
      ``0.368 * train_score + 0.632 * oob_score`` across splits.
    """

    def __init__(self, n_splits: int = 200, random_state: int | np.random.Generator | None = None):
        if isinstance(n_splits, bool) or not isinstance(n_splits, int | np.integer):
            raise TypeError("n_splits must be an integer.")
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1.")
        if (
            random_state is not None
            and not isinstance(random_state, np.random.Generator)
            and (isinstance(random_state, bool) or not isinstance(random_state, int | np.integer))
        ):
            raise TypeError("random_state must be an integer, numpy Generator, or None.")
        self.n_splits = n_splits
        self.random_state = random_state

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Generator[tuple[IntArray, IntArray], None, None]:
        X, y, groups = indexable(X, y, groups)
        n = len(X)
        if n < 2:
            raise ValueError("BootstrapCV requires at least 2 observations.")
        rng = (
            self.random_state
            if isinstance(self.random_state, np.random.Generator)
            else np.random.default_rng(self.random_state)
        )
        all_idx = np.arange(n)
        yielded = 0
        while yielded < self.n_splits:
            train = rng.integers(0, n, size=n)
            test = np.setdiff1d(all_idx, train)
            if len(test) == 0:
                continue
            yielded += 1
            yield train, test

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def _iter_test_indices(
        self, X: Any = None, y: Any = None, groups: Any = None
    ) -> Generator[IntArray, None, None]:
        for _, test in self.split(X, y, groups):
            yield test

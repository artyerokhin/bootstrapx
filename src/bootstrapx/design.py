"""Preflight diagnostics for one-level clustered resampling designs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

AnyArray = NDArray[Any]


@dataclass(frozen=True)
class ClusterDesignReport:
    """Observed design facts, not a guarantee of confidence-interval coverage.

    ``clusters_per_stratum`` and ``binary_cluster_counts_by_stratum`` follow
    NumPy's sorted order of unique stratum IDs. In the latter, each pair is
    ``(number of label-0 clusters, number of label-1 clusters)``.

    The optional one-class probabilities assume independent draws of whole
    clusters with replacement inside each observed stratum. They apply to a
    statistic that requires both binary classes (for example, ROC AUC), not
    to every statistic or to the original-sample collection process.
    """

    n_rows: int
    n_clusters: int
    n_strata: int
    clusters_per_stratum: tuple[int, ...]
    min_cluster_size: int
    median_cluster_size: float
    max_cluster_size: int
    cluster_size_cv: float
    binary_cluster_counts_by_stratum: tuple[tuple[int, int], ...] | None = None
    one_class_resample_probability: float | None = None
    probability_any_one_class_resample: float | None = None
    n_resamples_for_probability: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a detached mapping suitable for result metadata."""
        return asdict(self)


def _identifiers(ids: Any, name: str, n_rows: int | None = None) -> tuple[AnyArray, AnyArray]:
    """Validate identifiers and return their array and inverse unique codes."""
    # Preserve Python scalar types: np.asarray([1, "1"]) would otherwise
    # coerce both identifiers to the same string before uniqueness checking.
    values = np.asarray(ids, dtype=object)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    if n_rows is not None and len(values) != n_rows:
        raise ValueError(f"{name} must match data length ({n_rows}).")
    for value in values:
        if value is None or np.ndim(value) != 0:
            raise ValueError(f"{name} must not contain missing values.")
        try:
            is_missing = bool(value != value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain scalar, non-missing identifiers.") from exc
        if is_missing:
            raise ValueError(f"{name} must not contain missing values.")
        if isinstance(value, float | complex | np.floating | np.complexfloating):
            if not np.isfinite(value):
                raise ValueError(f"{name} must not contain infinite identifiers.")
    try:
        _, inverse = np.unique(values, return_inverse=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain mutually comparable identifiers.") from exc
    return values, inverse


def inspect_cluster_design(
    cluster_ids: Any,
    *,
    strata_ids: Any | None = None,
    binary_labels: Any | None = None,
    n_resamples: int = 9999,
) -> ClusterDesignReport:
    """Describe independent units and one-class risk before bootstrapping.

    ``cluster_ids`` identifies complete resampling units. ``strata_ids`` is
    optional and must be constant within each cluster. If ``binary_labels``
    is provided, it must contain 0/1 labels constant within each cluster;
    the report then computes the exact conditional probability that a whole-
    cluster resample has only one class. It never chooses a method or decides
    whether a reported interval has adequate coverage.
    """
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int | np.integer):
        raise TypeError("n_resamples must be a positive integer.")
    if n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer.")

    cluster_values, cluster_inverse = _identifiers(cluster_ids, "cluster_ids")
    n_rows = len(cluster_values)
    n_clusters = int(cluster_inverse.max()) + 1
    cluster_sizes = np.bincount(cluster_inverse, minlength=n_clusters)

    stratum_inverse: NDArray[np.intp]
    if strata_ids is None:
        stratum_inverse = np.zeros(n_rows, dtype=np.intp)
        n_strata = 1
    else:
        _, stratum_codes = _identifiers(strata_ids, "strata_ids", n_rows)
        stratum_inverse = np.asarray(stratum_codes, dtype=np.intp)
        n_strata = int(stratum_inverse.max()) + 1

    first_rows = np.full(n_clusters, n_rows, dtype=np.intp)
    np.minimum.at(first_rows, cluster_inverse, np.arange(n_rows))
    cluster_strata = stratum_inverse[first_rows]
    if np.any(cluster_strata[cluster_inverse] != stratum_inverse):
        raise ValueError("Each cluster_id must belong to exactly one stratum.")
    clusters_per_stratum = np.bincount(cluster_strata, minlength=n_strata)

    class_counts: tuple[tuple[int, int], ...] | None = None
    one_class_probability: float | None = None
    any_one_class_probability: float | None = None
    if binary_labels is not None:
        try:
            labels = np.asarray(binary_labels, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("binary_labels must contain only numeric 0/1 values.") from exc
        if (
            labels.shape != (n_rows,)
            or not np.all(np.isfinite(labels))
            or not np.all((labels == 0) | (labels == 1))
        ):
            raise ValueError("binary_labels must be a one-dimensional 0/1 array matching IDs.")
        cluster_labels = labels[first_rows].astype(np.intp)
        if np.any(cluster_labels[cluster_inverse] != labels):
            raise ValueError("binary_labels must be constant within each cluster.")

        by_stratum: list[tuple[int, int]] = []
        no_positive_probability = 1.0
        no_negative_probability = 1.0
        for stratum, count in enumerate(clusters_per_stratum):
            selected_labels = cluster_labels[cluster_strata == stratum]
            positive = int(selected_labels.sum())
            negative = int(count) - positive
            by_stratum.append((negative, positive))
            no_positive_probability *= (negative / count) ** int(count)
            no_negative_probability *= (positive / count) ** int(count)
        class_counts = tuple(by_stratum)
        one_class_probability = min(1.0, no_positive_probability + no_negative_probability)
        if one_class_probability == 1.0:
            any_one_class_probability = 1.0
        else:
            any_one_class_probability = -math.expm1(
                int(n_resamples) * math.log1p(-one_class_probability)
            )

    return ClusterDesignReport(
        n_rows=n_rows,
        n_clusters=n_clusters,
        n_strata=n_strata,
        clusters_per_stratum=tuple(int(count) for count in clusters_per_stratum),
        min_cluster_size=int(cluster_sizes.min()),
        median_cluster_size=float(np.median(cluster_sizes)),
        max_cluster_size=int(cluster_sizes.max()),
        cluster_size_cv=float(cluster_sizes.std(ddof=0) / cluster_sizes.mean()),
        binary_cluster_counts_by_stratum=class_counts,
        one_class_resample_probability=one_class_probability,
        probability_any_one_class_resample=any_one_class_probability,
        n_resamples_for_probability=int(n_resamples) if binary_labels is not None else None,
    )

"""Two-sample bootstrap comparisons for experiment workflows."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bootstrapx.stats.confidence import (
    ConfidenceInterval,
    basic_interval,
    bca_interval_from_jackknife,
    percentile_interval,
)
from bootstrapx.utils import (
    auto_batch_size,
    validate_bootstrap_distribution,
    validate_bootstrap_params,
    validate_data,
    validate_random_state,
)

FloatArray = NDArray[np.float64]
AnyArray = NDArray[Any]
Statistic = Callable[[FloatArray], float]
Effect = Callable[[float, float], float]

_EFFECTS = {"difference", "ratio", "relative_lift"}
_METHODS = {"percentile", "basic", "bca"}


@dataclass
class TwoSampleBootstrapResult:
    """Result of a control/treatment bootstrap comparison."""

    confidence_interval: ConfidenceInterval
    bootstrap_distribution: FloatArray
    estimate: float
    control_estimate: float
    treatment_estimate: float
    standard_error: float
    n_resamples: int
    method: str
    effect: str
    paired: bool
    resampling: str
    n_control: int
    n_treatment: int
    n_control_clusters: int | None = None
    n_treatment_clusters: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def theta_hat(self) -> float:
        """Alias for the observed effect estimate."""
        return self.estimate

    def __repr__(self) -> str:
        ci = self.confidence_interval
        return (
            f"TwoSampleBootstrapResult(effect={self.effect!r}, "
            f"estimate={self.estimate:.6g}, method={self.method!r}, "
            f"CI=[{ci.low:.6g}, {ci.high:.6g}])"
        )

    def to_dict(self, *, include_distribution: bool = False) -> dict[str, Any]:
        """Return a compact, mutation-safe comparison summary."""
        summary: dict[str, Any] = {
            "estimate": self.estimate,
            "control_estimate": self.control_estimate,
            "treatment_estimate": self.treatment_estimate,
            "standard_error": self.standard_error,
            "ci_low": self.confidence_interval.low,
            "ci_high": self.confidence_interval.high,
            "ci_method": self.confidence_interval.method,
            "method": self.method,
            "effect": self.effect,
            "paired": self.paired,
            "resampling": self.resampling,
            "n_control": self.n_control,
            "n_treatment": self.n_treatment,
            "n_control_clusters": self.n_control_clusters,
            "n_treatment_clusters": self.n_treatment_clusters,
            "n_resamples": self.n_resamples,
            "extra": deepcopy(self.extra),
            "metadata": deepcopy(self.metadata),
        }
        if include_distribution:
            summary["bootstrap_distribution"] = self.bootstrap_distribution.copy()
        return summary

    def to_frame(self) -> Any:
        """Return a one-row pandas DataFrame without the full distribution."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for TwoSampleBootstrapResult.to_frame(). "
                "Install with: pip install 'bootstrapx-lib[pandas]'"
            ) from exc
        return pd.DataFrame([self.to_dict()])


def _as_scalar(value: Any, *, source: str) -> float:
    values = np.asarray(value)
    if values.ndim != 0:
        raise ValueError(f"{source} must return exactly one scalar value.")
    try:
        result = float(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must return a numeric scalar value.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{source} must return a finite scalar value.")
    return result


def _ratio_denominator_is_zero(control_estimate: float) -> bool:
    """Return whether a finite floating-point denominator is exactly zero.

    Near-zero values can make a ratio statistically unstable, but rejecting
    them using an absolute tolerance would make the API depend on the units in
    which the same data is expressed. Non-zero denominators are therefore
    evaluated normally; overflow or another non-finite result is rejected by
    ``_evaluate_effect``.
    """
    return bool(control_estimate == 0.0)


def _resolve_effect(effect: str | Effect) -> tuple[Effect, str]:
    if isinstance(effect, str):
        name = effect.lower().strip()
        if name not in _EFFECTS:
            raise ValueError(f"Unknown effect {effect!r}. Choose from {sorted(_EFFECTS)}.")

        if name == "difference":
            return lambda control, treatment: treatment - control, name

        if name == "ratio":

            def ratio(control: float, treatment: float) -> float:
                if _ratio_denominator_is_zero(control):
                    raise ValueError(
                        "ratio is undefined because a control estimate is zero. "
                        "Use effect='difference' or a custom stabilized effect."
                    )
                return treatment / control

            return ratio, name

        def relative_lift(control: float, treatment: float) -> float:
            if _ratio_denominator_is_zero(control):
                raise ValueError(
                    "relative_lift is undefined because a control estimate is zero. "
                    "Use effect='difference' or a custom stabilized effect."
                )
            return (treatment - control) / control

        return relative_lift, name

    if not callable(effect):
        raise TypeError("effect must be a supported string or a callable.")
    return effect, getattr(effect, "__name__", "custom")


def _evaluate_statistic(statistic: Statistic, sample: FloatArray) -> float:
    try:
        value = statistic(sample)
    except ZeroDivisionError as exc:
        raise ValueError(
            "statistic is undefined because it divided by zero for an observed, "
            "resampled, or jackknife value."
        ) from exc
    return _as_scalar(value, source="statistic")


def _validate_comparison_samples(
    control: Any, treatment: Any, *, allow_2d: bool
) -> tuple[FloatArray, FloatArray]:
    control_array = validate_data(control, allow_2d=allow_2d)
    treatment_array = validate_data(treatment, allow_2d=allow_2d)
    if control_array.ndim != treatment_array.ndim:
        raise ValueError("control and treatment must have the same number of dimensions.")
    if control_array.ndim == 2:
        if control_array.shape[1] == 0 or treatment_array.shape[1] == 0:
            raise ValueError("Matrix samples must contain at least one feature column.")
        if control_array.shape[1] != treatment_array.shape[1]:
            raise ValueError("control and treatment must have the same number of feature columns.")
        try:
            import pandas as pd
        except ImportError:
            pass
        else:
            for name, sample in (("control", control), ("treatment", treatment)):
                if isinstance(sample, pd.DataFrame) and not sample.columns.is_unique:
                    raise ValueError(
                        f"{name} DataFrame must have unique column labels. "
                        "Rename duplicate columns before selecting metric features."
                    )
            if isinstance(control, pd.DataFrame) and isinstance(treatment, pd.DataFrame):
                if not control.columns.equals(treatment.columns):
                    raise ValueError(
                        "control and treatment DataFrames must have identical column labels "
                        "in the same order. Select matching numeric columns explicitly."
                    )
    return control_array, treatment_array


def _evaluate_effect(effect: Effect, control: float, treatment: float) -> float:
    try:
        value = effect(control, treatment)
    except ZeroDivisionError as exc:
        raise ValueError(
            "effect is undefined because it divided by zero for an observed or resampled value."
        ) from exc
    return _as_scalar(value, source="effect")


def _validate_cluster_ids(ids: Any, n: int, *, name: str) -> AnyArray:
    # Preserve labels before validation: NumPy's inferred string/float dtype
    # can merge 1 with "1", or round distinct large integer identifiers.
    values = np.asarray(ids, dtype=object)
    if values.ndim != 1 or len(values) != n:
        raise ValueError(f"{name} must be one-dimensional and match its sample length.")
    for identifier in values:
        if identifier is None or np.ndim(identifier) != 0:
            raise ValueError(f"{name} must contain scalar, non-missing identifiers.")
        try:
            if bool(identifier != identifier):
                raise ValueError(f"{name} must not contain missing values.")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain scalar, non-missing identifiers.") from exc
        if isinstance(identifier, float | complex | np.floating | np.complexfloating):
            if not np.isfinite(identifier):
                raise ValueError(f"{name} must not contain infinite identifiers.")
    try:
        unique = np.unique(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain mutually comparable identifiers.") from exc
    if len(unique) < 2:
        raise ValueError(f"{name} must contain at least two distinct clusters.")
    return values


def _validate_unit_ids(ids: Any, n: int, *, name: str) -> AnyArray:
    # Object dtype prevents mixed identifiers such as 1 and "1" from being
    # silently coerced into the same string before uniqueness checks.
    values = np.asarray(ids, dtype=object)
    if values.ndim != 1 or len(values) != n:
        raise ValueError(f"{name} must be one-dimensional and match its sample length.")
    unique: set[Any] = set()
    for identifier in values:
        if identifier is None or np.ndim(identifier) != 0:
            raise ValueError(f"{name} must contain scalar, non-missing identifiers.")
        try:
            missing = bool(identifier != identifier)
            hash(identifier)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must contain hashable, non-missing scalar identifiers."
            ) from exc
        if missing or (isinstance(identifier, float | np.floating) and not np.isfinite(identifier)):
            raise ValueError(f"{name} must not contain missing or infinite identifiers.")
        if identifier in unique:
            raise ValueError(
                f"{name} must be unique: each row must represent one analysis unit. "
                "Aggregate repeated observations or use cluster IDs instead."
            )
        unique.add(identifier)
    return values


def _validate_design_unit_ids(
    control_unit_ids: Any,
    treatment_unit_ids: Any,
    *,
    n_control: int,
    n_treatment: int,
    paired: bool,
    cluster_mode: bool,
) -> bool:
    supplied = control_unit_ids is not None or treatment_unit_ids is not None
    if not supplied:
        return False
    if control_unit_ids is None or treatment_unit_ids is None:
        raise ValueError("control_unit_ids and treatment_unit_ids must be provided together.")
    if cluster_mode:
        raise ValueError(
            "unit IDs validate one row per unit and cannot be combined with cluster IDs. "
            "For repeated-event input, use cluster IDs and check assignment integrity separately."
        )
    control = _validate_unit_ids(control_unit_ids, n_control, name="control_unit_ids")
    treatment = _validate_unit_ids(treatment_unit_ids, n_treatment, name="treatment_unit_ids")
    if paired:
        if not np.array_equal(control, treatment):
            raise ValueError(
                "Paired unit IDs must match in the same row order. "
                "Align corresponding units explicitly before analysis."
            )
    elif set(control).intersection(treatment):
        raise ValueError(
            "Independent samples must not share unit IDs. "
            "Check assignment integrity or choose a justified paired design."
        )
    return True


def _child_generators(
    random_state: int | np.random.Generator | None,
    count: int,
) -> list[np.random.Generator]:
    root = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    seeds = root.integers(0, np.iinfo(np.uint64).max, size=count, dtype=np.uint64)
    return [np.random.default_rng(seed) for seed in seeds]


def _iid_distribution(
    control: FloatArray,
    treatment: FloatArray,
    statistic: Statistic,
    effect: Effect,
    n_resamples: int,
    batch_size: int,
    random_state: int | np.random.Generator | None,
    *,
    paired: bool,
) -> FloatArray:
    distribution = np.empty(n_resamples, dtype=np.float64)
    if paired:
        (pair_rng,) = _child_generators(random_state, 1)
    else:
        control_rng, treatment_rng = _child_generators(random_state, 2)

    done = 0
    while done < n_resamples:
        size = min(batch_size, n_resamples - done)
        if paired:
            indices = pair_rng.integers(0, len(control), size=(size, len(control)))
            control_indices = treatment_indices = indices
        else:
            control_indices = control_rng.integers(0, len(control), size=(size, len(control)))
            treatment_indices = treatment_rng.integers(
                0, len(treatment), size=(size, len(treatment))
            )
        for offset in range(size):
            control_stat = _evaluate_statistic(statistic, control[control_indices[offset]])
            treatment_stat = _evaluate_statistic(statistic, treatment[treatment_indices[offset]])
            distribution[done + offset] = _evaluate_effect(effect, control_stat, treatment_stat)
        done += size
    return distribution


def _cluster_map(ids: AnyArray) -> tuple[AnyArray, dict[Any, NDArray[np.intp]]]:
    unique = np.unique(ids)
    mapping = {identifier: np.flatnonzero(ids == identifier) for identifier in unique}
    return unique, mapping


def _cluster_sample(
    data: FloatArray,
    mapping: dict[Any, NDArray[np.intp]],
    chosen: AnyArray,
) -> FloatArray:
    indices = np.concatenate([mapping[identifier] for identifier in chosen])
    return np.asarray(data[indices], dtype=np.float64)


def _cluster_distribution(
    control: FloatArray,
    treatment: FloatArray,
    control_ids: AnyArray,
    treatment_ids: AnyArray,
    statistic: Statistic,
    effect: Effect,
    n_resamples: int,
    batch_size: int,
    random_state: int | np.random.Generator | None,
) -> FloatArray:
    control_unique, control_map = _cluster_map(control_ids)
    treatment_unique, treatment_map = _cluster_map(treatment_ids)
    control_rng, treatment_rng = _child_generators(random_state, 2)
    distribution = np.empty(n_resamples, dtype=np.float64)

    done = 0
    while done < n_resamples:
        size = min(batch_size, n_resamples - done)
        control_choices = control_rng.choice(
            control_unique, size=(size, len(control_unique)), replace=True
        )
        treatment_choices = treatment_rng.choice(
            treatment_unique, size=(size, len(treatment_unique)), replace=True
        )
        for offset in range(size):
            control_sample = _cluster_sample(control, control_map, control_choices[offset])
            treatment_sample = _cluster_sample(treatment, treatment_map, treatment_choices[offset])
            distribution[done + offset] = _evaluate_effect(
                effect,
                _evaluate_statistic(statistic, control_sample),
                _evaluate_statistic(statistic, treatment_sample),
            )
        done += size
    return distribution


def _loo_effects(
    control: FloatArray,
    treatment: FloatArray,
    statistic: Statistic,
    effect: Effect,
    *,
    paired: bool,
    control_cluster_ids: AnyArray | None,
    treatment_cluster_ids: AnyArray | None,
) -> list[FloatArray]:
    if paired:
        effects = np.empty(len(control), dtype=np.float64)
        control_buffer = np.empty((len(control) - 1, *control.shape[1:]), dtype=control.dtype)
        treatment_buffer = np.empty(
            (len(treatment) - 1, *treatment.shape[1:]), dtype=treatment.dtype
        )
        for index in range(len(control)):
            control_buffer[:index] = control[:index]
            control_buffer[index:] = control[index + 1 :]
            treatment_buffer[:index] = treatment[:index]
            treatment_buffer[index:] = treatment[index + 1 :]
            effects[index] = _evaluate_effect(
                effect,
                _evaluate_statistic(statistic, control_buffer),
                _evaluate_statistic(statistic, treatment_buffer),
            )
        return [effects]

    control_full = _evaluate_statistic(statistic, control)
    treatment_full = _evaluate_statistic(statistic, treatment)

    if control_cluster_ids is None or treatment_cluster_ids is None:
        control_effects = np.empty(len(control), dtype=np.float64)
        treatment_effects = np.empty(len(treatment), dtype=np.float64)
        control_buffer = np.empty((len(control) - 1, *control.shape[1:]), dtype=control.dtype)
        treatment_buffer = np.empty(
            (len(treatment) - 1, *treatment.shape[1:]), dtype=treatment.dtype
        )
        for index in range(len(control)):
            control_buffer[:index] = control[:index]
            control_buffer[index:] = control[index + 1 :]
            control_effects[index] = _evaluate_effect(
                effect,
                _evaluate_statistic(statistic, control_buffer),
                treatment_full,
            )
        for index in range(len(treatment)):
            treatment_buffer[:index] = treatment[:index]
            treatment_buffer[index:] = treatment[index + 1 :]
            treatment_effects[index] = _evaluate_effect(
                effect,
                control_full,
                _evaluate_statistic(statistic, treatment_buffer),
            )
        return [control_effects, treatment_effects]

    control_unique = np.unique(control_cluster_ids)
    treatment_unique = np.unique(treatment_cluster_ids)
    control_effects = np.empty(len(control_unique), dtype=np.float64)
    treatment_effects = np.empty(len(treatment_unique), dtype=np.float64)
    for index, identifier in enumerate(control_unique):
        control_effects[index] = _evaluate_effect(
            effect,
            _evaluate_statistic(statistic, control[control_cluster_ids != identifier]),
            treatment_full,
        )
    for index, identifier in enumerate(treatment_unique):
        treatment_effects[index] = _evaluate_effect(
            effect,
            control_full,
            _evaluate_statistic(statistic, treatment[treatment_cluster_ids != identifier]),
        )
    return [control_effects, treatment_effects]


def bootstrap_two_sample(
    control: Any,
    treatment: Any,
    statistic: Statistic,
    *,
    effect: str | Effect = "difference",
    method: str = "bca",
    paired: bool = False,
    allow_2d: bool = False,
    control_cluster_ids: Any | None = None,
    treatment_cluster_ids: Any | None = None,
    control_unit_ids: Any | None = None,
    treatment_unit_ids: Any | None = None,
    metric_name: str | None = None,
    effect_unit: str | None = None,
    n_resamples: int = 9999,
    batch_size: int | None = None,
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = None,
) -> TwoSampleBootstrapResult:
    """Bootstrap an effect between control and treatment samples.

    The observed effect and every resampled effect are calculated as
    ``effect(statistic(control), statistic(treatment))``. Independent samples
    are resampled separately. With ``paired=True``, both samples use the same
    resampled indices. Supplying cluster IDs resamples complete clusters within
    each arm and is mutually exclusive with paired analysis.

    Parameters
    ----------
    control, treatment : array-like
        Finite one-dimensional samples, or numeric matrices with
        ``allow_2d=True``. Matrix rows are observations and columns are jointly
        observed features, always resampled together. Their order defines the
        direction of every built-in effect. DataFrames are converted to NumPy;
        matrix DataFrames must have unique column labels; two DataFrames must
        have identical labels and order.
    statistic : callable
        Scalar function applied separately to each arm, ``array -> float``.
        For matrix input, receives a 2-D array and must still return one scalar.
    effect : {"difference", "ratio", "relative_lift"} or callable
        Transformation of the two arm statistics. A callable receives
        ``(control_statistic, treatment_statistic)`` and returns one scalar.
        Difference is ``treatment - control``; ratio is
        ``treatment / control``; relative lift is
        ``(treatment - control) / control``.
    method : {"percentile", "basic", "bca"}
        Confidence-interval construction.
    paired : bool
        Resample corresponding rows together. The samples must have equal
        length and cluster IDs cannot be supplied.
        Pairing is positional: pandas indices are not used to align samples.
    allow_2d : bool
        Explicitly enable multicolumn input for composite scalar metrics.
        Defaults to False, preserving the one-dimensional input contract.
        Both arms must have the same dimensionality and feature count.
    control_cluster_ids, treatment_cluster_ids : array-like or None
        One cluster identifier per row. Both arrays are required for clustered
        analysis; complete clusters are resampled independently within each
        experiment arm. Labels must be scalar, non-missing, finite when numeric,
        and mutually comparable within each arm. Mixed string/numeric labels
        are rejected rather than coerced into one type.
    control_unit_ids, treatment_unit_ids : array-like or None
        Optional globally consistent identifiers for one-row-per-unit input.
        Both are required together and must be unique within each arm.
        Independent arms must be disjoint; paired arms must match in row order.
        Cannot be combined with cluster IDs. No IDs are stored in the result.
        Omitting IDs leaves correspondence/assignment validation to the caller.
    metric_name, effect_unit : str or None
        Optional non-empty reporting labels, for example ``"revenue/order"``
        and ``"USD/order"`` for a difference. Labels do not transform values or
        verify the metric definition. Ratios/lifts are dimensionless.
    n_resamples : int
        Number of bootstrap effects.
    batch_size : int or None
        Number of resamples processed per technical batch. Changing it does
        not change a seeded bootstrap distribution.
    confidence_level : float
        Requested interval level strictly between zero and one.
    random_state : int, numpy.random.Generator, or None
        Reproducible random-state source.

    Returns
    -------
    TwoSampleBootstrapResult
        Arm estimates, observed effect, interval, standard error, bootstrap
        distribution, and experiment-design metadata.

    Notes
    -----
    Ratio and relative-lift effects are rejected if the observed or any
    resampled control statistic is zero. Near-zero denominators are allowed
    because any fixed tolerance would depend on measurement units, but they
    can produce unstable intervals. No invalid replicates are silently
    discarded.
    """
    if not callable(statistic):
        raise TypeError("statistic must be callable.")
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    if not isinstance(paired, bool):
        raise TypeError("paired must be a boolean.")
    if not isinstance(allow_2d, bool):
        raise TypeError("allow_2d must be a boolean.")
    for name, label in (("metric_name", metric_name), ("effect_unit", effect_unit)):
        if label is not None:
            if not isinstance(label, str):
                raise TypeError(f"{name} must be a string or None.")
            if not label.strip():
                raise ValueError(f"{name} must be non-empty when provided.")

    method = method.lower().strip()
    if method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from {sorted(_METHODS)}.")
    effect_function, effect_name = _resolve_effect(effect)
    control_array, treatment_array = _validate_comparison_samples(
        control, treatment, allow_2d=allow_2d
    )

    cluster_mode = control_cluster_ids is not None or treatment_cluster_ids is not None
    if cluster_mode and (control_cluster_ids is None or treatment_cluster_ids is None):
        raise ValueError("control_cluster_ids and treatment_cluster_ids must be provided together.")
    if paired and cluster_mode:
        raise ValueError("paired=True cannot be combined with cluster IDs.")
    if paired and len(control_array) != len(treatment_array):
        raise ValueError("paired samples must contain the same number of observations.")
    unit_ids_validated = _validate_design_unit_ids(
        control_unit_ids,
        treatment_unit_ids,
        n_control=len(control_array),
        n_treatment=len(treatment_array),
        paired=paired,
        cluster_mode=cluster_mode,
    )

    control_ids: AnyArray | None = None
    treatment_ids: AnyArray | None = None
    if cluster_mode:
        control_ids = _validate_cluster_ids(
            control_cluster_ids, len(control_array), name="control_cluster_ids"
        )
        treatment_ids = _validate_cluster_ids(
            treatment_cluster_ids, len(treatment_array), name="treatment_cluster_ids"
        )

    n_units_control = len(np.unique(control_ids)) if control_ids is not None else len(control_array)
    n_units_treatment = (
        len(np.unique(treatment_ids)) if treatment_ids is not None else len(treatment_array)
    )
    if method == "bca" and min(n_units_control, n_units_treatment) < 3:
        unit_name = "clusters" if cluster_mode else "observations"
        raise ValueError(f"BCa requires at least three {unit_name} in each sample.")

    validate_bootstrap_params(
        method=method,
        n_observations=min(len(control_array), len(treatment_array)),
        n_resamples=n_resamples,
        batch_size=batch_size,
        confidence_level=confidence_level,
        ci_method=None,
        n_jobs=1,
        kwargs={},
    )
    validate_random_state(random_state)
    if batch_size is None:
        batch_size = auto_batch_size(control_array.size + treatment_array.size, n_resamples)

    control_estimate = _evaluate_statistic(statistic, control_array)
    treatment_estimate = _evaluate_statistic(statistic, treatment_array)
    estimate = _evaluate_effect(effect_function, control_estimate, treatment_estimate)

    if cluster_mode:
        assert control_ids is not None and treatment_ids is not None
        distribution = _cluster_distribution(
            control_array,
            treatment_array,
            control_ids,
            treatment_ids,
            statistic,
            effect_function,
            n_resamples,
            batch_size,
            random_state,
        )
    else:
        distribution = _iid_distribution(
            control_array,
            treatment_array,
            statistic,
            effect_function,
            n_resamples,
            batch_size,
            random_state,
            paired=paired,
        )
    distribution = validate_bootstrap_distribution(distribution, n_resamples)

    if method == "percentile":
        interval = percentile_interval(distribution, confidence_level)
    elif method == "basic":
        interval = basic_interval(distribution, estimate, confidence_level)
    else:
        jackknife_groups = _loo_effects(
            control_array,
            treatment_array,
            statistic,
            effect_function,
            paired=paired,
            control_cluster_ids=control_ids,
            treatment_cluster_ids=treatment_ids,
        )
        interval = bca_interval_from_jackknife(
            distribution,
            estimate,
            jackknife_groups,
            confidence_level,
        )

    from bootstrapx import __version__

    metadata = {
        "metric_name": metric_name or getattr(statistic, "__name__", type(statistic).__name__),
        "effect_unit": effect_unit,
        "resampling_unit": "cluster" if cluster_mode else "pair" if paired else "row",
        "n_control_units": n_units_control,
        "n_treatment_units": n_units_treatment,
        "n_features": control_array.shape[1] if control_array.ndim == 2 else 1,
        "unit_ids_validated": unit_ids_validated,
        "confidence_level": float(confidence_level),
        "batch_size": int(batch_size),
        "seed": int(random_state) if isinstance(random_state, int | np.integer) else None,
        "random_state_kind": "generator"
        if isinstance(random_state, np.random.Generator)
        else "seed"
        if random_state is not None
        else "unseeded",
        "package_version": __version__,
    }
    return TwoSampleBootstrapResult(
        confidence_interval=interval,
        bootstrap_distribution=distribution,
        estimate=estimate,
        control_estimate=control_estimate,
        treatment_estimate=treatment_estimate,
        standard_error=float(np.std(distribution, ddof=1)),
        n_resamples=n_resamples,
        method=method,
        effect=effect_name,
        paired=paired,
        resampling="cluster" if cluster_mode else "iid",
        n_control=len(control_array),
        n_treatment=len(treatment_array),
        n_control_clusters=n_units_control if cluster_mode else None,
        n_treatment_clusters=n_units_treatment if cluster_mode else None,
        metadata=metadata,
    )

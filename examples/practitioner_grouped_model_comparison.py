"""Compare fixed model predictions on a held-out set with repeated patient rows.

Run from the checkout:
PYTHONPATH=src python examples/practitioner_grouped_model_comparison.py
Requires scikit-learn. All patients and outcomes are synthetic.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from bootstrapx import bootstrap_two_sample


def make_holdout() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(706)
    n_patients = 400
    rows_per_patient = 5
    patient_ids = np.repeat(np.arange(n_patients), rows_per_patient)
    patient_effect = np.repeat(rng.normal(0, 0.9, n_patients), rows_per_patient)
    first_feature = rng.normal(size=len(patient_ids)) + 0.4 * patient_effect
    second_feature = rng.normal(size=len(patient_ids))
    probability = expit(0.9 * first_feature + 0.8 * second_feature + patient_effect)
    labels = rng.binomial(1, probability)
    features = np.column_stack((first_feature, second_feature))

    train_rows, test_rows = next(
        GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42).split(
            features, labels, groups=patient_ids
        )
    )
    train_patients = np.unique(patient_ids[train_rows])
    test_patients = patient_ids[test_rows]
    assert not np.intersect1d(train_patients, np.unique(test_patients)).size

    model_a = LogisticRegression(max_iter=200).fit(features[train_rows, :1], labels[train_rows])
    model_b = LogisticRegression(max_iter=200).fit(features[train_rows], labels[train_rows])
    predictions_a = model_a.predict_proba(features[test_rows, :1])[:, 1]
    predictions_b = model_b.predict_proba(features[test_rows])[:, 1]

    # One row contains the observed label and both models' predictions. The
    # same patient draw therefore moves all three columns together.
    held_out = np.column_stack((labels[test_rows], predictions_a, predictions_b))
    return held_out, test_patients, train_patients


def auc_gain(rows: np.ndarray) -> float:
    labels = rows[:, 0].astype(np.intp)
    return float(roc_auc_score(labels, rows[:, 2]) - roc_auc_score(labels, rows[:, 1]))


def auc_metric(rows: np.ndarray) -> float:
    return float(roc_auc_score(rows[:, 0].astype(np.intp), rows[:, 1]))


def brier_metric(rows: np.ndarray) -> float:
    return float(np.mean((rows[:, 1] - rows[:, 0]) ** 2))


def run(n_resamples: int = 999) -> dict[str, object]:
    held_out, test_patients, train_patients = make_holdout()
    result = bootstrap_two_sample(
        held_out[:, [0, 1]],
        held_out[:, [0, 2]],
        auc_metric,
        allow_2d=True,
        paired=True,
        paired_cluster_ids=test_patients,
        control_observation_ids=np.arange(len(held_out)),
        treatment_observation_ids=np.arange(len(held_out)),
        method="percentile",
        n_resamples=n_resamples,
        random_state=42,
    )
    brier_result = bootstrap_two_sample(
        held_out[:, [0, 1]],
        held_out[:, [0, 2]],
        brier_metric,
        allow_2d=True,
        paired=True,
        paired_cluster_ids=test_patients,
        method="percentile",
        n_resamples=n_resamples,
        random_state=42,
    )

    # SciPy can estimate the same clustered design by resampling patient
    # positions and reconstructing all corresponding rows in the statistic.
    patient_rows = [
        np.flatnonzero(test_patients == patient) for patient in np.unique(test_patients)
    ]

    def scipy_auc_gain(sampled_patients: np.ndarray) -> float:
        selected_rows = np.concatenate([patient_rows[int(i)] for i in sampled_patients])
        return auc_gain(held_out[selected_rows])

    reference = stats.bootstrap(
        (np.arange(len(patient_rows)),),
        scipy_auc_gain,
        vectorized=False,
        method="percentile",
        n_resamples=n_resamples,
        random_state=42,
    )
    return {
        "held_out": held_out,
        "test_patients": test_patients,
        "train_patients": train_patients,
        "bootstrapx": result,
        "scipy": reference,
        "brier": brier_result,
    }


if __name__ == "__main__":
    analysis = run()
    result = analysis["bootstrapx"]
    reference = analysis["scipy"]
    print(
        f"Held-out patients: {len(np.unique(analysis['test_patients']))}; "
        f"held-out rows: {len(analysis['held_out'])}"
    )
    print(f"Fixed-model AUC gain (B - A): {result.theta_hat:+.3f}")
    print(f"AUC A: {result.control_estimate:.3f}; AUC B: {result.treatment_estimate:.3f}")
    print(
        f"bootstrapx 95% percentile CI: "
        f"[{result.confidence_interval.low:+.3f}, {result.confidence_interval.high:+.3f}]"
    )
    print(
        f"SciPy 95% percentile CI: "
        f"[{reference.confidence_interval.low:+.3f}, "
        f"{reference.confidence_interval.high:+.3f}]"
    )
    brier = analysis["brier"]
    print(f"Brier difference (B - A; negative is better): {brier.estimate:+.3f}")
    print(f"Brier 95% CI: {brier.confidence_interval}")
    print(f"Brier 90% CI without new metric calls: {brier.interval(confidence_level=0.90)}")

"""Exploratory known-truth study of resampling design, not a release benchmark.

Run from the checkout with ``python benchmarks/study_design_aware_bootstrap.py``.
Only source code is tracked; the script prints Monte Carlo estimates and their
sampling error. It does not modify package code or produce stored results.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.stats import binom, norm


def interval_summary(hits: int, total: int) -> str:
    if total == 0:
        return "undefined"
    rate = hits / total
    mc_half_width = 1.96 * math.sqrt(rate * (1 - rate) / total)
    return f"{rate:.3f} ± {mc_half_width:.3f} (MC 95% half-width; {total} trials)"


def covers(draws: np.ndarray, truth: float) -> tuple[bool, float]:
    low, high = np.quantile(draws, [0.025, 0.975])
    return bool(low <= truth <= high), float(high - low)


def covers_basic(draws: np.ndarray, estimate: float, truth: float) -> tuple[bool, float]:
    lower_quantile, upper_quantile = np.quantile(draws, [0.025, 0.975])
    low, high = 2 * estimate - upper_quantile, 2 * estimate - lower_quantile
    return bool(low <= truth <= high), float(high - low)


def fixed_strata_study(
    trials: int, resamples: int, clusters_per_stratum: int, rng: np.random.Generator
) -> dict[str, object]:
    """Two fixed strata, equal-sized independent clusters, ICC 0.5.

    The target is 0.5 * (mu_0 + mu_1) = 0. Each stratum contributes a fixed
    number of clusters; cluster means have variance tau^2 + sigma^2 / k.
    """
    m, k = clusters_per_stratum, 5
    hits = dict.fromkeys(("row", "row_within_stratum", "cluster", "cluster_within_stratum"), 0)
    widths = dict.fromkeys(hits, 0.0)
    for _ in range(trials):
        cluster_effect = rng.normal(size=(2, m, 1))
        noise = rng.normal(size=(2, m, k))
        observations = np.array([-1.0, 1.0])[:, None, None] + cluster_effect + noise
        cluster_means = observations.mean(axis=2)

        rows = observations.reshape(-1)
        row_draws = rows[rng.integers(0, len(rows), size=(resamples, len(rows)))].mean(axis=1)
        within_row_draws = 0.5 * (
            observations[0]
            .reshape(-1)[rng.integers(0, m * k, size=(resamples, m * k))]
            .mean(axis=1)
            + observations[1]
            .reshape(-1)[rng.integers(0, m * k, size=(resamples, m * k))]
            .mean(axis=1)
        )
        units = cluster_means.reshape(-1)
        cluster_draws = units[rng.integers(0, len(units), size=(resamples, len(units)))].mean(
            axis=1
        )
        within_draws = 0.5 * (
            cluster_means[0][rng.integers(0, m, size=(resamples, m))].mean(axis=1)
            + cluster_means[1][rng.integers(0, m, size=(resamples, m))].mean(axis=1)
        )
        for name, draws in (
            ("row", row_draws),
            ("row_within_stratum", within_row_draws),
            ("cluster", cluster_draws),
            ("cluster_within_stratum", within_draws),
        ):
            hit, width = covers(draws, 0.0)
            hits[name] += int(hit)
            widths[name] += width

    return {
        "clusters_per_stratum": m,
        "true_se": math.sqrt((1.0 + 1.0 / k) / (2 * m)),
        "coverage": {name: interval_summary(count, trials) for name, count in hits.items()},
        "mean_width": {name: round(width / trials, 3) for name, width in widths.items()},
    }


def informative_cluster_size_study(
    trials: int, resamples: int, rng: np.random.Generator
) -> dict[str, object]:
    """Show that row- and unit-weighted targets differ when size is informative.

    A cluster with U > 0 has eight rows; otherwise it has two. Its mean is
    mu_h + U + independent noise. The equally weighted cluster target is zero,
    but the row-weighted target is E[K U] / E[K] = 6 / (5 sqrt(2 pi)).
    """
    m = 50
    row_truth = 6 / (5 * math.sqrt(2 * math.pi))
    cluster_hits = 0
    row_hits = 0
    mismatched_row_hits = 0
    for _ in range(trials):
        effect = rng.normal(size=(2, m))
        sizes = np.where(effect > 0, 8, 2)
        means = np.array([-1.0, 1.0])[:, None] + effect + rng.normal(size=(2, m)) / np.sqrt(sizes)
        selected = rng.integers(0, m, size=(2, resamples, m))
        chosen_means = np.take_along_axis(means[:, None, :], selected, axis=2)
        chosen_sizes = np.take_along_axis(sizes[:, None, :], selected, axis=2)
        cluster_draws = chosen_means.mean(axis=(0, 2))
        row_draws = (chosen_means * chosen_sizes).sum(axis=(0, 2)) / chosen_sizes.sum(axis=(0, 2))
        cluster_hits += int(covers(cluster_draws, 0.0)[0])
        row_hits += int(covers(row_draws, row_truth)[0])
        mismatched_row_hits += int(covers(row_draws, 0.0)[0])
    return {
        "clusters_per_stratum": m,
        "true_equal_cluster_mean": 0.0,
        "true_row_weighted_mean": round(row_truth, 3),
        "equal_cluster_interval_coverage_own_target": interval_summary(cluster_hits, trials),
        "row_weighted_interval_coverage_own_target": interval_summary(row_hits, trials),
        "row_weighted_interval_coverage_wrong_cluster_target": interval_summary(
            mismatched_row_hits, trials
        ),
    }


def auc_draws(
    kernel: np.ndarray, positive_weights: np.ndarray, negative_weights: np.ndarray
) -> np.ndarray:
    numerator = np.sum((positive_weights @ kernel) * negative_weights, axis=1)
    denominator = positive_weights.sum(axis=1) * negative_weights.sum(axis=1)
    return numerator / denominator


def rare_class_study(
    trials: int, resamples: int, n_patients: int, rng: np.random.Generator
) -> dict[str, object]:
    """Patient-level ROC AUC, duplicated records, and a constant-negative rule.

    AUC = P(score_+ > score_-), independent of prevalence in this model.
    Constant-negative accuracy = 1 - prevalence, which is not independent of
    prevalence. Five identical records per patient expose pseudo-replication.
    """
    p, shift, copies = 0.1, 1.5, 5
    true_auc = float(norm.cdf(shift / math.sqrt(2)))
    true_accuracy = 1 - p
    auc_valid = 0
    original_one_class = 0
    auc_hits = {
        "class_stratified_patient": 0,
        "class_stratified_patient_basic": 0,
        "class_stratified_row": 0,
    }
    auc_widths = dict.fromkeys(auc_hits, 0.0)
    few_positive = {"trials": 0, "hits": 0}
    more_positive = {"trials": 0, "hits": 0}
    accuracy_hits = {"patient": 0, "class_stratified_patient": 0}
    invalid_replicates = 0
    predicted_invalid_replicates = 0.0
    any_invalid = 0
    predicted_any_invalid = 0.0

    for _ in range(trials):
        labels = rng.binomial(1, p, size=n_patients)
        n_positive = int(labels.sum())
        n_negative = n_patients - n_positive

        # An always-negative classifier estimates population accuracy 1-p.
        ordinary_weights = rng.multinomial(
            n_patients, np.full(n_patients, 1 / n_patients), size=resamples
        )
        drawn_positives = ordinary_weights[:, labels == 1].sum(axis=1)
        ordinary_accuracy = 1 - drawn_positives / n_patients
        accuracy_hits["patient"] += int(covers(ordinary_accuracy, true_accuracy)[0])
        accuracy_hits["class_stratified_patient"] += int(
            1 - n_positive / n_patients == true_accuracy
        )

        invalid = (drawn_positives == 0) | (drawn_positives == n_patients)
        invalid_replicates += int(invalid.sum())
        any_invalid += int(invalid.any())
        q = (n_positive / n_patients) ** n_patients + (n_negative / n_patients) ** n_patients
        predicted_invalid_replicates += q * resamples
        predicted_any_invalid += 1 - (1 - q) ** resamples

        if not n_positive or not n_negative:
            original_one_class += 1
            continue

        scores = rng.normal(loc=shift * labels, size=n_patients)
        positive = scores[labels == 1]
        negative = scores[labels == 0]
        kernel = (positive[:, None] > negative[None, :]).astype(float)
        observed_auc = float(kernel.mean())
        auc_valid += 1

        for name, draw_size in (
            ("class_stratified_patient", 1),
            ("class_stratified_row", copies),
        ):
            positive_weights = rng.multinomial(
                n_positive * draw_size,
                np.full(n_positive, 1 / n_positive),
                size=resamples,
            )
            negative_weights = rng.multinomial(
                n_negative * draw_size,
                np.full(n_negative, 1 / n_negative),
                size=resamples,
            )
            draw = auc_draws(kernel, positive_weights, negative_weights)
            hit, width = covers(draw, true_auc)
            auc_hits[name] += int(hit)
            auc_widths[name] += width
            if name == "class_stratified_patient":
                bucket = few_positive if n_positive < 5 else more_positive
                bucket["trials"] += 1
                bucket["hits"] += int(hit)
                basic_hit, basic_width = covers_basic(draw, observed_auc, true_auc)
                auc_hits["class_stratified_patient_basic"] += int(basic_hit)
                auc_widths["class_stratified_patient_basic"] += basic_width

    return {
        "patients": n_patients,
        "positive_prevalence": p,
        "rows_per_patient": copies,
        "original_one_class_fraction": round(original_one_class / trials, 3),
        "predicted_original_one_class_fraction": round((1 - p) ** n_patients + p**n_patients, 3),
        "invalid_unstratified_replicate_fraction": round(
            invalid_replicates / (trials * resamples), 3
        ),
        "predicted_invalid_replicate_fraction": round(
            predicted_invalid_replicates / (trials * resamples), 3
        ),
        "fraction_runs_with_any_invalid_replicate": round(any_invalid / trials, 3),
        "predicted_fraction_runs_with_any_invalid": round(predicted_any_invalid / trials, 3),
        "auc_coverage_given_two_classes": {
            name: interval_summary(count, auc_valid) for name, count in auc_hits.items()
        },
        "auc_mean_width": {name: round(width / auc_valid, 3) for name, width in auc_widths.items()},
        "auc_percentile_coverage_by_positive_count": {
            "under_5": interval_summary(few_positive["hits"], few_positive["trials"]),
            "at_least_5": interval_summary(more_positive["hits"], more_positive["trials"]),
        },
        "constant_negative_accuracy_coverage": {
            name: interval_summary(count, trials) for name, count in accuracy_hits.items()
        },
        "exact_stratified_accuracy_coverage": round(
            float(binom.pmf(round(n_patients * p), n_patients, p)), 3
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--resamples", type=int, default=399)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--study", choices=("all", "fixed", "size", "rare"), default="all")
    args = parser.parse_args()
    if args.trials < 10 or args.resamples < 99:
        parser.error("Use at least 10 trials and 99 bootstrap resamples.")

    rng = np.random.default_rng(args.seed)
    print(f"seed={args.seed}, trials={args.trials}, resamples={args.resamples}")
    if args.study in ("all", "fixed"):
        print("\nFIXED TWO-STRATUM CLUSTER MEAN")
        for m in (10, 50):
            print(fixed_strata_study(args.trials, args.resamples, m, rng))
    if args.study in ("all", "size"):
        print("\nINFORMATIVE CLUSTER SIZE")
        print(informative_cluster_size_study(args.trials, args.resamples, rng))
    if args.study in ("all", "rare"):
        print("\nRARE-CLASS AUC AND ACCURACY")
        for n in (40, 200):
            print(rare_class_study(args.trials, args.resamples, n, rng))


if __name__ == "__main__":
    main()

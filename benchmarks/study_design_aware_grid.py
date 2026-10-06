"""Explicit-grid known-truth stress tests for clustered bootstrap intervals.

This is research, not a package benchmark or evidence of universal coverage.
Run from the checkout with::

    python benchmarks/study_design_aware_grid.py --study all

The scenario cells are explicit in ``main`` rather than selected from favorable
output. This is exploratory research, not a preregistered study. Each trial
draws a new dataset; the reported Monte Carlo half-width reflects only
simulation uncertainty, not uncertainty from finite bootstrap resamples. No
data or result files are saved.

Mean grid: 2 fixed, equally weighted strata; 5 observations per independent
cluster; Gaussian cluster and observation effects; 10/20/50 clusters per
stratum; intracluster correlation 0/.2/.5/.8. The true mean is zero. The
cluster-level Welch t interval is a model-specific reference, not bootstrapx.

AUC grid: 5 correlated observations per independent patient; one binary label
per patient; prevalence .05/.1; 40/200 patients; score ICC .25/.75/1. The
positive-class score shift is 1.5 and the true row-pair AUC is Phi(1.5/sqrt(2)).
Patients are resampled *within observed outcome class* for AUC only; this is
not a default recommendation for prevalence-dependent metrics. Coverage is
conditional on the original sample containing both classes. No invalid
bootstrap replicate is silently removed.
The optional ``auc_rows`` study compares patient and row resampling on the
same small datasets, for three score ICCs; it is separate because it is more
computationally expensive.

Size grid: 2 fixed strata with 20/50/200 clusters each; cluster sizes 2 or 8.
The size is either independent of the cluster effect or 8 when that effect is
positive and 2 otherwise. Both the equally weighted cluster mean and the
row-weighted mean are compared with their *own* population targets.

Skew grid: 2 fixed strata with 20/50 clusters each, equal cluster sizes and
centered lognormal cluster effects. Fixed population stratum weights are
(.5, .5) or (.2, .8); the weighted mean is part of the statistic, not inferred
from observed row counts.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy.stats import norm, t
from study_design_aware_bootstrap import covers, covers_basic


def _rate(hits: int, total: int) -> dict[str, float | int | None]:
    if not total:
        return {"hits": hits, "trials": total, "coverage": None, "mc_95_half_width": None}
    proportion = hits / total
    return {
        "hits": hits,
        "trials": total,
        "coverage": round(proportion, 4),
        "mc_95_half_width": round(1.96 * math.sqrt(proportion * (1 - proportion) / total), 4),
    }


def fixed_strata_mean(
    trials: int,
    resamples: int,
    clusters_per_stratum: int,
    icc: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Compare resampling units and interval rules under fixed-strata sampling."""
    m, rows_per_cluster = clusters_per_stratum, 5
    hits = dict.fromkeys(
        (
            "row_within_stratum_percentile",
            "cluster_within_stratum_percentile",
            "cluster_within_stratum_basic",
            "cluster_welch_t_reference",
        ),
        0,
    )
    width_sums = dict.fromkeys(hits, 0.0)
    for _ in range(trials):
        effects = rng.normal(size=(2, m, 1)) * math.sqrt(icc)
        errors = rng.normal(size=(2, m, rows_per_cluster)) * math.sqrt(1 - icc)
        observations = np.array([-1.0, 1.0])[:, None, None] + effects + errors
        cluster_means = observations.mean(axis=2)
        estimate = float(cluster_means.mean())

        row_draws = np.zeros(resamples)
        cluster_draws = np.zeros(resamples)
        for stratum in range(2):
            rows = observations[stratum].reshape(-1)
            row_draws += 0.5 * rows[rng.integers(0, len(rows), size=(resamples, len(rows)))].mean(
                axis=1
            )
            cluster_draws += 0.5 * cluster_means[stratum][
                rng.integers(0, m, size=(resamples, m))
            ].mean(axis=1)

        for name, result in (
            ("row_within_stratum_percentile", covers(row_draws, 0.0)),
            ("cluster_within_stratum_percentile", covers(cluster_draws, 0.0)),
            ("cluster_within_stratum_basic", covers_basic(cluster_draws, estimate, 0.0)),
        ):
            hit, width = result
            hits[name] += int(hit)
            width_sums[name] += width

        # Exact Gaussian cluster means justify this model-specific reference.
        stratum_variances = cluster_means.var(axis=1, ddof=1) / (4 * m)
        variance = float(stratum_variances.sum())
        degrees_of_freedom = variance**2 / float(np.square(stratum_variances).sum() / (m - 1))
        half_width = float(t.ppf(0.975, degrees_of_freedom) * math.sqrt(variance))
        hits["cluster_welch_t_reference"] += int(abs(estimate) <= half_width)
        width_sums["cluster_welch_t_reference"] += 2 * half_width

    return {
        "study": "fixed_strata_mean",
        "clusters_per_stratum": m,
        "icc": icc,
        "true_se": round(math.sqrt((icc + (1 - icc) / rows_per_cluster) / (2 * m)), 4),
        "coverage": {name: _rate(count, trials) for name, count in hits.items()},
        "mean_width": {name: round(width / trials, 4) for name, width in width_sums.items()},
    }


def _auc_from_patient_weights(
    patient_kernel: np.ndarray,
    positive_weights: np.ndarray,
    negative_weights: np.ndarray,
) -> np.ndarray:
    numerator = np.einsum(
        "bi,ij,bj->b", positive_weights, patient_kernel, negative_weights, optimize=True
    )
    return numerator / (positive_weights.sum(axis=1) * negative_weights.sum(axis=1))


def cluster_size_mean(
    trials: int,
    resamples: int,
    clusters_per_stratum: int,
    informative: bool,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Separate resampling coverage from row-vs-cluster estimand selection."""
    m = clusters_per_stratum
    row_truth = 6 / (5 * math.sqrt(2 * math.pi)) if informative else 0.0
    hits = {"cluster_own_target": 0, "row_own_target": 0, "row_wrong_cluster_target": 0}
    for _ in range(trials):
        effects = rng.normal(size=(2, m))
        if informative:
            sizes = np.where(effects > 0, 8, 2)
        else:
            sizes = np.where(rng.random(size=(2, m)) > 0.5, 8, 2)
        errors = rng.normal(size=(2, m, 8))
        means = (
            np.array([-1.0, 1.0])[:, None]
            + effects
            + np.sum(errors * (np.arange(8) < sizes[:, :, None]), axis=2) / sizes
        )
        selected = rng.integers(0, m, size=(2, resamples, m))
        chosen_means = np.take_along_axis(means[:, None, :], selected, axis=2)
        chosen_sizes = np.take_along_axis(sizes[:, None, :], selected, axis=2)
        cluster_draws = chosen_means.mean(axis=(0, 2))
        # The fixed strata receive equal *population* weight, independently
        # of the random number of sampled rows in each stratum.
        row_draws = 0.5 * (
            (chosen_means * chosen_sizes).sum(axis=2) / chosen_sizes.sum(axis=2)
        ).sum(axis=0)
        hits["cluster_own_target"] += int(covers(cluster_draws, 0.0)[0])
        hits["row_own_target"] += int(covers(row_draws, row_truth)[0])
        hits["row_wrong_cluster_target"] += int(covers(row_draws, 0.0)[0])
    return {
        "study": "cluster_size_mean",
        "clusters_per_stratum": m,
        "informative": informative,
        "cluster_truth": 0.0,
        "row_truth": round(row_truth, 4),
        "coverage": {name: _rate(count, trials) for name, count in hits.items()},
    }


def skewed_fixed_strata_mean(
    trials: int,
    resamples: int,
    clusters_per_stratum: int,
    first_stratum_weight: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Stress-test percentile/basic intervals under skew and fixed weights."""
    m = clusters_per_stratum
    weights = np.array([first_stratum_weight, 1 - first_stratum_weight])
    offsets = np.array([-1.0, 1.0])
    truth = float(weights @ offsets)
    centered_lognormal_sd = math.sqrt((math.e - 1) * math.e)
    hits = {"percentile": 0, "basic": 0}
    for _ in range(trials):
        effects = (rng.lognormal(size=(2, m)) - math.exp(0.5)) / centered_lognormal_sd
        errors = rng.normal(size=(2, m, 5)).mean(axis=2)
        means = offsets[:, None] + effects + errors
        estimate = float(weights @ means.mean(axis=1))
        drawn_means = np.zeros((2, resamples))
        for stratum in range(2):
            drawn_means[stratum] = means[stratum][rng.integers(0, m, size=(resamples, m))].mean(
                axis=1
            )
        draws = weights @ drawn_means
        hits["percentile"] += int(covers(draws, truth)[0])
        hits["basic"] += int(covers_basic(draws, estimate, truth)[0])
    return {
        "study": "skewed_fixed_strata_mean",
        "clusters_per_stratum": m,
        "first_stratum_weight": first_stratum_weight,
        "true_weighted_mean": round(truth, 4),
        "coverage": {name: _rate(count, trials) for name, count in hits.items()},
    }


def rare_class_auc(
    trials: int,
    resamples: int,
    patients: int,
    prevalence: float,
    score_icc: float,
    rng: np.random.Generator,
    *,
    row_baseline: bool = False,
) -> dict[str, object]:
    """Check conditional AUC coverage with a varying within-patient score ICC."""
    rows_per_patient, score_shift = 5, 1.5
    true_auc = float(norm.cdf(score_shift / math.sqrt(2)))
    hits = {"patient_percentile": 0, "patient_basic": 0}
    if row_baseline:
        hits["row_percentile"] = 0
    width_sums = dict.fromkeys(hits, 0.0)
    positive_count_buckets = {
        "1_to_4": {"hits": 0, "trials": 0},
        "5_to_9": {"hits": 0, "trials": 0},
        "10_plus": {"hits": 0, "trials": 0},
    }
    valid = 0
    predicted_invalid_fraction = 0.0
    predicted_any_invalid = 0.0
    for _ in range(trials):
        labels = rng.binomial(1, prevalence, size=patients)
        positive_count = int(labels.sum())
        negative_count = patients - positive_count
        q = (positive_count / patients) ** patients + (negative_count / patients) ** patients
        predicted_invalid_fraction += q
        predicted_any_invalid += 1 - (1 - q) ** resamples
        if not positive_count or not negative_count:
            continue
        valid += 1

        effects = rng.normal(size=(patients, 1)) * math.sqrt(score_icc)
        errors = rng.normal(size=(patients, rows_per_patient)) * math.sqrt(1 - score_icc)
        scores = score_shift * labels[:, None] + effects + errors
        positive_scores = scores[labels == 1]
        negative_scores = scores[labels == 0]
        # Mean over all row pairs for each pair of positive/negative patients.
        patient_kernel = (
            positive_scores[:, None, :, None] > negative_scores[None, :, None, :]
        ).mean(axis=(2, 3))
        estimate = float(patient_kernel.mean())
        positive_weights = rng.multinomial(
            positive_count, np.full(positive_count, 1 / positive_count), size=resamples
        )
        negative_weights = rng.multinomial(
            negative_count, np.full(negative_count, 1 / negative_count), size=resamples
        )
        draws = _auc_from_patient_weights(patient_kernel, positive_weights, negative_weights)
        percentile_hit, percentile_width = covers(draws, true_auc)
        basic_hit, basic_width = covers_basic(draws, estimate, true_auc)
        hits["patient_percentile"] += int(percentile_hit)
        hits["patient_basic"] += int(basic_hit)
        width_sums["patient_percentile"] += percentile_width
        width_sums["patient_basic"] += basic_width

        if row_baseline:
            positive_rows = positive_scores.reshape(-1)
            negative_rows = negative_scores.reshape(-1)
            row_kernel = (positive_rows[:, None] > negative_rows[None, :]).astype(float)
            positive_row_weights = rng.multinomial(
                len(positive_rows),
                np.full(len(positive_rows), 1 / len(positive_rows)),
                size=resamples,
            )
            negative_row_weights = rng.multinomial(
                len(negative_rows),
                np.full(len(negative_rows), 1 / len(negative_rows)),
                size=resamples,
            )
            row_draws = _auc_from_patient_weights(
                row_kernel, positive_row_weights, negative_row_weights
            )
            row_hit, row_width = covers(row_draws, true_auc)
            hits["row_percentile"] += int(row_hit)
            width_sums["row_percentile"] += row_width

        if positive_count < 5:
            bucket_name = "1_to_4"
        elif positive_count < 10:
            bucket_name = "5_to_9"
        else:
            bucket_name = "10_plus"
        bucket = positive_count_buckets[bucket_name]
        bucket["trials"] += 1
        bucket["hits"] += int(percentile_hit)

    return {
        "study": "rare_class_auc",
        "patients": patients,
        "prevalence": prevalence,
        "score_icc": score_icc,
        "true_auc": round(true_auc, 4),
        "original_one_class": _rate(trials - valid, trials),
        "predicted_original_one_class_fraction": round(
            (1 - prevalence) ** patients + prevalence**patients, 4
        ),
        "predicted_invalid_ordinary_bootstrap_fraction": round(
            predicted_invalid_fraction / trials, 4
        ),
        "predicted_runs_with_any_invalid_ordinary_bootstrap": round(
            predicted_any_invalid / trials, 4
        ),
        "coverage_conditional_on_two_classes": {
            name: _rate(count, valid) for name, count in hits.items()
        },
        "mean_width_conditional_on_two_classes": {
            name: round(width / valid, 4) if valid else None for name, width in width_sums.items()
        },
        "percentile_coverage_by_positive_patient_count": {
            name: _rate(bucket["hits"], bucket["trials"])
            for name, bucket in positive_count_buckets.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", choices=("all", "mean", "size", "skew", "auc", "auc_rows"), default="all"
    )
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--resamples", type=int, default=399)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    if args.trials < 10 or args.resamples < 99:
        parser.error("Use at least 10 trials and 99 bootstrap resamples.")

    print(json.dumps({"trials": args.trials, "resamples": args.resamples, "seed": args.seed}))
    rng = np.random.default_rng(args.seed)
    if args.study in ("all", "mean"):
        for clusters_per_stratum in (10, 20, 50):
            for icc in (0.0, 0.2, 0.5, 0.8):
                print(
                    json.dumps(
                        fixed_strata_mean(
                            args.trials, args.resamples, clusters_per_stratum, icc, rng
                        )
                    ),
                    flush=True,
                )
    if args.study in ("all", "size"):
        for clusters_per_stratum in (20, 50, 200):
            for informative in (False, True):
                print(
                    json.dumps(
                        cluster_size_mean(
                            args.trials, args.resamples, clusters_per_stratum, informative, rng
                        )
                    ),
                    flush=True,
                )
    if args.study in ("all", "skew"):
        for clusters_per_stratum in (20, 50):
            for first_stratum_weight in (0.5, 0.2):
                print(
                    json.dumps(
                        skewed_fixed_strata_mean(
                            args.trials,
                            args.resamples,
                            clusters_per_stratum,
                            first_stratum_weight,
                            rng,
                        )
                    ),
                    flush=True,
                )
    if args.study in ("all", "auc"):
        for patients in (40, 200):
            for prevalence in (0.05, 0.1):
                for score_icc in (0.25, 0.75, 1.0):
                    print(
                        json.dumps(
                            rare_class_auc(
                                args.trials,
                                resamples=args.resamples,
                                patients=patients,
                                prevalence=prevalence,
                                score_icc=score_icc,
                                rng=rng,
                            )
                        ),
                        flush=True,
                    )
    if args.study == "auc_rows":
        for score_icc in (0.25, 0.75, 1.0):
            print(
                json.dumps(
                    rare_class_auc(
                        args.trials,
                        resamples=args.resamples,
                        patients=40,
                        prevalence=0.1,
                        score_icc=score_icc,
                        rng=rng,
                        row_baseline=True,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()

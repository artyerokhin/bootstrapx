# Current Limitations

bootstrapx is intended to make common resampling workflows reliable and
reproducible. It does not remove the assumptions behind the selected bootstrap
method.

## Scalar statistics and effects

The public APIs expect exactly one finite scalar statistic per arm and one
finite scalar effect per resample. Vector-valued statistics, simultaneous
intervals, and covariance estimates are not yet supported.

Version 0.6.0 allows multicolumn input for one composite scalar metric through
`bootstrap_two_sample(..., allow_2d=True)`. This is not vector-valued output.
Optional row-unit ID checks detect duplicates/overlap or mismatched pairs, not
invalid randomization, missing telemetry, or all forms of selection bias.
See [Composite metrics](composite-metrics.md).

`DataFrame.bootstrap.summary()` evaluates columns independently. Separate
column intervals are not an interval or hypothesis test for the difference
between columns. Extract the two samples and call `bootstrap_two_sample()` for
an effect interval.

## Experiment comparisons

Independent, paired, and separately clustered control/treatment comparisons
are supported. The library does not provide p-values, sequential-testing
guarantees, CUPED/regression adjustment, multiple-testing correction, or
two-sample stratified resampling.

Ratio and relative-lift effects are undefined when a control estimate is zero
and can be unstable when it is merely close to zero. bootstrapx rejects
non-finite resampled effects instead of silently discarding them.

A real-world experiment can demonstrate correct analysis-unit selection and
effect interpretation, but its population effect is unknown. It therefore
cannot establish confidence-interval coverage. The
[Hillstrom case study](real-world-ab.md) is workflow evidence. The controlled
[product A/B reference](product-ab.md) exposes its generating effect and a
small smoke check, but the larger known-truth studies in
[Benchmarks](benchmarks.md) remain the release coverage evidence.

## Missing data

Input data must be finite. NaN and infinite values are rejected instead of
being silently dropped or imputed. Apply a documented missing-data policy
before calling bootstrapx.

## Machine-learning validation

`BootstrapCV` resamples independent rows and rejects non-None `groups`.
It is not a group-aware or time-series-safe cross-validator. Its OOB score
distribution is not automatically a confidence interval or a .632 estimate.
See [scikit-learn integration](integrations.md#scikit-learn).

## Dependent data

Block and sieve methods assume that their time-series model is appropriate.
Block length remains problem-dependent; compare nearby choices and check the
stability of the resulting interval. Sieve bootstrap assumes that an
autoregressive approximation is reasonable.

Cluster bootstrap resamples one grouping level. Multiway clustering,
hierarchical random effects, survey calibration weights, and finite-population
survey designs require additional methodology not currently implemented.

For clustered experiments, applying `np.mean` to raw events estimates an
event-weighted metric while clusters are the resampling unit. Aggregate to one
value per user first when the estimand is an equally weighted user-level mean.

## Monte Carlo and finite-sample uncertainty

Reported intervals do not include a separate estimate of Monte Carlo error.
Increase `n_resamples` and compare repeated seeds when interval endpoints are
decision-critical. BCa and studentized intervals can be unstable for very
small samples, nonsmooth statistics, or highly skewed distributions.

The 0.6 composite-metric release study also found material BCa undercoverage
for skewed and activity-correlated ratios despite close agreement with SciPy;
the lowest observed result was 87% for a nominal 95% interval with 24/30
clusters. That agreement is an implementation cross-check, not a coverage
guarantee. Baseline clustered ratios were near nominal at 100/120 clusters, but
the corresponding correlated case reached 94.7%, 95.0%, and 93.0% for
percentile, basic, and BCa. Their Monte Carlo intervals all included 95%.
Do not infer a universal minimum safe cluster count from two simulations.

## API maturity

bootstrapx is still in the `0.x` series. Changes are documented in the
changelog, but the public API is not yet covered by a 1.0 compatibility
guarantee. Pin the package version in production environments.

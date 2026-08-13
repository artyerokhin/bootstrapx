# Current Limitations

bootstrapx is intended to make common resampling workflows reliable and
reproducible. It does not remove the assumptions behind the selected bootstrap
method.

## Scalar, one-sample statistics

The top-level API expects exactly one finite scalar per resample. Vector-valued
statistics, simultaneous intervals, and covariance estimates are not yet
supported. Paired effects can be represented as a per-pair transformation.
Unpaired two-sample effects such as differences, ratios, and relative lift are
not yet a native workflow and require caller-owned resampling.

`DataFrame.bootstrap.summary()` evaluates columns independently. Separate
column intervals are not an interval or hypothesis test for the difference
between columns.

## Missing data

Input data must be finite. NaN and infinite values are rejected instead of
being silently dropped or imputed. Apply a documented missing-data policy
before calling bootstrapx.

## Dependent data

Block and sieve methods assume that their time-series model is appropriate.
Block length remains problem-dependent; compare nearby choices and check the
stability of the resulting interval. Sieve bootstrap assumes that an
autoregressive approximation is reasonable.

Cluster bootstrap resamples one grouping level. Multiway clustering,
hierarchical random effects, survey calibration weights, and finite-population
survey designs require additional methodology not currently implemented.

## Monte Carlo and finite-sample uncertainty

Reported intervals do not include a separate estimate of Monte Carlo error.
Increase `n_resamples` and compare repeated seeds when interval endpoints are
decision-critical. BCa and studentized intervals can be unstable for very
small samples, nonsmooth statistics, or highly skewed distributions.

## API maturity

bootstrapx is still in the `0.x` series. Changes are documented in the
changelog, but the public API is not yet covered by a 1.0 compatibility
guarantee. Pin the package version in production environments.

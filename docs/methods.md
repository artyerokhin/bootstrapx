# Methods

This page explains when to use each method.

## IID methods

### BCa (`method="bca"`)
Best default when you want strong coverage properties for a scalar statistic.

### Percentile (`method="percentile"`)
Good when you want a simple interval and your bootstrap distribution is already informative.

### Basic (`method="basic"`)
Useful as a classic alternative to percentile intervals.

### Studentized (`method="studentized"`)
Useful when you need bootstrap-t intervals and can afford higher compute cost.
Each outer estimate is paired with a standard error estimated from the same
outer sample. The default `n_inner=100` is a compromise; strongly skewed or
unstable statistics may require more inner and outer resamples.

### Bayesian (`method="bayesian"`)
Produces a Bayesian-bootstrap posterior by drawing Dirichlet weights. Built-in
weighted handling is available for `np.mean`, `np.nanmean`, and `np.average`.
Custom statistics must provide `weighted_statistic(data, weights)`. The
resulting percentile bounds are credible intervals, not frequentist confidence
intervals.

### Poisson / Bernoulli / Subsampling
Poisson uses Poisson(1) multiplier counts and is appropriate for smooth
functionals where the multiplier bootstrap is justified. Bernoulli uses a
finite-population-corrected random subset and assumes a smooth root-n
statistic. Subsampling estimates a centered, scaled root from samples of size
`subsample_size`; its default `rate=0.5` assumes root-n convergence. For
heavy-tailed estimators with another convergence rate, set `rate` from the
relevant statistical theory rather than relying on the default. Subsampling
theory also assumes `subsample_size < n` and usually works with a subsample
that is meaningfully smaller than the full data set.

## Time-series methods

### Moving Block Bootstrap (`mbb`)
Preserves local autocorrelation by resampling contiguous blocks.

### Circular Block Bootstrap (`cbb`)
Like MBB, but wraps around the series to reduce edge effects.

### Stationary Bootstrap (`stationary`)
Uses random block lengths and is often a strong default for stationary dependent data.

### Tapered Block Bootstrap (`tapered`)
Reduces block-boundary artifacts with tapering windows.

### Sieve Bootstrap (`sieve`)
Fits an AR approximation and resamples residual-driven trajectories. Use it when an AR representation is reasonable.

### Wild Bootstrap (`wild`)
Useful for heteroscedastic residual structures.

## Hierarchical methods

### Cluster Bootstrap (`cluster`)
Resample entire clusters, not rows, when observations within a cluster are dependent.

### Stratified Bootstrap (`strata`)
Use when the sampling design or inference target is stratified.

## Practical defaults

- General-purpose scalar statistic: `bca`
- Model metric on iid holdout set: `bca` or `percentile`
- Financial / dependent series: `stationary` or `mbb`
- Panel / user-session data: `cluster`
- Survey-like grouped data: `strata`

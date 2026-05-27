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

### Bayesian (`method="bayesian"`)
Useful for Bayesian bootstrap style uncertainty without assuming a parametric model.

### Poisson / Bernoulli / Subsampling
Useful for weighted or subsampling-based workflows, especially under heavy tails or sampling constraints.

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

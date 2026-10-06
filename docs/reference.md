# API Reference

## Top-level API

::: bootstrapx.bootstrap

::: bootstrapx.BootstrapResult

## Cluster-design diagnostics

::: bootstrapx.inspect_cluster_design

::: bootstrapx.ClusterDesignReport

## Experiment comparisons

::: bootstrapx.bootstrap_two_sample

::: bootstrapx.TwoSampleBootstrapResult

### Composite metric helper

Added in 0.6.0. See [Composite metrics](composite-metrics.md).

::: bootstrapx.RatioOfSums

::: bootstrapx.ConfidenceInterval

`BootstrapResult.to_dict()` excludes the potentially large bootstrap
distribution by default. Pass `include_distribution=True` when the full array
is required. `BootstrapResult.to_frame()` returns a compact one-row pandas
DataFrame.

`TwoSampleBootstrapResult` follows the same compact-export policy and adds arm
estimates, effect/design metadata, sample sizes, and optional cluster counts.

### Development additions

Not yet in the 0.6.0 PyPI package:

- `paired=True, paired_cluster_ids=...` selects shared whole-cluster draws;
  specify `method="percentile"` or `"basic"`. The default BCa is unsupported
  for this design and raises an error.
- `control_observation_ids` and `treatment_observation_ids` optionally validate
  unique, positionally matching row keys for paired comparisons. There is no
  automatic join/alignment or retention of IDs.
- `extra["design"]` reports counts/sizes for clustered results. Comparisons
  use `"paired"`, or separate `"control"`/`"treatment"` reports.
- `extra["distribution_diagnostics"]` contains `n_unique_values` and
  `is_degenerate`. These are exact facts, not a coverage assessment.
- `result.interval(confidence_level=0.95, method="percentile")` returns a new
  percentile/basic interval from saved draws. It does not mutate the result
  or call the metric. BCa reconstruction and specialized one-sample interval
  reinterpretation are unsupported.

See [Paired model comparison](paired-model-comparison.md) for a complete workflow.

## Integrations

::: bootstrapx.compat.sklearn_cv.BootstrapCV

::: bootstrapx.compat.pandas_accessor._BootstrapSeriesAccessor

::: bootstrapx.compat.pandas_accessor._BootstrapDataFrameAccessor

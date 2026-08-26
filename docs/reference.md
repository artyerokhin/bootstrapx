# API Reference

## Top-level API

::: bootstrapx.bootstrap

::: bootstrapx.BootstrapResult

## Experiment comparisons

::: bootstrapx.bootstrap_two_sample

::: bootstrapx.TwoSampleBootstrapResult

::: bootstrapx.ConfidenceInterval

`BootstrapResult.to_dict()` excludes the potentially large bootstrap
distribution by default. Pass `include_distribution=True` when the full array
is required. `BootstrapResult.to_frame()` returns a compact one-row pandas
DataFrame.

`TwoSampleBootstrapResult` follows the same compact-export policy and adds arm
estimates, effect/design metadata, sample sizes, and optional cluster counts.

## Integrations

::: bootstrapx.compat.sklearn_cv.BootstrapCV

::: bootstrapx.compat.pandas_accessor._BootstrapSeriesAccessor

::: bootstrapx.compat.pandas_accessor._BootstrapDataFrameAccessor

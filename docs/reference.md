# API Reference

## Top-level API

::: bootstrapx.bootstrap

::: bootstrapx.BootstrapResult

::: bootstrapx.ConfidenceInterval

`BootstrapResult.to_dict()` excludes the potentially large bootstrap
distribution by default. Pass `include_distribution=True` when the full array
is required. `BootstrapResult.to_frame()` returns a compact one-row pandas
DataFrame.

## Integrations

::: bootstrapx.compat.sklearn_cv.BootstrapCV

::: bootstrapx.compat.pandas_accessor._BootstrapSeriesAccessor

::: bootstrapx.compat.pandas_accessor._BootstrapDataFrameAccessor

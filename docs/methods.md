# Choose a Method

Choose the resampling design first and the interval construction second. A
more sophisticated interval cannot repair the wrong independence assumption.

## Decision table

| Situation | Method | Main setting | Main caution |
|---|---|---|---|
| Independent observations, general scalar statistic | `bca` | — | can be unstable for tiny samples or nonsmooth statistics |
| Independent observations, simple baseline | `percentile` | — | transformation and bias behavior can be weak |
| Independent observations, reflected interval | `basic` | — | relies on a useful error-distribution reflection |
| Bootstrap-t is scientifically justified | `studentized` | `n_inner=` | much more expensive; nested SE must be stable |
| Bayesian-bootstrap posterior | `bayesian` | `weighted_statistic=` for custom statistics | bounds are credible, not confidence, intervals |
| Smaller-sample asymptotics | `subsampling` | `subsample_size=`, `rate=` | rate and sample size require theory |
| Dependent stationary series | `stationary` | `mean_block=` | result is sensitive to dependence assumptions |
| Fixed-length local dependence | `mbb` / `cbb` | `block_length=` | compare several plausible block lengths |
| AR-like stationary series | `sieve` | `ar_order=` | inappropriate for dynamics an AR model cannot represent |
| Heteroscedastic residual workflow | `wild` | `fitted=`, `distribution=` | caller must supply a meaningful fitted structure |
| Repeated observations within groups | `cluster` | `cluster_ids=` | one grouping level only |
| Known strata in the sampling design | `strata` | `strata=` | strata must represent the actual design |

## IID intervals

`bca` is a reasonable starting point for many smooth scalar statistics, not a
universal best method. Compare it with `percentile` and investigate large
disagreements. `studentized` is useful only when the nested standard-error
estimate is meaningful and its extra cost is acceptable.

`bayesian` draws Dirichlet weights. `np.mean`, `np.nanmean`, and `np.average`
have built-in weighted handling; a custom statistic must provide
`weighted_statistic(data, weights)`.

`poisson`, `bernoulli`, and `subsampling` are specialist tools rather than
drop-in improvements over BCa. Their smoothness, finite-population, and
convergence-rate assumptions should come from the analysis design.

## Dependent data

Use `cluster` when dependence is explained by a grouping unit. Use block or
stationary methods when ordering and local serial dependence matter. Use
`strata` to preserve a known sampling composition, not merely because a useful
category exists in the dataset.

For detailed examples, continue to [Grouped and experiment data](ab-testing.md)
or [Time series](time-series.md).

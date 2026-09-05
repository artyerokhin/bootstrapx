# Real-world A/B Case Study

This walkthrough applies `bootstrap_two_sample()` to Kevin Hillstrom's public
email experiment. The source contains 64,000 customers randomly assigned to a
men's email, a women's email, or no email, followed by visit, conversion, and
spend outcomes.

The complete executable analysis is in the
[Hillstrom notebook](https://github.com/artyerokhin/bootstrapx/blob/main/notebooks/06_real_world_ab_hillstrom.ipynb).
It downloads the original CSV only when explicitly run, verifies the file with
SHA-256, and caches it under the ignored local `data/` directory. bootstrapx
does not redistribute the raw records because the source does not state a
formal redistribution license.

Original source: [MineThatData E-Mail Analytics and Data Mining
Challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html).

## Predefined analysis

The notebook compares `Womens E-Mail` with `No E-Mail`. Choosing one contrast
before looking at outcomes keeps the walkthrough focused and avoids presenting
two selected campaign comparisons as though no multiple-testing choice had
been made.

Each row is one randomized customer, so the samples are independent and no
cluster IDs are needed. Every assigned customer remains in the analysis,
including non-buyers with zero spend. Filtering to purchasers would condition
on a post-treatment outcome and change the estimand.

```python
result = bootstrap_two_sample(
    control["conversion"].to_numpy(dtype=float),
    treatment["conversion"].to_numpy(dtype=float),
    np.mean,
    effect="difference",
    method="bca",
    n_resamples=4_999,
    random_state=42,
)
```

## Reproduced results

These results come from the verified source file with SHA-256
`0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`.
The control contains 21,306 customers and the treatment contains 21,387.

| Outcome | Control | Treatment | Effect | 95% BCa interval |
|---|---:|---:|---:|---:|
| Visit rate | 10.617% | 15.140% | +4.523 pp | [+3.887, +5.151] pp |
| Conversion rate | 0.573% | 0.884% | +0.311 pp | [+0.152, +0.475] pp |
| Relative conversion lift | — | — | +54.3% | [+23.6%, +94.5%] |
| Spend per assigned customer | $0.653 | $1.077 | +$0.424 | [+$0.176, +$0.683] |

For this predefined contrast, all three absolute-effect intervals are above
zero. Relative lift is less stable because the control conversion rate is
small; the absolute percentage-point effect is the safer primary result.

## What this example establishes

The case study demonstrates a realistic workflow:

1. identify the randomization and analysis unit;
2. validate assignment labels and outcomes;
3. state one treatment-versus-control estimand;
4. preserve zero outcomes and analyze all assigned units;
5. report absolute effects and intervals before relative lift;
6. separate statistical uncertainty from business interpretation.

It does not prove 95% coverage because the population treatment effect is not
known for a real dataset. Coverage is assessed with known-truth simulations in
[Benchmarks](benchmarks.md). The result is also not a p-value, a
multiple-testing correction, or evidence that the campaign will generalize to
another customer population.

## Run it

From a bootstrapx checkout:

```bash
python -m pip install -e ".[pandas]" matplotlib jupyter
jupyter lab notebooks/06_real_world_ab_hillstrom.ipynb
```

The first execution downloads about 4 MB. Later executions reuse the verified
local cache. Delete `data/hillstrom.csv` to force a fresh download.

# Product A/B Reference

This is the primary end-to-end experiment example for bootstrapx. It uses
synthetic users so the data-generating process, true effect, analysis plan, and
decision rule are all explicit. The companion
[Hillstrom case study](real-world-ab.md) shows the additional limitations that
appear with archived real data.

The complete executable analysis is in the
[product A/B notebook](https://github.com/artyerokhin/bootstrapx/blob/main/notebooks/07_product_ab_reference.ipynb).
It requires no download and finishes in seconds on a typical laptop.

## Analysis plan

Before generating outcomes, the notebook fixes:

| Item | Choice |
|---|---|
| Randomization and analysis unit | User |
| Variants | 50,000 control and 50,000 treatment users |
| Primary metric | Seven-day conversion per assigned user |
| Estimand | Treatment minus control conversion rate |
| Interval | Two-sided 95% BCa, 4,999 resamples |
| Minimum practical effect | +0.5 percentage points |
| Metric decision rule | Lower interval endpoint must exceed +0.5 pp |

The simulated control conversion probability is 10.0%, and treatment is
11.2%. The true superpopulation effect is therefore +1.2 percentage points.
In a real experiment this truth would be unknown.

## Generate and validate the experiment

```python
rng = np.random.default_rng(1)
assignment = rng.permutation(
    np.repeat(["control", "treatment"], N_USERS // 2)
)
probability = np.where(
    assignment == "control", CONTROL_RATE, TREATMENT_RATE
)
converted = rng.binomial(1, probability)
```

The notebook verifies one row per user, exact assignment labels, finite binary
outcomes, and both nonempty arms before analysis. Production pipelines should
also perform their platform-specific sample-ratio, exposure, bot, and logging
checks.

## Estimate the primary effect

```python
result = bootstrap_two_sample(
    control,
    treatment,
    np.mean,
    effect="difference",
    method="bca",
    n_resamples=4_999,
    batch_size=100,
    random_state=42,
)
```

The saved deterministic run produces:

| Quantity | Result |
|---|---:|
| Control conversion | 9.884% |
| Treatment conversion | 11.058% |
| Estimated difference | +1.174 pp |
| 95% BCa interval | [+0.789, +1.546] pp |
| Known generating effect | +1.200 pp |

The interval contains the known generating effect. Its lower endpoint also
exceeds the predeclared +0.5 pp threshold, so the primary-metric rule passes in
this simulated run. That is not an automatic product launch: guardrail
metrics, implementation risk, cost, and business constraints remain separate
inputs to a real decision.

## Coverage is a different question

One successful interval cannot establish 95% coverage. The notebook includes
a fast 100-trial smoke check that exercises the known-truth loop and reports
its Monte Carlo standard error. Its deterministic result is 94 covered trials
out of 100, with about 2.4 percentage points of Monte Carlo standard error.

This small check is deliberately not used as release evidence. The larger,
versioned studies in [Benchmarks](benchmarks.md) contain the coverage claims,
failure accounting, and reproducible metadata.

## Run it

From a bootstrapx checkout:

```bash
python -m pip install -e ".[pandas]" jupyter
jupyter lab notebooks/07_product_ab_reference.ipynb
```

The notebook generates all data locally and does not access the network.

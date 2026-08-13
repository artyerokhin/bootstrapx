# Benchmarks

Benchmarks answer three different questions and should not be mixed:

1. **Runtime:** how long one configured call takes on one machine.
2. **Working memory:** how much memory that call allocates under a stated tool.
3. **Statistical behavior:** how often an interval covers a known parameter
   across independently generated datasets.

A speed benchmark cannot establish statistical correctness, and increasing
`n_resamples` reduces Monte Carlo noise but does not repair systematic
undercoverage caused by an unsuitable method.

## Audited 0.4.4 quick runtime

Measured on 2026-08-13 from the 0.4.4 release branch on Apple Silicon/macOS
15.7.4, Python 3.11.5, NumPy 1.26.4, and SciPy 1.11.1. Each cell uses
`np.mean`, 4,999 resamples, a fixed seed, and the median of five calls.

| Method | n | bootstrapx (ms) | SciPy (ms) | SciPy / bootstrapx |
|---|---:|---:|---:|---:|
| BCa | 200 | 5.37 | 11.29 | 2.10× |
| BCa | 500 | 14.19 | 15.56 | 1.10× |
| BCa | 1,000 | 33.42 | 31.81 | 0.95× |
| BCa | 2,000 | 61.12 | 69.43 | 1.14× |
| Percentile | 1,000 | 34.17 | 27.77 | 0.81× |
| Percentile | 5,000 | 141.85 | 342.48 | 2.41× |
| Percentile | 10,000 | 230.71 | 703.66 | 3.05× |

A ratio above 1 means bootstrapx was faster in that cell. The crossover is
real: the benchmark does not support a blanket “bootstrapx is faster” claim.
Runtime also depends on dependency versions, CPU, statistic, batch size, and
sample shape.

Auditable inputs: [speed CSV](https://github.com/artyerokhin/bootstrapx/blob/main/benchmarks/baselines/v0.4.4-speed-quick.csv)
and [environment metadata](https://github.com/artyerokhin/bootstrapx/blob/main/benchmarks/baselines/v0.4.4-environment.json).

Arbitrary-callable results were similarly mixed. For `n=1,000`, trimmed mean
took 111.44 ms versus SciPy's 135.14 ms, while IQR took 574.64 ms versus
600.13 ms. These numbers do not predict the cost of another callable.

## Memory measurement

The quick benchmark's `tracemalloc` peak for BCa was 0.225 MB versus 38.147 MB
at `n=500`, and 0.340 MB versus 152.565 MB at `n=2,000`. This supports the
narrow claim that batching greatly reduced allocations visible to
`tracemalloc` in this configuration.

Auditable input: [memory CSV](https://github.com/artyerokhin/bootstrapx/blob/main/benchmarks/baselines/v0.4.4-memory-quick.csv).

It is not a process-RSS or native-allocator guarantee. bootstrapx retains the
bootstrap distribution and method-specific arrays, so total memory is not
constant. A future release benchmark should add peak RSS alongside
`tracemalloc` before stronger memory claims are made.

## Optional Numba acceleration

```bash
pip install -e ".[numba]"
python benchmarks/bench_numba.py
```

This compares the same public `bootstrap` workflow against bootstrapx's Python
fallback. It reports first-process-call latency separately from warm runtime.
Only MBB, CBB, stationary, and the block-index portion of tapered bootstrap use
Numba. See [When Numba helps](integrations.md#when-numba-helps).

## Statistical coverage status

The existing broad coverage CSV and plots were generated before the 0.4.1–0.4.3
correctness fixes. They are useful historical diagnostics but are not valid
release-specific evidence for 0.4.4. They also reused the same integer seed for
data generation and resampling, so the benchmark script must separate those
random streams before the next published run.

Consequently, README and current documentation make no numeric 0.4.4 coverage
claim. The full release-candidate study should report:

- empirical coverage and a Monte Carlo uncertainty interval;
- invalid or failed trials rather than silently hiding them;
- package versions and commit hash;
- independent data-generation and resampling streams;
- all exclusions and finite-sample limitations.

## Reproduce the speed subset

```bash
python benchmarks/bench_speed.py --quick
```

When publishing a new result, record CPU, OS, Python, NumPy, SciPy and Numba
versions, commit hash, `n`, `n_resamples`, statistic, method, block setting,
warm-up policy, repeat count, and memory measurement tool.

## Run the 0.4.4 release suite

Create a clean environment from the release-candidate branch:

```bash
python3 -m venv .venv-bench
source .venv-bench/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,numba]"
```

Start with the quick profile. It verifies every benchmark pipeline and produces
real speed/Numba measurements, but its two-simulation coverage result is only a
smoke test:

```bash
python benchmarks/run_release.py \
  --profile quick \
  --output-dir benchmark_runs/v0.4.4-quick
```

Then run the release profile. It uses 300 datasets per coverage cell and can
take tens of minutes or longer depending on the machine:

```bash
python benchmarks/run_release.py \
  --profile release \
  --output-dir benchmark_runs/v0.4.4-release
```

If the process is interrupted, run the identical command with `--resume`:

```bash
python benchmarks/run_release.py \
  --profile release \
  --output-dir benchmark_runs/v0.4.4-release \
  --resume
```

For publishable statistical estimates, use 1,000 simulations per cell. This is
an overnight-style run rather than a normal CI job:

```bash
python benchmarks/run_release.py \
  --profile statistical \
  --output-dir benchmark_runs/v0.4.4-statistical
```

An optional 2,000-simulation coverage run is available separately:

```bash
python benchmarks/bench_coverage_accuracy.py \
  --full \
  --output-dir benchmark_runs/v0.4.4-coverage-full
```

All profiles preserve previous runs, record the exact environment and commit,
and checkpoint coverage after each configuration. Avoid other CPU-heavy work
during runtime measurements and keep the Mac connected to power. After the run,
share the chosen `benchmark_runs/v0.4.4-*` directory; README tables and plots
should be generated only from that directory.

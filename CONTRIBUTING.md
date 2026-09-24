# Contributing to bootstrapx

## Composite metrics development checks

The 0.6.0 work is unreleased. Install the checkout editable before using its API;
an installed 0.5.1 wheel does not include it. Additional local checks:

```bash
pytest tests/test_analysis_units.py tests/test_assigned_user_example.py tests/test_composite_benchmark.py
ruff check src tests examples benchmarks/bench_composite_metrics.py
python examples/assigned_users_composite_metrics.py
python benchmarks/bench_composite_metrics.py --profile quick --output-dir benchmark_runs/composite-quick
```

Example tests require pandas and skip when it is absent. Benchmark smoke/resume
checks require only core SciPy/NumPy. See the benchmark guide before long runs.

Thanks for helping make bootstrapx more useful for applied data science.

## Before opening a change

- Use a GitHub issue for statistical-method proposals or public API changes.
- Describe the practitioner workflow, assumptions, and expected output before
  proposing a new method.
- For correctness bugs, include the smallest reproducible example and, where
  possible, a reference implementation or statistical source.

## Local setup

```bash
git clone https://github.com/artyerokhin/bootstrapx.git
cd bootstrapx
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,pandas,sklearn,numba,docs]"
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Required checks

Run these before opening a pull request:

```bash
ruff format --check src tests benchmarks/bench_speed.py \
  benchmarks/bench_coverage_accuracy.py benchmarks/bench_numba.py \
  benchmarks/run_release.py benchmarks/plot_release_results.py \
  benchmarks/bench_two_sample.py benchmarks/bench_two_sample_coverage.py \
  benchmarks/run_comparison_release.py benchmarks/plot_comparison_results.py
ruff check src tests benchmarks/bench_speed.py \
  benchmarks/bench_coverage_accuracy.py benchmarks/bench_numba.py \
  benchmarks/run_release.py benchmarks/plot_release_results.py \
  benchmarks/bench_two_sample.py benchmarks/bench_two_sample_coverage.py \
  benchmarks/run_comparison_release.py benchmarks/plot_comparison_results.py
mypy src
pytest --cov=bootstrapx --cov-fail-under=85
pytest --doctest-modules src/bootstrapx
mkdocs build --strict
```

Also check typing with the optional Numba extra installed. CI covers both
the core and Numba-enabled typing environments.

Execute the offline release notebooks without changing their tracked outputs:

```bash
pip install nbclient nbformat ipykernel
BOOTSTRAPX_RUN_NOTEBOOKS=1 pytest tests/test_notebook_execution.py -k offline
```

The external Hillstrom download is checked separately to avoid tying ordinary
PR checks to the availability of its host:

```bash
pip install matplotlib
BOOTSTRAPX_RUN_NETWORK_NOTEBOOKS=1 pytest tests/test_notebook_execution.py -k hillstrom
```

On Windows, set the environment variable in PowerShell with
`$env:BOOTSTRAPX_RUN_NOTEBOOKS="1"` (or the corresponding network variable).
The real-data check is also available through the manually dispatched
**Verify real-data notebook** GitHub Actions workflow.

New statistical behavior should include both focused regression tests and a
simulation or invariant that demonstrates correctness. Avoid assertions that
depend on unstable wording from third-party libraries.

Release benchmark runs belong under ignored `benchmark_runs/` directories. Use
`python benchmarks/run_release.py --profile quick` for a pipeline check;
version only reviewed release evidence, together with its metadata and the
figures generated from it.

Changes to experiment comparisons must also run
`python benchmarks/run_comparison_release.py --profile quick`.

## Pull requests

- Keep changes focused and document user-visible behavior in `CHANGELOG.md`.
- Update examples and limitations when an API assumption changes.
- Do not commit generated coverage, build, benchmark-result, or site files.
- Use a separate branch; `main` is protected and accepts changes through PRs.

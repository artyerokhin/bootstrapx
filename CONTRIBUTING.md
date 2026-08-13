# Contributing to bootstrapx

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
  benchmarks/run_release.py
ruff check src tests benchmarks/bench_speed.py \
  benchmarks/bench_coverage_accuracy.py benchmarks/bench_numba.py \
  benchmarks/run_release.py
mypy src
pytest --cov=bootstrapx --cov-fail-under=85
pytest --doctest-modules src/bootstrapx
mkdocs build --strict
```

New statistical behavior should include both focused regression tests and a
simulation or invariant that demonstrates correctness. Avoid assertions that
depend on unstable wording from third-party libraries.

Release benchmark runs belong under ignored `benchmark_runs/` directories.
Use `python benchmarks/run_release.py --profile quick` for a pipeline check;
version only reviewed release baselines, together with their metadata.

## Pull requests

- Keep changes focused and document user-visible behavior in `CHANGELOG.md`.
- Update examples and limitations when an API assumption changes.
- Do not commit generated coverage, build, benchmark-result, or site files.
- Use a separate branch; `main` is protected and accepts changes through PRs.

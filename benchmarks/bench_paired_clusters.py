"""Known-truth coverage and matched runtime checks for paired whole-cluster draws.

Source-only research: prints results and progress, saves no artifacts. Includes
equal-sized Gaussian clusters and informative sizes with a known row-weighted
target. This is not a promise of uniform coverage or speed advantage.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

import bootstrapx
from bootstrapx import bootstrap_two_sample


def sample(rng: np.random.Generator, groups: int, informative: bool):
    effects = rng.normal(size=groups)
    sizes = np.where(effects > 0, 6, 2) if informative else np.full(groups, 5)
    ids = np.repeat(np.arange(groups), sizes)
    shared = np.repeat(rng.normal(0, 3, size=groups), sizes)
    a = shared + rng.normal(size=len(ids))
    b = a + 0.3 + np.repeat(effects, sizes) + rng.normal(0, 0.3, size=len(ids))
    truth = 0.3 + 1 / math.sqrt(2 * math.pi) if informative else 0.3
    return a, b, ids, truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--resamples", type=int, default=399)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    if args.trials < 10 or args.resamples < 99:
        parser.error("Use at least 10 datasets and 99 resamples.")
    print(f"seed={args.seed}; trials={args.trials}; resamples={args.resamples}", flush=True)
    print(
        f"Python={platform.python_version()}; platform={platform.platform()}; "
        f"bootstrapx={bootstrapx.__version__}; numpy={np.__version__}; scipy={scipy.__version__}"
    )
    package = Path(bootstrapx.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    print(f"package_source_sha256={digest.hexdigest()}")
    print(f"runner_sha256={hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}", flush=True)
    rng = np.random.default_rng(args.seed)
    for groups, informative in ((20, False), (100, False), (100, True)):
        hits = {"percentile": 0, "basic": 0}
        start = time.perf_counter()
        for trial in range(args.trials):
            a, b, ids, truth = sample(rng, groups, informative)
            result = bootstrap_two_sample(
                a,
                b,
                np.mean,
                paired=True,
                paired_cluster_ids=ids,
                method="percentile",
                n_resamples=args.resamples,
                random_state=rng,
            )
            for method in hits:
                ci = result.interval(method=method)
                hits[method] += int(ci.low <= truth <= ci.high)
            if (trial + 1) % 100 == 0:
                print(
                    f"groups={groups}, informative={informative}: {trial + 1} datasets", flush=True
                )
        print(
            f"groups={groups}; informative={informative}; seconds={time.perf_counter() - start:.2f}"
        )
        for method, count in hits.items():
            p = count / args.trials
            half = 1.96 * math.sqrt(p * (1 - p) / args.trials)
            print(f"  {method}: coverage={p:.4f}, MC 95% half-width={half:.4f}", flush=True)

    # Identical estimand and whole-cluster resampling, distinct RNG streams.
    a, b, ids, truth = sample(rng, 100, True)
    rows = [np.flatnonzero(ids == group) for group in np.unique(ids)]

    def reference_metric(selected):
        indices = np.concatenate([rows[int(group)] for group in selected])
        return float(b[indices].mean() - a[indices].mean())

    for name in ("bootstrapx", "scipy"):
        durations = []
        for _ in range(3):
            start = time.perf_counter()
            if name == "bootstrapx":
                output = bootstrap_two_sample(
                    a,
                    b,
                    np.mean,
                    paired=True,
                    paired_cluster_ids=ids,
                    method="percentile",
                    n_resamples=args.resamples,
                    random_state=42,
                )
            else:
                output = stats.bootstrap(
                    (np.arange(len(rows)),),
                    reference_metric,
                    vectorized=False,
                    method="percentile",
                    n_resamples=args.resamples,
                    random_state=42,
                )
            durations.append(time.perf_counter() - start)
        print(
            f"{name}: median matched runtime={np.median(durations):.4f}s; "
            f"CI={output.confidence_interval}"
        )


if __name__ == "__main__":
    main()

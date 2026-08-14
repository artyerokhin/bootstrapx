#!/usr/bin/env python3
"""Run benchmark suites sequentially so they do not compete for CPU or memory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _tracked_worktree_dirty() -> bool:
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    )
    return bool(output.strip())


def _verify_metadata(path: Path, expected_version: str, expected_commit: str) -> None:
    metadata = json.loads(path.read_text())
    if metadata.get("bootstrapx") != expected_version:
        raise RuntimeError(
            f"{path} measured bootstrapx {metadata.get('bootstrapx')!r}; "
            f"expected {expected_version!r}"
        )
    if metadata.get("git_commit") != expected_commit:
        raise RuntimeError(
            f"{path} measured commit {metadata.get('git_commit')!r}; expected {expected_commit!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("quick", "release", "statistical"),
        default="quick",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip finished speed/Numba suites and resume coverage checkpoints.",
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = (args.output_dir or Path(f"benchmark_runs/v0.4.4-{args.profile}-{stamp}")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    benchmark_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(benchmark_dir.parent / "src"))
    from bootstrapx import __version__

    commit = _git_commit()
    if args.profile != "quick" and _tracked_worktree_dirty():
        parser.error("release/statistical benchmarks require a clean tracked worktree")

    speed_dir = root / "speed"
    if not (args.resume and (speed_dir / "metadata.json").exists()):
        _run(
            [
                python,
                str(benchmark_dir / "bench_speed.py"),
                "--quick",
                "--output-dir",
                str(speed_dir),
            ]
        )
    else:
        print(f"Skipping completed speed suite: {speed_dir}")
    _verify_metadata(speed_dir / "metadata.json", __version__, commit)

    coverage_dir = root / "coverage"
    coverage_flag = "--smoke" if args.profile == "quick" else "--fast"
    if args.profile == "statistical":
        coverage_flag = ""
    coverage_command = [
        python,
        str(benchmark_dir / "bench_coverage_accuracy.py"),
        "--output-dir",
        str(coverage_dir),
    ]
    if coverage_flag:
        coverage_command.append(coverage_flag)
    if args.resume:
        coverage_command.append("--resume")
    _run(coverage_command)
    _verify_metadata(coverage_dir / "metadata.json", __version__, commit)

    numba_dir = root / "numba"
    if not (args.resume and (numba_dir / "metadata.json").exists()):
        n_resamples = "100" if args.profile == "quick" else "500"
        repeats = "3" if args.profile == "quick" else "5"
        _run(
            [
                python,
                str(benchmark_dir / "bench_numba.py"),
                "--n-resamples",
                n_resamples,
                "--repeats",
                repeats,
                "--output-dir",
                str(numba_dir),
            ]
        )
    else:
        print(f"Skipping completed Numba suite: {numba_dir}")
    _verify_metadata(numba_dir / "metadata.json", __version__, commit)

    print(f"\nAll requested benchmark results are in: {root}")


if __name__ == "__main__":
    main()

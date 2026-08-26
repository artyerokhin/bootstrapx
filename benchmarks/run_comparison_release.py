#!/usr/bin/env python3
"""Run 0.5.0 comparison benchmarks sequentially and verify their metadata."""

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
    parser.add_argument("--profile", choices=("quick", "release", "statistical"), default="quick")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = (args.output_dir or Path(f"benchmark_runs/v0.5.0-{args.profile}-{stamp}")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    benchmark_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(benchmark_dir.parent / "src"))
    from bootstrapx import __version__

    commit = _git_commit()
    if args.profile != "quick" and _tracked_worktree_dirty():
        parser.error("release/statistical benchmarks require a clean tracked worktree")

    runtime_dir = root / "runtime"
    if not (args.resume and (runtime_dir / "metadata.json").exists()):
        command = [
            python,
            str(benchmark_dir / "bench_two_sample.py"),
            "--output-dir",
            str(runtime_dir),
        ]
        if args.profile == "quick":
            command.append("--quick")
        _run(command)
    _verify_metadata(runtime_dir / "metadata.json", __version__, commit)

    coverage_dir = root / "coverage"
    coverage_command = [
        python,
        str(benchmark_dir / "bench_two_sample_coverage.py"),
        "--output-dir",
        str(coverage_dir),
    ]
    if args.profile == "quick":
        coverage_command.append("--smoke")
    elif args.profile == "release":
        coverage_command.append("--release")
    else:
        coverage_command.append("--statistical")
    if args.resume:
        coverage_command.append("--resume")
    _run(coverage_command)
    _verify_metadata(coverage_dir / "metadata.json", __version__, commit)
    print(f"\nAll requested 0.5.0 benchmark results are in: {root}")


if __name__ == "__main__":
    main()

"""Opt-in execution of unmodified release notebook cells in fresh kernels."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OFFLINE_NOTEBOOKS = [
    "04_ab_test_cluster_bootstrap.ipynb",
    "07_product_ab_reference.ipynb",
]


def _execute_notebook(name: str, checks: str) -> None:
    # These tools are required only for the explicitly selected execution
    # checks; ordinary core/minimum-dependency tests need no Jupyter stack.
    import nbformat
    from jupyter_client import KernelManager
    from nbclient import NotebookClient

    notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    notebook.cells.append(nbformat.v4.new_code_cell(checks))
    manager = KernelManager(kernel_name="python3")
    # A user-level python3 kernelspec may point at a different environment.
    # Select this test process's interpreter without registering a kernel.
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    NotebookClient(
        notebook,
        km=manager,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    ).execute(cleanup_kc=True)
    # The executed notebook stays in memory: never rewrite tracked outputs.


@pytest.mark.skipif(
    os.environ.get("BOOTSTRAPX_RUN_NOTEBOOKS") != "1",
    reason="Set BOOTSTRAPX_RUN_NOTEBOOKS=1 to execute offline notebooks.",
)
@pytest.mark.parametrize("name", OFFLINE_NOTEBOOKS)
def test_offline_notebook_executes(name: str) -> None:
    if name.startswith("04_"):
        checks = """
assert iid.estimate == clustered.estimate
assert clustered.confidence_interval.width > iid.confidence_interval.width
assert clustered.n_control_clusters == n_control_users
assert clustered.n_treatment_clusters == n_treatment_users
assert np.isfinite(clustered.bootstrap_distribution).all()
"""
    else:
        checks = """
assert np.isclose(result.control_estimate, 0.09884)
assert np.isclose(result.treatment_estimate, 0.11058)
assert np.isclose(result.estimate, 0.01174)
assert abs(interval.low - 0.007893) < 0.000005
assert abs(interval.high - 0.01546) < 0.000005
assert metric_rule_passes
assert interval.low < TRUE_EFFECT < interval.high
assert covered == 94
"""
    _execute_notebook(name, checks)


@pytest.mark.skipif(
    os.environ.get("BOOTSTRAPX_RUN_NETWORK_NOTEBOOKS") != "1",
    reason="Set BOOTSTRAPX_RUN_NETWORK_NOTEBOOKS=1 to check the external dataset.",
)
def test_hillstrom_notebook_executes_with_verified_source() -> None:
    _execute_notebook(
        "06_real_world_ab_hillstrom.ipynb",
        """
assert sha256(data_path) == EXPECTED_SHA256
assert len(data) == 64_000
assert len(analyses) == 4
assert np.isfinite(results[["effect", "ci_low", "ci_high"]].to_numpy()).all()
assert (results["ci_low"] <= results["ci_high"]).all()
""",
    )

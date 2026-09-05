"""Lightweight contracts for release-facing notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "notebooks" / "04_ab_test_cluster_bootstrap.ipynb",
    ROOT / "notebooks" / "06_real_world_ab_hillstrom.ipynb",
]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_release_notebook_code_compiles_without_recorded_errors(path: Path) -> None:
    notebook = json.loads(path.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert code_cells
    for index, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        compile(source, f"{path}:cell-{index}", "exec")
        assert not [
            output for output in cell.get("outputs", []) if output["output_type"] == "error"
        ]


def test_hillstrom_notebook_pins_source_and_does_not_embed_raw_data() -> None:
    path = ROOT / "notebooks" / "06_real_world_ab_hillstrom.ipynb"
    source = path.read_text()

    assert "Kevin_Hillstrom_MineThatData" in source
    assert "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece" in source
    assert "bootstrap_two_sample" in source
    assert path.stat().st_size < 200_000

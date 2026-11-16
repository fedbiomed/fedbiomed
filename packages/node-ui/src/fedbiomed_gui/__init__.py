# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Fed-BioMed Node Graphical User Interface."""

__version__ = "v6.5.0"

from pathlib import Path


def get_frontend_source_dir() -> Path:
    """Find React sources in an editable source checkout."""
    for root in Path(__file__).resolve().parents:
        ui_dir = root / "ui"
        if (root / "pyproject.toml").is_file() and (ui_dir / "package.json").is_file():
            return ui_dir

    raise FileNotFoundError(
        "React sources are unavailable in this installation. "
        "Rebuild the GUI from a source checkout."
    )

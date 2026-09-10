"""Framework-neutral launch environment projection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, MutableMapping

__all__ = ["prepend_python_path", "repository_root"]


def prepend_python_path(
    environment: MutableMapping[str, str], path: Path
) -> None:
    """Prepend one import root without leaving duplicate path entries."""

    current = environment.get("PYTHONPATH", "")
    values = [item for item in current.split(os.pathsep) if item]
    path_text = str(path)
    if path_text in values:
        values.remove(path_text)
    environment["PYTHONPATH"] = os.pathsep.join([path_text, *values])


def repository_root(environment: Mapping[str, str] | None = None) -> Path | None:
    """Find optional bundled sources without assuming an editable installation."""
    environment = os.environ if environment is None else environment
    configured = environment.get("MOEGAMBIT_REPOSITORY_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("MOEGAMBIT_REPOSITORY_ROOT must be an existing directory")
        return root
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "run_spare_single_rank.sh").is_file():
        return candidate
    return None

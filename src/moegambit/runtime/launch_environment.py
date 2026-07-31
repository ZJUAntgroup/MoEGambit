"""Framework-neutral launch environment projection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping

__all__ = ["prepend_python_path"]


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

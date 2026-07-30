"""Expose the canonical runtime and optional framework plugins to tests."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE_ROOT = REPOSITORY_ROOT / "src"

for source_root in (RUNTIME_SOURCE_ROOT,):
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)

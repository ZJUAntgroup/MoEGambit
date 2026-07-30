#!/usr/bin/env python3
"""Compatibility launcher that dispatches to a framework adapter."""

from __future__ import annotations

import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from moegambit.cli.elastic_compat import launcher_main as main
from moegambit.adapters.megatron.compat_launcher import LauncherControlAgent

__all__ = ["LauncherControlAgent", "main"]


if __name__ == "__main__":
    raise SystemExit(main())

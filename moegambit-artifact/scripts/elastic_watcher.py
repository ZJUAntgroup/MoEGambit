#!/usr/bin/env python3
"""Compatibility entry point for existing experiment scripts."""

from pathlib import Path
import runpy


runpy.run_path(
    Path(__file__).resolve().parents[1] / "src" / "elastic" / "elastic_watcher.py",
    run_name="__main__",
)

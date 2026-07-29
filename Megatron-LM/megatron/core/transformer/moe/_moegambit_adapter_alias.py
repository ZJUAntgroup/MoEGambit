"""Internal helper for one-release compatibility module aliases."""

from __future__ import annotations

import importlib
import sys


def install_alias(module_name: str, leaf: str):
    implementation = importlib.import_module(
        f"moegambit.adapters.megatron.moe.{leaf}"
    )
    sys.modules[module_name] = implementation
    return implementation

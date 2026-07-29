"""Lazy framework-adapter access.

The core package does not import framework implementations during discovery.
Concrete adapters are registered only when their implementation phase lands.
"""

from __future__ import annotations

from typing import Any

__all__ = ["FrameworkAdapter", "get_adapter", "available_adapters", "register_adapter"]


def __getattr__(name: str) -> Any:
    if name == "FrameworkAdapter":
        from .base import FrameworkAdapter

        return FrameworkAdapter
    if name in ("get_adapter", "available_adapters", "register_adapter"):
        from . import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

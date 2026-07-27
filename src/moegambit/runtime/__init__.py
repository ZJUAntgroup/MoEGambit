"""Recovery plans and lifecycle execution."""

from __future__ import annotations

from typing import Any

__all__ = ["RecoveryRuntime", "initialize"]


def __getattr__(name: str) -> Any:
    if name in ("RecoveryRuntime", "initialize"):
        from . import runtime

        return getattr(runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

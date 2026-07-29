"""Recovery plans and lifecycle execution."""

from __future__ import annotations

from typing import Any

__all__ = [
    "GroupManifest",
    "GroupSpec",
    "NoopOrdinalBarrier",
    "RecoveryRuntime",
    "RuntimeConfig",
    "TorchDistributedProtocol",
    "WatcherOrdinalBarrier",
    "discover_adapters",
    "initialize",
    "load_adapter",
    "wait_for_recovery_group_barrier",
]


def __getattr__(name: str) -> Any:
    if name in ("RecoveryRuntime", "initialize"):
        from . import runtime

        return getattr(runtime, name)
    if name == "RuntimeConfig":
        from .config import RuntimeConfig

        return RuntimeConfig
    if name in ("discover_adapters", "load_adapter"):
        from . import discovery

        return getattr(discovery, name)
    if name in (
        "GroupManifest",
        "GroupSpec",
        "NoopOrdinalBarrier",
        "TorchDistributedProtocol",
        "WatcherOrdinalBarrier",
        "wait_for_recovery_group_barrier",
    ):
        from . import distributed

        return getattr(distributed, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

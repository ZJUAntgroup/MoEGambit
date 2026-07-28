"""Generic runtime orchestration and wire protocols."""

from moegambit.runtime.config import RuntimeConfig
from moegambit.runtime.discovery import discover_adapters, load_adapter
from moegambit.runtime.distributed import (
    GroupManifest,
    GroupSpec,
    NoopOrdinalBarrier,
    TorchDistributedProtocol,
    WatcherOrdinalBarrier,
    wait_for_recovery_group_barrier,
)

__all__ = [
    "GroupManifest",
    "GroupSpec",
    "NoopOrdinalBarrier",
    "RuntimeConfig",
    "TorchDistributedProtocol",
    "WatcherOrdinalBarrier",
    "discover_adapters",
    "load_adapter",
    "wait_for_recovery_group_barrier",
]

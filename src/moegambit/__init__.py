"""MoEGambit: framework-agnostic elastic recovery for distributed training.

Import discipline is part of the public contract:

* importing :mod:`moegambit` never imports torch or a training framework;
* control-plane processes can run on GPU-less hosts;
* framework internals are confined to ``moegambit.adapters.<framework>``.

Phase B exposes configuration, capabilities, and typed errors lazily.  Runtime
construction is added in Phase C once the recovery executor exists.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0.dev0"

# The control-protocol version is deliberately independent of the package
# version.  A wire-incompatible change must bump this value.
PROTOCOL_VERSION = 1

__all__ = [
    "__version__",
    "PROTOCOL_VERSION",
    "AdapterCapabilities",
    "RuntimeConfig",
    "MoEGambitError",
]


def __getattr__(name: str) -> Any:
    if name == "AdapterCapabilities":
        from .capabilities import AdapterCapabilities

        return AdapterCapabilities
    if name == "RuntimeConfig":
        from .config import RuntimeConfig

        return RuntimeConfig
    if name == "MoEGambitError":
        from .errors import MoEGambitError

        return MoEGambitError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

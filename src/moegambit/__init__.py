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

from ._package_identity import assert_unambiguous_package as _check_package_identity

_check_package_identity(__file__)
del _check_package_identity

__version__ = "0.1.0.dev0"

# Callers that require the canonical core can verify this sentinel after
# import.  The temporary DeepSpeed-owned compatibility package does not expose
# it, so a wrong-root import fails an explicit bootstrap assertion.
PACKAGE_ROLE = "core"

# The control-protocol version is deliberately independent of the package
# version.  A wire-incompatible change must bump this value.
PROTOCOL_VERSION = 1

__all__ = [
    "__version__",
    "PROTOCOL_VERSION",
    "PACKAGE_ROLE",
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

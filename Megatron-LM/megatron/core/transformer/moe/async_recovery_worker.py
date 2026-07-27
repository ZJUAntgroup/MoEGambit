"""Compatibility alias for packaged MoEGambit module async_recovery_worker."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "async_recovery_worker")

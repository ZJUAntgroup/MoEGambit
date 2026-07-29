"""Compatibility alias for packaged MoEGambit module deferred_optimizer_load."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "deferred_optimizer_load")

"""Compatibility alias for packaged MoEGambit module stale_expert_restore."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "stale_expert_restore")

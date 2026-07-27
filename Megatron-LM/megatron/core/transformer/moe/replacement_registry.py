"""Compatibility alias for packaged MoEGambit module replacement_registry."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "replacement_registry")

"""Compatibility alias for packaged MoEGambit module group_rebuild."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "group_rebuild")

"""Compatibility alias for packaged MoEGambit module expert_directory."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "expert_directory")

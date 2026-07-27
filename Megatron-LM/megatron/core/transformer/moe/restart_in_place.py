"""Compatibility alias for packaged MoEGambit module restart_in_place."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "restart_in_place")

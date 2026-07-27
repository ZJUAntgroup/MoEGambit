"""Compatibility alias for packaged MoEGambit module recovery_controller."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "recovery_controller")

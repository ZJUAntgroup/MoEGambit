"""Compatibility alias for packaged MoEGambit module stage_safe_recovery."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "stage_safe_recovery")

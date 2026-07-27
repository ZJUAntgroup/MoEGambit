"""Compatibility alias for packaged MoEGambit module two_phase_recovery."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "two_phase_recovery")

"""Compatibility alias for packaged MoEGambit module fault_injection_framework."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "fault_injection_framework")

"""Compatibility alias for packaged MoEGambit module reintegration_barrier."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "reintegration_barrier")

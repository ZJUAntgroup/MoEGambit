"""Compatibility alias for packaged MoEGambit module iteration_invalidator."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "iteration_invalidator")

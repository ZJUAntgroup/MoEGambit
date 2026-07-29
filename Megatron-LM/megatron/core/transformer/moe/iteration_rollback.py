"""Compatibility alias for packaged MoEGambit module iteration_rollback."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "iteration_rollback")

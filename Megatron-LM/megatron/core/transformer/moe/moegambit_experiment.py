"""Compatibility alias for packaged MoEGambit module moegambit_experiment."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "moegambit_experiment")

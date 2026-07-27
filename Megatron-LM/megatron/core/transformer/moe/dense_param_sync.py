"""Compatibility alias for packaged MoEGambit module dense_param_sync."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "dense_param_sync")

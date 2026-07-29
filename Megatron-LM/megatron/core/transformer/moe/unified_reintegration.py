"""Compatibility alias for packaged MoEGambit module unified_reintegration."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "unified_reintegration")

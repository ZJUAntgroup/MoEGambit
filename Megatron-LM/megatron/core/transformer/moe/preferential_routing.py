"""Compatibility alias for packaged MoEGambit module preferential_routing."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "preferential_routing")

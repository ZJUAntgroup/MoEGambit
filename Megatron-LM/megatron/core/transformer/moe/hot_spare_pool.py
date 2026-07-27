"""Compatibility alias for packaged MoEGambit module hot_spare_pool."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "hot_spare_pool")

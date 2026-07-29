"""Compatibility alias for packaged MoEGambit module safe_point_group_repair."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "safe_point_group_repair")

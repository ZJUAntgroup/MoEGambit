"""Compatibility alias for packaged MoEGambit module dispatch_topology_refresh."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "dispatch_topology_refresh")

"""Compatibility alias for packaged MoEGambit module gap_aware_recovery_policy."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "gap_aware_recovery_policy")

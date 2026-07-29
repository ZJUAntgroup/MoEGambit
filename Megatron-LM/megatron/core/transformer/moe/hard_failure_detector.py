"""Compatibility alias for packaged MoEGambit module hard_failure_detector."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "hard_failure_detector")

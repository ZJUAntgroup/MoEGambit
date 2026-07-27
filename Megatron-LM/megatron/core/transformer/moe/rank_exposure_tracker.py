"""Compatibility alias for packaged MoEGambit module rank_exposure_tracker."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "rank_exposure_tracker")

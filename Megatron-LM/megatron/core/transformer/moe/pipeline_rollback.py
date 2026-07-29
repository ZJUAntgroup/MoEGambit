"""Compatibility alias for packaged MoEGambit module pipeline_rollback."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "pipeline_rollback")

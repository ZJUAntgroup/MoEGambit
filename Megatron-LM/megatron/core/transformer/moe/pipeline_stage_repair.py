"""Compatibility alias for packaged MoEGambit module pipeline_stage_repair."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "pipeline_stage_repair")

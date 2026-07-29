"""Compatibility alias for packaged MoEGambit module moc_pec_emulation."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "moc_pec_emulation")

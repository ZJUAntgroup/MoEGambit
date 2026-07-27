"""Compatibility alias for packaged MoEGambit module optimizer_commit_guard."""

from ._moegambit_adapter_alias import install_alias

_implementation = install_alias(__name__, "optimizer_commit_guard")

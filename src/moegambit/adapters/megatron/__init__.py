"""Megatron adapter, loaded only when explicitly selected."""

from .adapter import MEGATRON_CAPABILITIES, MegatronAdapterPlugin, build_megatron_adapter
from .engine import MegatronEngineAdapter

__all__ = [
    "MEGATRON_CAPABILITIES",
    "MegatronAdapterPlugin",
    "MegatronEngineAdapter",
    "build_megatron_adapter",
]

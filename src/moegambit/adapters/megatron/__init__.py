"""Megatron adapter, loaded only when explicitly selected."""

from .adapter import MEGATRON_CAPABILITIES, MegatronAdapterPlugin, build_megatron_adapter

__all__ = [
    "MEGATRON_CAPABILITIES",
    "MegatronAdapterPlugin",
    "build_megatron_adapter",
]

"""DeepSpeed adapter, loaded only when explicitly selected."""

from .engine import (
    DeepSpeedAdapter,
    DeepSpeedEngineAdapter,
    attach_engine,
)

__all__ = [
    "DeepSpeedAdapter",
    "DeepSpeedEngineAdapter",
    "attach_engine",
]

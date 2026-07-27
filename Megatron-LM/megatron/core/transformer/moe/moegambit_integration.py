"""Compatibility alias for the packaged Megatron MoE integration."""

from __future__ import annotations

import sys
import warnings

from moegambit.adapters.megatron import moe_integration as _implementation

warnings.warn(
    "megatron.core.transformer.moe.moegambit_integration is deprecated; "
    "use moegambit.adapters.megatron.moe_integration",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _implementation

"""Compatibility alias for the packaged Megatron recovery implementation.

Remove after the documented one-release deprecation window.  Replacing the
module object (rather than copying names) preserves private patch points used
by the existing Megatron integration while making the package implementation
the single source of truth.
"""

from __future__ import annotations

import sys
import warnings

from moegambit.adapters.megatron import elastic_client as _implementation

warnings.warn(
    "megatron.training.elastic_client is deprecated; recovery now lives in "
    "moegambit.adapters.megatron.elastic_client",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _implementation

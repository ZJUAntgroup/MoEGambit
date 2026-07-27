"""Explicit capability declaration for framework adapters.

Section 8.1 of the design doc.  Adapters must *declare* what they support.
The core is forbidden from probing with ``hasattr`` and guessing: a missing
capability has to produce a compatible fallback, never a partial recovery.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import FrozenSet, Mapping

__all__ = ["AdapterCapabilities", "SupportLevel"]


class SupportLevel(str):
    """Support tier reported by ``moegambit doctor`` (design doc 17.2)."""

    SUPPORTED = "supported"
    EXPERIMENTAL = "experimental"
    DETECTION_ONLY = "detection-only"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class AdapterCapabilities:
    """What a framework adapter can actually do.

    ``digest()`` is compared across every rank before a RecoveryPlan is
    generated.  Divergent capability sets mean the participants disagree about
    what recovery is even possible, which is a contract violation rather than
    something to negotiate at runtime.
    """

    static_world_replacement: bool = False
    selective_group_rebuild: bool = False
    full_group_rebuild: bool = False
    optimizer_memory_replication: bool = False
    peer_parameter_restore: bool = False
    moe_state_classification: bool = False
    two_phase_optimizer_restore: bool = False
    supported_zero_stages: FrozenSet[int] = field(default_factory=frozenset)
    supported_parallel_axes: FrozenSet[str] = field(default_factory=frozenset)

    def as_dict(self) -> Mapping[str, object]:
        return {
            "static_world_replacement": self.static_world_replacement,
            "selective_group_rebuild": self.selective_group_rebuild,
            "full_group_rebuild": self.full_group_rebuild,
            "optimizer_memory_replication": self.optimizer_memory_replication,
            "peer_parameter_restore": self.peer_parameter_restore,
            "moe_state_classification": self.moe_state_classification,
            "two_phase_optimizer_restore": self.two_phase_optimizer_restore,
            "supported_zero_stages": sorted(self.supported_zero_stages),
            "supported_parallel_axes": sorted(self.supported_parallel_axes),
        }

    def digest(self) -> str:
        """Stable hash used for the cross-rank capability agreement check."""
        blob = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def missing(self, required: "AdapterCapabilities") -> tuple:
        """Return the boolean capabilities ``required`` needs but we lack."""
        gaps = []
        for name, wanted in required.as_dict().items():
            if wanted is True and self.as_dict().get(name) is not True:
                gaps.append(name)
        return tuple(gaps)

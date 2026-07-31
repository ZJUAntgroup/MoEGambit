# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replacement Rank Registration Protocol for MOEGAMBIT-MoE.

This module implements a minimal protocol for registering spare / replacement
ranks that will take over the logical role of a failed rank.  The protocol
ensures that a replacement rank does **not** immediately join normal training
collectives — it must first register its readiness, wait for a safe-point
acknowledgement, and only then be integrated into the active training group.

State machine
-------------
Each replacement slot (keyed by the ``failed_rank`` it is replacing) goes
through the following states::

    NOT_PRESENT ──announce──▶ BOOTSTRAPPING ──ready──▶ READY_FOR_REPAIR
                                                            │
                                                      safe-point
                                                            │
                                                            ▼
                                                       INTEGRATED

* **NOT_PRESENT** — no replacement has been announced for this failed rank.
* **BOOTSTRAPPING** — a replacement rank has been announced and is performing
  initial setup (loading model skeleton, pulling dense parameters from healthy
  DP peers, loading stale expert weights from checkpoint).  It must NOT
  participate in any training collective.
* **READY_FOR_REPAIR** — the replacement rank has finished bootstrapping and
  is ready to be integrated.  It is still NOT participating in collectives.
  The system waits for a safe-point before proceeding.
* **INTEGRATED** — the replacement rank has been accepted at a safe-point and
  is now a full participant in training collectives.  The slot can be cleaned
  up after this.

Design principles
-----------------
* **No cluster scheduler** — this is a training-system-internal protocol.
  The caller (fault detector, job manager, etc.) is responsible for spawning
  the replacement process and calling ``announce_replacement()``.
* **Safe-point gating** — the ``mark_integrated()`` call should only happen
  inside a safe-point (e.g. after a ``torch.distributed.barrier()`` and
  before the next forward pass).
* **Role inheritance** — the replacement rank inherits the failed rank's
  logical role: its parallel-dimension coordinates (TP/DP/PP/EP ranks),
  its expert host mapping, and its recovery manifest / directory entry.
* **Integration with existing MOEGAMBIT-MoE stack**:
  - ``RankQuarantineRegistry`` (Step 4) quarantines the failed rank.
  - ``ActiveExpertDirectory`` (Step 5) tracks expert ↔ rank mapping.
  - ``RecoveryManifest`` (Step 5) provides checkpoint recovery info.
  - This module (Step 6) manages the replacement lifecycle.

Typical usage::

    from moegambit.adapters.megatron.moe.replacement_registry import (
        announce_replacement,
        announce_replacement_ready,
        query_replacement_status,
        mark_integrated,
    )

    # 1. Fault detected — quarantine the failed rank (Step 4)
    quarantine_rank(failed_rank=4, step=100)

    # 2. Spare worker spawned — announce replacement
    announce_replacement(failed_rank=4, replacement_rank=64, step=100)
    # Status: BOOTSTRAPPING

    # 3. Replacement finishes bootstrap — signal readiness
    announce_replacement_ready(failed_rank=4, step=105)
    # Status: READY_FOR_REPAIR

    # 4. Training loop reaches safe-point — integrate
    mark_integrated(failed_rank=4, step=110)
    # Status: INTEGRATED
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =====================================================================
# Replacement state enum
# =====================================================================

class ReplacementState(enum.IntEnum):
    """Lifecycle states for a replacement slot."""

    NOT_PRESENT = 0
    """No replacement has been announced for this failed rank."""

    BOOTSTRAPPING = 1
    """Replacement rank announced, performing initial setup.
    Must NOT participate in training collectives."""

    READY_FOR_REPAIR = 2
    """Bootstrap complete, waiting for safe-point integration.
    Must NOT participate in training collectives."""

    INTEGRATED = 3
    """Replacement accepted at safe-point, now a full participant."""


# Valid transitions (from → set of allowed destinations).
_VALID_TRANSITIONS: Dict[ReplacementState, Set[ReplacementState]] = {
    ReplacementState.NOT_PRESENT:      {ReplacementState.BOOTSTRAPPING},
    ReplacementState.BOOTSTRAPPING:    {ReplacementState.READY_FOR_REPAIR,
                                        ReplacementState.NOT_PRESENT},  # abort
    ReplacementState.READY_FOR_REPAIR: {ReplacementState.INTEGRATED,
                                        ReplacementState.NOT_PRESENT},  # abort
    ReplacementState.INTEGRATED:       {ReplacementState.NOT_PRESENT},  # cleanup
}


# =====================================================================
# Parallel role descriptor
# =====================================================================

@dataclass
class ParallelRoleDescriptor:
    """Describes the logical parallel-dimension role of a rank.

    This captures the position of a rank within Megatron's multi-dimensional
    parallelism grid.  A replacement rank must inherit exactly this role from
    the failed rank it is replacing.

    Megatron stores these as scattered global variables in ``parallel_state.py``.
    This dataclass provides a unified, serialisable snapshot.
    """

    global_rank: int = -1
    """Global rank in ``torch.distributed``."""

    tp_rank: int = 0
    """Rank within the tensor-model-parallel group."""

    dp_rank: int = 0
    """Rank within the data-parallel group."""

    pp_rank: int = 0
    """Rank within the pipeline-model-parallel group."""

    cp_rank: int = 0
    """Rank within the context-parallel group."""

    ep_rank: int = 0
    """Rank within the expert-model-parallel group."""

    tp_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1

    # Group membership — lists of global ranks in each group this rank belongs to.
    tp_group_ranks: List[int] = field(default_factory=list)
    dp_group_ranks: List[int] = field(default_factory=list)
    pp_group_ranks: List[int] = field(default_factory=list)
    ep_group_ranks: List[int] = field(default_factory=list)

    # Expert hosting info
    expert_ids: List[int] = field(default_factory=list)
    """Global expert IDs hosted on this rank (across all layers)."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "global_rank": self.global_rank,
            "tp_rank": self.tp_rank,
            "dp_rank": self.dp_rank,
            "pp_rank": self.pp_rank,
            "cp_rank": self.cp_rank,
            "ep_rank": self.ep_rank,
            "tp_size": self.tp_size,
            "dp_size": self.dp_size,
            "pp_size": self.pp_size,
            "cp_size": self.cp_size,
            "ep_size": self.ep_size,
            "tp_group_ranks": list(self.tp_group_ranks),
            "dp_group_ranks": list(self.dp_group_ranks),
            "pp_group_ranks": list(self.pp_group_ranks),
            "ep_group_ranks": list(self.ep_group_ranks),
            "expert_ids": list(self.expert_ids),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ParallelRoleDescriptor":
        """Deserialise from a dict."""
        return ParallelRoleDescriptor(
            global_rank=d.get("global_rank", -1),
            tp_rank=d.get("tp_rank", 0),
            dp_rank=d.get("dp_rank", 0),
            pp_rank=d.get("pp_rank", 0),
            cp_rank=d.get("cp_rank", 0),
            ep_rank=d.get("ep_rank", 0),
            tp_size=d.get("tp_size", 1),
            dp_size=d.get("dp_size", 1),
            pp_size=d.get("pp_size", 1),
            cp_size=d.get("cp_size", 1),
            ep_size=d.get("ep_size", 1),
            tp_group_ranks=d.get("tp_group_ranks", []),
            dp_group_ranks=d.get("dp_group_ranks", []),
            pp_group_ranks=d.get("pp_group_ranks", []),
            ep_group_ranks=d.get("ep_group_ranks", []),
            expert_ids=d.get("expert_ids", []),
        )

    def role_key(self) -> str:
        """Return a compact string identifying this role position.

        Format: ``tp{tp_rank}_dp{dp_rank}_pp{pp_rank}_cp{cp_rank}_ep{ep_rank}``
        """
        return (
            f"tp{self.tp_rank}_dp{self.dp_rank}_pp{self.pp_rank}"
            f"_cp{self.cp_rank}_ep{self.ep_rank}"
        )


# =====================================================================
# Replacement slot — per-failed-rank metadata
# =====================================================================

@dataclass
class ReplacementSlot:
    """Metadata for a single replacement operation.

    Each slot is keyed by the ``failed_rank`` it is replacing.
    """

    failed_rank: int
    """The global rank that failed and needs replacement."""

    replacement_rank: int = -1
    """The global rank of the replacement worker.  ``-1`` if not yet assigned."""

    state: ReplacementState = ReplacementState.NOT_PRESENT
    """Current lifecycle state."""

    # Role inheritance
    inherited_role: Optional[ParallelRoleDescriptor] = None
    """The parallel-dimension role inherited from the failed rank."""

    # Timestamps
    announced_step: int = -1
    """Training step when the replacement was announced."""

    ready_step: int = -1
    """Training step when the replacement signalled readiness."""

    integrated_step: int = -1
    """Training step when the replacement was integrated at safe-point."""

    announced_time: float = 0.0
    ready_time: float = 0.0
    integrated_time: float = 0.0

    reason: str = ""
    """Human-readable reason for the replacement."""

    def transition_to(self, new_state: ReplacementState, step: int = 0) -> None:
        """Perform a validated state transition.

        Raises:
            ValueError: If the transition is not allowed.
        """
        allowed = _VALID_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid replacement transition: {self.state.name} → {new_state.name}. "
                f"Allowed: {[s.name for s in allowed]}"
            )
        old = self.state
        self.state = new_state
        now = time.monotonic()

        if new_state == ReplacementState.BOOTSTRAPPING:
            self.announced_step = step
            self.announced_time = now
        elif new_state == ReplacementState.READY_FOR_REPAIR:
            self.ready_step = step
            self.ready_time = now
        elif new_state == ReplacementState.INTEGRATED:
            self.integrated_step = step
            self.integrated_time = now

        logger.info(
            "MOEGAMBIT-MoE replacement: failed_rank=%d, replacement_rank=%d, "
            "%s → %s (step=%d)",
            self.failed_rank, self.replacement_rank, old.name, new_state.name, step,
        )

    def is_participating(self) -> bool:
        """Return ``True`` only if the replacement is INTEGRATED.

        Before integration, the replacement rank must NOT participate in
        training collectives (AllToAll, AllReduce, etc.).
        """
        return self.state == ReplacementState.INTEGRATED

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "failed_rank": self.failed_rank,
            "replacement_rank": self.replacement_rank,
            "state": self.state.name,
            "inherited_role": self.inherited_role.to_dict() if self.inherited_role else None,
            "announced_step": self.announced_step,
            "ready_step": self.ready_step,
            "integrated_step": self.integrated_step,
            "reason": self.reason,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ReplacementSlot":
        """Deserialise from a dict."""
        role_data = d.get("inherited_role")
        role = ParallelRoleDescriptor.from_dict(role_data) if role_data else None
        slot = ReplacementSlot(
            failed_rank=d["failed_rank"],
            replacement_rank=d.get("replacement_rank", -1),
            state=ReplacementState[d.get("state", "NOT_PRESENT")],
            inherited_role=role,
            announced_step=d.get("announced_step", -1),
            ready_step=d.get("ready_step", -1),
            integrated_step=d.get("integrated_step", -1),
            reason=d.get("reason", ""),
        )
        return slot


# =====================================================================
# ReplacementRegistry — singleton per training job
# =====================================================================

class ReplacementRegistry:
    """Global registry of replacement rank operations.

    This is a **process-local** data structure — each surviving rank maintains
    its own copy.  In a real deployment, replacement decisions would be
    broadcast to all surviving ranks before being applied locally.

    The registry tracks:
    * Which failed ranks have pending replacements.
    * The lifecycle state of each replacement.
    * The inherited parallel role for each replacement.
    """

    def __init__(self) -> None:
        self._slots: Dict[int, ReplacementSlot] = {}

    # ------------------------------------------------------------------
    # Core protocol API
    # ------------------------------------------------------------------

    def announce_replacement(
        self,
        failed_rank: int,
        replacement_rank: int,
        step: int = 0,
        reason: str = "",
        role: Optional[ParallelRoleDescriptor] = None,
    ) -> ReplacementSlot:
        """Announce that a replacement rank will take over a failed rank.

        This transitions the slot from NOT_PRESENT → BOOTSTRAPPING.

        Args:
            failed_rank: The global rank that failed.
            replacement_rank: The global rank of the replacement worker.
            step: Current training step.
            reason: Human-readable reason.
            role: The parallel role descriptor inherited from the failed rank.
                If ``None``, the caller must set it later via
                ``set_inherited_role()`` before the replacement can be
                integrated.

        Returns:
            The created ``ReplacementSlot``.

        Raises:
            ValueError: If a replacement is already active for this failed rank.
        """
        if failed_rank in self._slots:
            existing = self._slots[failed_rank]
            if existing.state not in (ReplacementState.NOT_PRESENT,
                                       ReplacementState.INTEGRATED):
                raise ValueError(
                    f"Replacement already active for failed_rank={failed_rank} "
                    f"(state={existing.state.name}, replacement_rank={existing.replacement_rank})"
                )

        slot = ReplacementSlot(
            failed_rank=failed_rank,
            replacement_rank=replacement_rank,
            state=ReplacementState.NOT_PRESENT,
            inherited_role=role,
            reason=reason,
        )
        slot.transition_to(ReplacementState.BOOTSTRAPPING, step)
        self._slots[failed_rank] = slot

        logger.warning(
            "MOEGAMBIT-MoE: replacement announced — failed_rank=%d → replacement_rank=%d "
            "(step=%d, reason=%r)",
            failed_rank, replacement_rank, step, reason,
        )
        return slot

    def announce_replacement_ready(
        self,
        failed_rank: int,
        step: int = 0,
    ) -> ReplacementSlot:
        """Signal that the replacement rank has finished bootstrapping.

        This transitions the slot from BOOTSTRAPPING → READY_FOR_REPAIR.
        The replacement rank is still NOT participating in collectives.

        Args:
            failed_rank: The failed rank being replaced.
            step: Current training step.

        Returns:
            The updated ``ReplacementSlot``.

        Raises:
            KeyError: If no replacement is registered for this failed rank.
            ValueError: If the transition is invalid.
        """
        slot = self._get_slot_or_raise(failed_rank)
        slot.transition_to(ReplacementState.READY_FOR_REPAIR, step)
        return slot

    def mark_integrated(
        self,
        failed_rank: int,
        step: int = 0,
    ) -> ReplacementSlot:
        """Mark the replacement as integrated at a safe-point.

        This transitions the slot from READY_FOR_REPAIR → INTEGRATED.
        After this call, the replacement rank is a full participant in
        training collectives.

        This should ONLY be called inside a safe-point (after barrier,
        before next forward pass).

        Args:
            failed_rank: The failed rank being replaced.
            step: Current training step.

        Returns:
            The updated ``ReplacementSlot``.

        Raises:
            KeyError: If no replacement is registered for this failed rank.
            ValueError: If the transition is invalid.
        """
        slot = self._get_slot_or_raise(failed_rank)
        slot.transition_to(ReplacementState.INTEGRATED, step)

        logger.warning(
            "MOEGAMBIT-MoE: replacement INTEGRATED — failed_rank=%d, "
            "replacement_rank=%d (step=%d)",
            failed_rank, slot.replacement_rank, step,
        )
        return slot

    def abort_replacement(
        self,
        failed_rank: int,
        step: int = 0,
        reason: str = "",
    ) -> Optional[ReplacementSlot]:
        """Abort a pending replacement (BOOTSTRAPPING or READY_FOR_REPAIR → NOT_PRESENT).

        Args:
            failed_rank: The failed rank whose replacement is being aborted.
            step: Current training step.
            reason: Human-readable reason for abort.

        Returns:
            The slot if it was aborted, or ``None`` if no active replacement.
        """
        slot = self._slots.get(failed_rank)
        if slot is None:
            return None
        if slot.state in (ReplacementState.NOT_PRESENT, ReplacementState.INTEGRATED):
            return None

        old_state = slot.state
        slot.transition_to(ReplacementState.NOT_PRESENT, step)
        logger.warning(
            "MOEGAMBIT-MoE: replacement ABORTED — failed_rank=%d, "
            "replacement_rank=%d, was %s (step=%d, reason=%r)",
            failed_rank, slot.replacement_rank, old_state.name, step, reason,
        )
        return slot

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def query_replacement_status(
        self, failed_rank: int
    ) -> ReplacementState:
        """Query the replacement status for a failed rank.

        Returns:
            The current ``ReplacementState``.  Returns ``NOT_PRESENT`` if
            no replacement has been registered.
        """
        slot = self._slots.get(failed_rank)
        if slot is None:
            return ReplacementState.NOT_PRESENT
        return slot.state

    def get_slot(self, failed_rank: int) -> Optional[ReplacementSlot]:
        """Return the replacement slot for a failed rank, or ``None``."""
        return self._slots.get(failed_rank)

    def get_replacement_rank(self, failed_rank: int) -> Optional[int]:
        """Return the replacement rank for a failed rank, or ``None``."""
        slot = self._slots.get(failed_rank)
        if slot is None or slot.replacement_rank < 0:
            return None
        return slot.replacement_rank

    def get_inherited_role(self, failed_rank: int) -> Optional[ParallelRoleDescriptor]:
        """Return the inherited parallel role for a replacement."""
        slot = self._slots.get(failed_rank)
        if slot is None:
            return None
        return slot.inherited_role

    def is_replacement_participating(self, failed_rank: int) -> bool:
        """Return ``True`` only if the replacement is INTEGRATED.

        Before integration, the replacement rank must NOT participate in
        training collectives.
        """
        slot = self._slots.get(failed_rank)
        if slot is None:
            return False
        return slot.is_participating()

    def is_replacement_pending(self, failed_rank: int) -> bool:
        """Return ``True`` if a replacement is in progress (BOOTSTRAPPING or READY_FOR_REPAIR)."""
        slot = self._slots.get(failed_rank)
        if slot is None:
            return False
        return slot.state in (ReplacementState.BOOTSTRAPPING,
                              ReplacementState.READY_FOR_REPAIR)

    @property
    def pending_replacements(self) -> List[int]:
        """Return failed_ranks with pending (non-integrated) replacements."""
        return [
            fr for fr, slot in self._slots.items()
            if slot.state in (ReplacementState.BOOTSTRAPPING,
                              ReplacementState.READY_FOR_REPAIR)
        ]

    @property
    def ready_replacements(self) -> List[int]:
        """Return failed_ranks whose replacements are READY_FOR_REPAIR.

        These are the slots that can be integrated at the next safe-point.
        """
        return [
            fr for fr, slot in self._slots.items()
            if slot.state == ReplacementState.READY_FOR_REPAIR
        ]

    @property
    def integrated_replacements(self) -> List[int]:
        """Return failed_ranks whose replacements have been INTEGRATED."""
        return [
            fr for fr, slot in self._slots.items()
            if slot.state == ReplacementState.INTEGRATED
        ]

    @property
    def all_failed_ranks(self) -> List[int]:
        """Return all failed ranks that have (or had) replacement slots."""
        return sorted(self._slots.keys())

    @property
    def num_pending(self) -> int:
        return len(self.pending_replacements)

    @property
    def num_ready(self) -> int:
        return len(self.ready_replacements)

    @property
    def num_integrated(self) -> int:
        return len(self.integrated_replacements)

    def has_any_pending(self) -> bool:
        """Return ``True`` if any replacement is pending (not yet integrated)."""
        return self.num_pending > 0

    def has_any_ready(self) -> bool:
        """Return ``True`` if any replacement is ready for safe-point integration."""
        return self.num_ready > 0

    # ------------------------------------------------------------------
    # Role management
    # ------------------------------------------------------------------

    def set_inherited_role(
        self,
        failed_rank: int,
        role: ParallelRoleDescriptor,
    ) -> None:
        """Set or update the inherited parallel role for a replacement.

        Args:
            failed_rank: The failed rank being replaced.
            role: The parallel role descriptor.

        Raises:
            KeyError: If no replacement is registered for this failed rank.
        """
        slot = self._get_slot_or_raise(failed_rank)
        slot.inherited_role = role
        logger.info(
            "MOEGAMBIT-MoE: inherited role set for failed_rank=%d → %s",
            failed_rank, role.role_key(),
        )

    # ------------------------------------------------------------------
    # Integration with ExpertDirectory (Step 5)
    # ------------------------------------------------------------------

    def get_expert_directory_query(
        self, failed_rank: int
    ) -> Optional[Dict[str, Any]]:
        """Return expert directory query info for a replacement.

        This provides the replacement rank with the information it needs to
        query the ``ActiveExpertDirectory`` and ``RecoveryManifest`` for the
        experts it must host.

        Returns:
            A dict with keys:
            - ``failed_rank``: the failed rank
            - ``replacement_rank``: the replacement rank
            - ``expert_ids``: list of expert IDs to host
            - ``ep_rank``: EP rank index
            - ``ep_group_ranks``: EP group membership
            Or ``None`` if no replacement is registered.
        """
        slot = self._slots.get(failed_rank)
        if slot is None:
            return None

        role = slot.inherited_role
        if role is None:
            return {
                "failed_rank": failed_rank,
                "replacement_rank": slot.replacement_rank,
                "expert_ids": [],
                "ep_rank": -1,
                "ep_group_ranks": [],
            }

        return {
            "failed_rank": failed_rank,
            "replacement_rank": slot.replacement_rank,
            "expert_ids": list(role.expert_ids),
            "ep_rank": role.ep_rank,
            "ep_group_ranks": list(role.ep_group_ranks),
        }

    def get_recovery_manifest_query(
        self, failed_rank: int
    ) -> Optional[Dict[str, Any]]:
        """Return recovery manifest query info for a replacement.

        This provides the replacement rank with the information it needs to
        query the ``RecoveryManifest`` for checkpoint recovery sources.

        Returns:
            A dict with keys:
            - ``failed_rank``: the failed rank
            - ``replacement_rank``: the replacement rank
            - ``expert_ids``: list of expert IDs that need recovery
            - ``state``: current replacement state
            Or ``None`` if no replacement is registered.
        """
        slot = self._slots.get(failed_rank)
        if slot is None:
            return None

        expert_ids = []
        if slot.inherited_role is not None:
            expert_ids = list(slot.inherited_role.expert_ids)

        return {
            "failed_rank": failed_rank,
            "replacement_rank": slot.replacement_rank,
            "expert_ids": expert_ids,
            "state": slot.state.name,
        }

    # ------------------------------------------------------------------
    # Safe-point integration helpers
    # ------------------------------------------------------------------

    def collect_ready_for_integration(self) -> List[ReplacementSlot]:
        """Return all slots that are READY_FOR_REPAIR.

        This is called at a safe-point to determine which replacements
        should be integrated in this batch.

        Returns:
            List of ``ReplacementSlot`` objects ready for integration.
        """
        return [
            slot for slot in self._slots.values()
            if slot.state == ReplacementState.READY_FOR_REPAIR
        ]

    def integrate_all_ready(self, step: int = 0) -> List[ReplacementSlot]:
        """Integrate all READY_FOR_REPAIR replacements at once.

        This is a convenience method for safe-point processing.

        Args:
            step: Current training step.

        Returns:
            List of slots that were integrated.
        """
        ready = self.collect_ready_for_integration()
        for slot in ready:
            slot.transition_to(ReplacementState.INTEGRATED, step)
            logger.warning(
                "MOEGAMBIT-MoE: replacement INTEGRATED (batch) — failed_rank=%d, "
                "replacement_rank=%d (step=%d)",
                slot.failed_rank, slot.replacement_rank, step,
            )
        return ready

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_integrated(self) -> int:
        """Remove all INTEGRATED slots from the registry.

        Call this after the group rebuild is complete and the replacement
        ranks are fully operational.

        Returns:
            Number of slots removed.
        """
        to_remove = [
            fr for fr, slot in self._slots.items()
            if slot.state == ReplacementState.INTEGRATED
        ]
        for fr in to_remove:
            del self._slots[fr]
        if to_remove:
            logger.info(
                "MOEGAMBIT-MoE: cleaned up %d integrated replacement slots: %s",
                len(to_remove), to_remove,
            )
        return len(to_remove)

    def reset(self) -> None:
        """Clear all replacement slots."""
        self._slots.clear()

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a human-readable summary."""
        state_counts: Dict[str, int] = {}
        for slot in self._slots.values():
            name = slot.state.name
            state_counts[name] = state_counts.get(name, 0) + 1
        return {
            "total_slots": len(self._slots),
            "state_counts": state_counts,
            "pending": self.pending_replacements,
            "ready": self.ready_replacements,
            "integrated": self.integrated_replacements,
        }

    def __repr__(self) -> str:
        return (
            f"ReplacementRegistry(slots={len(self._slots)}, "
            f"pending={self.num_pending}, ready={self.num_ready}, "
            f"integrated={self.num_integrated})"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_slot_or_raise(self, failed_rank: int) -> ReplacementSlot:
        """Return the slot for a failed rank, or raise KeyError."""
        slot = self._slots.get(failed_rank)
        if slot is None:
            raise KeyError(
                f"No replacement registered for failed_rank={failed_rank}"
            )
        return slot


# =====================================================================
# Global singleton
# =====================================================================

_REGISTRY: Optional[ReplacementRegistry] = None


def get_replacement_registry() -> ReplacementRegistry:
    """Get or create the global ``ReplacementRegistry`` singleton."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ReplacementRegistry()
    return _REGISTRY


def clear_replacement_registry() -> None:
    """Reset the global registry (for testing)."""
    global _REGISTRY
    if _REGISTRY is not None:
        _REGISTRY.reset()
    _REGISTRY = None


# =====================================================================
# Convenience functions (top-level API)
# =====================================================================

def announce_replacement(
    failed_rank: int,
    replacement_rank: int,
    step: int = 0,
    reason: str = "",
    role: Optional[ParallelRoleDescriptor] = None,
) -> ReplacementSlot:
    """Announce a replacement rank for a failed rank.

    See ``ReplacementRegistry.announce_replacement()`` for details.
    """
    reg = get_replacement_registry()
    return reg.announce_replacement(
        failed_rank, replacement_rank, step, reason, role,
    )


def announce_replacement_ready(
    failed_rank: int,
    step: int = 0,
) -> ReplacementSlot:
    """Signal that the replacement rank has finished bootstrapping.

    See ``ReplacementRegistry.announce_replacement_ready()`` for details.
    """
    reg = get_replacement_registry()
    return reg.announce_replacement_ready(failed_rank, step)


def query_replacement_status(
    failed_rank: int,
) -> ReplacementState:
    """Query the replacement status for a failed rank.

    See ``ReplacementRegistry.query_replacement_status()`` for details.
    """
    reg = get_replacement_registry()
    return reg.query_replacement_status(failed_rank)


def mark_integrated(
    failed_rank: int,
    step: int = 0,
) -> ReplacementSlot:
    """Mark the replacement as integrated at a safe-point.

    See ``ReplacementRegistry.mark_integrated()`` for details.
    """
    reg = get_replacement_registry()
    return reg.mark_integrated(failed_rank, step)

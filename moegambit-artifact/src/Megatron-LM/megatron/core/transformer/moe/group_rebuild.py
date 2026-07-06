# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Safe-point Group Rebuild Protocol for MoEGambit.

This module implements a minimal, safe-point-gated protocol for rebuilding
NCCL process groups after a hard rank failure and replacement.  The protocol
ensures that:

* Groups are **never** rebuilt at arbitrary times — only at safe points
  (iteration boundary, gradient-accumulation boundary, or checkpoint boundary).
* The world size does **not** shrink — a replacement rank takes over the
  failed rank's logical slot.
* Only **affected** groups are rebuilt — groups that do not contain the
  failed/replacement rank are left untouched.

Protocol overview
-----------------
::

    ┌──────────────────────────────────────────────────────────────────┐
    │                     Training Loop                                │
    │                                                                  │
    │  ┌─────────┐   ┌──────────────┐   ┌──────────────┐              │
    │  │ forward  │──▶│   backward   │──▶│  optimizer   │──▶ ...      │
    │  └─────────┘   └──────────────┘   └──────────────┘              │
    │                                          │                       │
    │                                          ▼                       │
    │                                   ┌──────────────┐               │
    │                                   │  safe point  │               │
    │                                   │  (iter end)  │               │
    │                                   └──────┬───────┘               │
    │                                          │                       │
    │                          has_pending_group_repair()?             │
    │                                   yes ──▼── no ──▶ continue     │
    │                                          │                       │
    │                          maybe_rebuild_groups_at_safe_point()    │
    │                                          │                       │
    │                          ┌────────────────┼────────────────┐     │
    │                          ▼                ▼                ▼     │
    │                    invalidate       rebuild new      rebind      │
    │                    old handles      group handles    modules     │
    │                          │                │                │     │
    │                          └────────────────┼────────────────┘     │
    │                                          │                       │
    │                          finish_group_repair()                   │
    │                                          │                       │
    │                                   continue training              │
    └──────────────────────────────────────────────────────────────────┘

State machine
-------------
::

    IDLE ──detect fault──▶ PENDING_REPAIR
                                │
                          safe-point reached
                                │
                                ▼
                          REBUILDING ──success──▶ REBINDING
                                                      │
                                                 finish_group_repair()
                                                      │
                                                      ▼
                                                    IDLE

Scope (v1)
----------
* ✅ Replace a failed rank with a spare rank at safe-point
* ✅ Rebuild affected EP, Expert-DP, TP×EP, ETP×EP×PP groups
* ✅ Rebuild affected DP groups (replacement needs full training participation)
* ✅ Rebind MoE modules to new group handles
* ❌ World-size shrink (not supported)
* ❌ Fully dynamic arbitrary membership change (not supported)
* ❌ Hot-swap communicator at arbitrary time (not supported)
* ❌ ZeRO-2 (not considered in v1)

Integration with MoEGambit stack
------------------------------
* ``RankQuarantineRegistry`` (Step 4) — quarantines the failed rank.
* ``ReplacementRegistry`` (Step 6) — manages replacement lifecycle.
* ``ActiveExpertDirectory`` (Step 5) — updates rank↔expert mapping.
* This module (Step 7) — orchestrates the group rebuild.

Affected groups
---------------
When a rank in an EP group fails and is replaced, the following groups
must be rebuilt (because their membership changes):

1. **Expert Parallel (EP)** — ``_EXPERT_MODEL_PARALLEL_GROUP``
2. **Expert Tensor Parallel (ETP)** — ``_EXPERT_TENSOR_PARALLEL_GROUP``
3. **Expert Tensor × EP** — ``_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP``
4. **Expert Tensor × EP × PP** — ``_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP``
5. **Expert Data Parallel** — ``_EXPERT_DATA_PARALLEL_GROUP`` (+ Gloo variant)
6. **Data Parallel** — ``_DATA_PARALLEL_GROUP`` (+ Gloo variant)
7. **Data Parallel + CP** — ``_DATA_PARALLEL_GROUP_WITH_CP`` (+ Gloo variant)
8. **Tensor Parallel (TP)** — ``_TENSOR_MODEL_PARALLEL_GROUP``
9. **Tensor × Data Parallel** — ``_TENSOR_AND_DATA_PARALLEL_GROUP``
10. **Tensor × Data Parallel + CP** — ``_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP``

Groups that are NOT rebuilt (membership unchanged if replacement takes
the same logical slot):

* **Pipeline Parallel (PP)** — same PP group ranks
* **Context Parallel (CP)** — same CP group ranks
* **Embedding / Position Embedding** — same ranks

Note: When TP > 1, the TP group's NCCL communicator is bound to physical
rank IDs.  Since the replacement rank has a different physical rank ID,
the TP group must be rebuilt even though the logical membership is the
same.  The same applies to composite groups containing TP (TP×DP, etc.).

Note: In practice, since the replacement rank takes the exact same logical
slot as the failed rank, ALL groups that contained the failed rank need
rebuilding because the physical rank ID changes.  However, groups that
did NOT contain the failed rank are untouched.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =====================================================================
# Rebuild state enum
# =====================================================================

class GroupRebuildState(enum.IntEnum):
    """Lifecycle states for the group rebuild protocol."""

    IDLE = 0
    """No rebuild pending.  Normal training."""

    PENDING_REPAIR = 1
    """A hard-failed rank has been detected and a replacement is ready.
    Waiting for safe-point to begin rebuild."""

    REBUILDING = 2
    """Safe-point reached.  Old group handles are being invalidated and
    new groups are being created."""

    REBINDING = 3
    """New groups created.  Modules are being rebound to new handles."""


# Valid transitions
_VALID_TRANSITIONS: Dict[GroupRebuildState, Set[GroupRebuildState]] = {
    GroupRebuildState.IDLE:           {GroupRebuildState.PENDING_REPAIR},
    GroupRebuildState.PENDING_REPAIR: {GroupRebuildState.REBUILDING,
                                       GroupRebuildState.IDLE},  # cancel
    GroupRebuildState.REBUILDING:     {GroupRebuildState.REBINDING,
                                       GroupRebuildState.IDLE},  # abort
    GroupRebuildState.REBINDING:      {GroupRebuildState.IDLE},
}


# =====================================================================
# Affected group descriptor
# =====================================================================

# Canonical names for groups that may need rebuilding.
# These correspond to global variables in parallel_state.py.
EXPERT_GROUPS = frozenset({
    "EXPERT_MODEL_PARALLEL_GROUP",           # EP
    "EXPERT_TENSOR_PARALLEL_GROUP",          # ETP
    "EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP",  # ETP×EP
    "EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP",  # ETP×EP×PP
    "EXPERT_DATA_PARALLEL_GROUP",            # Expert DP (NCCL)
    "EXPERT_DATA_PARALLEL_GROUP_GLOO",       # Expert DP (Gloo)
})

DATA_PARALLEL_GROUPS = frozenset({
    "DATA_PARALLEL_GROUP",                   # DP (NCCL)
    "DATA_PARALLEL_GROUP_GLOO",              # DP (Gloo)
    "DATA_PARALLEL_GROUP_WITH_CP",           # DP+CP (NCCL)
    "DATA_PARALLEL_GROUP_WITH_CP_GLOO",      # DP+CP (Gloo)
})

TENSOR_PARALLEL_GROUPS = frozenset({
    "TENSOR_MODEL_PARALLEL_GROUP",           # TP (attention/FFN)
    "TENSOR_AND_DATA_PARALLEL_GROUP",        # TP×DP
    "TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP",  # TP×DP×CP
})

# All groups that may need rebuilding
ALL_REBUILDABLE_GROUPS = EXPERT_GROUPS | DATA_PARALLEL_GROUPS | TENSOR_PARALLEL_GROUPS


@dataclass
class GroupRebuildPlan:
    """Describes a planned group rebuild operation.

    This is a pure-data descriptor that captures what needs to happen
    at the next safe-point.  It does NOT hold actual ProcessGroup handles.
    """

    failed_rank: int
    """The global rank that failed."""

    replacement_rank: int
    """The global rank of the replacement worker."""

    affected_groups: List[str] = field(default_factory=list)
    """Names of groups that need rebuilding (from ALL_REBUILDABLE_GROUPS)."""

    old_ep_group_ranks: List[int] = field(default_factory=list)
    """The EP group ranks before rebuild (with failed rank)."""

    new_ep_group_ranks: List[int] = field(default_factory=list)
    """The EP group ranks after rebuild (with replacement rank)."""

    step: int = -1
    """Training step when the rebuild was requested."""

    created_time: float = field(default_factory=time.monotonic)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failed_rank": self.failed_rank,
            "replacement_rank": self.replacement_rank,
            "affected_groups": list(self.affected_groups),
            "old_ep_group_ranks": list(self.old_ep_group_ranks),
            "new_ep_group_ranks": list(self.new_ep_group_ranks),
            "step": self.step,
        }


# =====================================================================
# Module rebind descriptor
# =====================================================================

@dataclass
class ModuleRebindRecord:
    """Records which module attributes were rebound to new group handles.

    This is used for auditing and debugging — after a rebuild, we can
    verify that all expected modules were updated.
    """

    module_name: str = ""
    attribute_name: str = ""
    old_group_name: str = ""
    new_group_name: str = ""
    timestamp: float = field(default_factory=time.monotonic)


# =====================================================================
# GroupRebuildCoordinator — singleton per training job
# =====================================================================

class GroupRebuildCoordinator:
    """Orchestrates safe-point group rebuild after rank replacement.

    This is the central coordinator that:
    1. Accepts rebuild requests (from fault detector / replacement protocol).
    2. Gates rebuild execution to safe-points only.
    3. Tracks the rebuild lifecycle (IDLE → PENDING → REBUILDING → REBINDING → IDLE).
    4. Records which groups were rebuilt and which modules were rebound.

    The coordinator does NOT directly call ``torch.distributed.new_group()``
    or modify ``parallel_state`` globals — those operations are delegated to
    callback functions provided by the caller.  This keeps the coordinator
    testable without distributed init.

    Typical usage::

        coord = get_group_rebuild_coordinator()

        # Fault detected, replacement ready
        coord.request_rebuild(
            failed_rank=4, replacement_rank=64, step=100,
            affected_groups=["EXPERT_MODEL_PARALLEL_GROUP", ...],
            old_ep_group_ranks=[0, 2, 4, 6],
            new_ep_group_ranks=[0, 2, 64, 6],
        )

        # Training loop safe-point
        if coord.has_pending_group_repair():
            coord.maybe_rebuild_groups_at_safe_point(
                rebuild_fn=my_rebuild_function,
                rebind_fn=my_rebind_function,
            )
            coord.finish_group_repair(step=110)
    """

    def __init__(self) -> None:
        self._state: GroupRebuildState = GroupRebuildState.IDLE
        self._pending_plans: List[GroupRebuildPlan] = []
        self._active_plan: Optional[GroupRebuildPlan] = None
        self._history: List[Dict[str, Any]] = []
        self._rebind_records: List[ModuleRebindRecord] = []

    # ------------------------------------------------------------------
    # State query
    # ------------------------------------------------------------------

    @property
    def state(self) -> GroupRebuildState:
        return self._state

    def has_pending_group_repair(self) -> bool:
        """Return ``True`` if there is a pending rebuild request.

        This is the primary check called by the training loop at each
        potential safe-point.
        """
        return (
            self._state == GroupRebuildState.PENDING_REPAIR
            and len(self._pending_plans) > 0
        )

    def is_idle(self) -> bool:
        return self._state == GroupRebuildState.IDLE

    def is_rebuilding(self) -> bool:
        return self._state in (GroupRebuildState.REBUILDING,
                                GroupRebuildState.REBINDING)

    @property
    def pending_plans(self) -> List[GroupRebuildPlan]:
        return list(self._pending_plans)

    @property
    def active_plan(self) -> Optional[GroupRebuildPlan]:
        return self._active_plan

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    @property
    def rebind_records(self) -> List[ModuleRebindRecord]:
        return list(self._rebind_records)

    # ------------------------------------------------------------------
    # Request rebuild
    # ------------------------------------------------------------------

    def request_rebuild(
        self,
        failed_rank: int,
        replacement_rank: int,
        step: int = 0,
        affected_groups: Optional[List[str]] = None,
        old_ep_group_ranks: Optional[List[int]] = None,
        new_ep_group_ranks: Optional[List[int]] = None,
    ) -> GroupRebuildPlan:
        """Request a group rebuild for a failed→replacement rank swap.

        This transitions the coordinator from IDLE → PENDING_REPAIR.
        The actual rebuild will happen at the next safe-point.

        Args:
            failed_rank: The global rank that failed.
            replacement_rank: The global rank of the replacement.
            step: Current training step.
            affected_groups: Names of groups to rebuild.  If ``None``,
                defaults to all expert + data-parallel groups.
            old_ep_group_ranks: EP group ranks before rebuild.
            new_ep_group_ranks: EP group ranks after rebuild.

        Returns:
            The created ``GroupRebuildPlan``.

        Raises:
            ValueError: If the coordinator is not IDLE or PENDING_REPAIR.
        """
        if self._state not in (GroupRebuildState.IDLE,
                                GroupRebuildState.PENDING_REPAIR):
            raise ValueError(
                f"Cannot request rebuild in state {self._state.name}. "
                f"Must be IDLE or PENDING_REPAIR."
            )

        if affected_groups is None:
            affected_groups = sorted(ALL_REBUILDABLE_GROUPS)

        plan = GroupRebuildPlan(
            failed_rank=failed_rank,
            replacement_rank=replacement_rank,
            affected_groups=affected_groups,
            old_ep_group_ranks=old_ep_group_ranks or [],
            new_ep_group_ranks=new_ep_group_ranks or [],
            step=step,
        )
        self._pending_plans.append(plan)

        if self._state == GroupRebuildState.IDLE:
            self._transition_to(GroupRebuildState.PENDING_REPAIR)

        logger.warning(
            "MoEGambit group rebuild: requested — failed_rank=%d → replacement_rank=%d "
            "(step=%d, groups=%s)",
            failed_rank, replacement_rank, step, affected_groups,
        )
        return plan

    # ------------------------------------------------------------------
    # Safe-point execution
    # ------------------------------------------------------------------

    def maybe_rebuild_groups_at_safe_point(
        self,
        rebuild_fn=None,
        rebind_fn=None,
        step: int = 0,
    ) -> bool:
        """Execute group rebuild if pending and at a safe-point.

        This is the main entry point called by the training loop.  It:
        1. Checks if there is a pending rebuild.
        2. Transitions to REBUILDING.
        3. Calls ``rebuild_fn(plan)`` to create new group handles.
        4. Transitions to REBINDING.
        5. Calls ``rebind_fn(plan)`` to update module references.
        6. Does NOT call ``finish_group_repair()`` — the caller must do that.

        Args:
            rebuild_fn: Callable that takes a ``GroupRebuildPlan`` and
                performs the actual group teardown + creation.  If ``None``,
                the rebuild step is skipped (useful for testing).
            rebind_fn: Callable that takes a ``GroupRebuildPlan`` and
                rebinds module attributes to new group handles.  If ``None``,
                the rebind step is skipped.
            step: Current training step.

        Returns:
            ``True`` if a rebuild was executed, ``False`` otherwise.
        """
        if not self.has_pending_group_repair():
            return False

        # Merge all pending plans into one active plan
        # (In v1, we process them sequentially; in practice there's usually just one)
        if len(self._pending_plans) == 0:
            return False

        plan = self._pending_plans[0]
        self._active_plan = plan
        self._pending_plans = self._pending_plans[1:]

        # Phase 1: REBUILDING — invalidate old handles, create new ones
        self._transition_to(GroupRebuildState.REBUILDING)
        logger.warning(
            "MoEGambit group rebuild: REBUILDING — failed_rank=%d → replacement_rank=%d "
            "(step=%d)",
            plan.failed_rank, plan.replacement_rank, step,
        )

        if rebuild_fn is not None:
            rebuild_fn(plan)

        # Phase 2: REBINDING — update module references
        self._transition_to(GroupRebuildState.REBINDING)
        logger.warning(
            "MoEGambit group rebuild: REBINDING — updating module references (step=%d)",
            step,
        )

        if rebind_fn is not None:
            rebind_fn(plan)

        return True

    def finish_group_repair(self, step: int = 0) -> Optional[GroupRebuildPlan]:
        """Complete the group repair and return to IDLE.

        This should be called after ``maybe_rebuild_groups_at_safe_point()``
        and after verifying that the new groups are functional.

        Args:
            step: Current training step.

        Returns:
            The completed ``GroupRebuildPlan``, or ``None`` if no active plan.
        """
        if self._state not in (GroupRebuildState.REBINDING,
                                GroupRebuildState.REBUILDING):
            logger.warning(
                "finish_group_repair called in state %s, expected REBINDING or REBUILDING",
                self._state.name,
            )
            return None

        plan = self._active_plan
        self._active_plan = None

        # Record in history
        if plan is not None:
            record = plan.to_dict()
            record["completed_step"] = step
            record["completed_time"] = time.monotonic()
            self._history.append(record)

        # Check if more plans are pending
        if len(self._pending_plans) > 0:
            self._state = GroupRebuildState.PENDING_REPAIR
        else:
            self._transition_to(GroupRebuildState.IDLE)

        logger.warning(
            "MoEGambit group rebuild: FINISHED — returning to %s (step=%d, "
            "history_len=%d)",
            self._state.name, step, len(self._history),
        )
        return plan

    # ------------------------------------------------------------------
    # Cancel / abort
    # ------------------------------------------------------------------

    def cancel_pending(self) -> int:
        """Cancel all pending rebuild requests and return to IDLE.

        Returns:
            Number of plans cancelled.
        """
        count = len(self._pending_plans)
        self._pending_plans.clear()
        if self._state == GroupRebuildState.PENDING_REPAIR:
            self._transition_to(GroupRebuildState.IDLE)
        return count

    # ------------------------------------------------------------------
    # Rebind record tracking
    # ------------------------------------------------------------------

    def record_rebind(
        self,
        module_name: str,
        attribute_name: str,
        old_group_name: str = "",
        new_group_name: str = "",
    ) -> None:
        """Record a module attribute rebind for auditing."""
        self._rebind_records.append(ModuleRebindRecord(
            module_name=module_name,
            attribute_name=attribute_name,
            old_group_name=old_group_name,
            new_group_name=new_group_name,
        ))

    # ------------------------------------------------------------------
    # Affected group analysis
    # ------------------------------------------------------------------

    @staticmethod
    def compute_affected_groups(
        failed_rank: int,
        ep_group_ranks: List[int],
        dp_group_ranks: Optional[List[int]] = None,
    ) -> List[str]:
        """Determine which groups need rebuilding based on the failed rank.

        Args:
            failed_rank: The global rank that failed.
            ep_group_ranks: Ranks in the EP group.
            dp_group_ranks: Ranks in the DP group (if available).

        Returns:
            Sorted list of group names that need rebuilding.
        """
        affected = set()

        # If failed rank is in EP group → all expert groups need rebuild
        if failed_rank in ep_group_ranks:
            affected.update(EXPERT_GROUPS)

        # If failed rank is in DP group → all DP groups need rebuild
        if dp_group_ranks is not None and failed_rank in dp_group_ranks:
            affected.update(DATA_PARALLEL_GROUPS)
        else:
            # Conservative: always rebuild DP groups since replacement rank
            # needs to participate in gradient reduction
            affected.update(DATA_PARALLEL_GROUPS)

        return sorted(affected)

    @staticmethod
    def compute_new_group_ranks(
        old_ranks: List[int],
        failed_rank: int,
        replacement_rank: int,
    ) -> List[int]:
        """Compute new group ranks by replacing failed with replacement.

        Args:
            old_ranks: Original rank list.
            failed_rank: The rank to replace.
            replacement_rank: The new rank.

        Returns:
            New rank list with the replacement.
        """
        return [
            replacement_rank if r == failed_rank else r
            for r in old_ranks
        ]

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "state": self._state.name,
            "num_pending": len(self._pending_plans),
            "active_plan": self._active_plan.to_dict() if self._active_plan else None,
            "history_len": len(self._history),
            "rebind_records": len(self._rebind_records),
        }

    def __repr__(self) -> str:
        return (
            f"GroupRebuildCoordinator(state={self._state.name}, "
            f"pending={len(self._pending_plans)}, "
            f"history={len(self._history)})"
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all state (for testing)."""
        self._state = GroupRebuildState.IDLE
        self._pending_plans.clear()
        self._active_plan = None
        self._history.clear()
        self._rebind_records.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _transition_to(self, new_state: GroupRebuildState) -> None:
        allowed = _VALID_TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid rebuild transition: {self._state.name} → {new_state.name}. "
                f"Allowed: {[s.name for s in allowed]}"
            )
        old = self._state
        self._state = new_state
        logger.debug(
            "GroupRebuildCoordinator: %s → %s", old.name, new_state.name,
        )


# =====================================================================
# MoE module rebind helpers
# =====================================================================

# These are the module attributes that need updating after a group rebuild.
# Each entry is (attribute_name, pg_collection_field).
MOE_DISPATCHER_REBIND_MAP = [
    ("ep_group",    "ep"),
    ("tp_group",    "expt_tp"),
    ("tp_ep_group", "tp_ep"),
]

MOE_ROUTER_REBIND_MAP = [
    ("tp_group",      "tp"),
    ("cp_group",      "cp"),
    ("tp_cp_group",   "tp_cp"),
    ("tp_dp_cp_group", "tp_dp_cp"),
]

MOE_LAYER_REBIND_MAP = [
    ("ep_group",      "ep"),
    ("attn_tp_group", "tp"),
]

MOE_EXPERTS_REBIND_MAP = [
    ("ep_group", "ep"),
    ("tp_group", "expt_tp"),
    ("dp_group", "expt_dp"),
]


def rebind_moe_module_groups(
    module,
    new_pg_collection,
    rebind_map,
    module_name: str = "",
    coordinator: Optional[GroupRebuildCoordinator] = None,
) -> int:
    """Rebind process group attributes on a module to new handles.

    Args:
        module: The module whose attributes to update.
        new_pg_collection: A dict-like or object with the new group handles.
            Can be a ``ProcessGroupCollection`` or a plain dict.
        rebind_map: List of ``(attr_name, pg_collection_field)`` pairs.
        module_name: Name for logging/auditing.
        coordinator: Optional coordinator to record rebinds.

    Returns:
        Number of attributes rebound.
    """
    count = 0
    for attr_name, pg_field in rebind_map:
        if not hasattr(module, attr_name):
            continue

        # Get new group handle
        if isinstance(new_pg_collection, dict):
            new_group = new_pg_collection.get(pg_field)
        else:
            new_group = getattr(new_pg_collection, pg_field, None)

        if new_group is None:
            continue

        old_group = getattr(module, attr_name, None)
        setattr(module, attr_name, new_group)
        count += 1

        if coordinator is not None:
            coordinator.record_rebind(
                module_name=module_name,
                attribute_name=attr_name,
                old_group_name=f"old_{pg_field}",
                new_group_name=f"new_{pg_field}",
            )

        logger.debug(
            "Rebound %s.%s: %s → %s",
            module_name, attr_name, pg_field, pg_field,
        )

    return count


def rebind_dispatcher_derived_values(dispatcher) -> None:
    """Recompute derived values on a token dispatcher after group rebind.

    After rebinding ``ep_group`` and ``tp_group``, the dispatcher's cached
    ``ep_size``, ``tp_size``, ``tp_rank`` etc. are stale and must be updated.
    """
    if hasattr(dispatcher, 'ep_group') and dispatcher.ep_group is not None:
        try:
            dispatcher.ep_size = dispatcher.ep_group.size()
        except Exception:
            pass

    if hasattr(dispatcher, 'tp_group') and dispatcher.tp_group is not None:
        try:
            dispatcher.tp_size = dispatcher.tp_group.size()
            dispatcher.tp_rank = dispatcher.tp_group.rank()
        except Exception:
            pass


# =====================================================================
# Global singleton
# =====================================================================

_COORDINATOR: Optional[GroupRebuildCoordinator] = None


def get_group_rebuild_coordinator() -> GroupRebuildCoordinator:
    """Get or create the global ``GroupRebuildCoordinator`` singleton."""
    global _COORDINATOR
    if _COORDINATOR is None:
        _COORDINATOR = GroupRebuildCoordinator()
    return _COORDINATOR


def clear_group_rebuild_coordinator() -> None:
    """Reset the global coordinator (for testing)."""
    global _COORDINATOR
    if _COORDINATOR is not None:
        _COORDINATOR.reset()
    _COORDINATOR = None


# =====================================================================
# Top-level convenience API (training loop hooks)
# =====================================================================

def has_pending_group_repair() -> bool:
    """Check if there is a pending group rebuild.

    This is the primary hook for the training loop::

        if has_pending_group_repair():
            maybe_rebuild_groups_at_safe_point(...)
    """
    coord = get_group_rebuild_coordinator()
    return coord.has_pending_group_repair()


def maybe_rebuild_groups_at_safe_point(
    rebuild_fn=None,
    rebind_fn=None,
    step: int = 0,
) -> bool:
    """Execute group rebuild at safe-point if pending.

    See ``GroupRebuildCoordinator.maybe_rebuild_groups_at_safe_point()``.
    """
    coord = get_group_rebuild_coordinator()
    return coord.maybe_rebuild_groups_at_safe_point(
        rebuild_fn=rebuild_fn,
        rebind_fn=rebind_fn,
        step=step,
    )


def finish_group_repair(step: int = 0) -> Optional[GroupRebuildPlan]:
    """Complete the group repair and return to IDLE.

    See ``GroupRebuildCoordinator.finish_group_repair()``.
    """
    coord = get_group_rebuild_coordinator()
    return coord.finish_group_repair(step=step)

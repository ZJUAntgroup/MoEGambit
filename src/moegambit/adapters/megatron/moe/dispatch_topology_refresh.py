# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Token Dispatch Topology Refresh for MOEGAMBIT-MoE (Step 8).

After a safe-point group rebuild (Step 7), the MoE router, token dispatcher,
and expert host mapping must be refreshed so that:

* Failed ranks no longer appear in active dispatch targets.
* Replacement ranks that have completed integration can receive token traffic.
* The ``ActiveExpertDirectory`` reflects the new rank↔expert mapping.

Refresh flow
------------
::

    group rebuild completed (Step 7)
            │
            ▼
    refresh_dispatch_topology()
            │
            ├─ 1. refresh ActiveExpertDirectory
            │     └─ directory.refresh_from_placement(new_ep_group_ranks)
            │
            ├─ 2. refresh dispatcher routing metadata
            │     └─ (handled by group_rebuild rebind — no extra work here)
            │
            └─ 3. update replacement registry
                  └─ mark_integrated for ready replacements

Design principles
-----------------
* **Orchestration only** — this module does NOT create process groups or
  modify ``parallel_state`` globals.  It coordinates the *data-plane*
  refresh that must happen after the *control-plane* group rebuild.
* **Idempotent** — calling ``refresh_dispatch_topology()`` multiple times
  with the same arguments produces the same result.
* **Testable without distributed init** — all operations are on pure-Python
  / CPU-tensor data structures.

Integration with MOEGAMBIT-MoE stack
------------------------------
* ``ActiveExpertDirectory`` (Step 5) — live expert↔rank mapping.
* ``ReplacementRegistry`` (Step 6) — replacement lifecycle.
* ``GroupRebuildCoordinator`` (Step 7) — safe-point group rebuild.
* This module (Step 8) — post-rebuild topology refresh.

Dispatch target determination (Megatron native)
------------------------------------------------
Megatron uses an **implicit positional mapping** for expert-to-rank:

    expert_id = ep_rank * num_local_experts + local_offset

This mapping is encoded in:
* ``BaseMoELayer.local_expert_indices`` — computed in ``__init__``
* ``AlltoAllDispatcher.input_splits`` — computed per-forward via
  ``routing_map.sum(dim=0).reshape(ep_size, num_local_experts).sum(axis=1)``
* ``AllGatherDispatcher.local_map`` — sliced from ``routing_map`` per-forward

Typical usage::

    from moegambit.adapters.megatron.moe.dispatch_topology_refresh import (
        refresh_dispatch_topology,
        get_active_expert_host,
        is_rank_active_for_dispatch,
    )

    # After group rebuild completes:
    refresh_dispatch_topology(
        num_layers=4,
        num_experts=64,
        ep_size=8,
        new_ep_group_ranks=[0, 1, 2, 64, 4, 5, 6, 7],  # rank 3 → 64
        failed_rank=3,
        replacement_rank=64,
        step=110,
    )

    # Query the live mapping:
    host = get_active_expert_host(layer_id=2, expert_id=25)  # → 64

    # Check if a rank is active:
    assert is_rank_active_for_dispatch(64)   # replacement is active
    assert not is_rank_active_for_dispatch(3)  # failed rank is not
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =====================================================================
# DispatchTopologySnapshot — immutable view of current topology
# =====================================================================

@dataclass
class DispatchTopologySnapshot:
    """Immutable snapshot of the dispatch topology at a given point in time.

    This captures the state of the dispatch topology after a refresh.
    It is useful for auditing, logging, and testing.
    """

    num_layers: int = 0
    """Number of MoE layers."""

    num_experts: int = 0
    """Total number of global experts per layer."""

    ep_size: int = 1
    """Expert-parallel world size."""

    num_local_experts: int = 0
    """Number of experts per EP rank (= num_experts // ep_size)."""

    ep_group_ranks: List[int] = field(default_factory=list)
    """Ordered list of global ranks in the EP group."""

    active_ranks: FrozenSet[int] = field(default_factory=frozenset)
    """Set of ranks that are active for dispatch."""

    failed_rank: int = -1
    """The rank that was replaced (if any)."""

    replacement_rank: int = -1
    """The replacement rank (if any)."""

    step: int = -1
    """Training step when this snapshot was taken."""

    # Per-expert host mapping: expert_id → host_rank
    # (layer-independent since all layers have the same placement)
    expert_to_rank: Dict[int, int] = field(default_factory=dict)
    """Mapping from global expert ID to host rank."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "ep_size": self.ep_size,
            "num_local_experts": self.num_local_experts,
            "ep_group_ranks": list(self.ep_group_ranks),
            "active_ranks": sorted(self.active_ranks),
            "failed_rank": self.failed_rank,
            "replacement_rank": self.replacement_rank,
            "step": self.step,
            "expert_to_rank": dict(self.expert_to_rank),
        }

    def is_consistent(self) -> bool:
        """Check internal consistency of the snapshot.

        Returns:
            ``True`` if the snapshot is self-consistent.
        """
        # Every expert should map to a rank in ep_group_ranks
        for eid, rank in self.expert_to_rank.items():
            if rank not in self.ep_group_ranks:
                return False

        return True


# =====================================================================
# Core topology computation (pure functions)
# =====================================================================

def compute_expert_to_rank_mapping(
    num_experts: int,
    ep_size: int,
    ep_group_ranks: List[int],
) -> Dict[int, int]:
    """Compute the expert-to-rank mapping from EP configuration.

    This replicates Megatron's implicit positional mapping:
    ``expert_id = ep_rank * num_local_experts + local_offset``

    Args:
        num_experts: Total number of global experts.
        ep_size: Expert-parallel world size.
        ep_group_ranks: Ordered list of global ranks in the EP group.

    Returns:
        Dict mapping global expert ID → host global rank.
    """
    assert num_experts % ep_size == 0
    assert len(ep_group_ranks) == ep_size
    num_local = num_experts // ep_size

    mapping: Dict[int, int] = {}
    for ep_rank_idx, global_rank in enumerate(ep_group_ranks):
        for local_idx in range(num_local):
            expert_id = ep_rank_idx * num_local + local_idx
            mapping[expert_id] = global_rank

    return mapping


def compute_experts_on_rank(
    global_rank: int,
    num_experts: int,
    ep_size: int,
    ep_group_ranks: List[int],
) -> List[int]:
    """Return the expert IDs hosted on a given rank.

    Args:
        global_rank: The global rank to query.
        num_experts: Total number of global experts.
        ep_size: Expert-parallel world size.
        ep_group_ranks: Ordered list of global ranks in the EP group.

    Returns:
        Sorted list of global expert IDs on this rank.
        Empty list if the rank is not in the EP group.
    """
    if global_rank not in ep_group_ranks:
        return []

    num_local = num_experts // ep_size
    ep_rank_idx = ep_group_ranks.index(global_rank)
    return list(range(ep_rank_idx * num_local, (ep_rank_idx + 1) * num_local))


def compute_active_ranks(
    ep_group_ranks: List[int],
    failed_ranks: FrozenSet[int] = frozenset(),
) -> FrozenSet[int]:
    """Compute the set of active (non-failed) ranks.

    Args:
        ep_group_ranks: Ordered list of global ranks in the EP group.
        failed_ranks: Set of failed global ranks (already replaced).

    Returns:
        Frozenset of active global ranks.
    """
    return frozenset(r for r in ep_group_ranks if r not in failed_ranks)


# =====================================================================
# DispatchTopologyManager — singleton coordinator
# =====================================================================

class DispatchTopologyManager:
    """Manages the dispatch topology and coordinates refresh operations.

    This is the central coordinator for Step 8.  It:
    1. Maintains the current topology snapshot.
    2. Provides query APIs for expert host and rank activity.
    3. Orchestrates the refresh flow after a group rebuild.
    4. Integrates with all lower MOEGAMBIT-MoE layers.

    The manager does NOT directly modify router or dispatcher internals.
    It updates the shared data structures (directory, health mask, placement)
    that the router and dispatcher read during forward pass.
    """

    def __init__(self) -> None:
        self._snapshot: Optional[DispatchTopologySnapshot] = None
        self._history: List[DispatchTopologySnapshot] = []
        self._refresh_count: int = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise(
        self,
        num_layers: int,
        num_experts: int,
        ep_size: int,
        ep_group_ranks: Optional[List[int]] = None,
    ) -> DispatchTopologySnapshot:
        """Initialise the topology from the parallel configuration.

        This should be called once after ``initialize_model_parallel()``.

        Args:
            num_layers: Number of MoE layers.
            num_experts: Total number of global experts per layer.
            ep_size: Expert-parallel world size.
            ep_group_ranks: Ordered list of global ranks in the EP group.
                If ``None``, defaults to ``[0, 1, ..., ep_size-1]``.

        Returns:
            The initial ``DispatchTopologySnapshot``.
        """
        if ep_group_ranks is None:
            ep_group_ranks = list(range(ep_size))

        expert_to_rank = compute_expert_to_rank_mapping(
            num_experts, ep_size, ep_group_ranks,
        )

        snapshot = DispatchTopologySnapshot(
            num_layers=num_layers,
            num_experts=num_experts,
            ep_size=ep_size,
            num_local_experts=num_experts // ep_size,
            ep_group_ranks=list(ep_group_ranks),
            active_ranks=frozenset(ep_group_ranks),
            step=0,
            expert_to_rank=expert_to_rank,
        )
        self._snapshot = snapshot

        logger.info(
            "DispatchTopologyManager: initialised — %d layers, %d experts, "
            "ep_size=%d, ranks=%s",
            num_layers, num_experts, ep_size, ep_group_ranks,
        )
        return snapshot

    # ------------------------------------------------------------------
    # Core refresh API
    # ------------------------------------------------------------------

    def refresh_dispatch_topology(
        self,
        new_ep_group_ranks: List[int],
        failed_rank: int = -1,
        replacement_rank: int = -1,
        step: int = 0,
        *,
        re_enable_recovered_experts: bool = True,
        update_directory: bool = True,
        update_replacement_registry: bool = True,
        # Kept for API compat but no longer used:
        update_placement: bool = True,
        update_health_mask: bool = True,
    ) -> DispatchTopologySnapshot:
        """Refresh the dispatch topology after a group rebuild.

        This is the main entry point called after
        ``GroupRebuildCoordinator.finish_group_repair()``.

        The refresh flow:
        1. Recompute expert-to-rank mapping from new EP group ranks.
        2. Update ``ActiveExpertDirectory`` with new host ranks.
        3. Update ``ReplacementRegistry`` (mark integrated if ready).
        4. Take a new topology snapshot.

        Args:
            new_ep_group_ranks: New ordered list of global ranks in EP group.
            failed_rank: The rank that was replaced (-1 if N/A).
            replacement_rank: The replacement rank (-1 if N/A).
            step: Current training step.
            re_enable_recovered_experts: Unused (kept for API compat).
            update_directory: If ``True``, refresh the ActiveExpertDirectory.
            update_replacement_registry: If ``True``, update ReplacementRegistry.
            update_placement: Unused (kept for API compat).
            update_health_mask: Unused (kept for API compat).

        Returns:
            The new ``DispatchTopologySnapshot``.

        Raises:
            RuntimeError: If the manager has not been initialised.
        """
        if self._snapshot is None:
            raise RuntimeError(
                "DispatchTopologyManager not initialised. Call initialise() first."
            )

        num_layers = self._snapshot.num_layers
        num_experts = self._snapshot.num_experts
        ep_size = self._snapshot.ep_size

        assert len(new_ep_group_ranks) == ep_size, (
            f"new_ep_group_ranks length ({len(new_ep_group_ranks)}) != "
            f"ep_size ({ep_size})"
        )

        # 1. Recompute expert-to-rank mapping
        new_expert_to_rank = compute_expert_to_rank_mapping(
            num_experts, ep_size, new_ep_group_ranks,
        )

        # 2. Compute active ranks (all ranks in the new EP group are active)
        active = frozenset(new_ep_group_ranks)

        # 3. Update ActiveExpertDirectory
        if update_directory:
            self._refresh_directory(
                new_ep_group_ranks, failed_rank, replacement_rank, step,
            )

        # 4. Update ReplacementRegistry
        if update_replacement_registry and failed_rank >= 0:
            self._refresh_replacement_registry(
                failed_rank, replacement_rank, step,
            )

        # 5. Build new snapshot
        snapshot = DispatchTopologySnapshot(
            num_layers=num_layers,
            num_experts=num_experts,
            ep_size=ep_size,
            num_local_experts=num_experts // ep_size,
            ep_group_ranks=list(new_ep_group_ranks),
            active_ranks=active,
            failed_rank=failed_rank,
            replacement_rank=replacement_rank,
            step=step,
            expert_to_rank=new_expert_to_rank,
        )

        # Archive old snapshot
        if self._snapshot is not None:
            self._history.append(self._snapshot)

        self._snapshot = snapshot
        self._refresh_count += 1

        logger.warning(
            "MOEGAMBIT-MoE dispatch topology: REFRESHED (step=%d, "
            "failed_rank=%d → replacement_rank=%d, "
            "active_ranks=%s, refresh_count=%d)",
            step, failed_rank, replacement_rank,
            sorted(active), self._refresh_count,
        )

        return snapshot

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_active_expert_host(
        self, layer_id: int, expert_id: int
    ) -> Optional[int]:
        """Return the global rank hosting the given expert.

        This first checks the ``ActiveExpertDirectory`` (if available),
        then falls back to the topology snapshot.

        Args:
            layer_id: Transformer layer index.
            expert_id: Global expert index.

        Returns:
            The host global rank, or ``None`` if not found.
        """
        # Try directory first (it has per-layer granularity)
        directory = self._get_directory()
        if directory is not None:
            entry = directory.get_entry(layer_id, expert_id)
            if entry is not None:
                return entry.host_rank

        # Fall back to snapshot (layer-independent)
        if self._snapshot is not None:
            return self._snapshot.expert_to_rank.get(expert_id)

        return None

    def is_rank_active_for_dispatch(self, global_rank: int) -> bool:
        """Return ``True`` if the rank is active for dispatch.

        A rank is active if:
        * It is in the EP group.
        * If it is a replacement rank, it must be INTEGRATED.

        Args:
            global_rank: The global rank to check.

        Returns:
            ``True`` if the rank can receive dispatch traffic.
        """
        if self._snapshot is None:
            return False

        # Not in EP group → not active
        if global_rank not in self._snapshot.ep_group_ranks:
            return False

        # Check replacement registry — if this rank is a replacement,
        # it must be INTEGRATED to be active
        try:
            from moegambit.adapters.megatron.moe.replacement_registry import (
                get_replacement_registry,
                ReplacementState,
            )
            reg = get_replacement_registry()
            # Check if this rank is a replacement for any failed rank
            for fr in reg.all_failed_ranks:
                slot = reg.get_slot(fr)
                if slot is not None and slot.replacement_rank == global_rank:
                    if slot.state != ReplacementState.INTEGRATED:
                        return False
        except (ImportError, Exception):
            pass

        return True

    def get_experts_on_rank(self, global_rank: int) -> List[int]:
        """Return expert IDs hosted on the given rank.

        Args:
            global_rank: The global rank to query.

        Returns:
            Sorted list of global expert IDs.
        """
        if self._snapshot is None:
            return []

        return compute_experts_on_rank(
            global_rank,
            self._snapshot.num_experts,
            self._snapshot.ep_size,
            self._snapshot.ep_group_ranks,
        )

    def get_dispatchable_experts(self) -> List[int]:
        """Return all expert IDs that are currently dispatchable.

        After the soft-failure cleanup, all experts in the EP group are
        dispatchable (no health mask filtering).

        Returns:
            Sorted list of dispatchable global expert IDs.
        """
        if self._snapshot is None:
            return []

        return sorted(self._snapshot.expert_to_rank.keys())

    def get_non_dispatchable_experts(self) -> List[int]:
        """Return all expert IDs that are NOT currently dispatchable.

        After the soft-failure cleanup, this always returns an empty list
        since all experts are dispatchable.

        Returns:
            Empty list.
        """
        return []

    @property
    def snapshot(self) -> Optional[DispatchTopologySnapshot]:
        """Return the current topology snapshot."""
        return self._snapshot

    @property
    def history(self) -> List[DispatchTopologySnapshot]:
        """Return the history of topology snapshots."""
        return list(self._history)

    @property
    def refresh_count(self) -> int:
        """Return the number of topology refreshes performed."""
        return self._refresh_count

    # ------------------------------------------------------------------
    # Consistency check
    # ------------------------------------------------------------------

    def check_router_dispatcher_consistency(
        self,
        routing_map_expert_mask: Optional[Dict[int, bool]] = None,
    ) -> List[str]:
        """Check that the router's candidate set is consistent with dispatch topology.

        After the soft-failure cleanup, this is a simplified check that
        verifies all experts map to valid EP group ranks.

        Args:
            routing_map_expert_mask: Unused (kept for API compat).

        Returns:
            List of inconsistency descriptions (empty = consistent).
        """
        if self._snapshot is None:
            return ["DispatchTopologyManager not initialised"]

        issues: List[str] = []

        # Check each expert maps to a valid rank
        for eid in range(self._snapshot.num_experts):
            host = self._snapshot.expert_to_rank.get(eid)
            if host is None:
                issues.append(f"Expert {eid}: no host rank in mapping")
            elif host not in self._snapshot.ep_group_ranks:
                issues.append(
                    f"Expert {eid}: host rank {host} not in EP group "
                    f"{self._snapshot.ep_group_ranks}"
                )

        return issues

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a human-readable summary."""
        if self._snapshot is None:
            return {"initialised": False}

        return {
            "initialised": True,
            "num_layers": self._snapshot.num_layers,
            "num_experts": self._snapshot.num_experts,
            "ep_size": self._snapshot.ep_size,
            "ep_group_ranks": self._snapshot.ep_group_ranks,
            "active_ranks": sorted(self._snapshot.active_ranks),
            "dispatchable_experts": len(self.get_dispatchable_experts()),
            "refresh_count": self._refresh_count,
            "step": self._snapshot.step,
        }

    def __repr__(self) -> str:
        if self._snapshot is None:
            return "DispatchTopologyManager(uninitialised)"
        return (
            f"DispatchTopologyManager("
            f"experts={self._snapshot.num_experts}, "
            f"ep_size={self._snapshot.ep_size}, "
            f"active={len(self._snapshot.active_ranks)}, "
            f"refreshes={self._refresh_count})"
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all state (for testing)."""
        self._snapshot = None
        self._history.clear()
        self._refresh_count = 0

    # ------------------------------------------------------------------
    # Internal helpers — MOEGAMBIT-MoE layer integration
    # ------------------------------------------------------------------

    def _get_directory(self) -> Optional[Any]:
        """Get the global ActiveExpertDirectory, or None."""
        try:
            from moegambit.adapters.megatron.moe.expert_directory import (
                get_active_expert_directory,
            )
            return get_active_expert_directory()
        except (ImportError, Exception):
            return None

    def _get_quarantined_ranks(self) -> FrozenSet[int]:
        """Get the set of quarantined ranks from the quarantine registry."""
        try:
            from moegambit.adapters.megatron.moe.rank_quarantine import (
                get_rank_quarantine_registry,
            )
            reg = get_rank_quarantine_registry()
            return reg.quarantined_ranks
        except (ImportError, Exception):
            return frozenset()

    def _refresh_directory(
        self,
        new_ep_group_ranks: List[int],
        failed_rank: int,
        replacement_rank: int,
        step: int,
    ) -> None:
        """Refresh the ActiveExpertDirectory."""
        try:
            from moegambit.adapters.megatron.moe.expert_directory import (
                get_active_expert_directory,
            )
            directory = get_active_expert_directory()
            if directory is None:
                return

            # Refresh host-rank mapping
            directory.refresh_from_placement(new_ep_group_ranks)

            logger.debug(
                "Dispatch topology: refreshed ActiveExpertDirectory "
                "(new_ranks=%s, step=%d)",
                new_ep_group_ranks, step,
            )
        except (ImportError, Exception) as e:
            logger.debug("Dispatch topology: directory refresh skipped: %s", e)

    def _refresh_replacement_registry(
        self,
        failed_rank: int,
        replacement_rank: int,
        step: int,
    ) -> None:
        """Update the ReplacementRegistry after topology refresh."""
        try:
            from moegambit.adapters.megatron.moe.replacement_registry import (
                get_replacement_registry,
                ReplacementState,
            )
            reg = get_replacement_registry()
            status = reg.query_replacement_status(failed_rank)

            # If replacement is READY_FOR_REPAIR, mark as INTEGRATED
            if status == ReplacementState.READY_FOR_REPAIR:
                reg.mark_integrated(failed_rank, step)
                logger.info(
                    "Dispatch topology: marked replacement for "
                    "failed_rank=%d as INTEGRATED (step=%d)",
                    failed_rank, step,
                )
        except (ImportError, Exception) as e:
            logger.debug(
                "Dispatch topology: replacement registry update skipped: %s", e,
            )



# =====================================================================
# Global singleton
# =====================================================================

_MANAGER: Optional[DispatchTopologyManager] = None


def get_dispatch_topology_manager() -> DispatchTopologyManager:
    """Get or create the global ``DispatchTopologyManager`` singleton."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = DispatchTopologyManager()
    return _MANAGER


def clear_dispatch_topology_manager() -> None:
    """Reset the global manager (for testing)."""
    global _MANAGER
    if _MANAGER is not None:
        _MANAGER.reset()
    _MANAGER = None


# =====================================================================
# Top-level convenience API
# =====================================================================

def refresh_dispatch_topology(
    new_ep_group_ranks: List[int],
    failed_rank: int = -1,
    replacement_rank: int = -1,
    step: int = 0,
    **kwargs,
) -> DispatchTopologySnapshot:
    """Refresh the dispatch topology after a group rebuild.

    This is the primary hook called after ``finish_group_repair()``.

    See ``DispatchTopologyManager.refresh_dispatch_topology()`` for details.
    """
    mgr = get_dispatch_topology_manager()
    return mgr.refresh_dispatch_topology(
        new_ep_group_ranks=new_ep_group_ranks,
        failed_rank=failed_rank,
        replacement_rank=replacement_rank,
        step=step,
        **kwargs,
    )


def get_active_expert_host(
    layer_id: int,
    expert_id: int,
) -> Optional[int]:
    """Return the global rank hosting the given expert.

    See ``DispatchTopologyManager.get_active_expert_host()`` for details.
    """
    mgr = get_dispatch_topology_manager()
    return mgr.get_active_expert_host(layer_id, expert_id)


def is_rank_active_for_dispatch(global_rank: int) -> bool:
    """Return ``True`` if the rank is active for dispatch.

    See ``DispatchTopologyManager.is_rank_active_for_dispatch()`` for details.
    """
    mgr = get_dispatch_topology_manager()
    return mgr.is_rank_active_for_dispatch(global_rank)

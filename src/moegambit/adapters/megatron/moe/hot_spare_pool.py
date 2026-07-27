# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hot-Spare Node Pool for MOEGAMBIT-MoE.

This module manages a pool of hot-spare GPU ranks that stand by as
independent daemon processes, completely outside the training torchrun world.
When a training rank fails, the pool allocates a spare to replace it.

Architecture (Daemon Mode)
--------------------------
Hot-spare ranks are NOT launched by torchrun and do NOT participate in the
initial NCCL world.  They run as independent Python processes on dedicated
spare nodes, managed by ``hot_spare_daemon.py``.  On fault:

1. RecoveryController detects failure and calls ``allocate_spare()``.
2. The pool signals the daemon via TCPStore / socket.
3. The daemon joins a NEW NCCL process group created by
   ``GroupRebuildCoordinator`` at the next safe-point.
4. State restoration proceeds via DenseParamSync + StaleExpertRestore.

This design avoids all NCCL collective issues — spare ranks never participate
in any ``new_group()`` call until they are explicitly activated.

Integration with MOEGAMBIT-MoE Stack
------------------------------
* ``RecoveryController`` (Step 11) calls ``allocate_spare()`` on fault.
* ``ReplacementRegistry`` (Step 6) registers the allocated spare.
* ``GroupRebuildCoordinator`` (Step 7) rebuilds NCCL groups including the spare.
* ``DenseParamSync`` (Step 9) pulls current non-expert state to the spare.
* ``StaleExpertRestore`` (Step 10) loads expert shards from checkpoint.

Configuration
-------------
Enabled via ``--moe-moegambit-hot-spare-pool`` flag (requires ``--moe-moegambit-enable``).
The number of spare ranks is set via ``--moe-moegambit-num-hot-spares``.
Training torchrun world size is NOT affected (remains 64 GPUs for 8 nodes).

Usage::

    from megatron.core.transformer.moe.hot_spare_pool import (
        get_hot_spare_pool,
        initialize_hot_spare_pool,
    )

    # At training init:
    initialize_hot_spare_pool(
        num_spares=8,
        spare_addresses=["node9:gpu0", ..., "node9:gpu7"],
        control_store=store,
    )

    # On fault detection:
    pool = get_hot_spare_pool()
    spare_info = pool.allocate_spare(failed_rank=4, step=100)
    if spare_info is not None:
        # Signal daemon to join NCCL group rebuild
        ...
    else:
        # No spares available — fall back to checkpoint restart
        ...
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =====================================================================
# Spare rank states
# =====================================================================

class SpareState(enum.IntEnum):
    """Lifecycle states for a hot-spare rank."""

    STANDBY = 0
    """Spare is idle, pre-warmed, waiting for assignment.
    NOT participating in any training NCCL group."""

    ALLOCATED = 1
    """Spare has been assigned to replace a failed rank.
    Bootstrap in progress (loading state from peer/checkpoint).
    Still NOT in training NCCL groups."""

    ACTIVATING = 2
    """Spare is being integrated into NCCL groups at a safe-point.
    Transitional state during group_rebuild."""

    ACTIVE = 3
    """Spare is now a full training participant.
    It has been removed from the spare pool."""

    FAILED = 4
    """Spare itself failed before activation (rare).
    Removed from pool."""


# =====================================================================
# Spare slot metadata
# =====================================================================

@dataclass
class SpareSlot:
    """Metadata for a single hot-spare rank."""

    spare_rank: int
    """Global rank of this spare in the torch.distributed world."""

    state: SpareState = SpareState.STANDBY

    # Assignment info (set when allocated)
    assigned_to_failed_rank: int = -1
    """The failed rank this spare is replacing (-1 if unassigned)."""

    allocated_step: int = -1
    """Training step when this spare was allocated."""

    activated_step: int = -1
    """Training step when this spare joined training groups."""

    allocated_time: float = 0.0
    activated_time: float = 0.0

    # Pre-warming status
    checkpoint_loaded: bool = False
    """Whether the spare has pre-loaded the latest checkpoint."""

    model_skeleton_ready: bool = False
    """Whether the spare has initialized the model skeleton."""

    def is_available(self) -> bool:
        """Return True if this spare can be allocated."""
        return self.state == SpareState.STANDBY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spare_rank": self.spare_rank,
            "state": self.state.name,
            "assigned_to_failed_rank": self.assigned_to_failed_rank,
            "allocated_step": self.allocated_step,
            "activated_step": self.activated_step,
            "checkpoint_loaded": self.checkpoint_loaded,
            "model_skeleton_ready": self.model_skeleton_ready,
        }


# =====================================================================
# HotSparePool — the main pool manager
# =====================================================================

class HotSparePool:
    """Manages a pool of hot-spare GPU ranks for instant fault recovery.

    The pool tracks which spare ranks are available, allocates them on
    fault events, and coordinates their activation into training groups.

    NCCL Safety:
        Spare ranks are NEVER added to training NCCL groups by this module.
        Group integration is delegated to ``GroupRebuildCoordinator`` which
        operates only at safe-points.  This module only manages the logical
        allocation and readiness tracking.

    Communication:
        All coordination between the pool manager (running on training ranks)
        and spare ranks uses a Gloo-backend control process group or
        TCPStore — never NCCL.  This prevents any possibility of collective
        hangs from spare ranks.
    """

    def __init__(
        self,
        spare_ranks: List[int],
        world_size: int,
        training_world_size: int,
        control_store: Optional[Any] = None,
    ) -> None:
        """Initialize the hot-spare pool.

        Args:
            spare_ranks: List of global ranks designated as hot spares.
                These ranks must NOT be included in any training NCCL group.
            world_size: Total world size (training ranks + spare ranks).
            training_world_size: Number of active training ranks.
            control_store: Optional TCPStore for out-of-band communication
                with spare ranks.  If None, uses environment-based signaling.
        """
        self._world_size = world_size
        self._training_world_size = training_world_size
        self._control_store = control_store

        # Initialize spare slots
        self._slots: Dict[int, SpareSlot] = {}
        for rank in spare_ranks:
            self._slots[rank] = SpareSlot(spare_rank=rank)

        # Allocation history
        self._allocation_history: List[Dict[str, Any]] = []

        # Callbacks for spare rank communication (set by moegambit_integration)
        self._notify_spare_fn: Optional[Callable] = None
        self._query_spare_ready_fn: Optional[Callable] = None

        logger.info(
            "MOEGAMBIT-MoE HotSparePool initialized: %d spares, "
            "training_world=%d, total_world=%d, spare_ranks=%s",
            len(spare_ranks), training_world_size, world_size, spare_ranks,
        )

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    def set_callbacks(
        self,
        notify_spare_fn: Optional[Callable] = None,
        query_spare_ready_fn: Optional[Callable] = None,
    ) -> None:
        """Register callbacks for spare rank communication.

        Args:
            notify_spare_fn: Called to notify a spare rank of its assignment.
                Signature: (spare_rank, failed_rank, step, role_descriptor) -> bool
                Must use Gloo/TCPStore, NOT NCCL.
            query_spare_ready_fn: Called to check if a spare has finished
                bootstrap.
                Signature: (spare_rank) -> bool
                Must use Gloo/TCPStore, NOT NCCL.
        """
        if notify_spare_fn is not None:
            self._notify_spare_fn = notify_spare_fn
        if query_spare_ready_fn is not None:
            self._query_spare_ready_fn = query_spare_ready_fn

    # -----------------------------------------------------------------
    # Core allocation API
    # -----------------------------------------------------------------

    @property
    def num_available(self) -> int:
        """Number of spare ranks currently available for allocation."""
        return sum(1 for s in self._slots.values() if s.is_available())

    @property
    def num_total(self) -> int:
        """Total number of spare slots (including allocated/active)."""
        return len(self._slots)

    @property
    def available_spare_ranks(self) -> List[int]:
        """Return list of spare ranks in STANDBY state."""
        return sorted(
            s.spare_rank for s in self._slots.values() if s.is_available()
        )

    def has_available_spare(self) -> bool:
        """Return True if at least one spare is available."""
        return self.num_available > 0

    def can_handle_n_failures(self, n: int) -> bool:
        """Return True if the pool can handle n simultaneous failures."""
        return self.num_available >= n

    def allocate_spare(
        self,
        failed_rank: int,
        step: int = 0,
        reason: str = "",
    ) -> Optional[int]:
        """Allocate a spare rank to replace a failed training rank.

        This does NOT add the spare to any NCCL group — it only marks the
        spare as ALLOCATED and optionally notifies it via the control channel.

        The spare will be integrated into NCCL groups later at a safe-point
        by ``GroupRebuildCoordinator``.

        Args:
            failed_rank: The global rank that failed.
            step: Current training step.
            reason: Human-readable reason for allocation.

        Returns:
            The allocated spare rank, or None if no spares are available.
        """
        available = self.available_spare_ranks
        if not available:
            logger.warning(
                "MOEGAMBIT-MoE HotSparePool: no spares available for "
                "failed_rank=%d (step=%d). Pool exhausted (%d/%d used).",
                failed_rank, step,
                self.num_total - self.num_available, self.num_total,
            )
            return None

        # Select the first available spare (FIFO policy)
        # Future: could use topology-aware selection (same node, same switch)
        spare_rank = available[0]
        slot = self._slots[spare_rank]

        # Transition to ALLOCATED
        slot.state = SpareState.ALLOCATED
        slot.assigned_to_failed_rank = failed_rank
        slot.allocated_step = step
        slot.allocated_time = time.monotonic()

        # Record allocation
        record = {
            "spare_rank": spare_rank,
            "failed_rank": failed_rank,
            "step": step,
            "reason": reason,
            "timestamp": slot.allocated_time,
            "remaining_spares": self.num_available,
        }
        self._allocation_history.append(record)

        logger.warning(
            "MOEGAMBIT-MoE HotSparePool: allocated spare_rank=%d for "
            "failed_rank=%d (step=%d, reason=%r, remaining=%d/%d)",
            spare_rank, failed_rank, step, reason,
            self.num_available, self.num_total,
        )

        # Notify spare rank via control channel (Gloo/TCPStore, NOT NCCL)
        if self._notify_spare_fn is not None:
            try:
                self._notify_spare_fn(
                    spare_rank=spare_rank,
                    failed_rank=failed_rank,
                    step=step,
                )
            except Exception as exc:
                logger.error(
                    "MOEGAMBIT-MoE HotSparePool: failed to notify spare_rank=%d: %s",
                    spare_rank, exc,
                )

        # Also signal via TCPStore if available
        if self._control_store is not None:
            try:
                key = f"moegambit_spare_assign_{spare_rank}"
                value = f"{failed_rank}:{step}"
                self._control_store.set(key, value)
            except Exception as exc:
                logger.warning(
                    "MOEGAMBIT-MoE HotSparePool: TCPStore set failed for "
                    "spare_rank=%d: %s", spare_rank, exc,
                )

        return spare_rank

    def mark_activating(self, spare_rank: int, step: int = 0) -> bool:
        """Mark a spare as entering NCCL group integration (safe-point).

        Called by GroupRebuildCoordinator when it begins adding the spare
        to training process groups.

        Args:
            spare_rank: The spare rank being activated.
            step: Current training step.

        Returns:
            True if transition succeeded, False otherwise.
        """
        slot = self._slots.get(spare_rank)
        if slot is None:
            logger.error(
                "MOEGAMBIT-MoE HotSparePool: mark_activating called for "
                "unknown spare_rank=%d", spare_rank,
            )
            return False

        if slot.state != SpareState.ALLOCATED:
            logger.error(
                "MOEGAMBIT-MoE HotSparePool: mark_activating called for "
                "spare_rank=%d in state %s (expected ALLOCATED)",
                spare_rank, slot.state.name,
            )
            return False

        slot.state = SpareState.ACTIVATING
        logger.info(
            "MOEGAMBIT-MoE HotSparePool: spare_rank=%d → ACTIVATING (step=%d)",
            spare_rank, step,
        )
        return True

    def mark_active(self, spare_rank: int, step: int = 0) -> bool:
        """Mark a spare as fully integrated into training.

        Called after GroupRebuildCoordinator has successfully added the
        spare to all required NCCL groups and the replacement is INTEGRATED.

        Args:
            spare_rank: The spare rank that is now active.
            step: Current training step.

        Returns:
            True if transition succeeded, False otherwise.
        """
        slot = self._slots.get(spare_rank)
        if slot is None:
            return False

        if slot.state not in (SpareState.ALLOCATED, SpareState.ACTIVATING):
            logger.error(
                "MOEGAMBIT-MoE HotSparePool: mark_active called for "
                "spare_rank=%d in state %s", spare_rank, slot.state.name,
            )
            return False

        slot.state = SpareState.ACTIVE
        slot.activated_step = step
        slot.activated_time = time.monotonic()

        latency = slot.activated_time - slot.allocated_time
        logger.warning(
            "MOEGAMBIT-MoE HotSparePool: spare_rank=%d → ACTIVE "
            "(step=%d, allocation_to_active=%.2fs, remaining=%d/%d)",
            spare_rank, step, latency,
            self.num_available, self.num_total,
        )
        return True

    def mark_spare_failed(self, spare_rank: int, reason: str = "") -> bool:
        """Mark a spare rank as failed (before it was activated).

        Args:
            spare_rank: The spare rank that failed.
            reason: Human-readable reason.

        Returns:
            True if the spare was found and marked, False otherwise.
        """
        slot = self._slots.get(spare_rank)
        if slot is None:
            return False

        if slot.state == SpareState.ACTIVE:
            # Already active — this is a training rank failure, not a spare failure
            return False

        old_state = slot.state
        slot.state = SpareState.FAILED
        logger.warning(
            "MOEGAMBIT-MoE HotSparePool: spare_rank=%d FAILED (was %s, reason=%r)",
            spare_rank, old_state.name, reason,
        )
        return True

    # -----------------------------------------------------------------
    # Query API
    # -----------------------------------------------------------------

    def is_spare_rank(self, rank: int) -> bool:
        """Return True if the given rank is (or was) a spare."""
        return rank in self._slots

    def is_spare_in_standby(self, rank: int) -> bool:
        """Return True if the given rank is a spare in STANDBY."""
        slot = self._slots.get(rank)
        return slot is not None and slot.state == SpareState.STANDBY

    def get_slot(self, spare_rank: int) -> Optional[SpareSlot]:
        """Return the SpareSlot for a given rank, or None."""
        return self._slots.get(spare_rank)

    def get_spare_for_failed_rank(self, failed_rank: int) -> Optional[int]:
        """Return the spare rank allocated for a given failed rank."""
        for slot in self._slots.values():
            if (slot.assigned_to_failed_rank == failed_rank and
                    slot.state in (SpareState.ALLOCATED, SpareState.ACTIVATING)):
                return slot.spare_rank
        return None

    def is_spare_ready(self, spare_rank: int) -> bool:
        """Check if an allocated spare has finished bootstrap.

        Uses the registered query callback (Gloo/TCPStore based).
        """
        if self._query_spare_ready_fn is not None:
            try:
                return self._query_spare_ready_fn(spare_rank)
            except Exception:
                pass

        # Fallback: check TCPStore
        if self._control_store is not None:
            try:
                key = f"moegambit_spare_ready_{spare_rank}"
                value = self._control_store.get(key)
                return value == b"1" or value == "1"
            except Exception:
                pass

        return False

    # -----------------------------------------------------------------
    # Summary / monitoring
    # -----------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a summary of pool state for logging/monitoring."""
        state_counts: Dict[str, int] = {}
        for slot in self._slots.values():
            name = slot.state.name
            state_counts[name] = state_counts.get(name, 0) + 1

        return {
            "total_spares": self.num_total,
            "available": self.num_available,
            "state_counts": state_counts,
            "available_ranks": self.available_spare_ranks,
            "allocation_history_count": len(self._allocation_history),
        }

    def __repr__(self) -> str:
        return (
            f"HotSparePool(total={self.num_total}, "
            f"available={self.num_available}, "
            f"ranks={sorted(self._slots.keys())})"
        )


# =====================================================================
# Global singleton
# =====================================================================

_POOL: Optional[HotSparePool] = None


def get_hot_spare_pool() -> Optional[HotSparePool]:
    """Get the global HotSparePool singleton, or None if not initialized."""
    return _POOL


def initialize_hot_spare_pool(
    spare_ranks: List[int],
    world_size: int,
    training_world_size: int,
    control_store: Optional[Any] = None,
) -> HotSparePool:
    """Initialize the global HotSparePool singleton.

    Args:
        spare_ranks: List of global ranks designated as hot spares.
        world_size: Total world size (training + spares).
        training_world_size: Number of active training ranks.
        control_store: Optional TCPStore for control-plane communication.

    Returns:
        The initialized HotSparePool.
    """
    global _POOL
    _POOL = HotSparePool(
        spare_ranks=spare_ranks,
        world_size=world_size,
        training_world_size=training_world_size,
        control_store=control_store,
    )
    return _POOL


def clear_hot_spare_pool() -> None:
    """Reset the global pool (for testing)."""
    global _POOL
    _POOL = None

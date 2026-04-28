# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Minimal Fault-Injection & End-to-End Test Framework for BSR-MoE (Step 15).

This module provides a **self-contained, deterministic fault-injection
framework** that exercises the full BSR-MoE recovery stack without
requiring distributed init, GPUs, or real NCCL collectives.

It wires together BSR-MoE modules through their public singleton APIs
and callback interfaces, simulating the training loop and fault events
in a single process.

Fault injection primitives
--------------------------
The ``FaultInjector`` class provides four controllable fault primitives:

1. ``inject_hard_rank_failure(global_rank, step, reason)``
   — simulate a hard rank failure (cannot participate in collectives).
2. ``inject_replacement_ready(failed_rank, replacement_rank, step)``
   — simulate a replacement rank coming online and reporting ready.
3. ``inject_safe_point(step)``
   — simulate an iteration boundary (safe point).
4. ``run_training_steps(start_step, num_steps)``
   — simulate a sequence of normal training iterations.

End-to-end test scenario
------------------------
**Scenario B: hard-failed rank**
  Normal training → hard rank failure → pending group repair →
  replacement rank online → safe-point group rebuild → dispatch topology
  refresh → stale expert restore → deferred optimizer load → safe
  reintegration.

Recovery metrics
----------------
The ``RecoveryMetrics`` dataclass tracks timing and count metrics:

* ``time_to_pending_group_repair`` — wall-clock from fault to pending repair
* ``time_to_replacement_ready`` — wall-clock from fault to replacement ready
* ``time_to_safe_point_repair`` — wall-clock from fault to safe-point repair
* ``time_to_stale_runnable`` — wall-clock from fault to STALE_RUNNABLE
* ``time_to_fully_recovered`` — wall-clock from fault to FULLY_RECOVERED
* ``degraded_iterations`` — number of training iterations in degraded mode

Design principles
-----------------
* **Self-contained** — all BSR-MoE modules are imported via
  ``_import_from_file`` (no package-level imports needed).
* **Deterministic** — no randomness, no real collectives.
* **Single-process** — everything runs in one Python process.
* **Callback-driven** — the ``RecoveryController`` is wired with
  callbacks that delegate to the real BSR-MoE module singletons.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =====================================================================
# Recovery metrics
# =====================================================================

@dataclass
class RecoveryMetrics:
    """Tracks timing and count metrics for a single fault recovery."""

    failed_rank: int = -1
    """The rank that experienced the fault."""

    fault_type: str = ""
    """'hard' (failed)."""

    fault_timestamp: float = -1.0
    """Wall-clock time when the fault was injected."""

    # Timing metrics (wall-clock seconds from fault_timestamp)
    time_to_pending_group_repair: float = -1.0
    """Time from fault to pending group repair state."""

    time_to_replacement_ready: float = -1.0
    """Time from fault to replacement rank ready."""

    time_to_safe_point_repair: float = -1.0
    """Time from fault to safe-point repair execution."""

    time_to_stale_runnable: float = -1.0
    """Time from fault to experts entering STALE_RUNNABLE."""

    time_to_fully_recovered: float = -1.0
    """Time from fault to experts entering FULLY_RECOVERED."""

    # Count metrics
    degraded_iterations: int = 0
    """Number of training iterations spent in degraded mode."""

    def _elapsed(self, ts: float) -> float:
        """Compute elapsed time from fault_timestamp."""
        if self.fault_timestamp < 0 or ts < 0:
            return -1.0
        return ts - self.fault_timestamp

    def mark_pending_group_repair(self) -> None:
        self.time_to_pending_group_repair = self._elapsed(time.time())

    def mark_replacement_ready(self) -> None:
        self.time_to_replacement_ready = self._elapsed(time.time())

    def mark_safe_point_repair(self) -> None:
        self.time_to_safe_point_repair = self._elapsed(time.time())

    def mark_stale_runnable(self) -> None:
        self.time_to_stale_runnable = self._elapsed(time.time())

    def mark_fully_recovered(self) -> None:
        self.time_to_fully_recovered = self._elapsed(time.time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failed_rank": self.failed_rank,
            "fault_type": self.fault_type,
            "fault_timestamp": self.fault_timestamp,
            "time_to_pending_group_repair": self.time_to_pending_group_repair,
            "time_to_replacement_ready": self.time_to_replacement_ready,
            "time_to_safe_point_repair": self.time_to_safe_point_repair,
            "time_to_stale_runnable": self.time_to_stale_runnable,
            "time_to_fully_recovered": self.time_to_fully_recovered,
            "degraded_iterations": self.degraded_iterations,
        }


# =====================================================================
# Fault event record
# =====================================================================

@dataclass
class FaultEvent:
    """A single fault-injection event in the audit trail."""

    step: int = -1
    event_type: str = ""
    target_rank: int = -1
    target_layer: int = -1
    target_expert: int = -1
    replacement_rank: int = -1
    timestamp: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "event_type": self.event_type,
            "target_rank": self.target_rank,
            "target_layer": self.target_layer,
            "target_expert": self.target_expert,
            "replacement_rank": self.replacement_rank,
            "timestamp": self.timestamp,
            "details": dict(self.details),
        }


# =====================================================================
# Simulated training environment
# =====================================================================

@dataclass
class SimulatedEnvironment:
    """Describes the simulated MoE training environment.

    This is a lightweight description of the parallel topology used
    by the fault injector to compute expert placements, group ranks,
    etc.  No real distributed init is needed.
    """

    num_experts: int = 8
    """Total number of global experts."""

    num_layers: int = 2
    """Number of MoE layers."""

    ep_size: int = 4
    """Expert-parallel size."""

    ep_group_ranks: Optional[List[int]] = None
    """EP group ranks.  If None, defaults to [0, 1, ..., ep_size-1]."""

    dp_group_ranks: Optional[List[int]] = None
    """DP group ranks.  If None, defaults to [0, 1, ..., ep_size-1]."""

    checkpoint_step: int = 0
    """Step of the most recent checkpoint (for stale expert restore)."""

    checkpoint_dir: str = "/tmp/bsr_test_ckpt"
    """Simulated checkpoint directory."""

    def __post_init__(self):
        if self.ep_group_ranks is None:
            self.ep_group_ranks = list(range(self.ep_size))
        if self.dp_group_ranks is None:
            self.dp_group_ranks = list(range(self.ep_size))

    @property
    def num_local_experts(self) -> int:
        return self.num_experts // self.ep_size

    def experts_on_rank(self, global_rank: int) -> List[int]:
        """Compute which global expert IDs are hosted on a given rank."""
        if global_rank not in self.ep_group_ranks:
            return []
        ep_rank = self.ep_group_ranks.index(global_rank)
        nle = self.num_local_experts
        return list(range(ep_rank * nle, (ep_rank + 1) * nle))

    def rank_hosting_expert(self, expert_id: int) -> int:
        """Compute which global rank hosts a given expert."""
        nle = self.num_local_experts
        ep_rank = expert_id // nle
        if ep_rank >= len(self.ep_group_ranks):
            return -1
        return self.ep_group_ranks[ep_rank]


# =====================================================================
# Fault Injector
# =====================================================================

class FaultInjector:
    """Controllable fault-injection engine for BSR-MoE testing.

    Wires together BSR-MoE module singletons through the
    ``RecoveryController`` callback interface.  Provides fault
    primitives and tracks recovery metrics.

    Args:
        env: Simulated training environment description.
        modules: Dict of BSR-MoE module references (from _import_from_file).
            Expected keys: 'directory', 'replacement', 'group_rebuild',
            'topology', 'dense_sync', 'stale_restore', 'deferred_opt',
            'recovery_controller', 'reintegration_barrier'.
    """

    def __init__(
        self,
        env: SimulatedEnvironment,
        modules: Dict[str, Any],
    ) -> None:
        self._env = env
        self._modules = modules

        # Metrics per failed_rank
        self._metrics: Dict[int, RecoveryMetrics] = {}

        # Event log
        self._event_log: List[FaultEvent] = []

        # Degraded iteration counter
        self._degraded_iteration_count: int = 0

        # Current training step
        self._current_step: int = 0

        # Track which modules have been initialized
        self._initialized = False

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize all BSR-MoE module singletons and wire callbacks.

        Must be called before any fault injection.
        """
        env = self._env
        m = self._modules

        # 1. Clear all singletons
        self._clear_all_singletons()

        # 2. Initialize expert directory (Step 5)
        dir_mod = m['directory']
        directory = dir_mod.ActiveExpertDirectory.from_placement(
            num_layers=env.num_layers,
            num_experts=env.num_experts,
            ep_size=env.ep_size,
            ep_group_ranks=env.ep_group_ranks,
        )
        dir_mod.set_active_expert_directory(directory)

        # 3. Initialize replacement registry (Step 6)
        m['replacement'].get_replacement_registry()

        # 4. Initialize group rebuild coordinator (Step 7)
        m['group_rebuild'].get_group_rebuild_coordinator()

        # 5. Initialize dispatch topology manager (Step 8)
        topo_mod = m['topology']
        topo_mgr = topo_mod.get_dispatch_topology_manager()
        topo_mgr.initialise(
            num_layers=env.num_layers,
            num_experts=env.num_experts,
            ep_size=env.ep_size,
            ep_group_ranks=env.ep_group_ranks,
        )

        # 6. Initialize reintegration barrier (Step 14)
        m['reintegration_barrier'].get_reintegration_barrier()

        # 7. Initialize recovery controller (Step 11) and wire callbacks
        ctrl_mod = m['recovery_controller']
        ctrl = ctrl_mod.get_recovery_controller()
        self._wire_callbacks(ctrl)

        self._initialized = True

    def _wire_callbacks(self, ctrl) -> None:
        """Wire RecoveryController callbacks to BSR-MoE module singletons."""
        env = self._env
        m = self._modules

        def health_mark_stale_runnable_fn(*, expert_ids, step=-1):
            # Track via ExpertRecoveryTracker (already in controller)
            pass

        def health_mark_healthy_fn(*, expert_ids, step=-1):
            # Track via ExpertRecoveryTracker (already in controller)
            pass

        def replacement_announce_fn(*, failed_rank, replacement_rank, step=-1):
            rep_mod = m['replacement']
            rep_mod.announce_replacement(
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                step=step,
            )

        def replacement_query_fn(*, failed_rank):
            rep_mod = m['replacement']
            return rep_mod.query_replacement_status(failed_rank)

        def replacement_integrate_fn(*, failed_rank, replacement_rank, step=-1):
            rep_mod = m['replacement']
            rep_mod.mark_integrated(failed_rank, step=step)

        def group_rebuild_request_fn(
            *, failed_rank, replacement_rank, step=-1,
            ep_group_ranks=None, dp_group_ranks=None,
        ):
            gb_mod = m['group_rebuild']
            coord = gb_mod.get_group_rebuild_coordinator()
            coord.request_rebuild(
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                step=step,
                old_ep_group_ranks=ep_group_ranks,
                new_ep_group_ranks=gb_mod.GroupRebuildCoordinator.compute_new_group_ranks(
                    ep_group_ranks or [], failed_rank, replacement_rank,
                ),
            )

        def group_rebuild_execute_fn(*, failed_rank, replacement_rank, step=-1):
            gb_mod = m['group_rebuild']
            coord = gb_mod.get_group_rebuild_coordinator()
            coord.maybe_rebuild_groups_at_safe_point(
                rebuild_fn=lambda plan: None,
                rebind_fn=lambda plan: None,
                step=step,
            )

        def group_rebuild_finish_fn(*, failed_rank, replacement_rank, step=-1):
            gb_mod = m['group_rebuild']
            coord = gb_mod.get_group_rebuild_coordinator()
            coord.finish_group_repair(step=step)

        def topology_refresh_fn(
            *, failed_rank, replacement_rank, step=-1, expert_ids=None,
        ):
            topo_mod = m['topology']
            mgr = topo_mod.get_dispatch_topology_manager()
            new_ep = list(env.ep_group_ranks)
            if failed_rank in new_ep:
                idx = new_ep.index(failed_rank)
                new_ep[idx] = replacement_rank
            mgr.refresh_dispatch_topology(
                new_ep_group_ranks=new_ep,
                failed_rank=failed_rank,
                replacement_rank=replacement_rank,
                step=step,
                update_directory=True,
                update_replacement_registry=False,
            )

        def dense_sync_fn(*, failed_rank, replacement_rank, step=-1):
            # Simulated: no real broadcast needed
            pass

        def expert_restore_fn(
            *, failed_rank, replacement_rank, step=-1, expert_ids=None,
        ):
            # Simulate stale expert restore: mark experts as recovering
            if expert_ids and failed_rank in self._metrics:
                self._metrics[failed_rank].mark_stale_runnable()

        ctrl.register_callbacks(
            health_mark_stale_runnable_fn=health_mark_stale_runnable_fn,
            health_mark_healthy_fn=health_mark_healthy_fn,
            replacement_announce_fn=replacement_announce_fn,
            replacement_query_fn=replacement_query_fn,
            replacement_integrate_fn=replacement_integrate_fn,
            group_rebuild_request_fn=group_rebuild_request_fn,
            group_rebuild_execute_fn=group_rebuild_execute_fn,
            group_rebuild_finish_fn=group_rebuild_finish_fn,
            topology_refresh_fn=topology_refresh_fn,
            dense_sync_fn=dense_sync_fn,
            expert_restore_fn=expert_restore_fn,
        )

    def _clear_all_singletons(self) -> None:
        """Clear all BSR-MoE module singletons."""
        m = self._modules
        m['directory'].clear_active_expert_directory()
        m['replacement'].clear_replacement_registry()
        m['group_rebuild'].clear_group_rebuild_coordinator()
        m['topology'].clear_dispatch_topology_manager()
        m['recovery_controller'].clear_recovery_controller()
        m['reintegration_barrier'].clear_reintegration_barrier()
        if hasattr(m.get('deferred_opt', None), 'clear_deferred_optimizer_loader'):
            m['deferred_opt'].clear_deferred_optimizer_loader()
        if hasattr(m.get('stale_restore', None), 'clear_stale_expert_restore_coordinator'):
            m['stale_restore'].clear_stale_expert_restore_coordinator()
        if hasattr(m.get('stale_restore', None), 'clear_optimizer_update_barrier'):
            m['stale_restore'].clear_optimizer_update_barrier()

    # -----------------------------------------------------------------
    # Fault injection primitives
    # -----------------------------------------------------------------

    def inject_hard_rank_failure(
        self,
        global_rank: int,
        step: int = -1,
        reason: str = "test_hard_failure",
    ) -> RecoveryMetrics:
        """Simulate a hard rank failure.

        This triggers the full hard-failure flow through the
        RecoveryController.

        Returns the RecoveryMetrics for this fault.
        """
        expert_ids = self._env.experts_on_rank(global_rank)

        # Create metrics
        metrics = RecoveryMetrics(
            failed_rank=global_rank,
            fault_type="hard",
            fault_timestamp=time.time(),
        )
        self._metrics[global_rank] = metrics

        # Trigger through recovery controller
        ctrl_mod = self._modules['recovery_controller']
        ctrl = ctrl_mod.get_recovery_controller()
        ctrl.on_hard_rank_failure(
            global_rank,
            reason=reason,
            step=step,
            expert_ids=expert_ids,
            ep_group_ranks=list(self._env.ep_group_ranks),
            dp_group_ranks=list(self._env.dp_group_ranks),
        )

        metrics.mark_pending_group_repair()

        self._log_event("hard_rank_failure", step=step,
                        target_rank=global_rank,
                        details={"reason": reason, "expert_ids": expert_ids})

        return metrics

    def inject_replacement_ready(
        self,
        failed_rank: int,
        replacement_rank: int,
        step: int = -1,
    ) -> None:
        """Simulate a replacement rank coming online and reporting ready."""
        ctrl_mod = self._modules['recovery_controller']
        ctrl = ctrl_mod.get_recovery_controller()

        # 1. Assign replacement
        ctrl.on_replacement_assigned(failed_rank, replacement_rank, step=step)

        # 2. Mark ready in replacement registry
        rep_mod = self._modules['replacement']
        rep_mod.announce_replacement_ready(failed_rank, step=step)

        # 3. Notify controller
        ctrl.on_replacement_ready(failed_rank, step=step)

        # Update metrics
        if failed_rank in self._metrics:
            self._metrics[failed_rank].mark_replacement_ready()

        # Begin reintegration barrier tracking
        barrier_mod = self._modules['reintegration_barrier']
        barrier = barrier_mod.get_reintegration_barrier()
        expert_ids = self._env.experts_on_rank(failed_rank)
        barrier.begin_reintegration(
            failed_rank=failed_rank,
            replacement_rank=replacement_rank,
            expert_ids=expert_ids,
            step=step,
        )
        barrier.mark_precondition(failed_rank, "replacement_ready", step=step)

        self._log_event("replacement_ready", step=step,
                        target_rank=failed_rank,
                        replacement_rank=replacement_rank)

    def inject_safe_point(self, step: int) -> bool:
        """Simulate an iteration boundary (safe point).

        Calls RecoveryController.before_iteration() which may trigger
        the full repair sequence.

        Returns True if a repair was executed.
        """
        self._current_step = step

        ctrl_mod = self._modules['recovery_controller']
        ctrl = ctrl_mod.get_recovery_controller()

        repaired = ctrl.before_iteration(step=step)

        if repaired:
            self._complete_post_repair_steps(step)

        # Track degraded iterations
        if ctrl.is_degraded:
            self._degraded_iteration_count += 1
            for m in self._metrics.values():
                if m.time_to_fully_recovered < 0:
                    m.degraded_iterations += 1

        self._log_event("safe_point", step=step,
                        details={"repaired": repaired})

        return repaired

    def _complete_post_repair_steps(self, step: int) -> None:
        """Complete post-repair steps after safe-point repair."""
        barrier_mod = self._modules['reintegration_barrier']
        barrier = barrier_mod.get_reintegration_barrier()

        for failed_rank in list(self._metrics.keys()):
            metrics = self._metrics[failed_rank]
            if metrics.time_to_safe_point_repair >= 0:
                continue  # Already processed

            metrics.mark_safe_point_repair()

            # Mark remaining barrier preconditions
            for precond in [
                "groups_repaired",
                "directory_refreshed",
                "topology_refreshed",
                "experts_restorable",
            ]:
                barrier.mark_precondition(failed_rank, precond, step=step)

            # Simulate deferred optimizer load
            self._simulate_deferred_optimizer_load(failed_rank, step)

            # Execute reintegration via barrier
            if barrier.can_reintegrate(failed_rank):
                barrier.execute_reintegration(
                    failed_rank, step=step,
                    re_enable_routing_fn=lambda eids, s: None,
                    clear_degraded_fn=lambda eids, s: None,
                )

    def _simulate_deferred_optimizer_load(
        self,
        failed_rank: int,
        step: int,
    ) -> None:
        """Simulate deferred optimizer state loading."""
        expert_ids = self._env.experts_on_rank(failed_rank)
        deferred_mod = self._modules.get('deferred_opt')
        if deferred_mod is None:
            if failed_rank in self._metrics:
                self._metrics[failed_rank].mark_fully_recovered()
            return

        loader = deferred_mod.get_deferred_optimizer_loader()

        # Submit load requests
        for layer in range(self._env.num_layers):
            for eid in expert_ids:
                loader.submit_load(
                    layer_id=layer,
                    expert_id=eid,
                    checkpoint_step=self._env.checkpoint_step,
                    step=step,
                )

        # Execute loads (simulated)
        loader.execute_pending_loads(
            load_fn=lambda req: True,
            step=step,
        )

        # Finalize
        stale_mod = self._modules.get('stale_restore')
        opt_barrier = None
        if stale_mod and hasattr(stale_mod, 'get_optimizer_update_barrier'):
            opt_barrier = stale_mod.get_optimizer_update_barrier()

        loader.finalize_loaded(
            barrier=opt_barrier,
            health_managers=None,
            step=step,
        )

        # Update metrics
        if failed_rank in self._metrics:
            self._metrics[failed_rank].mark_fully_recovered()

    # -----------------------------------------------------------------
    # Simulated training loop
    # -----------------------------------------------------------------

    def run_training_steps(self, start_step: int, num_steps: int) -> None:
        """Simulate a sequence of normal training iterations."""
        for step in range(start_step, start_step + num_steps):
            self.inject_safe_point(step)
            self._current_step = step

    # -----------------------------------------------------------------
    # Finalize: call after_iteration to complete reintegration
    # -----------------------------------------------------------------

    def finalize_iteration(self, step: int) -> None:
        """Call after_iteration and then before_iteration to finalize."""
        ctrl_mod = self._modules['recovery_controller']
        ctrl = ctrl_mod.get_recovery_controller()
        ctrl.after_iteration(step=step)
        ctrl.before_iteration(step=step + 1)

    # -----------------------------------------------------------------
    # Accessors
    # -----------------------------------------------------------------

    def get_metrics(self, failed_rank: int) -> Optional[RecoveryMetrics]:
        """Get recovery metrics for a specific failed rank."""
        return self._metrics.get(failed_rank)

    @property
    def all_metrics(self) -> Dict[int, RecoveryMetrics]:
        """All recovery metrics."""
        return dict(self._metrics)

    @property
    def event_log(self) -> List[FaultEvent]:
        """Event log (read-only copy)."""
        return list(self._event_log)

    @property
    def degraded_iteration_count(self) -> int:
        return self._degraded_iteration_count

    def get_controller_phase(self) -> str:
        """Get current RecoveryController phase name."""
        ctrl_mod = self._modules['recovery_controller']
        ctrl = ctrl_mod.get_recovery_controller()
        return ctrl.phase.name

    # -----------------------------------------------------------------
    # Summary / reset
    # -----------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "current_step": self._current_step,
            "num_faults": len(self._metrics),
            "degraded_iterations": self._degraded_iteration_count,
            "controller_phase": self.get_controller_phase() if self._initialized else "N/A",
            "num_events": len(self._event_log),
        }

    def reset(self) -> None:
        """Reset all state."""
        if self._initialized:
            self._clear_all_singletons()
        self._metrics.clear()
        self._event_log.clear()
        self._degraded_iteration_count = 0
        self._current_step = 0
        self._initialized = False

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _log_event(
        self,
        event_type: str,
        *,
        step: int = -1,
        target_rank: int = -1,
        target_layer: int = -1,
        target_expert: int = -1,
        replacement_rank: int = -1,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        event = FaultEvent(
            step=step,
            event_type=event_type,
            target_rank=target_rank,
            target_layer=target_layer,
            target_expert=target_expert,
            replacement_rank=replacement_rank,
            timestamp=time.time(),
            details=details or {},
        )
        self._event_log.append(event)


# =====================================================================
# E2E Scenario Runner
# =====================================================================

def run_scenario_b_hard_failure(
    env: SimulatedEnvironment,
    modules: Dict[str, Any],
    *,
    normal_steps: int = 5,
    degraded_steps: int = 3,
    failed_rank: int = 2,
    replacement_rank: int = 11,
) -> Tuple[FaultInjector, RecoveryMetrics]:
    """Run Scenario B: hard-failed rank.

    Steps:
    1. Normal training for ``normal_steps`` iterations.
    2. Inject hard rank failure.
    3. System enters pending group repair.
    4. Run ``degraded_steps`` degraded iterations.
    5. Replacement rank comes online.
    6. Safe-point group rebuild.
    7. Dispatch topology refresh.
    8. Stale expert restore.
    9. Deferred optimizer-state load.
    10. Safe reintegration.

    Returns:
        (FaultInjector, RecoveryMetrics) for inspection.
    """
    injector = FaultInjector(env, modules)
    injector.initialize()

    # 1. Normal training
    injector.run_training_steps(0, normal_steps)

    # 2. Inject hard rank failure
    fault_step = normal_steps
    metrics = injector.inject_hard_rank_failure(
        failed_rank, step=fault_step, reason="test_scenario_b",
    )

    # 3. Verify pending group repair
    assert injector.get_controller_phase() == "PENDING_GROUP_REPAIR"

    # 4. Run degraded iterations
    injector.run_training_steps(fault_step + 1, degraded_steps)

    # 5. Replacement rank comes online
    replacement_step = fault_step + degraded_steps + 1
    injector.inject_replacement_ready(
        failed_rank, replacement_rank, step=replacement_step,
    )

    # 6-10. Safe-point repair (triggers full sequence)
    repair_step = replacement_step + 1
    repaired = injector.inject_safe_point(repair_step)
    assert repaired, "Safe-point repair should have been executed"

    # Finalize reintegration
    injector.finalize_iteration(repair_step)

    return injector, metrics

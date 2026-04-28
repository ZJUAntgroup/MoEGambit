# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""BSR-MoE End-to-End Fault Injection & Test Framework (Phase 14).

This module provides a **minimal, deterministic** fault injection framework
for validating the entire BSR-MoE recovery pipeline in unit tests.  It
orchestrates all BSR-MoE subsystems (RecoveryController, IterationInvalidator,
RollbackReplayManager, OptimizerCommitGuard, HardFailureDetector,
PipelineRollbackCoordinator) through a simulated training loop.

Design goals
------------
* **No torch / no distributed** — runs in the same stub environment as
  other BSR-MoE unit tests.
* **Deterministic** — fault timing is step-based, not wall-clock-based.
* **Observable** — every recovery phase transition is recorded with
  timestamps so that latency metrics can be computed.
* **Composable** — multiple fault scenarios can be chained.

Architecture
------------
::

    ┌──────────────────────────────────────────────────┐
    │                 SimulatedTrainingLoop             │
    │  for step in range(N):                           │
    │      injector.maybe_inject(step)                 │
    │      invalidator.begin_iteration(step)           │
    │      commit_guard.begin_iteration(step)          │
    │      controller.before_iteration(step)           │
    │      ... simulated forward/backward ...          │
    │      if commit_guard.should_commit():             │
    │          ... optimizer.step() ...                 │
    │      invalidator.end_iteration(step)             │
    │      commit_guard.end_iteration(step)            │
    │      controller.after_iteration(step)            │
    │      metrics.record_step(step, controller)       │
    └──────────────────────────────────────────────────┘

Metrics collected
-----------------
* ``time_to_detect``       — fault step → first phase transition
* ``time_to_rollback``     — fault step → ROLLBACK_PENDING entered
* ``time_to_replacement``  — fault step → WAITING_FOR_REPLACEMENT entered
* ``time_to_group_repair`` — fault step → group_rebuild_finish_fn called
* ``time_to_pp_repair``    — fault step → pipeline_stage_repair_fn called
* ``time_to_dispatch``     — fault step → topology_refresh_fn called
* ``time_to_resume``       — fault step → HEALTHY_TRAINING restored
* ``degraded_steps``       — number of steps spent in non-HEALTHY phase
* ``invalidated_steps``    — number of steps where iteration was invalidated
* ``rollback_count``       — total rollbacks performed
* ``replay_count``         — total replays completed
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =====================================================================
# Fault scenario definitions
# =====================================================================

class FaultType(enum.Enum):
    """Types of faults that can be injected."""
    HARD_FAILURE = "hard_failure"
    PIPELINE_STAGE_FAILURE = "pipeline_stage_failure"
    REPLACEMENT_READY = "replacement_ready"
    SAFE_POINT_REACHED = "safe_point_reached"


@dataclass
class FaultScenario:
    """A single fault injection event.

    Attributes:
        step: Training step at which to inject.
        fault_type: Type of fault to inject.
        failed_rank: The rank that fails.
        replacement_rank: The replacement rank (for REPLACEMENT_READY).
        failed_stage: Pipeline stage index (for PIPELINE_STAGE_FAILURE).
        pp_group_ranks: PP group ranks (for PIPELINE_STAGE_FAILURE).
        expert_ids: Affected expert IDs.
        ep_group_ranks: EP group ranks.
        dp_group_ranks: DP group ranks.
        mid_iteration: Whether the fault occurs mid-iteration.
        reason: Human-readable reason.
    """
    step: int = -1
    fault_type: FaultType = FaultType.HARD_FAILURE
    failed_rank: int = -1
    replacement_rank: int = -1
    failed_stage: int = -1
    pp_group_ranks: List[int] = field(default_factory=list)
    expert_ids: List[int] = field(default_factory=list)
    ep_group_ranks: List[int] = field(default_factory=list)
    dp_group_ranks: List[int] = field(default_factory=list)
    mid_iteration: bool = False
    reason: str = ""


# =====================================================================
# Recovery metrics collector
# =====================================================================

@dataclass
class RecoveryMetrics:
    """Metrics collected during a single fault-recovery cycle.

    All ``time_*`` fields are in **training steps** (not wall-clock),
    measured relative to the fault injection step.  A value of -1
    means the event has not occurred.
    """
    fault_step: int = -1
    fault_type: str = ""
    failed_rank: int = -1

    # Step-based latencies (relative to fault_step)
    time_to_detect: int = -1
    time_to_rollback: int = -1
    time_to_replacement: int = -1
    time_to_group_repair: int = -1
    time_to_pp_repair: int = -1
    time_to_dispatch_refresh: int = -1
    time_to_resume: int = -1

    # Wall-clock timestamps (absolute)
    ts_fault: float = 0.0
    ts_detect: float = 0.0
    ts_rollback: float = 0.0
    ts_replacement: float = 0.0
    ts_group_repair: float = 0.0
    ts_pp_repair: float = 0.0
    ts_dispatch_refresh: float = 0.0
    ts_resume: float = 0.0

    # Counters
    degraded_steps: int = 0
    invalidated_steps: int = 0
    rollback_count: int = 0
    replay_count: int = 0

    # Throughput
    steps_before_fault: int = 0
    steps_after_resume: int = 0

    def wall_clock_latencies(self) -> Dict[str, float]:
        """Return wall-clock latencies in seconds."""
        base = self.ts_fault if self.ts_fault > 0 else 0.0
        result = {}
        for name in ('detect', 'rollback', 'replacement', 'group_repair',
                      'pp_repair', 'dispatch_refresh', 'resume'):
            ts = getattr(self, f'ts_{name}', 0.0)
            if ts > 0 and base > 0:
                result[f'wall_{name}_s'] = ts - base
            else:
                result[f'wall_{name}_s'] = -1.0
        return result

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'fault_step': self.fault_step,
            'fault_type': self.fault_type,
            'failed_rank': self.failed_rank,
            'time_to_detect': self.time_to_detect,
            'time_to_rollback': self.time_to_rollback,
            'time_to_replacement': self.time_to_replacement,
            'time_to_group_repair': self.time_to_group_repair,
            'time_to_pp_repair': self.time_to_pp_repair,
            'time_to_dispatch_refresh': self.time_to_dispatch_refresh,
            'time_to_resume': self.time_to_resume,
            'degraded_steps': self.degraded_steps,
            'invalidated_steps': self.invalidated_steps,
            'rollback_count': self.rollback_count,
            'replay_count': self.replay_count,
            'steps_before_fault': self.steps_before_fault,
            'steps_after_resume': self.steps_after_resume,
        }
        d.update(self.wall_clock_latencies())
        return d

    def format_report(self) -> str:
        """Format a human-readable report."""
        lines = [
            f"=== Recovery Metrics Report ===",
            f"Fault: type={self.fault_type}, rank={self.failed_rank}, "
            f"step={self.fault_step}",
            f"",
            f"Step-based latencies (steps from fault):",
            f"  detect:           {self.time_to_detect}",
            f"  rollback:         {self.time_to_rollback}",
            f"  replacement:      {self.time_to_replacement}",
            f"  group_repair:     {self.time_to_group_repair}",
            f"  pp_repair:        {self.time_to_pp_repair}",
            f"  dispatch_refresh: {self.time_to_dispatch_refresh}",
            f"  resume:           {self.time_to_resume}",
            f"",
            f"Counters:",
            f"  degraded_steps:   {self.degraded_steps}",
            f"  invalidated_steps:{self.invalidated_steps}",
            f"  rollback_count:   {self.rollback_count}",
            f"  replay_count:     {self.replay_count}",
            f"",
            f"Throughput:",
            f"  steps_before_fault: {self.steps_before_fault}",
            f"  steps_after_resume: {self.steps_after_resume}",
        ]
        wc = self.wall_clock_latencies()
        lines.append("")
        lines.append("Wall-clock latencies (seconds):")
        for k, v in wc.items():
            lines.append(f"  {k}: {v:.6f}" if v >= 0 else f"  {k}: N/A")
        lines.append("=" * 32)
        return "\n".join(lines)


class RecoveryMetricsCollector:
    """Collects recovery metrics by observing RecoveryController events.

    Usage::

        collector = RecoveryMetricsCollector()
        # ... run simulated training loop ...
        collector.on_fault_injected(step, fault_type, failed_rank)
        collector.on_phase_change(step, old_phase, new_phase)
        collector.on_callback_invoked(step, callback_name)
        collector.on_step_completed(step, is_degraded, is_invalidated)
        # ... after recovery ...
        metrics = collector.finalize()
        print(metrics.format_report())
    """

    def __init__(self) -> None:
        self._current: Optional[RecoveryMetrics] = None
        self._completed: List[RecoveryMetrics] = []
        self._resumed: bool = False

    def on_fault_injected(
        self,
        step: int,
        fault_type: str,
        failed_rank: int,
    ) -> None:
        """Record a fault injection event."""
        m = RecoveryMetrics(
            fault_step=step,
            fault_type=fault_type,
            failed_rank=failed_rank,
            ts_fault=time.monotonic(),
            steps_before_fault=step,
        )
        self._current = m
        self._resumed = False

    def on_phase_change(
        self,
        step: int,
        old_phase: str,
        new_phase: str,
    ) -> None:
        """Record a phase transition."""
        if self._current is None:
            return
        m = self._current
        now = time.monotonic()
        delta = step - m.fault_step

        # First transition after fault = detection
        if m.time_to_detect < 0:
            m.time_to_detect = delta
            m.ts_detect = now

        if new_phase == 'ROLLBACK_PENDING' and m.time_to_rollback < 0:
            m.time_to_rollback = delta
            m.ts_rollback = now

        if new_phase == 'WAITING_FOR_REPLACEMENT' and m.time_to_replacement < 0:
            m.time_to_replacement = delta
            m.ts_replacement = now

        if new_phase == 'HEALTHY_TRAINING' and old_phase == 'REINTEGRATED':
            m.time_to_resume = delta
            m.ts_resume = now
            self._resumed = True

    def on_callback_invoked(
        self,
        step: int,
        callback_name: str,
    ) -> None:
        """Record a callback invocation."""
        if self._current is None:
            return
        m = self._current
        now = time.monotonic()
        delta = step - m.fault_step

        if callback_name == 'group_rebuild_finish_fn' and m.time_to_group_repair < 0:
            m.time_to_group_repair = delta
            m.ts_group_repair = now

        if callback_name == 'pipeline_stage_repair_fn' and m.time_to_pp_repair < 0:
            m.time_to_pp_repair = delta
            m.ts_pp_repair = now

        if callback_name == 'topology_refresh_fn' and m.time_to_dispatch_refresh < 0:
            m.time_to_dispatch_refresh = delta
            m.ts_dispatch_refresh = now

    def on_step_completed(
        self,
        step: int,
        is_degraded: bool,
        is_invalidated: bool,
    ) -> None:
        """Record per-step status."""
        if self._current is None:
            return
        if is_degraded:
            self._current.degraded_steps += 1
        if is_invalidated:
            self._current.invalidated_steps += 1
        if self._resumed:
            self._current.steps_after_resume += 1

    def on_rollback(self) -> None:
        """Record a rollback event."""
        if self._current is not None:
            self._current.rollback_count += 1

    def on_replay_completed(self) -> None:
        """Record a replay completion."""
        if self._current is not None:
            self._current.replay_count += 1

    def finalize(self) -> Optional[RecoveryMetrics]:
        """Finalize and return the current metrics."""
        if self._current is not None:
            self._completed.append(self._current)
            result = self._current
            self._current = None
            return result
        return None

    @property
    def all_metrics(self) -> List[RecoveryMetrics]:
        """All completed metrics."""
        result = list(self._completed)
        if self._current is not None:
            result.append(self._current)
        return result

    def reset(self) -> None:
        self._current = None
        self._completed.clear()
        self._resumed = False


# =====================================================================
# Fault injector
# =====================================================================

class FaultInjector:
    """Deterministic, step-based fault injector.

    Holds a list of ``FaultScenario`` objects and injects them at the
    specified training steps.  Works with any ``RecoveryController``
    instance (no Megatron dependencies).

    Usage::

        injector = FaultInjector(controller)
        injector.add_scenario(FaultScenario(
            step=50, fault_type=FaultType.HARD_FAILURE,
            failed_rank=4, expert_ids=[8, 9], ...
        ))
        injector.add_scenario(FaultScenario(
            step=60, fault_type=FaultType.REPLACEMENT_READY,
            failed_rank=4, replacement_rank=64,
        ))

        for step in range(100):
            injector.maybe_inject(step)
            controller.before_iteration(step)
            ...
    """

    def __init__(self, controller, *, metrics: Optional[RecoveryMetricsCollector] = None) -> None:
        self._controller = controller
        self._scenarios: List[FaultScenario] = []
        self._injected: Dict[int, bool] = {}  # scenario index → injected
        self._metrics = metrics

    def add_scenario(self, scenario: FaultScenario) -> int:
        """Add a fault scenario.  Returns the scenario index."""
        idx = len(self._scenarios)
        self._scenarios.append(scenario)
        self._injected[idx] = False
        return idx

    def maybe_inject(self, step: int) -> List[int]:
        """Inject any scenarios scheduled for this step.

        Returns list of scenario indices that were injected.
        """
        injected = []
        for idx, scenario in enumerate(self._scenarios):
            if self._injected[idx]:
                continue
            if step < scenario.step:
                continue

            self._injected[idx] = True
            self._inject_one(scenario, step)
            injected.append(idx)

        return injected

    def _inject_one(self, scenario: FaultScenario, step: int) -> None:
        """Inject a single fault scenario."""
        ctrl = self._controller
        ft = scenario.fault_type

        logger.warning(
            "FaultInjector: injecting %s at step %d (rank=%d)",
            ft.value, step, scenario.failed_rank,
        )

        if ft == FaultType.HARD_FAILURE:
            if self._metrics:
                self._metrics.on_fault_injected(
                    step, ft.value, scenario.failed_rank,
                )
            ctrl.on_hard_rank_failure(
                failed_rank=scenario.failed_rank,
                reason=scenario.reason or "injected_hard_failure",
                step=step,
                expert_ids=scenario.expert_ids or None,
                ep_group_ranks=scenario.ep_group_ranks or None,
                dp_group_ranks=scenario.dp_group_ranks or None,
                mid_iteration=scenario.mid_iteration,
            )

        elif ft == FaultType.PIPELINE_STAGE_FAILURE:
            if self._metrics:
                self._metrics.on_fault_injected(
                    step, ft.value, scenario.failed_rank,
                )
            ctrl.on_pipeline_stage_failure(
                failed_stage=scenario.failed_stage,
                failed_rank=scenario.failed_rank,
                step=step,
                pp_group_ranks=scenario.pp_group_ranks or None,
                reason=scenario.reason or "injected_pipeline_failure",
                expert_ids=scenario.expert_ids or None,
                ep_group_ranks=scenario.ep_group_ranks or None,
                dp_group_ranks=scenario.dp_group_ranks or None,
                mid_iteration=scenario.mid_iteration,
            )

        elif ft == FaultType.REPLACEMENT_READY:
            ctrl.on_replacement_assigned(
                failed_rank=scenario.failed_rank,
                replacement_rank=scenario.replacement_rank,
                step=step,
            )
            ctrl.on_replacement_ready(
                failed_rank=scenario.failed_rank,
                step=step,
            )

        elif ft == FaultType.SAFE_POINT_REACHED:
            # No-op: safe point is handled by controller.before_iteration()
            pass

    @property
    def num_scenarios(self) -> int:
        return len(self._scenarios)

    @property
    def num_injected(self) -> int:
        return sum(1 for v in self._injected.values() if v)

    def reset(self) -> None:
        self._scenarios.clear()
        self._injected.clear()


# =====================================================================
# Simulated training loop
# =====================================================================

@dataclass
class TrainingLoopConfig:
    """Configuration for the simulated training loop."""
    num_steps: int = 100
    pp_size: int = 1
    pp_rank: int = 0
    num_microbatches: int = 1
    ep_group_ranks: List[int] = field(default_factory=lambda: [0, 2, 4, 6])
    dp_group_ranks: List[int] = field(default_factory=lambda: [0, 4, 8])
    pp_group_ranks: List[int] = field(default_factory=list)
    num_experts: int = 16
    num_layers: int = 4


@dataclass
class TrainingLoopResult:
    """Result of a simulated training loop run."""
    total_steps: int = 0
    healthy_steps: int = 0
    degraded_steps: int = 0
    invalidated_steps: int = 0
    repairs_executed: int = 0
    rollbacks: int = 0
    replays_completed: int = 0
    final_phase: str = ""
    metrics: List[RecoveryMetrics] = field(default_factory=list)
    callback_log: List[Tuple[int, str]] = field(default_factory=list)
    phase_log: List[Tuple[int, str, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def run_simulated_training_loop(
    controller,
    injector: FaultInjector,
    config: TrainingLoopConfig,
    *,
    invalidator=None,
    rollback_manager=None,
    commit_guard=None,
    pipeline_coordinator=None,
    metrics_collector: Optional[RecoveryMetricsCollector] = None,
) -> TrainingLoopResult:
    """Run a simulated training loop with fault injection.

    This function orchestrates all BSR-MoE subsystems through a
    deterministic training loop.  It does NOT perform any real
    computation — it only drives the state machines.

    Args:
        controller: RecoveryController instance.
        injector: FaultInjector with pre-configured scenarios.
        config: Training loop configuration.
        invalidator: IterationInvalidator instance (optional).
        rollback_manager: RollbackReplayManager instance (optional).
        commit_guard: OptimizerCommitGuard instance (optional).
        pipeline_coordinator: PipelineRollbackCoordinator (optional).
        metrics_collector: RecoveryMetricsCollector (optional).

    Returns:
        TrainingLoopResult with all collected data.
    """
    result = TrainingLoopResult()
    mc = metrics_collector

    # Install an event observer on the controller to track phase changes
    original_transition = controller._transition_to

    def _observed_transition(new_phase, event_type="", step=-1, **details):
        old_name = controller._phase.name
        original_transition(new_phase, event_type=event_type, step=step, **details)
        new_name = new_phase.name
        result.phase_log.append((step, old_name, new_name))
        if mc:
            mc.on_phase_change(step, old_name, new_name)

    controller._transition_to = _observed_transition

    # Install callback observers
    _install_callback_observers(controller, result, mc)

    try:
        for step in range(config.num_steps):
            # --- Pre-iteration ---

            # Invalidator begin
            if invalidator is not None:
                invalidator.begin_iteration(step)

            # Commit guard begin
            if commit_guard is not None:
                commit_guard.begin_iteration(step)

            # Pipeline coordinator begin
            if pipeline_coordinator is not None and config.pp_size > 1:
                pipeline_coordinator.begin_iteration(
                    step, pp_rank=config.pp_rank,
                    pp_size=config.pp_size,
                    num_microbatches=config.num_microbatches,
                )

            # Rollback manager snapshot
            if rollback_manager is not None:
                rollback_manager.take_snapshot(
                    iteration=step,
                    consumed_train_samples=step * 8,  # simulated
                )

            # Controller safe-point hook
            repair_done = controller.before_iteration(step=step)
            if repair_done:
                result.repairs_executed += 1

            # --- Simulated forward/backward ---
            # Inject faults here (mid-iteration) so that
            # before_iteration() has already cleared the previous
            # iteration's invalidation flag.
            injector.maybe_inject(step)

            # --- Optimizer commit check ---
            is_invalid = False
            if invalidator is not None:
                is_invalid = invalidator.is_current_iteration_invalid()
            # Also check controller's own invalidation flag (set by
            # on_hard_rank_failure with mid_iteration=True).
            if not is_invalid and controller.iteration_was_invalidated:
                is_invalid = True

            if commit_guard is not None:
                should_commit = commit_guard.should_commit()
                if should_commit:
                    commit_guard.mark_committed(step)
                else:
                    commit_guard.mark_skipped(step, reason="iteration_invalid")
            elif is_invalid:
                pass  # skip optimizer

            # --- Rollback if needed ---
            if is_invalid and rollback_manager is not None:
                if rollback_manager.has_valid_snapshot:
                    did_rollback = rollback_manager.rollback(
                        args=_MockArgs(step * 8),
                    )
                    if did_rollback:
                        result.rollbacks += 1
                        if mc:
                            mc.on_rollback()

            # --- Replay check ---
            if rollback_manager is not None and rollback_manager.is_replay_pending():
                rollback_manager.complete_replay()
                result.replays_completed += 1
                if mc:
                    mc.on_replay_completed()

            # --- Post-iteration ---
            if invalidator is not None:
                invalidator.end_iteration(step)

            if commit_guard is not None:
                commit_guard.end_iteration(step)

            if pipeline_coordinator is not None and config.pp_size > 1:
                pipeline_coordinator.end_iteration(step)

            controller.after_iteration(step=step)

            # --- Record metrics ---
            is_degraded = controller.is_degraded
            result.total_steps += 1
            if is_degraded:
                result.degraded_steps += 1
            else:
                result.healthy_steps += 1
            if is_invalid:
                result.invalidated_steps += 1

            if mc:
                mc.on_step_completed(step, is_degraded, is_invalid)

    except Exception as e:
        result.errors.append(str(e))
        logger.error("SimulatedTrainingLoop: exception at step: %s", e)

    # Finalize
    result.final_phase = controller.phase.name
    if mc:
        m = mc.finalize()
        if m is not None:
            result.metrics.append(m)

    # Restore original transition
    controller._transition_to = original_transition

    return result


def _install_callback_observers(controller, result, mc):
    """Wrap controller callbacks to record invocations."""
    callback_names = [
        '_health_mark_stale_runnable_fn', '_health_mark_healthy_fn',
        '_replacement_announce_fn', '_replacement_integrate_fn',
        '_group_rebuild_request_fn', '_group_rebuild_execute_fn',
        '_group_rebuild_finish_fn', '_topology_refresh_fn',
        '_dense_sync_fn', '_expert_restore_fn',
        '_pipeline_stage_repair_fn', '_pipeline_rollback_fn',
        '_microbatch_invalidation_fn',
    ]

    for attr_name in callback_names:
        original = getattr(controller, attr_name, None)
        if original is None:
            continue

        cb_name = attr_name.lstrip('_')

        def _make_wrapper(orig_fn, name):
            def wrapper(**kwargs):
                step = kwargs.get('step', -1)
                result.callback_log.append((step, name))
                if mc:
                    mc.on_callback_invoked(step, name)
                return orig_fn(**kwargs)
            return wrapper

        setattr(controller, attr_name, _make_wrapper(original, cb_name))


class _MockArgs:
    """Minimal mock for Megatron args used by RollbackReplayManager."""
    def __init__(self, consumed_train_samples=0):
        self.consumed_train_samples = consumed_train_samples


# =====================================================================
# Pre-built scenario factories
# =====================================================================

def build_pp1_hard_failure_scenario(
    *,
    fault_step: int = 50,
    replacement_step: int = 60,
    failed_rank: int = 4,
    replacement_rank: int = 64,
    expert_ids: Optional[List[int]] = None,
    ep_group_ranks: Optional[List[int]] = None,
    dp_group_ranks: Optional[List[int]] = None,
    mid_iteration: bool = True,
) -> List[FaultScenario]:
    """Build a PP=1 hard failure + replacement scenario.

    Returns a list of two FaultScenario objects:
    1. Hard failure at ``fault_step``
    2. Replacement ready at ``replacement_step``
    """
    return [
        FaultScenario(
            step=fault_step,
            fault_type=FaultType.HARD_FAILURE,
            failed_rank=failed_rank,
            expert_ids=expert_ids or [8, 9],
            ep_group_ranks=ep_group_ranks or [0, 2, 4, 6],
            dp_group_ranks=dp_group_ranks or [0, 4, 8],
            mid_iteration=mid_iteration,
            reason="simulated_hard_failure",
        ),
        FaultScenario(
            step=replacement_step,
            fault_type=FaultType.REPLACEMENT_READY,
            failed_rank=failed_rank,
            replacement_rank=replacement_rank,
        ),
    ]


def build_pp_gt1_pipeline_failure_scenario(
    *,
    fault_step: int = 50,
    replacement_step: int = 60,
    failed_rank: int = 4,
    replacement_rank: int = 64,
    failed_stage: int = 2,
    pp_group_ranks: Optional[List[int]] = None,
    expert_ids: Optional[List[int]] = None,
    ep_group_ranks: Optional[List[int]] = None,
    dp_group_ranks: Optional[List[int]] = None,
    mid_iteration: bool = True,
) -> List[FaultScenario]:
    """Build a PP>1 pipeline stage failure + replacement scenario.

    Returns a list of two FaultScenario objects:
    1. Pipeline stage failure at ``fault_step``
    2. Replacement ready at ``replacement_step``
    """
    return [
        FaultScenario(
            step=fault_step,
            fault_type=FaultType.PIPELINE_STAGE_FAILURE,
            failed_rank=failed_rank,
            failed_stage=failed_stage,
            pp_group_ranks=pp_group_ranks or [0, 2, 4, 6],
            expert_ids=expert_ids or [8, 9],
            ep_group_ranks=ep_group_ranks or [0, 2, 4, 6],
            dp_group_ranks=dp_group_ranks or [0, 4, 8],
            mid_iteration=mid_iteration,
            reason="simulated_pipeline_stage_failure",
        ),
        FaultScenario(
            step=replacement_step,
            fault_type=FaultType.REPLACEMENT_READY,
            failed_rank=failed_rank,
            replacement_rank=replacement_rank,
        ),
    ]


# =====================================================================
# Log format specification
# =====================================================================

LOG_FORMAT_SPEC = """
BSR-MoE Recovery Log Format
============================

All BSR-MoE log messages use the ``BSR-MoE`` prefix for easy grep.

Phase transitions:
    BSR-MoE controller: {OLD_PHASE} → {NEW_PHASE} (event={EVENT}, step={STEP})

Fault injection:
    FaultInjector: injecting {FAULT_TYPE} at step {STEP} (rank={RANK})

Callback invocations:
    BSR-MoE {callback_name}: {SUCCESS|FAILED} — {details}

Metrics report:
    === Recovery Metrics Report ===
    Fault: type={TYPE}, rank={RANK}, step={STEP}
    Step-based latencies (steps from fault):
      detect:           {N}
      rollback:         {N}
      ...
    Counters:
      degraded_steps:   {N}
      ...

Grep patterns:
    grep "BSR-MoE controller:" log.txt     # Phase transitions
    grep "FaultInjector:" log.txt           # Fault injections
    grep "Recovery Metrics" log.txt         # Metrics reports
    grep "FAILED" log.txt                   # Failures
    grep "INVALIDATED" log.txt              # Iteration invalidations
"""

# =====================================================================
# Troubleshooting guide
# =====================================================================

TROUBLESHOOTING_GUIDE = """
BSR-MoE Recovery Troubleshooting Guide
========================================

1. Controller stuck in PENDING_GROUP_REPAIR
   - Check: Was on_replacement_assigned() called?
   - Check: Is the replacement rank reachable?
   - Fix: Ensure the job scheduler calls bsr_announce_replacement_ready()

2. Controller stuck in WAITING_FOR_REPLACEMENT
   - Check: Was on_replacement_ready() called?
   - Check: Did the replacement rank finish bootstrapping?
   - Fix: Verify dense param sync and model loading completed

3. Controller stuck in SAFE_POINT_REPAIR
   - Check: Was before_iteration() called at the next iteration boundary?
   - Check: Are all repair callbacks registered?
   - Fix: Ensure the training loop calls bsr_before_iteration(step)

4. Pipeline repair failed (PP>1)
   - Check: Are pp_group_ranks set in the FaultRecord?
   - Check: Did pipeline_stage_repair_fn raise an exception?
   - Fix: Verify PipelineStageRepairer has valid PP group ranks

5. Iteration not invalidated after hard failure
   - Check: Was mid_iteration=True passed to on_hard_rank_failure()?
   - Check: Is the IterationInvalidator wired to the HardFailureDetector?
   - Fix: Ensure bsr_report_hard_failure() is called with mid_iteration=True

6. Optimizer committed tainted gradients
   - Check: Is OptimizerCommitGuard.should_commit() called before optimizer.step()?
   - Check: Is is_iteration_invalid_fn wired?
   - Fix: Add bsr_should_commit_optimizer() check before optimizer.step()

7. Rollback not performed
   - Check: Was take_snapshot() called at iteration start?
   - Check: Does the RollbackReplayManager have a valid snapshot?
   - Fix: Ensure bsr_snapshot_iteration() is called at each iteration start

8. Replay stuck
   - Check: Was complete_replay() called after successful replay?
   - Check: Has max_replay_attempts been exceeded?
   - Fix: Call bsr_complete_replay() after successful train_step on replay

9. Metrics show time_to_resume = -1
   - Check: Did the controller reach HEALTHY_TRAINING?
   - Check: Were all recovery steps completed?
   - Fix: Run more training steps after replacement_ready injection

10. PP>1: ROLLBACK_PENDING not entered
    - Check: Was mid_iteration=True and pp_group_ranks has >1 rank?
    - Check: Was on_pipeline_stage_failure() called (not on_hard_rank_failure)?
    - Fix: Use on_pipeline_stage_failure() for PP>1 failures
"""

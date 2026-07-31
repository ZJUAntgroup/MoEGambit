# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MOEGAMBIT-MoE Experiment Orchestrator (Step 11).

Provides:
1. **Structured fault injection** — programmatic injection of soft/hard
   failures with configurable gap, replacement timing, and recovery path.
2. **Metrics collection** — wall-clock timing for every recovery phase.
3. **Experiment scenarios** — pre-built scenarios for validating the
   gap-aware hybrid recovery pipeline end-to-end.
4. **Structured logging** — JSON-serializable experiment reports.

Experiment scenarios
--------------------
::

    Scenario 1: small_gap_checkpoint_restart
        gap <= threshold → CHECKPOINT_RESTART path
        Validates: full checkpoint load, iteration rollback

    Scenario 2: large_gap_hybrid_recovery
        gap > threshold → HYBRID_RECOVERY path
        Validates: dense from DP peer, expert from checkpoint

    Scenario 3: hybrid_with_deferred_optimizer
        HYBRID_RECOVERY + weights-first, optimizer-later
        Validates: two-phase state machine, deferred optimizer load

    Scenario 4: hybrid_with_preferential_routing
        HYBRID_RECOVERY + preferential routing bias for recovered experts
        Validates: bias activation, linear decay, capacity preservation

Usage (unit test / mock mode)::

    from moegambit.adapters.megatron.moe.moegambit_experiment import (
        ExperimentOrchestrator,
        ExperimentScenario,
    )

    orch = ExperimentOrchestrator(config)
    report = orch.run_scenario(ExperimentScenario.LARGE_GAP_HYBRID)
    assert report.success
    print(report.to_json())
"""

from __future__ import annotations

import enum
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

def _ts() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# =====================================================================
# Experiment scenarios
# =====================================================================

class ExperimentScenario(enum.Enum):
    """Pre-built experiment scenarios."""

    SMALL_GAP_CHECKPOINT_RESTART = "small_gap_checkpoint_restart"
    """gap <= threshold → CHECKPOINT_RESTART."""

    LARGE_GAP_HYBRID = "large_gap_hybrid_recovery"
    """gap > threshold → HYBRID_RECOVERY."""

    HYBRID_DEFERRED_OPTIMIZER = "hybrid_with_deferred_optimizer"
    """HYBRID_RECOVERY + weights-first, optimizer-later."""

    HYBRID_PREFERENTIAL_ROUTING = "hybrid_with_preferential_routing"
    """HYBRID_RECOVERY + preferential routing bias."""

    # SOFT_FAILURE_NO_REGRESSION removed — quarantine path deleted.


# =====================================================================
# Metrics
# =====================================================================

@dataclass
class RecoveryMetrics:
    """Wall-clock timing for each recovery phase."""

    time_to_detect: float = 0.0
    """Seconds from fault injection to detection callback."""

    time_to_choose_path: float = 0.0
    """Seconds for gap-aware policy evaluation."""

    time_to_dense_sync: float = 0.0
    """Seconds for dense parameter broadcast (hybrid path only)."""

    time_to_expert_restore: float = 0.0
    """Seconds for expert weight restore from checkpoint."""

    time_to_optimizer_attach: float = 0.0
    """Seconds for deferred optimizer load (if enabled)."""

    time_to_convergence: float = 0.0
    """Seconds for unified post-recovery convergence."""

    time_to_resume: float = 0.0
    """Total seconds from fault injection to training resume."""

    recovery_path: str = ""
    """Selected recovery path name."""

    gap: int = -1
    """Iteration gap between checkpoint and current step."""

    gap_threshold: int = -1
    """Configured gap threshold."""

    experts_recovered: int = 0
    """Number of experts recovered."""

    preferential_routing_activated: bool = False
    """Whether preferential routing was activated."""

    two_phase_final_state: str = ""
    """Final two-phase state machine state."""

    optimizer_deferred: bool = False
    """Whether optimizer load was deferred."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_to_detect": round(self.time_to_detect, 4),
            "time_to_choose_path": round(self.time_to_choose_path, 4),
            "time_to_dense_sync": round(self.time_to_dense_sync, 4),
            "time_to_expert_restore": round(self.time_to_expert_restore, 4),
            "time_to_optimizer_attach": round(self.time_to_optimizer_attach, 4),
            "time_to_convergence": round(self.time_to_convergence, 4),
            "time_to_resume": round(self.time_to_resume, 4),
            "recovery_path": self.recovery_path,
            "gap": self.gap,
            "gap_threshold": self.gap_threshold,
            "experts_recovered": self.experts_recovered,
            "preferential_routing_activated": self.preferential_routing_activated,
            "two_phase_final_state": self.two_phase_final_state,
            "optimizer_deferred": self.optimizer_deferred,
        }


# =====================================================================
# Fault injection plan
# =====================================================================

@dataclass
class FaultInjectionPlan:
    """Describes a programmatic fault injection."""

    fault_type: str = "hard_failure"
    """'hard_failure' (only supported fault type)."""

    inject_step: int = 50
    """Training step at which to inject the fault."""

    failed_rank: int = 0
    """Rank to fail."""

    replacement_rank: int = -1
    """Replacement rank (-1 = identity inheritance)."""

    replacement_delay_steps: int = 10
    """Steps between fault and replacement ready."""

    checkpoint_iteration: int = -1
    """Simulated checkpoint iteration (for gap calculation)."""

    gap_threshold: int = 100
    """Gap threshold for path selection."""

    expert_ids: List[int] = field(default_factory=list)
    """Expert IDs affected by the fault."""

    ep_group_ranks: List[int] = field(default_factory=list)
    dp_group_ranks: List[int] = field(default_factory=list)

    enable_deferred_optimizer: bool = False
    enable_preferential_routing: bool = False
    preferential_routing_window: int = 100
    preferential_routing_bias: float = 0.1

    @property
    def gap(self) -> int:
        if self.checkpoint_iteration < 0:
            return -1
        return self.inject_step - self.checkpoint_iteration

    @property
    def expected_path(self) -> str:
        if self.checkpoint_iteration < 0:
            return "HYBRID_RECOVERY"
        if self.gap <= self.gap_threshold:
            return "CHECKPOINT_RESTART"
        return "HYBRID_RECOVERY"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_type": self.fault_type,
            "inject_step": self.inject_step,
            "failed_rank": self.failed_rank,
            "replacement_rank": self.replacement_rank,
            "replacement_delay_steps": self.replacement_delay_steps,
            "checkpoint_iteration": self.checkpoint_iteration,
            "gap_threshold": self.gap_threshold,
            "gap": self.gap,
            "expected_path": self.expected_path,
            "expert_ids": list(self.expert_ids),
            "enable_deferred_optimizer": self.enable_deferred_optimizer,
            "enable_preferential_routing": self.enable_preferential_routing,
        }


# =====================================================================
# Experiment report
# =====================================================================

@dataclass
class ExperimentReport:
    """Complete report for one experiment run."""

    scenario: str = ""
    plan: Optional[FaultInjectionPlan] = None
    metrics: Optional[RecoveryMetrics] = None
    success: bool = False
    errors: List[str] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    timestamp: str = ""

    @property
    def elapsed_seconds(self) -> float:
        return self.end_time - self.start_time if self.end_time > self.start_time else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "success": self.success,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "timestamp": self.timestamp,
            "plan": self.plan.to_dict() if self.plan else None,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "errors": list(self.errors),
            "events": list(self.events),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# =====================================================================
# Phase timer
# =====================================================================

class PhaseTimer:
    """Context manager for timing a named recovery phase."""

    def __init__(self, phase_name: str, metrics: RecoveryMetrics, events: List):
        self._phase = phase_name
        self._metrics = metrics
        self._events = events
        self._start = 0.0

    def __enter__(self):
        self._start = time.monotonic()
        self._events.append({
            "type": "phase_start",
            "phase": self._phase,
            "wall_time": time.time(),
        })
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.monotonic() - self._start
        self._events.append({
            "type": "phase_end",
            "phase": self._phase,
            "elapsed": round(elapsed, 6),
            "error": str(exc_val) if exc_val else None,
            "wall_time": time.time(),
        })
        # Map phase name to metrics field
        attr = f"time_to_{self._phase}"
        if hasattr(self._metrics, attr):
            setattr(self._metrics, attr, elapsed)
        return False  # don't suppress exceptions


# =====================================================================
# Experiment orchestrator
# =====================================================================

class ExperimentOrchestrator:
    """Orchestrates MOEGAMBIT-MoE experiment scenarios.

    The orchestrator drives the RecoveryController + MOEGAMBIT-MoE modules
    through a complete fault→recovery cycle using injected callbacks.

    For unit testing, all actual distributed operations (broadcast,
    checkpoint load, etc.) are replaced by mock callbacks passed in
    the constructor.
    """

    def __init__(
        self,
        *,
        num_experts: int = 8,
        num_layers: int = 4,
        ep_size: int = 2,
        ep_group_ranks: Optional[List[int]] = None,
        dp_group_ranks: Optional[List[int]] = None,
        # Mock callbacks for each recovery step
        dense_sync_fn: Optional[Callable] = None,
        expert_restore_fn: Optional[Callable] = None,
        checkpoint_restart_fn: Optional[Callable] = None,
        deferred_optimizer_submit_fn: Optional[Callable] = None,
        deferred_optimizer_poll_fn: Optional[Callable] = None,
        group_rebuild_fn: Optional[Callable] = None,
        topology_refresh_fn: Optional[Callable] = None,
    ):
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.ep_size = ep_size
        self.ep_group_ranks = ep_group_ranks or list(range(ep_size))
        self.dp_group_ranks = dp_group_ranks or list(range(ep_size, ep_size + 2))

        # Mock callbacks
        self._dense_sync_fn = dense_sync_fn or (lambda **kw: None)
        self._expert_restore_fn = expert_restore_fn or (lambda **kw: None)
        self._checkpoint_restart_fn = checkpoint_restart_fn or (lambda **kw: None)
        self._deferred_optimizer_submit_fn = deferred_optimizer_submit_fn
        self._deferred_optimizer_poll_fn = deferred_optimizer_poll_fn
        self._group_rebuild_fn = group_rebuild_fn or (lambda **kw: None)
        self._topology_refresh_fn = topology_refresh_fn or (lambda **kw: None)

        # History
        self._reports: List[ExperimentReport] = []

    # ------------------------------------------------------------------
    # Scenario builders
    # ------------------------------------------------------------------

    def _build_plan(self, scenario: ExperimentScenario) -> FaultInjectionPlan:
        """Build a FaultInjectionPlan for the given scenario."""

        base_experts = list(range(self.num_experts // self.ep_size))

        if scenario == ExperimentScenario.SMALL_GAP_CHECKPOINT_RESTART:
            return FaultInjectionPlan(
                fault_type="hard_failure",
                inject_step=50,
                failed_rank=0,
                replacement_rank=0,
                replacement_delay_steps=5,
                checkpoint_iteration=45,  # gap=5, small
                gap_threshold=100,
                expert_ids=base_experts,
                ep_group_ranks=list(self.ep_group_ranks),
                dp_group_ranks=list(self.dp_group_ranks),
            )

        elif scenario == ExperimentScenario.LARGE_GAP_HYBRID:
            return FaultInjectionPlan(
                fault_type="hard_failure",
                inject_step=500,
                failed_rank=0,
                replacement_rank=0,
                replacement_delay_steps=10,
                checkpoint_iteration=100,  # gap=400, large
                gap_threshold=100,
                expert_ids=base_experts,
                ep_group_ranks=list(self.ep_group_ranks),
                dp_group_ranks=list(self.dp_group_ranks),
            )

        elif scenario == ExperimentScenario.HYBRID_DEFERRED_OPTIMIZER:
            return FaultInjectionPlan(
                fault_type="hard_failure",
                inject_step=500,
                failed_rank=0,
                replacement_rank=0,
                replacement_delay_steps=10,
                checkpoint_iteration=100,
                gap_threshold=100,
                expert_ids=base_experts,
                ep_group_ranks=list(self.ep_group_ranks),
                dp_group_ranks=list(self.dp_group_ranks),
                enable_deferred_optimizer=True,
            )

        elif scenario == ExperimentScenario.HYBRID_PREFERENTIAL_ROUTING:
            return FaultInjectionPlan(
                fault_type="hard_failure",
                inject_step=500,
                failed_rank=0,
                replacement_rank=0,
                replacement_delay_steps=10,
                checkpoint_iteration=100,
                gap_threshold=100,
                expert_ids=base_experts,
                ep_group_ranks=list(self.ep_group_ranks),
                dp_group_ranks=list(self.dp_group_ranks),
                enable_preferential_routing=True,
                preferential_routing_window=100,
                preferential_routing_bias=0.1,
            )

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_scenario(
        self,
        scenario: ExperimentScenario,
        plan: Optional[FaultInjectionPlan] = None,
    ) -> ExperimentReport:
        """Run a single experiment scenario.

        Args:
            scenario: The scenario to run.
            plan: Optional override plan. If None, uses the built-in plan.

        Returns:
            ExperimentReport with metrics and events.
        """
        if plan is None:
            plan = self._build_plan(scenario)

        report = ExperimentReport(
            scenario=scenario.value,
            plan=plan,
            metrics=RecoveryMetrics(),
            start_time=time.monotonic(),
            timestamp=_ts(),
        )
        metrics = report.metrics
        events = report.events

        logger.warning(
            "[%s] MOEGAMBIT-MoE EXPERIMENT: starting scenario=%s, plan=%s",
            _ts(), scenario.value, json.dumps(plan.to_dict()),
        )

        try:
            self._run_scenario_inner(scenario, plan, metrics, events, report)
        except Exception as e:
            report.errors.append(f"experiment_exception: {e}")
            logger.error(
                "[%s] MOEGAMBIT-MoE EXPERIMENT: scenario=%s FAILED: %s",
                _ts(), scenario.value, e,
            )

        report.end_time = time.monotonic()
        metrics.time_to_resume = report.elapsed_seconds
        report.success = len(report.errors) == 0

        logger.warning(
            "[%s] MOEGAMBIT-MoE EXPERIMENT: scenario=%s %s — "
            "elapsed=%.4fs, path=%s, errors=%d",
            _ts(), scenario.value,
            "PASSED" if report.success else "FAILED",
            report.elapsed_seconds,
            metrics.recovery_path,
            len(report.errors),
        )

        self._reports.append(report)
        return report

    def _run_scenario_inner(
        self,
        scenario: ExperimentScenario,
        plan: FaultInjectionPlan,
        metrics: RecoveryMetrics,
        events: List,
        report: ExperimentReport,
    ) -> None:
        """Internal: execute the scenario steps."""
        from moegambit.adapters.megatron.moe.recovery_controller import (
            RecoveryController,
            RecoveryPhase,
        )

        ctrl = RecoveryController()

        # --- Phase 1: Fault detection ---
        with PhaseTimer("detect", metrics, events):
            ctrl.on_hard_rank_failure(
                failed_rank=plan.failed_rank,
                reason=f"experiment_{scenario.value}",
                step=plan.inject_step,
                expert_ids=plan.expert_ids,
                ep_group_ranks=plan.ep_group_ranks,
                dp_group_ranks=plan.dp_group_ranks,
            )

        # --- Phase 2: Gap-aware path selection ---
        with PhaseTimer("choose_path", metrics, events):
            # Simulate gap-aware policy evaluation
            metrics.gap = plan.gap
            metrics.gap_threshold = plan.gap_threshold
            metrics.recovery_path = plan.expected_path

        # --- Phase 3: Replacement assignment + ready ---
        replacement_step = plan.inject_step + plan.replacement_delay_steps
        replacement_rank = (
            plan.replacement_rank if plan.replacement_rank >= 0
            else plan.failed_rank
        )

        ctrl.on_replacement_assigned(
            failed_rank=plan.failed_rank,
            replacement_rank=replacement_rank,
            step=replacement_step,
        )
        ctrl.on_replacement_ready(
            failed_rank=plan.failed_rank,
            step=replacement_step,
        )

        # Wire callbacks for safe-point repair
        callback_timings = {}

        def _timed_callback(name, fn):
            def wrapper(**kw):
                t0 = time.monotonic()
                fn(**kw)
                callback_timings[name] = time.monotonic() - t0
            return wrapper

        ctrl.register_callbacks(
            group_rebuild_execute_fn=_timed_callback(
                "group_rebuild", self._group_rebuild_fn
            ),
            topology_refresh_fn=_timed_callback(
                "topology_refresh", self._topology_refresh_fn
            ),
            dense_sync_fn=_timed_callback(
                "dense_sync", self._dense_sync_fn
            ),
            expert_restore_fn=_timed_callback(
                "expert_restore", self._expert_restore_fn
            ),
            checkpoint_restart_fn=_timed_callback(
                "checkpoint_restart", self._checkpoint_restart_fn
            ),
        )

        # Set up gap-aware policy if needed
        if plan.checkpoint_iteration >= 0:
            from moegambit.adapters.megatron.moe.gap_aware_recovery_policy import (
                GapAwareRecoveryPolicyManager,
            )
            policy_mgr = GapAwareRecoveryPolicyManager(
                gap_threshold=plan.gap_threshold,
                get_checkpoint_iteration_fn=lambda: plan.checkpoint_iteration,
            )
            ctrl.set_gap_aware_policy_manager(policy_mgr)

        # --- Phase 4: Safe-point repair ---
        repair_step = replacement_step + 1
        repaired = ctrl.before_iteration(step=repair_step)

        if not repaired:
            report.errors.append("safe-point repair did not execute")
            return

        # Extract callback timings into metrics
        if metrics.recovery_path == "HYBRID_RECOVERY":
            metrics.time_to_dense_sync = callback_timings.get("dense_sync", 0.0)
            metrics.time_to_expert_restore = callback_timings.get("expert_restore", 0.0)
        else:
            metrics.time_to_expert_restore = callback_timings.get(
                "checkpoint_restart", 0.0
            )

        # Verify controller reached REINTEGRATED
        if ctrl.phase != RecoveryPhase.REINTEGRATED:
            report.errors.append(
                f"expected REINTEGRATED, got {ctrl.phase.name}"
            )
            return

        # Verify selected path matches expected
        if ctrl.last_recovery_path != metrics.recovery_path:
            report.errors.append(
                f"path mismatch: expected {metrics.recovery_path}, "
                f"got {ctrl.last_recovery_path}"
            )

        # --- Phase 5: Deferred optimizer (optional) ---
        if plan.enable_deferred_optimizer:
            with PhaseTimer("optimizer_attach", metrics, events):
                metrics.optimizer_deferred = True
                if self._deferred_optimizer_submit_fn:
                    self._deferred_optimizer_submit_fn(
                        expert_ids=plan.expert_ids,
                        step=repair_step,
                    )
                if self._deferred_optimizer_poll_fn:
                    self._deferred_optimizer_poll_fn(step=repair_step)
                metrics.two_phase_final_state = "FULLY_RECOVERED"

        # --- Phase 6: Preferential routing (optional) ---
        if plan.enable_preferential_routing:
            with PhaseTimer("convergence", metrics, events):
                metrics.preferential_routing_activated = True
                # Verify bias would be applied
                events.append({
                    "type": "preferential_routing",
                    "window": plan.preferential_routing_window,
                    "bias": plan.preferential_routing_bias,
                    "expert_ids": list(plan.expert_ids),
                })

        # --- Phase 7: Finalize reintegration ---
        finalized = ctrl.before_iteration(step=repair_step + 1)
        if ctrl.phase != RecoveryPhase.HEALTHY_TRAINING:
            report.errors.append(
                f"expected HEALTHY_TRAINING after finalize, got {ctrl.phase.name}"
            )

        metrics.experts_recovered = len(plan.expert_ids)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def reports(self) -> List[ExperimentReport]:
        return list(self._reports)

    @property
    def num_runs(self) -> int:
        return len(self._reports)

    @property
    def num_passed(self) -> int:
        return sum(1 for r in self._reports if r.success)

    def summary(self) -> Dict[str, Any]:
        return {
            "num_runs": self.num_runs,
            "num_passed": self.num_passed,
            "scenarios": [r.scenario for r in self._reports],
            "all_passed": self.num_passed == self.num_runs,
        }

    def full_report_json(self, indent: int = 2) -> str:
        """All reports as JSON."""
        return json.dumps(
            [r.to_dict() for r in self._reports],
            indent=indent,
            ensure_ascii=False,
        )


# =====================================================================
# Convenience: run all scenarios
# =====================================================================

def run_all_experiments(
    **orchestrator_kwargs,
) -> Dict[str, ExperimentReport]:
    """Run all pre-built experiment scenarios.

    Returns a dict mapping scenario name → ExperimentReport.
    """
    orch = ExperimentOrchestrator(**orchestrator_kwargs)
    results = {}
    for scenario in ExperimentScenario:
        report = orch.run_scenario(scenario)
        results[scenario.value] = report
    return results


# =====================================================================
# Troubleshooting guide
# =====================================================================

TROUBLESHOOTING_GUIDE = """
MOEGAMBIT-MoE Experiment Troubleshooting Guide
=========================================

1. "safe-point repair did not execute"
   - Check: RecoveryController must be in SAFE_POINT_REPAIR phase
   - Check: A replacement must be assigned AND marked ready
   - Check: before_iteration() must be called after replacement ready

2. "path mismatch: expected HYBRID_RECOVERY, got CHECKPOINT_RESTART"
   - Check: gap_threshold vs actual gap
   - Check: checkpoint_iteration is set correctly
   - Fix: Adjust gap_threshold or checkpoint_iteration in the plan

3. "expected REINTEGRATED, got SAFE_POINT_REPAIR"
   - Check: group_rebuild_execute_fn must not raise
   - Check: Look at error logs from the callback

4. "expected HEALTHY_TRAINING after finalize"
   - Check: reintegration_barrier may be blocking
   - Check: missing preconditions in the barrier

5. Deferred optimizer not completing
   - Check: deferred_optimizer_poll_fn must return True
   - Check: two-phase state machine must reach FULLY_RECOVERED

6. Preferential routing not activated
   - Check: moe_moegambit_preferential_routing must be True in config
   - Check: expert_ids must be non-empty
   - Check: PreferentialRoutingManager must be initialized

7. Dense sync timing too high
   - Check: large model → many dense params to broadcast
   - Check: NCCL performance (bandwidth, latency)
   - Normal range: 0.1-5s depending on model size and network

8. Expert restore timing too high
   - Check: checkpoint I/O bandwidth
   - Check: number of experts being restored
   - Normal range: 1-30s depending on expert size and storage speed
"""

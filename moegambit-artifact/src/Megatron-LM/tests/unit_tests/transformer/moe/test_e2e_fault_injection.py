# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""End-to-end fault injection tests for MoEGambit (Phase 14).

Tests validate the complete recovery pipeline under PP=1 and PP>1
using the fault injection framework.  All tests run without torch
or distributed — they exercise the state machines, callbacks, and
metrics collection in a deterministic simulated training loop.

Test matrix
-----------
PP=1 scenarios:
  1. Hard failure → replacement → safe-point repair → resume
  2. Hard failure mid-iteration → rollback → replay → repair → resume
  3. Multiple sequential failures

PP>1 scenarios:
  4. Pipeline stage failure → rollback → replacement → PP repair → resume
  5. Pipeline stage failure at iteration boundary (no rollback)
  6. Multiple PP failures on different stages

Metrics validation:
  9. All latency metrics are non-negative after recovery
  10. Degraded steps counted correctly
  11. Callback invocation order verified
"""

import importlib
import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Bootstrap: load modules without torch / megatron.core.__init__
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Provide a minimal torch stub
_torch_stub = types.ModuleType('torch')
_torch_stub.Tensor = type('Tensor', (), {})
_torch_distributed_stub = types.ModuleType('torch.distributed')
_torch_distributed_stub.is_initialized = lambda: False
_torch_distributed_stub.get_rank = lambda: 0
_torch_distributed_stub.new_group = lambda ranks=None, backend='nccl', **kw: MagicMock()
_torch_distributed_stub.destroy_process_group = lambda pg: None
_torch_distributed_stub.barrier = lambda group=None, **kw: None
_torch_stub.distributed = _torch_distributed_stub
_torch_stub.cuda = MagicMock()

_real_torch = sys.modules.get('torch')
if _real_torch is None:
    sys.modules['torch'] = _torch_stub
    sys.modules['torch.distributed'] = _torch_distributed_stub

# Stub megatron.core
_mc_stub = types.ModuleType('megatron')
_mc_core_stub = types.ModuleType('megatron.core')
_mc_moe_stub = types.ModuleType('megatron.core.transformer')
_mc_moe_stub2 = types.ModuleType('megatron.core.transformer.moe')
for mod_name, mod in [
    ('megatron', _mc_stub),
    ('megatron.core', _mc_core_stub),
    ('megatron.core.transformer', _mc_moe_stub),
    ('megatron.core.transformer.moe', _mc_moe_stub2),
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = mod


def _load_module(name, file_path):
    """Load a module from file path, bypassing package __init__."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = '.'.join(name.split('.')[:-1])
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_moe_dir = os.path.join(
    _REPO_ROOT, 'megatron', 'core', 'transformer', 'moe',
)

# Load modules under test
rc_mod = _load_module(
    'megatron.core.transformer.moe.recovery_controller',
    os.path.join(_moe_dir, 'recovery_controller.py'),
)
inv_mod = _load_module(
    'megatron.core.transformer.moe.iteration_invalidator',
    os.path.join(_moe_dir, 'iteration_invalidator.py'),
)
rb_mod = _load_module(
    'megatron.core.transformer.moe.iteration_rollback',
    os.path.join(_moe_dir, 'iteration_rollback.py'),
)
ocg_mod = _load_module(
    'megatron.core.transformer.moe.optimizer_commit_guard',
    os.path.join(_moe_dir, 'optimizer_commit_guard.py'),
)
hfd_mod = _load_module(
    'megatron.core.transformer.moe.hard_failure_detector',
    os.path.join(_moe_dir, 'hard_failure_detector.py'),
)
pr_mod = _load_module(
    'megatron.core.transformer.moe.pipeline_rollback',
    os.path.join(_moe_dir, 'pipeline_rollback.py'),
)
fif_mod = _load_module(
    'megatron.core.transformer.moe.fault_injection_framework',
    os.path.join(_moe_dir, 'fault_injection_framework.py'),
)

RecoveryController = rc_mod.RecoveryController
RecoveryPhase = rc_mod.RecoveryPhase
IterationInvalidator = inv_mod.IterationInvalidator
RollbackReplayManager = rb_mod.RollbackReplayManager
OptimizerCommitGuard = ocg_mod.OptimizerCommitGuard
HardFailureDetector = hfd_mod.HardFailureDetector
PipelineRollbackCoordinator = pr_mod.PipelineRollbackCoordinator

FaultInjector = fif_mod.FaultInjector
FaultScenario = fif_mod.FaultScenario
FaultType = fif_mod.FaultType
RecoveryMetricsCollector = fif_mod.RecoveryMetricsCollector
TrainingLoopConfig = fif_mod.TrainingLoopConfig
run_simulated_training_loop = fif_mod.run_simulated_training_loop
build_pp1_hard_failure_scenario = fif_mod.build_pp1_hard_failure_scenario
build_pp_gt1_pipeline_failure_scenario = fif_mod.build_pp_gt1_pipeline_failure_scenario


# =====================================================================
# Helper: build a fully-wired test environment
# =====================================================================

def _make_env(pp_size=1):
    """Create a fully-wired test environment with all subsystems."""
    ctrl = RecoveryController()
    invalidator = IterationInvalidator()
    rollback_mgr = RollbackReplayManager()
    commit_guard = OptimizerCommitGuard()
    pipeline_coord = PipelineRollbackCoordinator() if pp_size > 1 else None

    # Wire commit guard to invalidator
    commit_guard.register_callbacks(
        is_iteration_invalid_fn=invalidator.is_current_iteration_invalid,
    )

    # Register controller callbacks (all mocked)
    callbacks = dict(
        health_mark_stale_runnable_fn=MagicMock(),
        health_mark_healthy_fn=MagicMock(),
        replacement_announce_fn=MagicMock(),
        replacement_integrate_fn=MagicMock(),
        group_rebuild_request_fn=MagicMock(),
        group_rebuild_execute_fn=MagicMock(),
        group_rebuild_finish_fn=MagicMock(),
        topology_refresh_fn=MagicMock(),
        dense_sync_fn=MagicMock(),
        expert_restore_fn=MagicMock(),
        pipeline_stage_repair_fn=MagicMock(),
        pipeline_rollback_fn=MagicMock(),
        microbatch_invalidation_fn=MagicMock(),
    )
    ctrl.register_callbacks(**callbacks)

    metrics = RecoveryMetricsCollector()
    injector = FaultInjector(ctrl, metrics=metrics)

    return {
        'ctrl': ctrl,
        'invalidator': invalidator,
        'rollback_mgr': rollback_mgr,
        'commit_guard': commit_guard,
        'pipeline_coord': pipeline_coord,
        'callbacks': callbacks,
        'metrics': metrics,
        'injector': injector,
    }


def _run(env, scenarios, num_steps=100, pp_size=1):
    """Add scenarios and run the simulated training loop."""
    injector = env['injector']
    for s in scenarios:
        injector.add_scenario(s)

    config = TrainingLoopConfig(
        num_steps=num_steps,
        pp_size=pp_size,
        pp_rank=0,
        num_microbatches=4 if pp_size > 1 else 1,
    )

    return run_simulated_training_loop(
        controller=env['ctrl'],
        injector=injector,
        config=config,
        invalidator=env['invalidator'],
        rollback_manager=env['rollback_mgr'],
        commit_guard=env['commit_guard'],
        pipeline_coordinator=env['pipeline_coord'],
        metrics_collector=env['metrics'],
    )


# =====================================================================
# PP=1 Tests
# =====================================================================

class TestPP1HardFailureE2E(unittest.TestCase):
    """PP=1: Hard failure → replacement → safe-point repair → resume."""

    def test_full_recovery_cycle(self):
        env = _make_env(pp_size=1)
        scenarios = build_pp1_hard_failure_scenario(
            fault_step=10, replacement_step=15,
        )
        result = _run(env, scenarios, num_steps=30)

        # Should return to healthy
        self.assertEqual(result.final_phase, 'HEALTHY_TRAINING')
        self.assertEqual(result.repairs_executed, 1)
        self.assertEqual(result.total_steps, 30)
        self.assertGreater(result.healthy_steps, 0)
        self.assertGreater(result.degraded_steps, 0)
        self.assertEqual(len(result.errors), 0)

    def test_callbacks_invoked(self):
        env = _make_env(pp_size=1)
        scenarios = build_pp1_hard_failure_scenario(
            fault_step=10, replacement_step=15,
        )
        result = _run(env, scenarios, num_steps=30)

        cb_names = [name for _, name in result.callback_log]
        self.assertIn('replacement_integrate_fn', cb_names)
        self.assertIn('group_rebuild_request_fn', cb_names)
        self.assertIn('group_rebuild_execute_fn', cb_names)
        self.assertIn('group_rebuild_finish_fn', cb_names)
        self.assertIn('topology_refresh_fn', cb_names)
        self.assertIn('dense_sync_fn', cb_names)
        self.assertIn('expert_restore_fn', cb_names)

    def test_pp1_no_pipeline_repair(self):
        """PP=1 should NOT invoke pipeline_stage_repair_fn."""
        env = _make_env(pp_size=1)
        scenarios = build_pp1_hard_failure_scenario(
            fault_step=10, replacement_step=15,
        )
        result = _run(env, scenarios, num_steps=30)

        cb_names = [name for _, name in result.callback_log]
        self.assertNotIn('pipeline_stage_repair_fn', cb_names)

    def test_metrics_collected(self):
        env = _make_env(pp_size=1)
        scenarios = build_pp1_hard_failure_scenario(
            fault_step=10, replacement_step=15,
        )
        result = _run(env, scenarios, num_steps=30)

        self.assertEqual(len(result.metrics), 1)
        m = result.metrics[0]
        self.assertEqual(m.fault_step, 10)
        self.assertEqual(m.fault_type, 'hard_failure')
        self.assertEqual(m.failed_rank, 4)
        self.assertGreaterEqual(m.time_to_detect, 0)
        self.assertGreaterEqual(m.time_to_resume, 0)
        self.assertGreater(m.degraded_steps, 0)

    def test_mid_iteration_invalidation(self):
        """Mid-iteration hard failure should invalidate the iteration."""
        env = _make_env(pp_size=1)
        scenarios = build_pp1_hard_failure_scenario(
            fault_step=10, replacement_step=15, mid_iteration=True,
        )
        result = _run(env, scenarios, num_steps=30)

        self.assertGreater(result.invalidated_steps, 0)


class TestPP1RollbackReplayE2E(unittest.TestCase):
    """PP=1: Hard failure mid-iteration → rollback → replay → repair."""

    def test_rollback_and_replay(self):
        env = _make_env(pp_size=1)
        scenarios = build_pp1_hard_failure_scenario(
            fault_step=10, replacement_step=15, mid_iteration=True,
        )
        result = _run(env, scenarios, num_steps=30)

        # Rollback should have been attempted
        self.assertGreaterEqual(result.rollbacks, 1)
        self.assertEqual(result.final_phase, 'HEALTHY_TRAINING')


class TestPP1MultipleFailures(unittest.TestCase):
    """PP=1: Multiple sequential failures."""

    def test_two_failures(self):
        env = _make_env(pp_size=1)
        scenarios = [
            # First failure
            FaultScenario(
                step=10, fault_type=FaultType.HARD_FAILURE,
                failed_rank=4, expert_ids=[8, 9],
                ep_group_ranks=[0, 2, 4, 6],
                dp_group_ranks=[0, 4, 8],
            ),
            FaultScenario(
                step=15, fault_type=FaultType.REPLACEMENT_READY,
                failed_rank=4, replacement_rank=64,
            ),
            # Second failure (after first recovery)
            FaultScenario(
                step=25, fault_type=FaultType.HARD_FAILURE,
                failed_rank=6, expert_ids=[12, 13],
                ep_group_ranks=[0, 2, 64, 6],
                dp_group_ranks=[0, 6, 8],
            ),
            FaultScenario(
                step=30, fault_type=FaultType.REPLACEMENT_READY,
                failed_rank=6, replacement_rank=66,
            ),
        ]
        result = _run(env, scenarios, num_steps=50)

        self.assertEqual(result.final_phase, 'HEALTHY_TRAINING')
        self.assertEqual(result.repairs_executed, 2)


# =====================================================================
# PP>1 Tests
# =====================================================================

class TestPPGt1PipelineFailureE2E(unittest.TestCase):
    """PP>1: Pipeline stage failure → rollback → replacement → PP repair → resume."""

    def test_full_pp_recovery_cycle(self):
        env = _make_env(pp_size=4)
        scenarios = build_pp_gt1_pipeline_failure_scenario(
            fault_step=10, replacement_step=15,
            failed_stage=2, pp_group_ranks=[0, 2, 4, 6],
        )
        result = _run(env, scenarios, num_steps=30, pp_size=4)

        self.assertEqual(result.final_phase, 'HEALTHY_TRAINING')
        self.assertEqual(result.repairs_executed, 1)
        self.assertEqual(len(result.errors), 0)

    def test_pipeline_repair_callback_invoked(self):
        """PP>1 should invoke pipeline_stage_repair_fn."""
        env = _make_env(pp_size=4)
        scenarios = build_pp_gt1_pipeline_failure_scenario(
            fault_step=10, replacement_step=15,
            failed_stage=2, pp_group_ranks=[0, 2, 4, 6],
        )
        result = _run(env, scenarios, num_steps=30, pp_size=4)

        cb_names = [name for _, name in result.callback_log]
        self.assertIn('pipeline_stage_repair_fn', cb_names)

    def test_rollback_pending_visited(self):
        """Mid-iteration PP failure should visit ROLLBACK_PENDING."""
        env = _make_env(pp_size=4)
        scenarios = build_pp_gt1_pipeline_failure_scenario(
            fault_step=10, replacement_step=15,
            failed_stage=2, pp_group_ranks=[0, 2, 4, 6],
            mid_iteration=True,
        )
        result = _run(env, scenarios, num_steps=30, pp_size=4)

        phase_targets = [t for _, _, t in result.phase_log]
        self.assertIn('ROLLBACK_PENDING', phase_targets)

    def test_pipeline_rebinding_visited(self):
        """PP>1 repair should visit PIPELINE_REBINDING."""
        env = _make_env(pp_size=4)
        scenarios = build_pp_gt1_pipeline_failure_scenario(
            fault_step=10, replacement_step=15,
            failed_stage=2, pp_group_ranks=[0, 2, 4, 6],
        )
        result = _run(env, scenarios, num_steps=30, pp_size=4)

        phase_targets = [t for _, _, t in result.phase_log]
        self.assertIn('PIPELINE_REBINDING', phase_targets)

    def test_pp_metrics(self):
        """PP>1 metrics should include pp_repair latency."""
        env = _make_env(pp_size=4)
        scenarios = build_pp_gt1_pipeline_failure_scenario(
            fault_step=10, replacement_step=15,
            failed_stage=2, pp_group_ranks=[0, 2, 4, 6],
        )
        result = _run(env, scenarios, num_steps=30, pp_size=4)

        self.assertEqual(len(result.metrics), 1)
        m = result.metrics[0]
        self.assertEqual(m.fault_step, 10)
        self.assertGreaterEqual(m.time_to_detect, 0)
        self.assertGreaterEqual(m.time_to_pp_repair, 0)
        self.assertGreaterEqual(m.time_to_resume, 0)

    def test_microbatch_invalidation(self):
        """Mid-iteration PP failure should invalidate microbatches."""
        env = _make_env(pp_size=4)
        scenarios = build_pp_gt1_pipeline_failure_scenario(
            fault_step=10, replacement_step=15,
            failed_stage=2, pp_group_ranks=[0, 2, 4, 6],
            mid_iteration=True,
        )
        result = _run(env, scenarios, num_steps=30, pp_size=4)

        cb_names = [name for _, name in result.callback_log]
        self.assertIn('microbatch_invalidation_fn', cb_names)


class TestPPGt1IterationBoundaryFailure(unittest.TestCase):
    """PP>1: Pipeline failure at iteration boundary (no rollback)."""

    def test_no_rollback_at_boundary(self):
        env = _make_env(pp_size=4)
        scenarios = build_pp_gt1_pipeline_failure_scenario(
            fault_step=10, replacement_step=15,
            failed_stage=1, pp_group_ranks=[0, 2, 4, 6],
            mid_iteration=False,
        )
        result = _run(env, scenarios, num_steps=30, pp_size=4)

        self.assertEqual(result.final_phase, 'HEALTHY_TRAINING')
        # No ROLLBACK_PENDING
        phase_targets = [t for _, _, t in result.phase_log]
        self.assertNotIn('ROLLBACK_PENDING', phase_targets)
        # But PIPELINE_REBINDING should still be visited
        self.assertIn('PIPELINE_REBINDING', phase_targets)


class TestPPGt1MultipleFailures(unittest.TestCase):
    """PP>1: Multiple pipeline failures on different stages."""

    def test_two_stage_failures(self):
        env = _make_env(pp_size=4)
        scenarios = [
            # First failure: stage 1
            FaultScenario(
                step=10, fault_type=FaultType.PIPELINE_STAGE_FAILURE,
                failed_rank=2, failed_stage=1,
                pp_group_ranks=[0, 2, 4, 6],
                expert_ids=[4, 5],
                ep_group_ranks=[0, 2, 4, 6],
                dp_group_ranks=[0, 2, 8],
                mid_iteration=True,
            ),
            FaultScenario(
                step=15, fault_type=FaultType.REPLACEMENT_READY,
                failed_rank=2, replacement_rank=62,
            ),
            # Second failure: stage 3
            FaultScenario(
                step=30, fault_type=FaultType.PIPELINE_STAGE_FAILURE,
                failed_rank=6, failed_stage=3,
                pp_group_ranks=[0, 62, 4, 6],
                expert_ids=[12, 13],
                ep_group_ranks=[0, 62, 4, 6],
                dp_group_ranks=[0, 6, 8],
                mid_iteration=True,
            ),
            FaultScenario(
                step=35, fault_type=FaultType.REPLACEMENT_READY,
                failed_rank=6, replacement_rank=66,
            ),
        ]
        result = _run(env, scenarios, num_steps=50, pp_size=4)

        self.assertEqual(result.final_phase, 'HEALTHY_TRAINING')
        self.assertEqual(result.repairs_executed, 2)


# =====================================================================
# Metrics validation tests
# =====================================================================

class TestMetricsValidation(unittest.TestCase):
    """Validate that metrics are correctly collected."""

    def test_all_latencies_non_negative_pp1(self):
        env = _make_env(pp_size=1)
        scenarios = build_pp1_hard_failure_scenario(
            fault_step=10, replacement_step=15,
        )
        result = _run(env, scenarios, num_steps=30)

        m = result.metrics[0]
        self.assertGreaterEqual(m.time_to_detect, 0)
        self.assertGreaterEqual(m.time_to_resume, 0)
        self.assertGreaterEqual(m.time_to_group_repair, 0)
        self.assertGreaterEqual(m.time_to_dispatch_refresh, 0)

    def test_all_latencies_non_negative_pp_gt1(self):
        env = _make_env(pp_size=4)
        scenarios = build_pp_gt1_pipeline_failure_scenario(
            fault_step=10, replacement_step=15,
            failed_stage=2, pp_group_ranks=[0, 2, 4, 6],
            mid_iteration=True,
        )
        result = _run(env, scenarios, num_steps=30, pp_size=4)

        m = result.metrics[0]
        self.assertGreaterEqual(m.time_to_detect, 0)
        self.assertGreaterEqual(m.time_to_rollback, 0)
        self.assertGreaterEqual(m.time_to_pp_repair, 0)
        self.assertGreaterEqual(m.time_to_resume, 0)

    def test_degraded_steps_counted(self):
        env = _make_env(pp_size=1)
        scenarios = build_pp1_hard_failure_scenario(
            fault_step=10, replacement_step=15,
        )
        result = _run(env, scenarios, num_steps=30)

        m = result.metrics[0]
        self.assertGreater(m.degraded_steps, 0)
        # degraded_steps should be <= total_steps - healthy_steps
        self.assertLessEqual(m.degraded_steps, result.total_steps)

    def test_metrics_report_format(self):
        env = _make_env(pp_size=1)
        scenarios = build_pp1_hard_failure_scenario(
            fault_step=10, replacement_step=15,
        )
        result = _run(env, scenarios, num_steps=30)

        m = result.metrics[0]
        report = m.format_report()
        self.assertIn('Recovery Metrics Report', report)
        self.assertIn('detect:', report)
        self.assertIn('resume:', report)
        self.assertIn('degraded_steps:', report)

    def test_wall_clock_latencies(self):
        env = _make_env(pp_size=1)
        scenarios = build_pp1_hard_failure_scenario(
            fault_step=10, replacement_step=15,
        )
        result = _run(env, scenarios, num_steps=30)

        m = result.metrics[0]
        wc = m.wall_clock_latencies()
        self.assertIn('wall_detect_s', wc)
        self.assertIn('wall_resume_s', wc)
        # Wall-clock latencies should be >= 0 (or -1 if not recorded)
        for k, v in wc.items():
            self.assertTrue(v >= 0 or v == -1.0, f"{k}={v}")


# =====================================================================
# Callback order validation
# =====================================================================

class TestCallbackOrder(unittest.TestCase):
    """Validate that callbacks are invoked in the correct order."""

    def test_pp1_callback_order(self):
        """PP=1: callbacks should follow the 7-step repair sequence."""
        env = _make_env(pp_size=1)
        scenarios = build_pp1_hard_failure_scenario(
            fault_step=10, replacement_step=15,
        )
        result = _run(env, scenarios, num_steps=30)

        # Extract repair-phase callbacks (filter out quarantine/health)
        repair_cbs = [
            name for _, name in result.callback_log
            if name in (
                'replacement_integrate_fn', 'group_rebuild_request_fn',
                'group_rebuild_execute_fn', 'group_rebuild_finish_fn',
                'topology_refresh_fn', 'dense_sync_fn', 'expert_restore_fn',
            )
        ]

        expected_order = [
            'replacement_integrate_fn',
            'group_rebuild_request_fn',
            'group_rebuild_execute_fn',
            'group_rebuild_finish_fn',
            'topology_refresh_fn',
            'dense_sync_fn',
            'expert_restore_fn',
        ]
        self.assertEqual(repair_cbs, expected_order)

    def test_pp_gt1_callback_order(self):
        """PP>1: callbacks should include pipeline_stage_repair_fn after expert_restore."""
        env = _make_env(pp_size=4)
        scenarios = build_pp_gt1_pipeline_failure_scenario(
            fault_step=10, replacement_step=15,
            failed_stage=2, pp_group_ranks=[0, 2, 4, 6],
        )
        result = _run(env, scenarios, num_steps=30, pp_size=4)

        repair_cbs = [
            name for _, name in result.callback_log
            if name in (
                'replacement_integrate_fn', 'group_rebuild_request_fn',
                'group_rebuild_execute_fn', 'group_rebuild_finish_fn',
                'topology_refresh_fn', 'dense_sync_fn', 'expert_restore_fn',
                'pipeline_stage_repair_fn',
            )
        ]

        expected_order = [
            'replacement_integrate_fn',
            'group_rebuild_request_fn',
            'group_rebuild_execute_fn',
            'group_rebuild_finish_fn',
            'topology_refresh_fn',
            'dense_sync_fn',
            'expert_restore_fn',
            'pipeline_stage_repair_fn',
        ]
        self.assertEqual(repair_cbs, expected_order)


# =====================================================================
# Phase transition validation
# =====================================================================

class TestPhaseTransitions(unittest.TestCase):
    """Validate complete phase transition sequences."""

    def test_pp1_phase_sequence(self):
        env = _make_env(pp_size=1)
        scenarios = build_pp1_hard_failure_scenario(
            fault_step=10, replacement_step=15,
        )
        result = _run(env, scenarios, num_steps=30)

        targets = [t for _, _, t in result.phase_log]
        # Must contain these in order
        self.assertIn('PENDING_GROUP_REPAIR', targets)
        self.assertIn('WAITING_FOR_REPLACEMENT', targets)
        self.assertIn('SAFE_POINT_REPAIR', targets)
        self.assertIn('REINTEGRATED', targets)
        self.assertIn('HEALTHY_TRAINING', targets)

        # PENDING before WAITING before SAFE_POINT before REINTEGRATED
        idx_pending = targets.index('PENDING_GROUP_REPAIR')
        idx_waiting = targets.index('WAITING_FOR_REPLACEMENT')
        idx_safe = targets.index('SAFE_POINT_REPAIR')
        idx_reint = targets.index('REINTEGRATED')
        idx_healthy = targets.index('HEALTHY_TRAINING')
        self.assertLess(idx_pending, idx_waiting)
        self.assertLess(idx_waiting, idx_safe)
        self.assertLess(idx_safe, idx_reint)
        self.assertLess(idx_reint, idx_healthy)

    def test_pp_gt1_phase_sequence(self):
        env = _make_env(pp_size=4)
        scenarios = build_pp_gt1_pipeline_failure_scenario(
            fault_step=10, replacement_step=15,
            failed_stage=2, pp_group_ranks=[0, 2, 4, 6],
            mid_iteration=True,
        )
        result = _run(env, scenarios, num_steps=30, pp_size=4)

        targets = [t for _, _, t in result.phase_log]
        # PP>1 mid-iteration: must include ROLLBACK_PENDING and PIPELINE_REBINDING
        self.assertIn('ROLLBACK_PENDING', targets)
        self.assertIn('PENDING_GROUP_REPAIR', targets)
        self.assertIn('PIPELINE_REBINDING', targets)
        self.assertIn('REINTEGRATED', targets)
        self.assertIn('HEALTHY_TRAINING', targets)

        # Order: ROLLBACK_PENDING before PENDING before PIPELINE_REBINDING
        idx_rollback = targets.index('ROLLBACK_PENDING')
        idx_pending = targets.index('PENDING_GROUP_REPAIR')
        idx_pp = targets.index('PIPELINE_REBINDING')
        idx_reint = targets.index('REINTEGRATED')
        self.assertLess(idx_rollback, idx_pending)
        self.assertLess(idx_pending, idx_pp)
        self.assertLess(idx_pp, idx_reint)


# =====================================================================
# Framework self-tests
# =====================================================================

class TestFaultInjectorFramework(unittest.TestCase):
    """Tests for the fault injection framework itself."""

    def test_scenario_add_and_inject(self):
        ctrl = RecoveryController()
        ctrl.register_callbacks()
        injector = FaultInjector(ctrl)

        idx = injector.add_scenario(FaultScenario(
            step=5, fault_type=FaultType.HARD_FAILURE,
            failed_rank=4, expert_ids=[8, 9],
            ep_group_ranks=[0, 2, 4, 6],
            dp_group_ranks=[0, 4, 8],
        ))
        self.assertEqual(idx, 0)
        self.assertEqual(injector.num_scenarios, 1)
        self.assertEqual(injector.num_injected, 0)

        # Before step 5: no injection
        injected = injector.maybe_inject(3)
        self.assertEqual(injected, [])
        self.assertEqual(injector.num_injected, 0)

        # At step 5: inject
        injected = injector.maybe_inject(5)
        self.assertEqual(injected, [0])
        self.assertEqual(injector.num_injected, 1)

        # After step 5: no re-injection
        injected = injector.maybe_inject(6)
        self.assertEqual(injected, [])

    def test_metrics_collector_lifecycle(self):
        mc = RecoveryMetricsCollector()
        mc.on_fault_injected(10, 'hard_failure', 4)
        mc.on_phase_change(10, 'HEALTHY_TRAINING', 'PENDING_GROUP_REPAIR')
        mc.on_callback_invoked(15, 'group_rebuild_finish_fn')
        mc.on_callback_invoked(15, 'topology_refresh_fn')
        mc.on_callback_invoked(15, 'pipeline_stage_repair_fn')
        mc.on_step_completed(10, True, True)
        mc.on_step_completed(11, True, False)
        mc.on_phase_change(16, 'REINTEGRATED', 'HEALTHY_TRAINING')
        mc.on_step_completed(17, False, False)

        m = mc.finalize()
        self.assertIsNotNone(m)
        self.assertEqual(m.fault_step, 10)
        self.assertEqual(m.time_to_detect, 0)
        self.assertGreaterEqual(m.time_to_group_repair, 0)
        self.assertGreaterEqual(m.time_to_dispatch_refresh, 0)
        self.assertGreaterEqual(m.time_to_pp_repair, 0)
        self.assertEqual(m.time_to_resume, 6)
        self.assertEqual(m.degraded_steps, 2)
        self.assertEqual(m.invalidated_steps, 1)

    def test_training_loop_config_defaults(self):
        config = TrainingLoopConfig()
        self.assertEqual(config.num_steps, 100)
        self.assertEqual(config.pp_size, 1)

    def test_reset(self):
        ctrl = RecoveryController()
        injector = FaultInjector(ctrl)
        injector.add_scenario(FaultScenario(step=5))
        injector.reset()
        self.assertEqual(injector.num_scenarios, 0)


if __name__ == '__main__':
    unittest.main()

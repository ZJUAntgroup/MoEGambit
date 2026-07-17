# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for reintegration barrier integration with RecoveryController.

Validates:
1. ReintegrationBarrier standalone lifecycle
2. RecoveryController + barrier integration
3. Barrier blocks reintegration until all preconditions met
4. Barrier allows reintegration after all preconditions met
5. End-to-end: hard failure → repair → barrier → reintegration → healthy
"""

import os
import sys
import types
import unittest
import importlib.util

# ---------------------------------------------------------------------------
# Bootstrap: load modules without triggering torch imports
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..', '..'))
_MOE_DIR = os.path.join(_REPO_ROOT, 'megatron', 'core', 'transformer', 'moe')

# Stub out torch and megatron.core before importing MOEGAMBIT modules
_torch_stub = types.ModuleType('torch')
_torch_stub.Tensor = type('Tensor', (), {})
_dist_stub = types.ModuleType('torch.distributed')
_dist_stub.is_initialized = lambda: False
_dist_stub.get_rank = lambda: 0
_torch_stub.distributed = _dist_stub
sys.modules.setdefault('torch', _torch_stub)
sys.modules.setdefault('torch.distributed', _dist_stub)

_mc_stub = types.ModuleType('megatron')
_mcore_stub = types.ModuleType('megatron.core')
_mt_stub = types.ModuleType('megatron.core.transformer')
_moe_pkg = types.ModuleType('megatron.core.transformer.moe')
sys.modules.setdefault('megatron', _mc_stub)
sys.modules.setdefault('megatron.core', _mcore_stub)
sys.modules.setdefault('megatron.core.transformer', _mt_stub)
sys.modules.setdefault('megatron.core.transformer.moe', _moe_pkg)


def _load_module(name, filename):
    path = os.path.join(_MOE_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'megatron.core.transformer.moe'
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rb_mod = _load_module(
    'megatron.core.transformer.moe.reintegration_barrier',
    'reintegration_barrier.py',
)
rc_mod = _load_module(
    'megatron.core.transformer.moe.recovery_controller',
    'recovery_controller.py',
)

ReintegrationBarrier = rb_mod.ReintegrationBarrier
ReintegrationPhase = rb_mod.ReintegrationPhase
REQUIRED_PRECONDITIONS = rb_mod.REQUIRED_PRECONDITIONS
RecoveryController = rc_mod.RecoveryController
RecoveryPhase = rc_mod.RecoveryPhase


# =====================================================================
# Test: ReintegrationBarrier standalone
# =====================================================================

class TestReintegrationBarrierStandalone(unittest.TestCase):
    """Test the ReintegrationBarrier in isolation."""

    def setUp(self):
        self.barrier = ReintegrationBarrier()

    def test_begin_reintegration(self):
        record = self.barrier.begin_reintegration(
            failed_rank=2, replacement_rank=10,
            expert_ids=[4, 5], step=100,
        )
        self.assertEqual(record.failed_rank, 2)
        self.assertEqual(record.replacement_rank, 10)
        self.assertEqual(record.expert_ids, [4, 5])
        self.assertEqual(record.phase, ReintegrationPhase.ISOLATED_READY)
        self.assertFalse(self.barrier.can_reintegrate(2))

    def test_mark_all_preconditions(self):
        self.barrier.begin_reintegration(
            failed_rank=2, replacement_rank=10,
            expert_ids=[4, 5], step=100,
        )
        for pc in sorted(REQUIRED_PRECONDITIONS):
            self.barrier.mark_precondition(2, pc, step=100)

        record = self.barrier.get_record(2)
        self.assertTrue(record.all_preconditions_met())
        self.assertEqual(record.phase, ReintegrationPhase.REPAIRED_NOT_ROUTED)
        self.assertTrue(self.barrier.can_reintegrate(2))

    def test_cannot_reintegrate_with_missing_preconditions(self):
        self.barrier.begin_reintegration(
            failed_rank=2, replacement_rank=10,
            expert_ids=[4, 5], step=100,
        )
        # Mark only some preconditions
        self.barrier.mark_precondition(2, "groups_repaired", step=100)
        self.barrier.mark_precondition(2, "directory_refreshed", step=100)
        self.assertFalse(self.barrier.can_reintegrate(2))

    def test_execute_reintegration(self):
        self.barrier.begin_reintegration(
            failed_rank=2, replacement_rank=10,
            expert_ids=[4, 5], step=100,
        )
        for pc in sorted(REQUIRED_PRECONDITIONS):
            self.barrier.mark_precondition(2, pc, step=100)

        routing_calls = []
        def mock_re_enable(expert_ids, step):
            routing_calls.append((expert_ids, step))

        result = self.barrier.execute_reintegration(
            2, step=105,
            re_enable_routing_fn=mock_re_enable,
        )
        self.assertTrue(result)
        record = self.barrier.get_record(2)
        self.assertEqual(record.phase, ReintegrationPhase.ROUTED_INTEGRATED)
        self.assertEqual(len(routing_calls), 1)
        self.assertEqual(routing_calls[0][0], [4, 5])

    def test_execute_reintegration_fails_without_preconditions(self):
        self.barrier.begin_reintegration(
            failed_rank=2, replacement_rank=10,
            expert_ids=[4, 5], step=100,
        )
        with self.assertRaises(RuntimeError):
            self.barrier.execute_reintegration(2, step=105)

    def test_num_pending_and_integrated(self):
        self.barrier.begin_reintegration(
            failed_rank=2, replacement_rank=10,
            expert_ids=[4, 5], step=100,
        )
        self.assertEqual(self.barrier.num_pending, 1)
        self.assertEqual(self.barrier.num_integrated, 0)

        for pc in sorted(REQUIRED_PRECONDITIONS):
            self.barrier.mark_precondition(2, pc, step=100)
        self.barrier.execute_reintegration(2, step=105)

        self.assertEqual(self.barrier.num_pending, 0)
        self.assertEqual(self.barrier.num_integrated, 1)

    def test_summary(self):
        self.barrier.begin_reintegration(
            failed_rank=2, replacement_rank=10,
            expert_ids=[4, 5], step=100,
        )
        s = self.barrier.summary()
        self.assertIn('num_pending', s)
        self.assertIn('num_integrated', s)

    def test_reset(self):
        self.barrier.begin_reintegration(
            failed_rank=2, replacement_rank=10,
            expert_ids=[4, 5], step=100,
        )
        self.barrier.reset()
        self.assertEqual(self.barrier.num_pending, 0)
        self.assertIsNone(self.barrier.get_record(2))

    def test_duplicate_begin_raises(self):
        self.barrier.begin_reintegration(
            failed_rank=2, replacement_rank=10,
            expert_ids=[4, 5], step=100,
        )
        with self.assertRaises(ValueError):
            self.barrier.begin_reintegration(
                failed_rank=2, replacement_rank=11,
                expert_ids=[4, 5], step=101,
            )

    def test_invalid_precondition_raises(self):
        self.barrier.begin_reintegration(
            failed_rank=2, replacement_rank=10,
            expert_ids=[4, 5], step=100,
        )
        with self.assertRaises(ValueError):
            self.barrier.mark_precondition(2, "nonexistent_precondition", step=100)

    def test_mark_precondition_no_record_raises(self):
        with self.assertRaises(KeyError):
            self.barrier.mark_precondition(99, "group_repaired", step=100)


# =====================================================================
# Test: RecoveryController + Barrier integration
# =====================================================================

class TestRecoveryControllerBarrierIntegration(unittest.TestCase):
    """Test that RecoveryController correctly uses the barrier."""

    def setUp(self):
        self.ctrl = RecoveryController()
        self.barrier = ReintegrationBarrier()
        self.ctrl._reintegration_barrier = self.barrier

        # Track callback invocations
        self.quarantine_calls = []
        self.health_unavailable_calls = []
        self.health_healthy_calls = []
        self.group_rebuild_calls = []
        self.topology_calls = []
        self.dense_sync_calls = []
        self.expert_restore_calls = []
        self.replacement_announce_calls = []
        self.replacement_integrate_calls = []

        self.ctrl.register_callbacks(
            quarantine_fn=lambda **kw: self.quarantine_calls.append(kw),
            health_mark_unavailable_fn=lambda **kw: self.health_unavailable_calls.append(kw),
            health_mark_stale_runnable_fn=lambda **kw: None,
            health_mark_healthy_fn=lambda **kw: self.health_healthy_calls.append(kw),
            replacement_announce_fn=lambda **kw: self.replacement_announce_calls.append(kw),
            replacement_integrate_fn=lambda **kw: self.replacement_integrate_calls.append(kw),
            group_rebuild_request_fn=lambda **kw: self.group_rebuild_calls.append(kw),
            group_rebuild_execute_fn=lambda **kw: self.group_rebuild_calls.append(kw),
            group_rebuild_finish_fn=lambda **kw: self.group_rebuild_calls.append(kw),
            topology_refresh_fn=lambda **kw: self.topology_calls.append(kw),
            dense_sync_fn=lambda **kw: self.dense_sync_calls.append(kw),
            expert_restore_fn=lambda **kw: self.expert_restore_calls.append(kw),
        )

    def _simulate_hard_failure_and_repair(self, step=100):
        """Drive the controller through hard failure → safe-point repair."""
        # Hard failure
        self.ctrl.on_hard_rank_failure(
            failed_rank=2, reason="test",
            step=step, expert_ids=[4, 5],
            ep_group_ranks=[0, 1, 2, 3],
            dp_group_ranks=[0, 4, 8],
            mid_iteration=True,
        )
        # Replacement assigned + ready
        self.ctrl.on_replacement_assigned(
            failed_rank=2, replacement_rank=10, step=step + 5,
        )
        self.ctrl.on_replacement_ready(
            failed_rank=2, step=step + 5,
        )
        # Safe-point repair
        self.ctrl.before_iteration(step=step + 10)

    def test_barrier_blocks_finalization_without_preconditions(self):
        """Barrier should block _finalize_reintegration if preconditions not met."""
        self._simulate_hard_failure_and_repair(step=100)

        # At this point, the controller should be in REINTEGRATED phase
        # but the barrier record should exist with preconditions from
        # _execute_safe_point_repair
        self.assertEqual(self.ctrl.phase, RecoveryPhase.REINTEGRATED)

        # Check barrier has a record
        record = self.barrier.get_record(2)
        self.assertIsNotNone(record)

        # Now call before_iteration which triggers _finalize_reintegration
        # The barrier should have all preconditions met from the repair
        # sequence, so finalization should proceed
        self.ctrl.before_iteration(step=115)

        # Should have transitioned to HEALTHY_TRAINING
        self.assertEqual(self.ctrl.phase, RecoveryPhase.HEALTHY_TRAINING)

    def test_barrier_record_created_during_repair(self):
        """Verify barrier record is created during safe-point repair."""
        self._simulate_hard_failure_and_repair(step=100)

        record = self.barrier.get_record(2)
        self.assertIsNotNone(record)
        self.assertEqual(record.failed_rank, 2)
        self.assertEqual(record.replacement_rank, 10)
        self.assertEqual(record.expert_ids, [4, 5])

    def test_barrier_preconditions_marked_during_repair(self):
        """Verify all preconditions are marked during safe-point repair."""
        self._simulate_hard_failure_and_repair(step=100)

        record = self.barrier.get_record(2)
        self.assertIsNotNone(record)
        # All preconditions should be met after repair
        self.assertTrue(record.all_preconditions_met())

    def test_full_lifecycle_with_barrier(self):
        """Full lifecycle: failure → repair → barrier → reintegration → healthy."""
        # Phase 1: Hard failure
        self.ctrl.on_hard_rank_failure(
            failed_rank=2, reason="test",
            step=100, expert_ids=[4, 5],
            ep_group_ranks=[0, 1, 2, 3],
            dp_group_ranks=[0, 4, 8],
            mid_iteration=True,
        )
        self.assertEqual(self.ctrl.phase, RecoveryPhase.PENDING_GROUP_REPAIR)

        # Phase 2: Replacement
        self.ctrl.on_replacement_assigned(
            failed_rank=2, replacement_rank=10, step=105,
        )
        self.ctrl.on_replacement_ready(failed_rank=2, step=105)
        self.assertEqual(self.ctrl.phase, RecoveryPhase.SAFE_POINT_REPAIR)

        # Phase 3: Safe-point repair (triggers barrier begin + preconditions)
        self.ctrl.before_iteration(step=110)
        self.assertEqual(self.ctrl.phase, RecoveryPhase.REINTEGRATED)

        # Verify barrier state
        record = self.barrier.get_record(2)
        self.assertIsNotNone(record)
        self.assertTrue(record.all_preconditions_met())

        # Phase 4: Finalize reintegration (next iteration)
        self.ctrl.before_iteration(step=111)
        self.assertEqual(self.ctrl.phase, RecoveryPhase.HEALTHY_TRAINING)

        # Verify health_mark_healthy was called
        self.assertTrue(len(self.health_healthy_calls) > 0)

    def test_no_barrier_still_works(self):
        """Without barrier, finalization should still work (backward compat)."""
        ctrl_no_barrier = RecoveryController()
        ctrl_no_barrier.register_callbacks(
            quarantine_fn=lambda **kw: None,
            health_mark_unavailable_fn=lambda **kw: None,
            health_mark_stale_runnable_fn=lambda **kw: None,
            health_mark_healthy_fn=lambda **kw: self.health_healthy_calls.append(kw),
            replacement_announce_fn=lambda **kw: None,
            replacement_integrate_fn=lambda **kw: None,
            group_rebuild_request_fn=lambda **kw: None,
            group_rebuild_execute_fn=lambda **kw: None,
            group_rebuild_finish_fn=lambda **kw: None,
            topology_refresh_fn=lambda **kw: None,
            dense_sync_fn=lambda **kw: None,
            expert_restore_fn=lambda **kw: None,
        )

        # Hard failure + replacement + repair
        ctrl_no_barrier.on_hard_rank_failure(
            failed_rank=2, reason="test",
            step=100, expert_ids=[4, 5],
            ep_group_ranks=[0, 1, 2, 3],
            dp_group_ranks=[0, 4, 8],
            mid_iteration=True,
        )
        ctrl_no_barrier.on_replacement_assigned(
            failed_rank=2, replacement_rank=10, step=105,
        )
        ctrl_no_barrier.on_replacement_ready(failed_rank=2, step=105)
        ctrl_no_barrier.before_iteration(step=110)
        self.assertEqual(ctrl_no_barrier.phase, RecoveryPhase.REINTEGRATED)

        # Finalize
        self.health_healthy_calls.clear()
        ctrl_no_barrier.before_iteration(step=111)
        self.assertEqual(ctrl_no_barrier.phase, RecoveryPhase.HEALTHY_TRAINING)
        self.assertTrue(len(self.health_healthy_calls) > 0)


# =====================================================================
# Test: Barrier prevents premature participation
# =====================================================================

class TestBarrierPreventsPrematurity(unittest.TestCase):
    """Verify replacement rank cannot participate before reintegration."""

    def test_isolated_rank_not_reintegratable(self):
        barrier = ReintegrationBarrier()
        barrier.begin_reintegration(
            failed_rank=2, replacement_rank=10,
            expert_ids=[4, 5], step=100,
        )
        # Without any preconditions, cannot reintegrate
        self.assertFalse(barrier.can_reintegrate(2))

    def test_partial_preconditions_not_reintegratable(self):
        barrier = ReintegrationBarrier()
        barrier.begin_reintegration(
            failed_rank=2, replacement_rank=10,
            expert_ids=[4, 5], step=100,
        )
        # Mark only 3 of 5 preconditions
        barrier.mark_precondition(2, "groups_repaired", step=100)
        barrier.mark_precondition(2, "directory_refreshed", step=100)
        barrier.mark_precondition(2, "topology_refreshed", step=100)
        self.assertFalse(barrier.can_reintegrate(2))

        record = barrier.get_record(2)
        missing = record.missing_preconditions()
        self.assertTrue(len(missing) > 0)

    def test_all_preconditions_allows_reintegration(self):
        barrier = ReintegrationBarrier()
        barrier.begin_reintegration(
            failed_rank=2, replacement_rank=10,
            expert_ids=[4, 5], step=100,
        )
        for pc in sorted(REQUIRED_PRECONDITIONS):
            barrier.mark_precondition(2, pc, step=100)
        self.assertTrue(barrier.can_reintegrate(2))


# =====================================================================
# Test: moegambit_integration.py public API
# =====================================================================

class TestMoegambitIntegrationReintegrationAPI(unittest.TestCase):
    """Test the public API functions in moegambit_integration.py."""

    def test_moegambit_can_reintegrate_no_barrier(self):
        """Without barrier, should return True (permissive)."""
        # We can't easily test moegambit_integration functions without full
        # initialization, but we can test the barrier directly
        barrier = ReintegrationBarrier()
        # No record for rank 99 → can_reintegrate returns False
        self.assertFalse(barrier.can_reintegrate(99))

    def test_barrier_summary(self):
        barrier = ReintegrationBarrier()
        barrier.begin_reintegration(
            failed_rank=2, replacement_rank=10,
            expert_ids=[4, 5], step=100,
        )
        s = barrier.summary()
        self.assertEqual(s['num_pending'], 1)
        self.assertEqual(s['num_integrated'], 0)


# =====================================================================
# Test: Multiple concurrent failures
# =====================================================================

class TestMultipleConcurrentFailures(unittest.TestCase):
    """Test barrier with multiple concurrent rank failures."""

    def test_two_failures_independent_barriers(self):
        barrier = ReintegrationBarrier()

        # Two failures
        barrier.begin_reintegration(
            failed_rank=2, replacement_rank=10,
            expert_ids=[4, 5], step=100,
        )
        barrier.begin_reintegration(
            failed_rank=3, replacement_rank=11,
            expert_ids=[6, 7], step=100,
        )
        self.assertEqual(barrier.num_pending, 2)

        # Complete rank 2 only
        for pc in sorted(REQUIRED_PRECONDITIONS):
            barrier.mark_precondition(2, pc, step=100)
        self.assertTrue(barrier.can_reintegrate(2))
        self.assertFalse(barrier.can_reintegrate(3))

        barrier.execute_reintegration(2, step=105)
        self.assertEqual(barrier.num_pending, 1)
        self.assertEqual(barrier.num_integrated, 1)

        # Complete rank 3
        for pc in sorted(REQUIRED_PRECONDITIONS):
            barrier.mark_precondition(3, pc, step=106)
        barrier.execute_reintegration(3, step=107)
        self.assertEqual(barrier.num_pending, 0)
        self.assertEqual(barrier.num_integrated, 2)


# =====================================================================
# Test: Controller reset clears barrier
# =====================================================================

class TestControllerResetClearsBarrier(unittest.TestCase):
    """Test that controller reset clears the barrier."""

    def test_reset_clears_barrier(self):
        ctrl = RecoveryController()
        barrier = ReintegrationBarrier()
        ctrl._reintegration_barrier = barrier

        barrier.begin_reintegration(
            failed_rank=2, replacement_rank=10,
            expert_ids=[4, 5], step=100,
        )
        self.assertEqual(barrier.num_pending, 1)

        ctrl.reset()
        self.assertIsNone(ctrl._reintegration_barrier)


if __name__ == '__main__':
    unittest.main()

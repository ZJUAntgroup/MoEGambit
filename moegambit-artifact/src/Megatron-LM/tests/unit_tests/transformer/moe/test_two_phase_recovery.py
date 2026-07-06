# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for Two-Phase Weights-First Recovery Protocol.

Tests cover:
1. TwoPhaseState transitions and validation
2. ExpertRecoverySession lifecycle and timing metrics
3. TwoPhaseRecoveryCoordinator full lifecycle
4. Update barrier semantics (forward/backward/optimizer step)
5. Skip-optimizer shortcut
6. Integration with DeferredOptimizerLoader state machine
7. Time-to-trainable metric (weights-first is faster)
8. Config flag behavior
"""

import importlib
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Bootstrap: mock torch and megatron.core to avoid GPU dependency
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_real_torch = None
try:
    import torch as _real_torch
except ImportError:
    pass

if _real_torch is None:
    _mock_torch = MagicMock()
    _mock_torch.Tensor = MagicMock
    _mock_torch.device = MagicMock
    sys.modules['torch'] = _mock_torch
    sys.modules['torch.distributed'] = MagicMock()
    sys.modules['torch.cuda'] = MagicMock()

if 'megatron.core' not in sys.modules:
    sys.modules['megatron'] = MagicMock()
    sys.modules['megatron.core'] = MagicMock()
    sys.modules['megatron.core.transformer'] = MagicMock()
    sys.modules['megatron.core.transformer.moe'] = MagicMock()

def _load_module(mod_name, file_path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__name__ = mod_name
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

_moe_dir = os.path.join(_REPO_ROOT, "megatron", "core", "transformer", "moe")

# Load modules under test
tp_mod = _load_module(
    "megatron.core.transformer.moe.two_phase_recovery",
    os.path.join(_moe_dir, "two_phase_recovery.py"),
)

ser_mod = _load_module(
    "megatron.core.transformer.moe.stale_expert_restore",
    os.path.join(_moe_dir, "stale_expert_restore.py"),
)

dol_mod = _load_module(
    "megatron.core.transformer.moe.deferred_optimizer_load",
    os.path.join(_moe_dir, "deferred_optimizer_load.py"),
)

ed_mod = _load_module(
    "megatron.core.transformer.moe.expert_directory",
    os.path.join(_moe_dir, "expert_directory.py"),
)

# Aliases
TwoPhaseState = tp_mod.TwoPhaseState
ExpertRecoverySession = tp_mod.ExpertRecoverySession
TwoPhaseRecoveryCoordinator = tp_mod.TwoPhaseRecoveryCoordinator

ExpertRestoreEntry = ser_mod.ExpertRestoreEntry
ExpertRestorePlan = ser_mod.ExpertRestorePlan
ExpertRestoreResult = ser_mod.ExpertRestoreResult
OptimizerUpdateBarrier = ser_mod.OptimizerUpdateBarrier
restore_expert_weights = ser_mod.restore_expert_weights

DeferredOptimizerLoader = dol_mod.DeferredOptimizerLoader
OptimizerLoadState = dol_mod.OptimizerLoadState


# =====================================================================
# Test: TwoPhaseState transitions
# =====================================================================

class TestTwoPhaseStateTransitions(unittest.TestCase):

    def test_valid_forward_transitions(self):
        """All forward transitions in the happy path are valid."""
        session = ExpertRecoverySession(layer_id=1, expert_id=2)
        self.assertEqual(session.state, TwoPhaseState.NOT_STARTED)

        session.transition_to(TwoPhaseState.WEIGHTS_LOADING)
        self.assertEqual(session.state, TwoPhaseState.WEIGHTS_LOADING)

        session.transition_to(TwoPhaseState.WEIGHTS_READY)
        self.assertEqual(session.state, TwoPhaseState.WEIGHTS_READY)

        session.transition_to(TwoPhaseState.OPTIMIZER_PENDING)
        self.assertEqual(session.state, TwoPhaseState.OPTIMIZER_PENDING)

        session.transition_to(TwoPhaseState.FULLY_RECOVERED)
        self.assertEqual(session.state, TwoPhaseState.FULLY_RECOVERED)

        session.transition_to(TwoPhaseState.COMPLETED)
        self.assertEqual(session.state, TwoPhaseState.COMPLETED)

    def test_invalid_transition_raises(self):
        """Invalid transitions raise ValueError."""
        session = ExpertRecoverySession(layer_id=1, expert_id=2)
        with self.assertRaises(ValueError):
            session.transition_to(TwoPhaseState.WEIGHTS_READY)  # skip LOADING

    def test_skip_optimizer_transition(self):
        """WEIGHTS_READY → FULLY_RECOVERED (skip optimizer) is valid."""
        session = ExpertRecoverySession(layer_id=1, expert_id=2)
        session.transition_to(TwoPhaseState.WEIGHTS_LOADING)
        session.transition_to(TwoPhaseState.WEIGHTS_READY)
        session.transition_to(TwoPhaseState.FULLY_RECOVERED)
        self.assertEqual(session.state, TwoPhaseState.FULLY_RECOVERED)

    def test_abort_from_weights_loading(self):
        """WEIGHTS_LOADING → NOT_STARTED (abort) is valid."""
        session = ExpertRecoverySession(layer_id=1, expert_id=2)
        session.transition_to(TwoPhaseState.WEIGHTS_LOADING)
        session.transition_to(TwoPhaseState.NOT_STARTED)
        self.assertEqual(session.state, TwoPhaseState.NOT_STARTED)

    def test_completed_is_terminal(self):
        """No transitions from COMPLETED."""
        session = ExpertRecoverySession(layer_id=1, expert_id=2)
        session.transition_to(TwoPhaseState.WEIGHTS_LOADING)
        session.transition_to(TwoPhaseState.WEIGHTS_READY)
        session.transition_to(TwoPhaseState.FULLY_RECOVERED)
        session.transition_to(TwoPhaseState.COMPLETED)
        with self.assertRaises(ValueError):
            session.transition_to(TwoPhaseState.NOT_STARTED)


# =====================================================================
# Test: Update barrier semantics
# =====================================================================

class TestUpdateBarrierSemantics(unittest.TestCase):

    def test_not_started_blocks_everything(self):
        session = ExpertRecoverySession(layer_id=1, expert_id=2)
        self.assertFalse(session.is_forward_allowed)
        self.assertFalse(session.is_optimizer_step_allowed)

    def test_weights_loading_blocks_everything(self):
        session = ExpertRecoverySession(layer_id=1, expert_id=2)
        session.transition_to(TwoPhaseState.WEIGHTS_LOADING)
        self.assertFalse(session.is_forward_allowed)
        self.assertFalse(session.is_optimizer_step_allowed)

    def test_weights_ready_allows_forward_blocks_optimizer(self):
        session = ExpertRecoverySession(layer_id=1, expert_id=2)
        session.transition_to(TwoPhaseState.WEIGHTS_LOADING)
        session.transition_to(TwoPhaseState.WEIGHTS_READY)
        self.assertTrue(session.is_forward_allowed)
        self.assertFalse(session.is_optimizer_step_allowed)

    def test_optimizer_pending_allows_forward_blocks_optimizer(self):
        session = ExpertRecoverySession(layer_id=1, expert_id=2)
        session.transition_to(TwoPhaseState.WEIGHTS_LOADING)
        session.transition_to(TwoPhaseState.WEIGHTS_READY)
        session.transition_to(TwoPhaseState.OPTIMIZER_PENDING)
        self.assertTrue(session.is_forward_allowed)
        self.assertFalse(session.is_optimizer_step_allowed)

    def test_fully_recovered_allows_everything(self):
        session = ExpertRecoverySession(layer_id=1, expert_id=2)
        session.transition_to(TwoPhaseState.WEIGHTS_LOADING)
        session.transition_to(TwoPhaseState.WEIGHTS_READY)
        session.transition_to(TwoPhaseState.FULLY_RECOVERED)
        self.assertTrue(session.is_forward_allowed)
        self.assertTrue(session.is_optimizer_step_allowed)

    def test_completed_allows_everything(self):
        session = ExpertRecoverySession(layer_id=1, expert_id=2)
        session.transition_to(TwoPhaseState.WEIGHTS_LOADING)
        session.transition_to(TwoPhaseState.WEIGHTS_READY)
        session.transition_to(TwoPhaseState.FULLY_RECOVERED)
        session.transition_to(TwoPhaseState.COMPLETED)
        self.assertTrue(session.is_forward_allowed)
        self.assertTrue(session.is_optimizer_step_allowed)


# =====================================================================
# Test: TwoPhaseRecoveryCoordinator full lifecycle
# =====================================================================

class TestTwoPhaseCoordinatorLifecycle(unittest.TestCase):

    def setUp(self):
        self.coord = TwoPhaseRecoveryCoordinator(defer_optimizer=True)

    def test_full_lifecycle(self):
        """Full happy path: NOT_STARTED → ... → COMPLETED."""
        expert_ids = [(1, 2), (1, 3)]

        # Phase 1: begin
        sessions = self.coord.begin_recovery(
            expert_ids=expert_ids,
            failed_rank=1, replacement_rank=10, step=100,
        )
        self.assertEqual(len(sessions), 2)
        self.assertEqual(self.coord.num_active, 2)
        for s in sessions:
            self.assertEqual(s.state, TwoPhaseState.WEIGHTS_LOADING)

        # Phase 1 complete: weights restored
        count = self.coord.on_weights_restored(expert_ids, step=105)
        self.assertEqual(count, 2)
        self.assertEqual(self.coord.num_weights_ready, 2)
        self.assertTrue(self.coord.all_weights_ready)

        # Phase 2: optimizer submitted
        count = self.coord.on_optimizer_submitted(expert_ids, step=105)
        self.assertEqual(count, 2)
        self.assertEqual(self.coord.num_optimizer_pending, 2)

        # Phase 2 complete: optimizer loaded
        count = self.coord.on_optimizer_loaded(expert_ids, step=110)
        self.assertEqual(count, 2)
        self.assertEqual(self.coord.num_fully_recovered, 2)

        # Promotion
        count = self.coord.on_promoted_healthy(expert_ids, step=111)
        self.assertEqual(count, 2)
        self.assertTrue(self.coord.all_completed)
        self.assertEqual(self.coord.num_active, 0)

    def test_skip_optimizer_phase(self):
        """WEIGHTS_READY → FULLY_RECOVERED via skip_optimizer_phase."""
        expert_ids = [(1, 2)]
        self.coord.begin_recovery(expert_ids=expert_ids, step=100)
        self.coord.on_weights_restored(expert_ids, step=105)

        count = self.coord.skip_optimizer_phase(expert_ids, step=105)
        self.assertEqual(count, 1)
        session = self.coord.get_session(1, 2)
        self.assertEqual(session.state, TwoPhaseState.FULLY_RECOVERED)
        self.assertTrue(session.is_optimizer_step_allowed)

    def test_query_api(self):
        """Query methods return correct results."""
        self.coord.begin_recovery(
            expert_ids=[(1, 2), (1, 3), (2, 2)], step=100,
        )
        self.coord.on_weights_restored([(1, 2)], step=105)

        self.assertEqual(self.coord.get_state(1, 2), TwoPhaseState.WEIGHTS_READY)
        self.assertEqual(self.coord.get_state(1, 3), TwoPhaseState.WEIGHTS_LOADING)
        self.assertEqual(self.coord.get_state(2, 2), TwoPhaseState.WEIGHTS_LOADING)
        self.assertEqual(self.coord.get_state(99, 99), TwoPhaseState.NOT_STARTED)

        self.assertTrue(self.coord.is_forward_allowed(1, 2))
        self.assertFalse(self.coord.is_forward_allowed(1, 3))
        self.assertTrue(self.coord.is_forward_allowed(99, 99))  # no session

        self.assertFalse(self.coord.is_optimizer_step_allowed(1, 2))
        self.assertTrue(self.coord.is_optimizer_step_allowed(99, 99))

        loading = self.coord.get_experts_in_state(TwoPhaseState.WEIGHTS_LOADING)
        self.assertEqual(sorted(loading), [(1, 3), (2, 2)])

    def test_summary(self):
        """summary() returns correct structure."""
        self.coord.begin_recovery(expert_ids=[(1, 2)], step=100)
        self.coord.on_weights_restored([(1, 2)], step=105)
        s = self.coord.summary()
        self.assertEqual(s['total_sessions'], 1)
        self.assertEqual(s['active'], 1)
        self.assertIn('WEIGHTS_READY', s['by_state'])
        self.assertTrue(s['defer_optimizer'])

    def test_reset(self):
        """reset() clears all sessions."""
        self.coord.begin_recovery(expert_ids=[(1, 2)], step=100)
        self.coord.reset()
        self.assertEqual(self.coord.num_active, 0)
        self.assertEqual(self.coord.get_state(1, 2), TwoPhaseState.NOT_STARTED)


# =====================================================================
# Test: Timing metrics
# =====================================================================

class TestTimingMetrics(unittest.TestCase):

    def test_time_to_trainable(self):
        """time_to_trainable measures Phase 1 duration only."""
        session = ExpertRecoverySession(layer_id=1, expert_id=2)
        session.recovery_start_time = 100.0
        session.transition_to(TwoPhaseState.WEIGHTS_LOADING)

        session.weights_ready_time = 100.5
        session.transition_to(TwoPhaseState.WEIGHTS_READY)

        self.assertAlmostEqual(session.time_to_trainable, 0.5)

    def test_optimizer_load_elapsed(self):
        """optimizer_load_elapsed measures Phase 2 duration."""
        session = ExpertRecoverySession(layer_id=1, expert_id=2)
        session.recovery_start_time = 100.0
        session.transition_to(TwoPhaseState.WEIGHTS_LOADING)
        session.weights_ready_time = 100.5
        session.transition_to(TwoPhaseState.WEIGHTS_READY)
        session.optimizer_submit_time = 100.5
        session.transition_to(TwoPhaseState.OPTIMIZER_PENDING)
        session.fully_recovered_time = 102.0
        session.transition_to(TwoPhaseState.FULLY_RECOVERED)

        self.assertAlmostEqual(session.optimizer_load_elapsed, 1.5)
        self.assertAlmostEqual(session.total_recovery_elapsed, 2.0)

    def test_weights_first_faster_than_sync(self):
        """Demonstrate that time-to-trainable < total recovery time.

        This is the key property: the expert re-joins training after
        Phase 1 (weights only), while Phase 2 (optimizer) runs in
        the background.
        """
        coord = TwoPhaseRecoveryCoordinator(defer_optimizer=True)
        expert_ids = [(1, 2), (1, 3)]

        # Simulate Phase 1: 0.5s
        coord.begin_recovery(expert_ids=expert_ids, step=100)
        for key in expert_ids:
            s = coord.get_session(*key)
            s.recovery_start_time = 100.0

        coord.on_weights_restored(expert_ids, step=105)
        for key in expert_ids:
            s = coord.get_session(*key)
            s.weights_ready_time = 100.5

        # Simulate Phase 2: 2.0s more
        coord.on_optimizer_submitted(expert_ids, step=105)
        for key in expert_ids:
            s = coord.get_session(*key)
            s.optimizer_submit_time = 100.5

        coord.on_optimizer_loaded(expert_ids, step=110)
        for key in expert_ids:
            s = coord.get_session(*key)
            s.fully_recovered_time = 102.5

        # Verify: time-to-trainable (0.5s) < total recovery (2.5s)
        for key in expert_ids:
            s = coord.get_session(*key)
            self.assertAlmostEqual(s.time_to_trainable, 0.5)
            self.assertAlmostEqual(s.total_recovery_elapsed, 2.5)
            self.assertLess(s.time_to_trainable, s.total_recovery_elapsed)

    def test_to_dict(self):
        """to_dict includes all timing fields."""
        session = ExpertRecoverySession(
            layer_id=1, expert_id=2,
            recovery_start_time=100.0,
            weights_ready_time=100.5,
            recovery_start_step=100,
            weights_ready_step=105,
        )
        session.transition_to(TwoPhaseState.WEIGHTS_LOADING)
        session.transition_to(TwoPhaseState.WEIGHTS_READY)
        d = session.to_dict()
        self.assertEqual(d['state'], 'WEIGHTS_READY')
        self.assertAlmostEqual(d['time_to_trainable'], 0.5)
        self.assertEqual(d['recovery_start_step'], 100)
        self.assertEqual(d['weights_ready_step'], 105)


# =====================================================================
# Test: Integration with DeferredOptimizerLoader
# =====================================================================

class TestTwoPhaseWithDeferredLoader(unittest.TestCase):

    def test_full_two_phase_with_deferred_loader(self):
        """End-to-end: two-phase coordinator + deferred optimizer loader."""
        coord = TwoPhaseRecoveryCoordinator(defer_optimizer=True)
        loader = DeferredOptimizerLoader()
        barrier = OptimizerUpdateBarrier()

        expert_ids = [(1, 2), (1, 3)]

        # Phase 1: begin + weights restored
        coord.begin_recovery(expert_ids=expert_ids, step=100)
        coord.on_weights_restored(expert_ids, step=105)

        # Verify: forward allowed, optimizer blocked
        for key in expert_ids:
            self.assertTrue(coord.is_forward_allowed(*key))
            self.assertFalse(coord.is_optimizer_step_allowed(*key))

        # Phase 2: submit optimizer loads
        plan = ExpertRestorePlan(failed_rank=1, replacement_rank=10, step=105)
        plan.entries.append(ExpertRestoreEntry(layer_id=1, expert_id=2))
        plan.entries.append(ExpertRestoreEntry(layer_id=1, expert_id=3))
        loader.submit_from_restore_plan(plan, step=105)
        coord.on_optimizer_submitted(expert_ids, step=105)

        # Execute loads (dry-run)
        barrier._blocked_expert_ids = {2, 3}
        barrier._blocked_params = {"p1", "p2"}
        num_exec, num_fin = loader.poll_and_finalize(
            step=110, load_fn=None,
            barrier=barrier, health_managers={},
        )
        self.assertEqual(num_exec, 2)
        self.assertEqual(num_fin, 2)

        # Drive two-phase: check finalized requests
        finalized_keys = []
        for key in expert_ids:
            req = loader.get_request(key[0], key[1])
            if req and req.state == OptimizerLoadState.FINALIZED:
                finalized_keys.append(key)
        coord.on_optimizer_loaded(finalized_keys, step=110)

        # Verify: optimizer step now allowed
        for key in expert_ids:
            self.assertTrue(coord.is_optimizer_step_allowed(*key))
            self.assertEqual(
                coord.get_state(*key), TwoPhaseState.FULLY_RECOVERED,
            )

    def test_skip_optimizer_with_deferred_loader(self):
        """When defer_optimizer=False, skip optimizer phase."""
        coord = TwoPhaseRecoveryCoordinator(defer_optimizer=False)
        expert_ids = [(1, 2)]

        coord.begin_recovery(expert_ids=expert_ids, step=100)
        coord.on_weights_restored(expert_ids, step=105)

        # Skip optimizer phase
        coord.skip_optimizer_phase(expert_ids, step=105)

        self.assertTrue(coord.is_optimizer_step_allowed(1, 2))
        self.assertEqual(coord.get_state(1, 2), TwoPhaseState.FULLY_RECOVERED)


# =====================================================================
# Test: Global singleton
# =====================================================================

class TestTwoPhaseGlobalSingleton(unittest.TestCase):

    def test_singleton(self):
        tp_mod.clear_two_phase_recovery_coordinator()
        c1 = tp_mod.get_two_phase_recovery_coordinator()
        c2 = tp_mod.get_two_phase_recovery_coordinator()
        self.assertIs(c1, c2)
        tp_mod.clear_two_phase_recovery_coordinator()

    def test_clear_resets(self):
        tp_mod.clear_two_phase_recovery_coordinator()
        coord = tp_mod.get_two_phase_recovery_coordinator()
        coord.begin_recovery(expert_ids=[(1, 2)], step=100)
        self.assertEqual(coord.num_active, 1)
        tp_mod.clear_two_phase_recovery_coordinator()
        coord2 = tp_mod.get_two_phase_recovery_coordinator()
        self.assertEqual(coord2.num_active, 0)


if __name__ == '__main__':
    unittest.main()

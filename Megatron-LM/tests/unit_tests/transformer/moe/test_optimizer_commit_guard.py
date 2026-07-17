# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for MOEGAMBIT-MoE Optimizer Commit Guard.

Tests cover:
1. OptimizerCommitGuard basic lifecycle
2. should_commit() blocking when iteration is invalid
3. Commit/skip tracking
4. Phase tracking
5. Partial commit recovery interface
6. Inverse compensation interface (future)
7. Integration with IterationInvalidator + HardFailureDetector
8. End-to-end: failure during forward → guard blocks optimizer → rollback
"""

import importlib
import importlib.util
import os
import sys
import types
import unittest

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if "torch" not in sys.modules:
    _torch_stub = types.ModuleType("torch")
    _torch_stub.Tensor = type("Tensor", (), {})
    _torch_distributed = types.ModuleType("torch.distributed")
    _torch_distributed.is_initialized = lambda: False
    _torch_distributed.get_rank = lambda: 0
    _torch_stub.distributed = _torch_distributed
    sys.modules["torch"] = _torch_stub
    sys.modules["torch.distributed"] = _torch_distributed


def _load_module(dotted_path: str):
    parts = dotted_path.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            pkg = types.ModuleType(parent)
            pkg.__path__ = []
            pkg.__package__ = parent
            sys.modules[parent] = pkg
    rel_path = os.path.join(_REPO_ROOT, *parts) + ".py"
    spec = importlib.util.spec_from_file_location(dotted_path, rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_path] = mod
    spec.loader.exec_module(mod)
    return mod


_ocg_mod = _load_module("megatron.core.transformer.moe.optimizer_commit_guard")
_inv_mod = _load_module("megatron.core.transformer.moe.iteration_invalidator")
_hfd_mod = _load_module("megatron.core.transformer.moe.hard_failure_detector")
_rc_mod = _load_module("megatron.core.transformer.moe.recovery_controller")
_rb_mod = _load_module("megatron.core.transformer.moe.iteration_rollback")

OptimizerCommitGuard = _ocg_mod.OptimizerCommitGuard
CommitPhase = _ocg_mod.CommitPhase
CommitRecord = _ocg_mod.CommitRecord
IterationInvalidator = _inv_mod.IterationInvalidator
HardFailureDetector = _hfd_mod.HardFailureDetector
FailureSource = _hfd_mod.FailureSource
RecoveryController = _rc_mod.RecoveryController
RecoveryPhase = _rc_mod.RecoveryPhase
RollbackReplayManager = _rb_mod.RollbackReplayManager


# =====================================================================
# Test: Basic lifecycle
# =====================================================================

class TestCommitGuardBasic(unittest.TestCase):

    def setUp(self):
        self.guard = OptimizerCommitGuard()

    def test_initial_state(self):
        self.assertFalse(self.guard.in_iteration)
        self.assertEqual(self.guard.phase, CommitPhase.NOT_STARTED)
        self.assertFalse(self.guard.is_blocked)
        self.assertEqual(self.guard.total_commits, 0)

    def test_iteration_lifecycle(self):
        self.guard.begin_iteration(100)
        self.assertTrue(self.guard.in_iteration)
        self.assertEqual(self.guard.current_step, 100)
        self.guard.end_iteration(100)
        self.assertFalse(self.guard.in_iteration)

    def test_should_commit_default_true(self):
        """Without invalidation callback, should_commit always True."""
        self.guard.begin_iteration(100)
        self.assertTrue(self.guard.should_commit())

    def test_mark_committed(self):
        self.guard.begin_iteration(100)
        self.guard.mark_committed(100)
        self.assertEqual(self.guard.phase, CommitPhase.COMMITTED)
        self.assertEqual(self.guard.total_commits, 1)

    def test_mark_skipped(self):
        self.guard.begin_iteration(100)
        self.guard.mark_skipped(100, reason="test")
        self.assertEqual(self.guard.total_skips, 1)

    def test_summary(self):
        self.guard.begin_iteration(50)
        s = self.guard.summary()
        self.assertEqual(s["current_step"], 50)
        self.assertEqual(s["phase"], "NOT_STARTED")
        self.assertFalse(s["is_blocked"])

    def test_reset(self):
        self.guard.begin_iteration(100)
        self.guard.mark_committed(100)
        self.guard.reset()
        self.assertEqual(self.guard.total_commits, 0)
        self.assertFalse(self.guard.in_iteration)


# =====================================================================
# Test: Blocking when iteration is invalid
# =====================================================================

class TestCommitGuardBlocking(unittest.TestCase):

    def setUp(self):
        self.guard = OptimizerCommitGuard()
        self.inv = IterationInvalidator()
        self.guard.register_callbacks(
            is_iteration_invalid_fn=self.inv.is_current_iteration_invalid,
        )

    def test_should_commit_true_when_valid(self):
        """should_commit returns True when iteration is valid."""
        self.inv.begin_iteration(100)
        self.guard.begin_iteration(100)
        self.assertTrue(self.guard.should_commit())
        self.assertFalse(self.guard.is_blocked)

    def test_should_commit_false_when_invalid(self):
        """should_commit returns False when iteration is invalidated."""
        self.inv.begin_iteration(100)
        self.guard.begin_iteration(100)
        # Invalidate the iteration
        self.inv.invalidate(step=100, failed_rank=3, reason="nccl error")
        self.assertFalse(self.guard.should_commit())
        self.assertTrue(self.guard.is_blocked)
        self.assertEqual(self.guard.total_blocks, 1)

    def test_blocking_is_idempotent(self):
        """Multiple should_commit calls when invalid increment block count."""
        self.inv.begin_iteration(100)
        self.guard.begin_iteration(100)
        self.inv.invalidate(step=100)
        self.assertFalse(self.guard.should_commit())
        self.assertFalse(self.guard.should_commit())
        self.assertEqual(self.guard.total_blocks, 2)

    def test_next_iteration_unblocked(self):
        """After invalidation, next iteration is not blocked."""
        # Iteration 100: invalidated
        self.inv.begin_iteration(100)
        self.guard.begin_iteration(100)
        self.inv.invalidate(step=100)
        self.assertFalse(self.guard.should_commit())
        self.inv.end_iteration(100)
        self.guard.end_iteration(100)

        # Iteration 101: clean
        self.inv.begin_iteration(101)
        self.guard.begin_iteration(101)
        self.assertTrue(self.guard.should_commit())
        self.assertFalse(self.guard.is_blocked)


# =====================================================================
# Test: Phase tracking
# =====================================================================

class TestPhaseTracking(unittest.TestCase):

    def test_phase_progression(self):
        guard = OptimizerCommitGuard()
        guard.begin_iteration(100)

        guard.mark_phase(CommitPhase.GRADS_PREPARED)
        self.assertEqual(guard.phase, CommitPhase.GRADS_PREPARED)

        guard.mark_phase(CommitPhase.GRADS_CLIPPED)
        self.assertEqual(guard.phase, CommitPhase.GRADS_CLIPPED)

        guard.mark_phase(CommitPhase.MAIN_PARAMS_UPDATED)
        self.assertEqual(guard.phase, CommitPhase.MAIN_PARAMS_UPDATED)

        guard.mark_phase(CommitPhase.MODEL_PARAMS_WRITTEN)
        self.assertEqual(guard.phase, CommitPhase.MODEL_PARAMS_WRITTEN)

        guard.mark_committed(100)
        self.assertEqual(guard.phase, CommitPhase.COMMITTED)


# =====================================================================
# Test: Partial commit recovery
# =====================================================================

class TestPartialCommitRecovery(unittest.TestCase):

    def test_no_recovery_before_main_params_updated(self):
        """No recovery needed if failure before MAIN_PARAMS_UPDATED."""
        guard = OptimizerCommitGuard()
        guard.begin_iteration(100)
        guard.mark_phase(CommitPhase.GRADS_PREPARED)
        result = guard.recover_from_partial_commit(100)
        self.assertFalse(result)

    def test_recovery_after_main_params_updated(self):
        """Recovery triggered if failure after MAIN_PARAMS_UPDATED."""
        recover_calls = []

        def recover_fn(step, phase):
            recover_calls.append((step, phase))

        guard = OptimizerCommitGuard()
        guard.register_callbacks(recover_fn=recover_fn)
        guard.begin_iteration(100)
        guard.mark_phase(CommitPhase.MAIN_PARAMS_UPDATED)
        result = guard.recover_from_partial_commit(100)
        self.assertTrue(result)
        self.assertEqual(len(recover_calls), 1)
        self.assertEqual(recover_calls[0][1], CommitPhase.MAIN_PARAMS_UPDATED)
        self.assertEqual(guard.total_recoveries, 1)

    def test_no_recovery_after_committed(self):
        """No recovery needed if already committed."""
        guard = OptimizerCommitGuard()
        guard.begin_iteration(100)
        guard.mark_committed(100)
        result = guard.recover_from_partial_commit(100)
        self.assertFalse(result)


# =====================================================================
# Test: Inverse compensation (future interface)
# =====================================================================

class TestInverseCompensation(unittest.TestCase):

    def test_inverse_compensation_returns_false(self):
        """v1: inverse compensation always returns False."""
        guard = OptimizerCommitGuard()
        guard.begin_iteration(100)
        result = guard.request_inverse_compensation(
            step=100,
            rollback_optimizer_state=True,
            rollback_main_params=True,
        )
        self.assertFalse(result)


# =====================================================================
# Test: CommitRecord
# =====================================================================

class TestCommitRecord(unittest.TestCase):

    def test_defaults(self):
        r = CommitRecord()
        self.assertEqual(r.step, -1)
        self.assertFalse(r.committed)
        self.assertFalse(r.blocked)

    def test_to_dict(self):
        r = CommitRecord(step=100, committed=True,
                         phase_reached=CommitPhase.COMMITTED)
        d = r.to_dict()
        self.assertEqual(d["step"], 100)
        self.assertTrue(d["committed"])
        self.assertEqual(d["phase_reached"], "COMMITTED")


# =====================================================================
# Test: Integration — full failure → guard blocks → rollback
# =====================================================================

class TestFullIntegration(unittest.TestCase):
    """End-to-end: Detector → Invalidator → Guard blocks optimizer → Rollback."""

    def setUp(self):
        self.ctrl = RecoveryController()
        self.inv = IterationInvalidator()
        self.detector = HardFailureDetector()
        self.guard = OptimizerCommitGuard()
        self.mgr = RollbackReplayManager()

        # Wire
        self.detector.register_callbacks(
            on_hard_failure_fn=self.ctrl.on_hard_rank_failure,
            on_iteration_invalid_fn=self.inv.invalidate,
        )
        self.ctrl.register_callbacks(
            quarantine_fn=lambda **kw: None,
            health_mark_unavailable_fn=lambda **kw: None,
        )
        self.guard.register_callbacks(
            is_iteration_invalid_fn=self.inv.is_current_iteration_invalid,
        )

    def test_forward_failure_blocks_optimizer(self):
        """
        Scenario:
        1. Iteration 100 starts
        2. Forward fails (NCCL error)
        3. Guard blocks optimizer.step()
        4. Rollback restores state
        5. Replay succeeds, optimizer commits
        """

        class MockArgs:
            consumed_train_samples = 5000
            consumed_valid_samples = 0

        args = MockArgs()
        iteration = 100

        # --- Iteration 100 starts ---
        self.inv.begin_iteration(iteration)
        self.ctrl.before_iteration(step=iteration)
        self.guard.begin_iteration(iteration)
        self.mgr.take_snapshot(
            iteration=iteration,
            consumed_train_samples=args.consumed_train_samples,
        )

        # --- Forward FAILS ---
        self.detector.report_collective_failure(
            failed_rank=3, step=iteration, mid_iteration=True,
            exception=RuntimeError("NCCL error"),
        )

        # --- Guard blocks optimizer ---
        self.assertFalse(self.guard.should_commit())
        self.assertTrue(self.guard.is_blocked)
        self.guard.mark_skipped(iteration, reason="iteration_invalidated")

        # --- Verify: no optimizer commit happened ---
        self.assertEqual(self.guard.total_commits, 0)
        self.assertEqual(self.guard.total_skips, 1)
        self.assertEqual(self.guard.total_blocks, 1)

        # --- Rollback ---
        self.mgr.rollback(args)
        self.assertEqual(args.consumed_train_samples, 5000)

        # --- End invalidated iteration ---
        self.inv.end_iteration(iteration)
        self.guard.end_iteration(iteration)

        # --- Replay iteration 100 ---
        self.inv.begin_iteration(iteration)
        self.ctrl.before_iteration(step=iteration)
        self.guard.begin_iteration(iteration)

        # No failure this time
        self.assertTrue(self.guard.should_commit())
        self.assertFalse(self.guard.is_blocked)

        # Optimizer commits
        self.guard.mark_committed(iteration)
        self.assertEqual(self.guard.total_commits, 1)

        self.mgr.complete_replay()
        self.inv.end_iteration(iteration)
        self.guard.end_iteration(iteration)

    def test_no_half_updated_params(self):
        """
        Verify: when iteration is invalidated, optimizer.step() is
        never called, so params remain at previous iteration's state.

        This simulates the training loop logic:
        - forward_backward_func() raises NCCL error
        - guard.should_commit() returns False
        - optimizer.step() is NOT called
        - params are unchanged
        """
        iteration = 200
        self.inv.begin_iteration(iteration)
        self.guard.begin_iteration(iteration)

        # Simulate: forward fails, invalidation triggered
        self.inv.invalidate(step=iteration, failed_rank=5)

        # Guard blocks
        self.assertFalse(self.guard.should_commit())

        # Verify phase is still NOT_STARTED (optimizer never entered)
        self.assertEqual(self.guard.phase, CommitPhase.NOT_STARTED)

        # No commit, no partial update
        self.assertEqual(self.guard.total_commits, 0)

    def test_rollback_params_consistent_with_previous_iteration(self):
        """
        Simulate 3 iterations:
        - Iter 1: success (commit)
        - Iter 2: failure mid-forward (guard blocks, rollback)
        - Iter 2 replay: success (commit)

        Verify consumed_samples is correct throughout.
        """

        class MockArgs:
            consumed_train_samples = 0
            consumed_valid_samples = 0

        args = MockArgs()
        batch_size = 128

        # --- Iteration 1: success ---
        self.inv.begin_iteration(1)
        self.guard.begin_iteration(1)
        self.mgr.take_snapshot(iteration=1,
                               consumed_train_samples=args.consumed_train_samples)
        # forward/backward succeed
        self.assertTrue(self.guard.should_commit())
        self.guard.mark_committed(1)
        args.consumed_train_samples += batch_size
        self.inv.end_iteration(1)
        self.guard.end_iteration(1)
        self.assertEqual(args.consumed_train_samples, 128)

        # --- Iteration 2: failure ---
        self.inv.begin_iteration(2)
        self.guard.begin_iteration(2)
        self.mgr.take_snapshot(iteration=2,
                               consumed_train_samples=args.consumed_train_samples)
        # forward fails
        self.inv.invalidate(step=2, failed_rank=3)
        self.assertFalse(self.guard.should_commit())
        self.guard.mark_skipped(2)
        # rollback
        self.mgr.rollback(args)
        self.assertEqual(args.consumed_train_samples, 128)  # restored
        self.inv.end_iteration(2)
        self.guard.end_iteration(2)

        # --- Iteration 2 replay: success ---
        self.inv.begin_iteration(2)
        self.guard.begin_iteration(2)
        # forward/backward succeed
        self.assertTrue(self.guard.should_commit())
        self.guard.mark_committed(2)
        args.consumed_train_samples += batch_size
        self.mgr.complete_replay()
        self.inv.end_iteration(2)
        self.guard.end_iteration(2)
        self.assertEqual(args.consumed_train_samples, 256)

        # Totals
        self.assertEqual(self.guard.total_commits, 2)
        self.assertEqual(self.guard.total_skips, 1)
        self.assertEqual(self.guard.total_blocks, 1)


# =====================================================================
# Test: Singleton management
# =====================================================================

class TestSingletons(unittest.TestCase):

    def test_guard_singleton(self):
        _ocg_mod.clear_optimizer_commit_guard()
        g1 = _ocg_mod.get_optimizer_commit_guard()
        g2 = _ocg_mod.get_optimizer_commit_guard()
        self.assertIs(g1, g2)
        _ocg_mod.clear_optimizer_commit_guard()
        g3 = _ocg_mod.get_optimizer_commit_guard()
        self.assertIsNot(g1, g3)


if __name__ == "__main__":
    unittest.main()

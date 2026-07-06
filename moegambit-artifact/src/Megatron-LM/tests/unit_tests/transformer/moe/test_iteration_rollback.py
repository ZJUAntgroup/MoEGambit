# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for MoEGambit Iteration Rollback & Replay Manager.

Tests cover:
1. IterationSnapshot creation and fields
2. RollbackReplayManager snapshot/rollback/replay lifecycle
3. Data iterator rewind/advance integration
4. Forward-failure rollback + replay
5. Backward-failure rollback + replay
6. consumed_samples and iteration consistency
7. Max replay attempt limiting
8. Integration with HardFailureDetector + IterationInvalidator
"""

import importlib
import importlib.util
import os
import sys
import types
import unittest

# ---------------------------------------------------------------------------
# Bootstrap: load target modules without triggering torch imports
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Stub out torch
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
    """Load a module by dotted path, creating parent packages as needed."""
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


# Load modules under test
_rb_mod = _load_module("megatron.core.transformer.moe.iteration_rollback")
_hfd_mod = _load_module("megatron.core.transformer.moe.hard_failure_detector")
_inv_mod = _load_module("megatron.core.transformer.moe.iteration_invalidator")
_rc_mod = _load_module("megatron.core.transformer.moe.recovery_controller")

RollbackReplayManager = _rb_mod.RollbackReplayManager
IterationSnapshot = _rb_mod.IterationSnapshot
HardFailureDetector = _hfd_mod.HardFailureDetector
FailureSource = _hfd_mod.FailureSource
IterationInvalidator = _inv_mod.IterationInvalidator
RecoveryController = _rc_mod.RecoveryController
RecoveryPhase = _rc_mod.RecoveryPhase


# =====================================================================
# Mock: RerunDataIterator-like object
# =====================================================================

class MockDataIterator:
    """Simulates RerunDataIterator with rewind/advance."""

    def __init__(self, data):
        self._data = list(data)
        self._pos = 0
        self._saved = []
        self._replaying = False
        self._replay_pos = 0
        self.rewind_count = 0
        self.advance_count = 0

    def __next__(self):
        if self._replaying:
            if self._replay_pos >= len(self._saved):
                raise StopIteration
            val = self._saved[self._replay_pos]
            self._replay_pos += 1
            return val
        if self._pos >= len(self._data):
            raise StopIteration
        val = self._data[self._pos]
        self._pos += 1
        self._saved.append(val)
        return val

    def rewind(self):
        """Replay saved microbatches from the beginning."""
        self._replaying = True
        self._replay_pos = 0
        self.rewind_count += 1

    def advance(self):
        """Drop saved microbatches and move forward."""
        self._replaying = False
        self._saved = []
        self.advance_count += 1

    @property
    def saved_count(self):
        return len(self._saved)


# =====================================================================
# Mock: args-like namespace
# =====================================================================

class MockArgs:
    def __init__(self):
        self.consumed_train_samples = 0
        self.consumed_valid_samples = 0
        self.iteration = 0


# =====================================================================
# Test: IterationSnapshot
# =====================================================================

class TestIterationSnapshot(unittest.TestCase):

    def test_defaults(self):
        snap = IterationSnapshot()
        self.assertEqual(snap.iteration, -1)
        self.assertEqual(snap.consumed_train_samples, 0)
        self.assertFalse(snap.valid)

    def test_fields(self):
        snap = IterationSnapshot(
            iteration=100,
            consumed_train_samples=5000,
            consumed_valid_samples=200,
            num_floating_point_operations_so_far=1000000,
            valid=True,
        )
        self.assertEqual(snap.iteration, 100)
        self.assertEqual(snap.consumed_train_samples, 5000)
        self.assertTrue(snap.valid)

    def test_to_dict(self):
        snap = IterationSnapshot(iteration=42, valid=True)
        d = snap.to_dict()
        self.assertEqual(d["iteration"], 42)
        self.assertTrue(d["valid"])


# =====================================================================
# Test: RollbackReplayManager — basic lifecycle
# =====================================================================

class TestRollbackReplayManagerBasic(unittest.TestCase):

    def setUp(self):
        self.mgr = RollbackReplayManager()
        self.args = MockArgs()

    def test_initial_state(self):
        self.assertFalse(self.mgr.has_valid_snapshot)
        self.assertFalse(self.mgr.is_replay_pending())
        self.assertEqual(self.mgr.total_rollbacks, 0)

    def test_snapshot(self):
        self.mgr.take_snapshot(
            iteration=100,
            consumed_train_samples=5000,
            consumed_valid_samples=200,
            num_floating_point_operations_so_far=1000000,
        )
        self.assertTrue(self.mgr.has_valid_snapshot)
        snap = self.mgr.snapshot
        self.assertEqual(snap.iteration, 100)
        self.assertEqual(snap.consumed_train_samples, 5000)

    def test_rollback_no_snapshot(self):
        """Rollback without snapshot returns False."""
        result = self.mgr.rollback(self.args)
        self.assertFalse(result)

    def test_rollback_restores_consumed_samples(self):
        """Rollback restores consumed_train_samples."""
        self.mgr.take_snapshot(iteration=100, consumed_train_samples=5000)
        self.args.consumed_train_samples = 5128  # advanced during failed iter
        result = self.mgr.rollback(self.args)
        self.assertTrue(result)
        self.assertEqual(self.args.consumed_train_samples, 5000)

    def test_rollback_restores_fp_ops(self):
        """Rollback restores num_floating_point_operations_so_far."""
        self.mgr.take_snapshot(
            iteration=100,
            consumed_train_samples=5000,
            num_floating_point_operations_so_far=1000000,
        )
        fp_ops_ref = [1050000]  # advanced during failed iter
        self.mgr.rollback(self.args, num_fp_ops_ref=fp_ops_ref)
        self.assertEqual(fp_ops_ref[0], 1000000)

    def test_rollback_sets_replay_pending(self):
        """Rollback sets replay_pending flag."""
        self.mgr.take_snapshot(iteration=100, consumed_train_samples=5000)
        self.mgr.rollback(self.args)
        self.assertTrue(self.mgr.is_replay_pending())

    def test_complete_replay_clears_flag(self):
        """complete_replay clears replay_pending."""
        self.mgr.take_snapshot(iteration=100, consumed_train_samples=5000)
        self.mgr.rollback(self.args)
        self.assertTrue(self.mgr.is_replay_pending())
        self.mgr.complete_replay()
        self.assertFalse(self.mgr.is_replay_pending())

    def test_advance_normal_path(self):
        """advance on normal path resets replay counter."""
        self.mgr.take_snapshot(iteration=100, consumed_train_samples=5000)
        self.mgr.advance()
        self.assertFalse(self.mgr.is_replay_pending())

    def test_summary(self):
        self.mgr.take_snapshot(iteration=50, consumed_train_samples=2500)
        s = self.mgr.summary()
        self.assertTrue(s["has_valid_snapshot"])
        self.assertEqual(s["snapshot_iteration"], 50)
        self.assertFalse(s["replay_pending"])

    def test_reset(self):
        self.mgr.take_snapshot(iteration=100, consumed_train_samples=5000)
        self.mgr.rollback(self.args)
        self.mgr.reset()
        self.assertFalse(self.mgr.has_valid_snapshot)
        self.assertFalse(self.mgr.is_replay_pending())
        self.assertEqual(self.mgr.total_rollbacks, 0)


# =====================================================================
# Test: Data iterator rewind/advance
# =====================================================================

class TestDataIteratorIntegration(unittest.TestCase):

    def setUp(self):
        self.mgr = RollbackReplayManager()
        self.args = MockArgs()

    def test_rollback_rewinds_data_iterator(self):
        """Rollback calls rewind() on data iterator."""
        di = MockDataIterator([10, 20, 30, 40])
        # Consume 2 items (simulating 2 microbatches in forward)
        next(di)
        next(di)
        self.assertEqual(di.saved_count, 2)

        self.mgr.take_snapshot(iteration=100, consumed_train_samples=5000)
        self.mgr.rollback(self.args, data_iterators=di)

        self.assertEqual(di.rewind_count, 1)
        # After rewind, replaying the same data
        self.assertEqual(next(di), 10)
        self.assertEqual(next(di), 20)

    def test_complete_replay_advances_data_iterator(self):
        """complete_replay calls advance() on data iterator."""
        di = MockDataIterator([10, 20, 30, 40])
        next(di)
        next(di)

        self.mgr.take_snapshot(iteration=100, consumed_train_samples=5000)
        self.mgr.rollback(self.args, data_iterators=di)
        self.mgr.complete_replay(data_iterators=di)

        self.assertEqual(di.advance_count, 1)

    def test_advance_normal_advances_data_iterator(self):
        """advance() on normal path calls advance() on data iterator."""
        di = MockDataIterator([10, 20, 30])
        next(di)

        self.mgr.advance(data_iterators=di)
        self.assertEqual(di.advance_count, 1)

    def test_rollback_with_list_of_iterators(self):
        """Rollback works with a list of data iterators (virtual PP)."""
        di1 = MockDataIterator([1, 2])
        di2 = MockDataIterator([3, 4])
        next(di1)
        next(di2)

        self.mgr.take_snapshot(iteration=100, consumed_train_samples=5000)
        self.mgr.rollback(self.args, data_iterators=[di1, di2])

        self.assertEqual(di1.rewind_count, 1)
        self.assertEqual(di2.rewind_count, 1)


# =====================================================================
# Test: Forward failure → rollback + replay
# =====================================================================

class TestForwardFailureRollbackReplay(unittest.TestCase):
    """Simulate a failure during forward pass, rollback, and replay."""

    def test_forward_failure_rollback_replay(self):
        """
        Scenario:
        1. Iteration 100 starts, snapshot taken
        2. Forward pass consumes 4 microbatches, then fails
        3. Rollback restores consumed_samples, rewinds data
        4. Replay re-executes with same data
        5. Replay succeeds, iteration completes
        """
        mgr = RollbackReplayManager()
        args = MockArgs()
        args.consumed_train_samples = 5000
        di = MockDataIterator(list(range(100)))

        # --- Iteration 100 starts ---
        iteration = 100
        mgr.take_snapshot(
            iteration=iteration,
            consumed_train_samples=args.consumed_train_samples,
        )

        # Simulate forward: consume 4 microbatches
        batch1 = [next(di) for _ in range(4)]
        self.assertEqual(batch1, [0, 1, 2, 3])
        # Simulate partial consumed_samples update (shouldn't happen
        # before optimizer.step, but let's be safe)
        args.consumed_train_samples = 5128  # hypothetical

        # --- Forward FAILS ---
        # Rollback
        result = mgr.rollback(args, data_iterators=di)
        self.assertTrue(result)
        self.assertTrue(mgr.is_replay_pending())

        # Verify state restored
        self.assertEqual(args.consumed_train_samples, 5000)

        # --- Replay iteration 100 ---
        # Data iterator replays the same 4 microbatches
        replay_batch = [next(di) for _ in range(4)]
        self.assertEqual(replay_batch, [0, 1, 2, 3])

        # Replay succeeds
        mgr.complete_replay(data_iterators=di)
        self.assertFalse(mgr.is_replay_pending())

        # Now advance consumed_samples normally
        args.consumed_train_samples = 5128
        iteration += 1
        self.assertEqual(iteration, 101)
        self.assertEqual(args.consumed_train_samples, 5128)


# =====================================================================
# Test: Backward failure → rollback + replay
# =====================================================================

class TestBackwardFailureRollbackReplay(unittest.TestCase):
    """Simulate a failure during backward pass, rollback, and replay."""

    def test_backward_failure_rollback_replay(self):
        """
        Scenario:
        1. Iteration 200 starts, snapshot taken
        2. Forward pass completes (all microbatches consumed)
        3. Backward pass fails mid-way
        4. Rollback restores state, rewinds data
        5. Replay succeeds
        """
        mgr = RollbackReplayManager()
        args = MockArgs()
        args.consumed_train_samples = 10000
        num_fp_ops = [500000]
        di = MockDataIterator(list(range(100)))

        # --- Iteration 200 starts ---
        iteration = 200
        mgr.take_snapshot(
            iteration=iteration,
            consumed_train_samples=args.consumed_train_samples,
            num_floating_point_operations_so_far=num_fp_ops[0],
        )

        # Forward: consume 4 microbatches
        batch = [next(di) for _ in range(4)]
        self.assertEqual(batch, [0, 1, 2, 3])

        # Backward: partially completes, then fails
        # (no state change to args at this point — optimizer hasn't stepped)

        # --- Backward FAILS ---
        result = mgr.rollback(args, data_iterators=di, num_fp_ops_ref=num_fp_ops)
        self.assertTrue(result)
        self.assertEqual(args.consumed_train_samples, 10000)
        self.assertEqual(num_fp_ops[0], 500000)

        # --- Replay ---
        replay_batch = [next(di) for _ in range(4)]
        self.assertEqual(replay_batch, [0, 1, 2, 3])

        mgr.complete_replay(data_iterators=di)
        self.assertFalse(mgr.is_replay_pending())
        self.assertEqual(mgr.total_replays_completed, 1)


# =====================================================================
# Test: Max replay attempts
# =====================================================================

class TestMaxReplayAttempts(unittest.TestCase):

    def test_max_replay_exceeded(self):
        """After max_replay_attempts, rollback returns False."""
        mgr = RollbackReplayManager()
        mgr.max_replay_attempts = 2
        args = MockArgs()
        args.consumed_train_samples = 5000

        mgr.take_snapshot(iteration=100, consumed_train_samples=5000)

        # First rollback: attempt 1
        self.assertTrue(mgr.rollback(args))
        self.assertFalse(mgr.exceeded_max_replays)

        # Second rollback: attempt 2
        self.assertTrue(mgr.rollback(args))
        self.assertFalse(mgr.exceeded_max_replays)

        # Third rollback: attempt 3 > max 2
        self.assertFalse(mgr.rollback(args))
        self.assertTrue(mgr.exceeded_max_replays)

    def test_successful_replay_resets_counter(self):
        """Successful replay resets the replay counter."""
        mgr = RollbackReplayManager()
        mgr.max_replay_attempts = 2
        args = MockArgs()
        args.consumed_train_samples = 5000

        mgr.take_snapshot(iteration=100, consumed_train_samples=5000)
        mgr.rollback(args)
        mgr.complete_replay()

        # Counter should be reset
        self.assertEqual(mgr.replay_count, 0)

        # Can rollback again for a new failure
        mgr.take_snapshot(iteration=101, consumed_train_samples=5128)
        self.assertTrue(mgr.rollback(args))


# =====================================================================
# Test: Iteration ID and consumed_samples consistency
# =====================================================================

class TestConsistency(unittest.TestCase):
    """Verify iteration ID and consumed_samples stay consistent."""

    def test_multi_iteration_with_one_failure(self):
        """
        Run 5 iterations, fail on iteration 3, replay it, continue.
        Verify iteration IDs and consumed_samples are correct.
        """
        mgr = RollbackReplayManager()
        args = MockArgs()
        args.consumed_train_samples = 0
        batch_size = 128
        di = MockDataIterator(list(range(1000)))

        iterations_completed = []

        for iteration in range(1, 6):
            # Snapshot
            mgr.take_snapshot(
                iteration=iteration,
                consumed_train_samples=args.consumed_train_samples,
            )

            # Consume data
            next(di)

            # Simulate failure on iteration 3
            if iteration == 3 and not mgr.is_replay_pending():
                mgr.rollback(args, data_iterators=di)
                # Re-consume data (replay)
                next(di)
                mgr.complete_replay(data_iterators=di)

            # Normal completion
            args.consumed_train_samples += batch_size
            iterations_completed.append(iteration)

            # Advance
            mgr.advance(data_iterators=di)

        self.assertEqual(iterations_completed, [1, 2, 3, 4, 5])
        self.assertEqual(args.consumed_train_samples, 5 * batch_size)

    def test_consumed_samples_not_double_counted(self):
        """
        After rollback, consumed_samples is restored to pre-iteration
        value, so the replay doesn't double-count.
        """
        mgr = RollbackReplayManager()
        args = MockArgs()
        args.consumed_train_samples = 1000
        batch_size = 128

        # Snapshot at iteration start
        mgr.take_snapshot(iteration=10, consumed_train_samples=1000)

        # Simulate: consumed_samples was NOT yet incremented (it happens
        # after train_step in the training loop), but let's test the
        # rollback restores correctly even if it was
        args.consumed_train_samples = 1128  # hypothetical mid-step update

        # Rollback
        mgr.rollback(args)
        self.assertEqual(args.consumed_train_samples, 1000)

        # Replay succeeds, now increment normally
        args.consumed_train_samples += batch_size
        self.assertEqual(args.consumed_train_samples, 1128)
        mgr.complete_replay()


# =====================================================================
# Test: Cleanup callback
# =====================================================================

class TestCleanupCallback(unittest.TestCase):

    def test_cleanup_fn_called_on_rollback(self):
        """cleanup_fn is called during rollback."""
        cleanup_calls = []

        def cleanup_fn(snapshot):
            cleanup_calls.append(snapshot.iteration)

        mgr = RollbackReplayManager()
        mgr.register_callbacks(cleanup_fn=cleanup_fn)
        args = MockArgs()

        mgr.take_snapshot(iteration=100, consumed_train_samples=5000)
        mgr.rollback(args)

        self.assertEqual(cleanup_calls, [100])

    def test_pipeline_rollback_fn_called(self):
        """pipeline_rollback_fn (PP>1 placeholder) is called."""
        pp_calls = []

        def pp_fn(snapshot):
            pp_calls.append(snapshot.iteration)

        mgr = RollbackReplayManager()
        mgr.register_callbacks(pipeline_rollback_fn=pp_fn)
        args = MockArgs()

        mgr.take_snapshot(iteration=200, consumed_train_samples=10000)
        mgr.rollback(args)

        self.assertEqual(pp_calls, [200])


# =====================================================================
# Test: Integration with Detector + Invalidator + Controller
# =====================================================================

class TestFullIntegration(unittest.TestCase):
    """End-to-end: Detector → Invalidator → Controller → Rollback."""

    def setUp(self):
        self.ctrl = RecoveryController()
        self.inv = IterationInvalidator()
        self.detector = HardFailureDetector()
        self.mgr = RollbackReplayManager()

        # Wire detector → controller + invalidator
        self.detector.register_callbacks(
            on_hard_failure_fn=self.ctrl.on_hard_rank_failure,
            on_iteration_invalid_fn=self.inv.invalidate,
        )
        self.ctrl.register_callbacks(
            quarantine_fn=lambda **kw: None,
            health_mark_unavailable_fn=lambda **kw: None,
        )

    def test_full_flow_forward_failure(self):
        """
        Full flow:
        1. Iteration 100 starts
        2. Snapshot taken
        3. Forward pass fails (NCCL error)
        4. Detector reports failure → invalidator marks invalid
        5. Rollback restores state
        6. Replay succeeds
        """
        args = MockArgs()
        args.consumed_train_samples = 5000
        di = MockDataIterator(list(range(100)))
        iteration = 100

        # 1. Begin iteration
        self.inv.begin_iteration(iteration)
        self.ctrl.before_iteration(step=iteration)

        # 2. Snapshot
        self.mgr.take_snapshot(
            iteration=iteration,
            consumed_train_samples=args.consumed_train_samples,
        )

        # 3. Forward: consume data
        batch = [next(di) for _ in range(4)]
        self.assertEqual(batch, [0, 1, 2, 3])

        # 4. Forward FAILS — NCCL error
        exc = RuntimeError("NCCL SystemError")
        self.detector.report_collective_failure(
            failed_rank=3, step=iteration, mid_iteration=True,
            exception=exc, expert_ids=[6, 7],
        )

        # 5. Check invalidation
        self.assertTrue(self.inv.is_current_iteration_invalid())
        self.assertTrue(self.ctrl.iteration_was_invalidated)
        self.assertEqual(self.ctrl.phase, RecoveryPhase.PENDING_GROUP_REPAIR)

        # 6. Rollback
        result = self.mgr.rollback(args, data_iterators=di)
        self.assertTrue(result)
        self.assertEqual(args.consumed_train_samples, 5000)
        self.assertTrue(self.mgr.is_replay_pending())

        # 7. End invalidated iteration
        self.inv.end_iteration(iteration)

        # 8. Next loop: begin new iteration (same step — replay)
        self.inv.begin_iteration(iteration)
        self.ctrl.before_iteration(step=iteration)
        self.assertFalse(self.inv.is_current_iteration_invalid())

        # 9. Replay: consume same data
        replay_batch = [next(di) for _ in range(4)]
        self.assertEqual(replay_batch, [0, 1, 2, 3])

        # 10. Replay succeeds
        self.mgr.complete_replay(data_iterators=di)
        self.assertFalse(self.mgr.is_replay_pending())
        self.assertEqual(self.mgr.total_replays_completed, 1)

    def test_full_flow_backward_failure(self):
        """
        Full flow with backward failure:
        1. Forward completes
        2. Backward fails
        3. Rollback + replay
        """
        args = MockArgs()
        args.consumed_train_samples = 10000
        di = MockDataIterator(list(range(100)))
        iteration = 200

        # Begin
        self.inv.begin_iteration(iteration)
        self.ctrl.before_iteration(step=iteration)
        self.mgr.take_snapshot(
            iteration=iteration,
            consumed_train_samples=args.consumed_train_samples,
        )

        # Forward: all microbatches consumed
        batch = [next(di) for _ in range(4)]

        # Backward FAILS
        self.detector.report_collective_failure(
            failed_rank=5, step=iteration, mid_iteration=True,
            exception=RuntimeError("NCCL timeout"),
        )

        # Invalidated
        self.assertTrue(self.inv.is_current_iteration_invalid())

        # Rollback
        self.mgr.rollback(args, data_iterators=di)
        self.assertEqual(args.consumed_train_samples, 10000)

        # End + replay
        self.inv.end_iteration(iteration)
        self.inv.begin_iteration(iteration)

        replay_batch = [next(di) for _ in range(4)]
        self.assertEqual(replay_batch, [0, 1, 2, 3])

        self.mgr.complete_replay(data_iterators=di)
        self.assertFalse(self.mgr.is_replay_pending())


# =====================================================================
# Test: Singleton management
# =====================================================================

class TestSingletons(unittest.TestCase):

    def test_manager_singleton(self):
        _rb_mod.clear_rollback_replay_manager()
        m1 = _rb_mod.get_rollback_replay_manager()
        m2 = _rb_mod.get_rollback_replay_manager()
        self.assertIs(m1, m2)
        _rb_mod.clear_rollback_replay_manager()
        m3 = _rb_mod.get_rollback_replay_manager()
        self.assertIsNot(m1, m3)


if __name__ == "__main__":
    unittest.main()

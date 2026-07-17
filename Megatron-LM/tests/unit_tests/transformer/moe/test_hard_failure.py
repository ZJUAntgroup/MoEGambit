# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for MOEGAMBIT-MoE Hard Failure Detection & Iteration Invalidation.

These tests are designed to run WITHOUT torch/NCCL dependencies by using
importlib to load the target modules directly, bypassing megatron.core.__init__
which imports torch.
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

# Stub out torch so that any transitive import doesn't fail
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
    # Ensure parent packages exist as namespace stubs
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


# Load the three modules under test
_hfd_mod = _load_module("megatron.core.transformer.moe.hard_failure_detector")
_inv_mod = _load_module("megatron.core.transformer.moe.iteration_invalidator")
_rc_mod = _load_module("megatron.core.transformer.moe.recovery_controller")

HardFailureDetector = _hfd_mod.HardFailureDetector
HardFailureRecord = _hfd_mod.HardFailureRecord
FailureSource = _hfd_mod.FailureSource

IterationInvalidator = _inv_mod.IterationInvalidator
InvalidationRecord = _inv_mod.InvalidationRecord

RecoveryController = _rc_mod.RecoveryController
RecoveryPhase = _rc_mod.RecoveryPhase
FaultRecord = _rc_mod.FaultRecord


# =====================================================================
# Test: HardFailureDetector
# =====================================================================

class TestHardFailureDetector(unittest.TestCase):
    """Tests for HardFailureDetector."""

    def setUp(self):
        self.detector = HardFailureDetector()
        self.callback_calls = []
        self.invalid_calls = []

        def on_hard_failure(**kwargs):
            self.callback_calls.append(kwargs)

        def on_iteration_invalid(**kwargs):
            self.invalid_calls.append(kwargs)

        self.detector.register_callbacks(
            on_hard_failure_fn=on_hard_failure,
            on_iteration_invalid_fn=on_iteration_invalid,
        )

    def test_report_failure_basic(self):
        """New failure returns True and invokes callback."""
        result = self.detector.report_failure(
            42, source=FailureSource.COLLECTIVE_ERROR,
            reason="nccl timeout", step=100, mid_iteration=True,
        )
        self.assertTrue(result)
        self.assertEqual(self.detector.num_failures, 1)
        self.assertTrue(self.detector.is_failed(42))
        self.assertIn(42, self.detector.failed_ranks)

        # Callback was invoked
        self.assertEqual(len(self.callback_calls), 1)
        self.assertEqual(self.callback_calls[0]["failed_rank"], 42)
        self.assertEqual(self.callback_calls[0]["mid_iteration"], True)

    def test_report_failure_idempotent(self):
        """Duplicate report returns False and does NOT re-invoke callback."""
        self.detector.report_failure(42, step=100)
        result = self.detector.report_failure(42, step=101)
        self.assertFalse(result)
        self.assertEqual(len(self.callback_calls), 1)

    def test_mid_iteration_triggers_invalidation_callback(self):
        """mid_iteration=True triggers the on_iteration_invalid callback."""
        self.detector.report_failure(
            7, source=FailureSource.RANK_EXIT,
            step=50, mid_iteration=True, reason="process exited",
        )
        self.assertEqual(len(self.invalid_calls), 1)
        self.assertEqual(self.invalid_calls[0]["step"], 50)
        self.assertEqual(self.invalid_calls[0]["failed_rank"], 7)

    def test_no_invalidation_when_not_mid_iteration(self):
        """mid_iteration=False does NOT trigger invalidation callback."""
        self.detector.report_failure(7, step=50, mid_iteration=False)
        self.assertEqual(len(self.invalid_calls), 0)

    def test_report_collective_failure(self):
        """Convenience method sets source=COLLECTIVE_ERROR."""
        exc = RuntimeError("NCCL timeout")
        self.detector.report_collective_failure(
            3, reason="test", step=200, exception=exc,
        )
        self.assertTrue(self.detector.is_failed(3))
        record = self.detector.get_record(3)
        self.assertEqual(record.source, FailureSource.COLLECTIVE_ERROR)
        self.assertEqual(record.exception_type, "RuntimeError")

    def test_report_pipeline_stage_failure(self):
        """PP>1 placeholder sets source=P2P_TIMEOUT."""
        self.detector.report_pipeline_stage_failure(
            5, reason="p2p timeout", step=300,
        )
        record = self.detector.get_record(5)
        self.assertEqual(record.source, FailureSource.P2P_TIMEOUT)

    def test_has_mid_iteration_failure(self):
        """has_mid_iteration_failure reflects recorded failures."""
        self.assertFalse(self.detector.has_mid_iteration_failure)
        self.detector.report_failure(1, mid_iteration=False)
        self.assertFalse(self.detector.has_mid_iteration_failure)
        self.detector.report_failure(2, mid_iteration=True)
        self.assertTrue(self.detector.has_mid_iteration_failure)

    def test_clear_rank(self):
        """clear_rank removes a rank from the detected set."""
        self.detector.report_failure(10, step=1)
        self.assertTrue(self.detector.is_failed(10))
        self.detector.clear_rank(10)
        self.assertFalse(self.detector.is_failed(10))
        self.assertEqual(self.detector.num_failures, 0)

    def test_reset(self):
        """reset clears all records."""
        self.detector.report_failure(1)
        self.detector.report_failure(2)
        self.detector.reset()
        self.assertEqual(self.detector.num_failures, 0)

    def test_summary(self):
        """summary returns correct structure."""
        self.detector.report_failure(5, mid_iteration=True)
        s = self.detector.summary()
        self.assertEqual(s["num_failures"], 1)
        self.assertEqual(s["failed_ranks"], [5])
        self.assertTrue(s["has_mid_iteration_failure"])

    def test_multiple_failures(self):
        """Multiple distinct ranks are all tracked."""
        for r in [1, 3, 5, 7]:
            self.detector.report_failure(r, step=r * 10)
        self.assertEqual(self.detector.num_failures, 4)
        self.assertEqual(self.detector.failed_ranks, {1, 3, 5, 7})
        self.assertEqual(len(self.callback_calls), 4)

    def test_no_callback_when_not_registered(self):
        """No crash when callbacks are not registered."""
        detector = HardFailureDetector()  # no callbacks
        result = detector.report_failure(1, mid_iteration=True)
        self.assertTrue(result)
        self.assertTrue(detector.is_failed(1))


# =====================================================================
# Test: IterationInvalidator
# =====================================================================

class TestIterationInvalidator(unittest.TestCase):
    """Tests for IterationInvalidator."""

    def setUp(self):
        self.inv = IterationInvalidator()

    def test_lifecycle_clean(self):
        """Clean iteration: begin → not invalid → end."""
        self.inv.begin_iteration(100)
        self.assertFalse(self.inv.is_current_iteration_invalid())
        self.assertFalse(self.inv.should_skip_optimizer_step())
        self.assertTrue(self.inv.in_iteration)
        self.inv.end_iteration(100)
        self.assertFalse(self.inv.in_iteration)

    def test_invalidate_mid_iteration(self):
        """Invalidation during iteration is detected."""
        self.inv.begin_iteration(200)
        result = self.inv.invalidate(step=200, failed_rank=5, reason="test")
        self.assertTrue(result)
        self.assertTrue(self.inv.is_current_iteration_invalid())
        self.assertTrue(self.inv.should_skip_optimizer_step())
        self.assertEqual(self.inv.total_invalidations, 1)

    def test_invalidate_idempotent(self):
        """Second invalidation in same iteration returns False."""
        self.inv.begin_iteration(300)
        self.assertTrue(self.inv.invalidate(step=300, failed_rank=1))
        self.assertFalse(self.inv.invalidate(step=300, failed_rank=2))
        self.assertEqual(self.inv.total_invalidations, 1)

    def test_begin_clears_previous_invalidation(self):
        """begin_iteration resets the invalid flag."""
        self.inv.begin_iteration(100)
        self.inv.invalidate(step=100, failed_rank=1)
        self.assertTrue(self.inv.is_current_iteration_invalid())

        # Next iteration clears it
        self.inv.end_iteration(100)
        self.inv.begin_iteration(101)
        self.assertFalse(self.inv.is_current_iteration_invalid())

    def test_invalid_flag_persists_after_end(self):
        """_invalid flag persists after end_iteration (for training loop check)."""
        self.inv.begin_iteration(100)
        self.inv.invalidate(step=100, failed_rank=1)
        self.inv.end_iteration(100)
        # Still invalid — training loop reads this AFTER end_iteration
        self.assertTrue(self.inv.is_current_iteration_invalid())

    def test_history_tracking(self):
        """Invalidation records are archived in history."""
        for step in range(5):
            self.inv.begin_iteration(step)
            self.inv.invalidate(step=step, failed_rank=step)
            self.inv.end_iteration(step)
        self.assertEqual(self.inv.total_invalidations, 5)

    def test_last_invalidation(self):
        """last_invalidation returns the most recent record."""
        self.inv.begin_iteration(10)
        self.inv.invalidate(step=10, failed_rank=3, reason="boom")
        rec = self.inv.last_invalidation
        self.assertIsNotNone(rec)
        self.assertEqual(rec.step, 10)
        self.assertEqual(rec.failed_rank, 3)
        self.assertEqual(rec.reason, "boom")

    def test_summary(self):
        """summary returns correct structure."""
        self.inv.begin_iteration(50)
        self.inv.invalidate(step=50)
        s = self.inv.summary()
        self.assertEqual(s["current_step"], 50)
        self.assertTrue(s["is_invalid"])
        self.assertTrue(s["in_iteration"])
        self.assertEqual(s["total_invalidations"], 1)

    def test_reset(self):
        """reset clears all state."""
        self.inv.begin_iteration(1)
        self.inv.invalidate(step=1)
        self.inv.reset()
        self.assertFalse(self.inv.is_current_iteration_invalid())
        self.assertEqual(self.inv.total_invalidations, 0)
        self.assertFalse(self.inv.in_iteration)


# =====================================================================
# Test: RecoveryController hard failure integration
# =====================================================================

class TestRecoveryControllerHardFailure(unittest.TestCase):
    """Tests for RecoveryController hard failure + iteration invalidation."""

    def setUp(self):
        self.ctrl = RecoveryController()
        self.quarantine_calls = []
        self.unavailable_calls = []

        self.ctrl.register_callbacks(
            quarantine_fn=lambda **kw: self.quarantine_calls.append(kw),
            health_mark_unavailable_fn=lambda **kw: self.unavailable_calls.append(kw),
        )

    def test_hard_failure_transitions_to_pending(self):
        """Hard failure from HEALTHY → PENDING_GROUP_REPAIR."""
        self.ctrl.on_hard_rank_failure(
            failed_rank=3, reason="nccl error", step=100,
            expert_ids=[6, 7],
        )
        self.assertEqual(self.ctrl.phase, RecoveryPhase.PENDING_GROUP_REPAIR)
        self.assertEqual(self.ctrl.num_active_faults, 1)

    def test_hard_failure_mid_iteration_sets_invalidation(self):
        """mid_iteration=True sets iteration_was_invalidated."""
        self.ctrl.on_hard_rank_failure(
            failed_rank=3, step=100, mid_iteration=True,
        )
        self.assertTrue(self.ctrl.iteration_was_invalidated)
        self.assertEqual(self.ctrl.invalidated_step, 100)

    def test_hard_failure_no_mid_iteration_no_invalidation(self):
        """mid_iteration=False does NOT set iteration_was_invalidated."""
        self.ctrl.on_hard_rank_failure(
            failed_rank=3, step=100, mid_iteration=False,
        )
        self.assertFalse(self.ctrl.iteration_was_invalidated)

    def test_before_iteration_clears_invalidation(self):
        """before_iteration clears the invalidation flag."""
        self.ctrl.on_hard_rank_failure(
            failed_rank=3, step=100, mid_iteration=True,
        )
        self.assertTrue(self.ctrl.iteration_was_invalidated)
        self.ctrl.before_iteration(step=101)
        self.assertFalse(self.ctrl.iteration_was_invalidated)

    def test_escalation_soft_to_hard(self):
        """Soft fault can be escalated to hard fault."""
        self.ctrl.on_rank_quarantined(
            failed_rank=5, step=50, expert_ids=[10, 11],
        )
        self.assertEqual(self.ctrl.phase, RecoveryPhase.DEGRADED_ISOLATION)

        self.ctrl.on_hard_rank_failure(
            failed_rank=5, step=55, expert_ids=[10, 11],
            mid_iteration=True,
        )
        self.assertEqual(self.ctrl.phase, RecoveryPhase.PENDING_GROUP_REPAIR)
        record = self.ctrl.get_fault_record(5)
        self.assertEqual(record.fault_type, "hard")
        self.assertTrue(record.mid_iteration)

    def test_hard_failure_idempotent(self):
        """Duplicate hard failure for same rank is a no-op."""
        self.ctrl.on_hard_rank_failure(failed_rank=3, step=100)
        self.ctrl.on_hard_rank_failure(failed_rank=3, step=101)
        self.assertEqual(self.ctrl.num_active_faults, 1)

    def test_fault_record_mid_iteration_field(self):
        """FaultRecord.mid_iteration is set correctly."""
        self.ctrl.on_hard_rank_failure(
            failed_rank=7, step=200, mid_iteration=True,
        )
        record = self.ctrl.get_fault_record(7)
        self.assertTrue(record.mid_iteration)
        d = record.to_dict()
        self.assertTrue(d["mid_iteration"])

    def test_full_lifecycle_with_invalidation(self):
        """Full lifecycle: hard failure → replacement → repair → healthy."""
        # 1. Hard failure mid-iteration
        self.ctrl.on_hard_rank_failure(
            failed_rank=2, step=100, mid_iteration=True,
            expert_ids=[4, 5], ep_group_ranks=[0, 1, 2, 3],
        )
        self.assertEqual(self.ctrl.phase, RecoveryPhase.PENDING_GROUP_REPAIR)
        self.assertTrue(self.ctrl.iteration_was_invalidated)

        # 2. Next iteration clears invalidation
        self.ctrl.before_iteration(step=101)
        self.assertFalse(self.ctrl.iteration_was_invalidated)

        # 3. Replacement assigned
        self.ctrl.on_replacement_assigned(
            failed_rank=2, replacement_rank=99, step=101,
        )
        self.assertEqual(self.ctrl.phase, RecoveryPhase.WAITING_FOR_REPLACEMENT)

        # 4. Replacement ready
        self.ctrl.on_replacement_ready(failed_rank=2, step=102)
        self.assertEqual(self.ctrl.phase, RecoveryPhase.SAFE_POINT_REPAIR)

        # 5. Safe-point repair at next iteration boundary
        repaired = self.ctrl.before_iteration(step=103)
        self.assertTrue(repaired)
        self.assertEqual(self.ctrl.phase, RecoveryPhase.REINTEGRATED)

        # 6. Finalize at next iteration
        self.ctrl.before_iteration(step=104)
        self.assertEqual(self.ctrl.phase, RecoveryPhase.HEALTHY_TRAINING)
        self.assertEqual(self.ctrl.num_active_faults, 0)
        self.assertEqual(self.ctrl.num_completed_recoveries, 1)

    def test_reset_clears_invalidation(self):
        """reset() clears iteration invalidation state."""
        self.ctrl.on_hard_rank_failure(
            failed_rank=1, step=50, mid_iteration=True,
        )
        self.assertTrue(self.ctrl.iteration_was_invalidated)
        self.ctrl.reset()
        self.assertFalse(self.ctrl.iteration_was_invalidated)
        self.assertEqual(self.ctrl.invalidated_step, -1)


# =====================================================================
# Test: Integration — Detector + Invalidator + Controller wired together
# =====================================================================

class TestIntegration(unittest.TestCase):
    """End-to-end integration: Detector → Controller → Invalidator."""

    def setUp(self):
        self.ctrl = RecoveryController()
        self.inv = IterationInvalidator()
        self.detector = HardFailureDetector()

        # Wire: detector → controller.on_hard_rank_failure
        #        detector → invalidator.invalidate
        self.detector.register_callbacks(
            on_hard_failure_fn=self.ctrl.on_hard_rank_failure,
            on_iteration_invalid_fn=self.inv.invalidate,
        )

        # Register no-op callbacks on controller to avoid errors
        self.ctrl.register_callbacks(
            quarantine_fn=lambda **kw: None,
            health_mark_unavailable_fn=lambda **kw: None,
        )

    def test_collective_failure_mid_iteration(self):
        """Collective failure mid-iteration invalidates iteration and
        transitions controller to PENDING_GROUP_REPAIR."""
        # Simulate: iteration 100 starts
        self.inv.begin_iteration(100)
        self.ctrl.before_iteration(step=100)

        # Simulate: NCCL error during forward pass
        exc = RuntimeError("NCCL SystemError: Connection reset by peer")
        self.detector.report_collective_failure(
            failed_rank=3, reason="nccl error", step=100,
            mid_iteration=True, exception=exc,
            expert_ids=[6, 7],
        )

        # Check: iteration is invalid
        self.assertTrue(self.inv.is_current_iteration_invalid())
        self.assertTrue(self.inv.should_skip_optimizer_step())

        # Check: controller is in PENDING_GROUP_REPAIR
        self.assertEqual(self.ctrl.phase, RecoveryPhase.PENDING_GROUP_REPAIR)
        self.assertTrue(self.ctrl.iteration_was_invalidated)

        # Check: detector recorded the failure
        self.assertTrue(self.detector.is_failed(3))
        record = self.detector.get_record(3)
        self.assertTrue(record.mid_iteration)
        self.assertEqual(record.source, FailureSource.COLLECTIVE_ERROR)

    def test_simulated_rank_exit_mid_iteration(self):
        """Simulate rank exit detected via Gloo probe mid-iteration."""
        self.inv.begin_iteration(100)
        self.ctrl.before_iteration(step=100)

        # Gloo probe detects rank 5 has exited
        self.detector.report_failure(
            5, source=FailureSource.RANK_EXIT,
            reason="process exited", step=100,
            mid_iteration=True, expert_ids=[10, 11],
        )

        # Iteration is invalid
        self.assertTrue(self.inv.is_current_iteration_invalid())

        # Controller is in PENDING_GROUP_REPAIR
        self.assertEqual(self.ctrl.phase, RecoveryPhase.PENDING_GROUP_REPAIR)

        # End iteration (training loop would skip optimizer)
        self.inv.end_iteration(100)

        # Next iteration: invalidation is cleared
        self.inv.begin_iteration(101)
        self.ctrl.before_iteration(step=101)
        self.assertFalse(self.inv.is_current_iteration_invalid())
        self.assertFalse(self.ctrl.iteration_was_invalidated)

        # Controller is still in PENDING_GROUP_REPAIR (waiting for replacement)
        self.assertEqual(self.ctrl.phase, RecoveryPhase.PENDING_GROUP_REPAIR)

    def test_failure_not_mid_iteration(self):
        """Failure detected at iteration boundary does NOT invalidate."""
        self.inv.begin_iteration(100)
        self.ctrl.before_iteration(step=100)
        self.inv.end_iteration(100)

        # Failure detected between iterations (e.g. during checkpoint)
        self.detector.report_failure(
            8, source=FailureSource.EXTERNAL_MONITOR,
            reason="node down", step=100,
            mid_iteration=False,
        )

        # Iteration is NOT invalid (failure was not mid-iteration)
        # Note: begin_iteration(100) already reset the flag, and
        # mid_iteration=False means invalidate() was not called
        self.assertFalse(self.inv.is_current_iteration_invalid())
        self.assertFalse(self.ctrl.iteration_was_invalidated)

        # But controller still transitions to PENDING_GROUP_REPAIR
        self.assertEqual(self.ctrl.phase, RecoveryPhase.PENDING_GROUP_REPAIR)

    def test_full_recovery_flow(self):
        """Full flow: detect → invalidate → replacement → repair → healthy."""
        # Step 100: failure mid-iteration
        self.inv.begin_iteration(100)
        self.ctrl.before_iteration(step=100)
        self.detector.report_collective_failure(
            failed_rank=2, step=100, mid_iteration=True,
            expert_ids=[4, 5], ep_group_ranks=[0, 1, 2, 3],
        )
        self.assertTrue(self.inv.is_current_iteration_invalid())
        self.inv.end_iteration(100)

        # Step 101: invalidation cleared, still pending
        self.inv.begin_iteration(101)
        self.ctrl.before_iteration(step=101)
        self.assertFalse(self.inv.is_current_iteration_invalid())
        self.assertEqual(self.ctrl.phase, RecoveryPhase.PENDING_GROUP_REPAIR)

        # Replacement assigned and ready
        self.ctrl.on_replacement_assigned(
            failed_rank=2, replacement_rank=99, step=101,
        )
        self.ctrl.on_replacement_ready(failed_rank=2, step=101)
        self.assertEqual(self.ctrl.phase, RecoveryPhase.SAFE_POINT_REPAIR)
        self.inv.end_iteration(101)

        # Step 102: safe-point repair
        self.inv.begin_iteration(102)
        repaired = self.ctrl.before_iteration(step=102)
        self.assertTrue(repaired)
        self.assertEqual(self.ctrl.phase, RecoveryPhase.REINTEGRATED)
        self.inv.end_iteration(102)

        # Step 103: finalize
        self.inv.begin_iteration(103)
        self.ctrl.before_iteration(step=103)
        self.assertEqual(self.ctrl.phase, RecoveryPhase.HEALTHY_TRAINING)
        self.assertEqual(self.ctrl.num_completed_recoveries, 1)

    def test_multiple_failures_same_iteration(self):
        """Multiple ranks fail in the same iteration."""
        self.inv.begin_iteration(100)
        self.ctrl.before_iteration(step=100)

        # Two ranks fail
        self.detector.report_failure(
            1, source=FailureSource.COLLECTIVE_ERROR,
            step=100, mid_iteration=True,
        )
        # Second failure: controller is already in PENDING_GROUP_REPAIR,
        # so the transition is a no-op, but the fault is still recorded
        # in the detector.
        self.detector.report_failure(
            2, source=FailureSource.COLLECTIVE_ERROR,
            step=100, mid_iteration=True,
        )

        self.assertTrue(self.inv.is_current_iteration_invalid())
        self.assertEqual(self.detector.num_failures, 2)
        # Note: v1 controller handles one fault at a time, so only
        # the first rank's fault record is in active_faults via the
        # controller. The second rank's failure is recorded in the
        # detector but the controller transition is a no-op because
        # it's already in PENDING_GROUP_REPAIR.


# =====================================================================
# Test: Singleton management
# =====================================================================

class TestSingletons(unittest.TestCase):
    """Tests for global singleton get/clear functions."""

    def test_detector_singleton(self):
        _hfd_mod.clear_hard_failure_detector()
        d1 = _hfd_mod.get_hard_failure_detector()
        d2 = _hfd_mod.get_hard_failure_detector()
        self.assertIs(d1, d2)
        _hfd_mod.clear_hard_failure_detector()
        d3 = _hfd_mod.get_hard_failure_detector()
        self.assertIsNot(d1, d3)

    def test_invalidator_singleton(self):
        _inv_mod.clear_iteration_invalidator()
        i1 = _inv_mod.get_iteration_invalidator()
        i2 = _inv_mod.get_iteration_invalidator()
        self.assertIs(i1, i2)
        _inv_mod.clear_iteration_invalidator()
        i3 = _inv_mod.get_iteration_invalidator()
        self.assertIsNot(i1, i3)

    def test_controller_singleton(self):
        _rc_mod.clear_recovery_controller()
        c1 = _rc_mod.get_recovery_controller()
        c2 = _rc_mod.get_recovery_controller()
        self.assertIs(c1, c2)
        _rc_mod.clear_recovery_controller()
        c3 = _rc_mod.get_recovery_controller()
        self.assertIsNot(c1, c3)


# =====================================================================
# Test: HardFailureRecord & FaultRecord dataclasses
# =====================================================================

class TestDataclasses(unittest.TestCase):
    """Tests for dataclass fields and serialization."""

    def test_hard_failure_record_defaults(self):
        r = HardFailureRecord()
        self.assertEqual(r.failed_rank, -1)
        self.assertEqual(r.source, FailureSource.UNKNOWN)
        self.assertFalse(r.mid_iteration)

    def test_hard_failure_record_fields(self):
        r = HardFailureRecord(
            failed_rank=5,
            source=FailureSource.GLOO_PROBE,
            reason="gloo detected exit",
            step=42,
            mid_iteration=True,
            exception_type="RuntimeError",
        )
        self.assertEqual(r.failed_rank, 5)
        self.assertEqual(r.source, FailureSource.GLOO_PROBE)
        self.assertTrue(r.mid_iteration)

    def test_fault_record_to_dict(self):
        r = FaultRecord(
            failed_rank=3,
            fault_type="hard",
            reason="test",
            fault_step=100,
            mid_iteration=True,
            expert_ids=[6, 7],
        )
        d = r.to_dict()
        self.assertEqual(d["failed_rank"], 3)
        self.assertEqual(d["fault_type"], "hard")
        self.assertTrue(d["mid_iteration"])
        self.assertEqual(d["expert_ids"], [6, 7])

    def test_invalidation_record(self):
        r = InvalidationRecord(step=50, failed_rank=2, reason="boom")
        self.assertEqual(r.step, 50)
        self.assertEqual(r.failed_rank, 2)


if __name__ == "__main__":
    unittest.main()

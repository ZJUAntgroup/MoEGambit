# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Tests for MoEGambit Stage-Safe Recovery Protocol (Step 10).

"""Unit tests for the stage-safe recovery protocol for PP > 1.

Covers:
    1. StageRecoveryPhase transitions
    2. StageRecoveryResult data class
    3. StageSafeRecoveryProtocol — full closed-loop execution
    4. Utility functions (compute_new_pp_ranks, identify_failed_stage)
    5. Global singleton management
    6. Dual-path semantics (HYBRID_RECOVERY vs CHECKPOINT_RESTART)
    7. Error handling and partial failure
    8. RecoveryController PP-aware integration
    9. moegambit_integration wiring (moegambit_execute_stage_safe_recovery)
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Path setup — ensure repo root is on sys.path
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from megatron.core.transformer.moe.stage_safe_recovery import (
    StageRecoveryPhase,
    StageRecoveryResult,
    StageSafeRecoveryProtocol,
    compute_new_pp_ranks,
    identify_failed_stage,
    get_stage_safe_recovery_protocol,
    clear_stage_safe_recovery_protocol,
    _VALID_PHASE_TRANSITIONS,
)


# =====================================================================
# 1. StageRecoveryPhase
# =====================================================================

class TestStageRecoveryPhase(unittest.TestCase):
    """Test the phase enum and valid transitions."""

    def test_phase_values(self):
        self.assertEqual(StageRecoveryPhase.IDLE, 0)
        self.assertEqual(StageRecoveryPhase.FAILURE_DETECTED, 1)
        self.assertEqual(StageRecoveryPhase.TRAINING_RESUMED, 7)

    def test_all_phases_have_transitions(self):
        for phase in StageRecoveryPhase:
            self.assertIn(phase, _VALID_PHASE_TRANSITIONS)

    def test_idle_can_only_go_to_failure_detected(self):
        targets = _VALID_PHASE_TRANSITIONS[StageRecoveryPhase.IDLE]
        self.assertEqual(targets, {StageRecoveryPhase.FAILURE_DETECTED})

    def test_training_resumed_returns_to_idle(self):
        targets = _VALID_PHASE_TRANSITIONS[StageRecoveryPhase.TRAINING_RESUMED]
        self.assertEqual(targets, {StageRecoveryPhase.IDLE})


# =====================================================================
# 2. StageRecoveryResult
# =====================================================================

class TestStageRecoveryResult(unittest.TestCase):
    """Test the result data class."""

    def test_default_not_success(self):
        r = StageRecoveryResult()
        self.assertFalse(r.success)

    def test_success_when_training_resumed_no_errors(self):
        r = StageRecoveryResult(
            phase_reached=StageRecoveryPhase.TRAINING_RESUMED,
        )
        self.assertTrue(r.success)

    def test_not_success_with_errors(self):
        r = StageRecoveryResult(
            phase_reached=StageRecoveryPhase.TRAINING_RESUMED,
            errors=["something went wrong"],
        )
        self.assertFalse(r.success)

    def test_to_dict(self):
        r = StageRecoveryResult(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            recovery_path="HYBRID_RECOVERY",
            phase_reached=StageRecoveryPhase.TRAINING_RESUMED,
        )
        d = r.to_dict()
        self.assertEqual(d["failed_rank"], 2)
        self.assertEqual(d["failed_stage"], 1)
        self.assertEqual(d["replacement_rank"], 5)
        self.assertEqual(d["step"], 100)
        self.assertEqual(d["recovery_path"], "HYBRID_RECOVERY")
        self.assertEqual(d["phase_reached"], "TRAINING_RESUMED")
        self.assertTrue(d["success"])


# =====================================================================
# 3. StageSafeRecoveryProtocol — full closed-loop
# =====================================================================

class TestProtocolFullLoop(unittest.TestCase):
    """Test the full recovery protocol with mock callbacks."""

    def setUp(self):
        self.protocol = StageSafeRecoveryProtocol()
        self.calls = []

    def _make_callback(self, name):
        def cb(**kwargs):
            self.calls.append((name, kwargs))
        return cb

    def test_full_hybrid_recovery(self):
        result = self.protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
            expert_ids=[4, 5],
            recovery_path="HYBRID_RECOVERY",
            invalidate_iteration_fn=self._make_callback("invalidate"),
            rollback_fn=self._make_callback("rollback"),
            wait_replacement_fn=self._make_callback("wait"),
            group_repair_fn=self._make_callback("group_repair"),
            topology_refresh_fn=self._make_callback("topology"),
            dense_sync_fn=self._make_callback("dense_sync"),
            expert_restore_fn=self._make_callback("expert_restore"),
            convergence_fn=self._make_callback("convergence"),
            p2p_rebind_fn=self._make_callback("p2p_rebind"),
            reintegration_fn=self._make_callback("reintegration"),
        )

        self.assertTrue(result.success)
        self.assertEqual(result.phase_reached, StageRecoveryPhase.TRAINING_RESUMED)
        self.assertEqual(result.failed_rank, 2)
        self.assertEqual(result.failed_stage, 1)
        self.assertEqual(result.replacement_rank, 5)
        self.assertEqual(result.recovery_path, "HYBRID_RECOVERY")
        self.assertTrue(result.iteration_invalidated)
        self.assertTrue(result.rollback_success)
        self.assertTrue(result.replacement_ready)
        self.assertTrue(result.p2p_rebound)
        self.assertTrue(result.training_resumed)

        # Verify callback order
        names = [c[0] for c in self.calls]
        self.assertEqual(names, [
            "invalidate", "rollback", "wait",
            "group_repair", "topology", "dense_sync", "expert_restore",
            "convergence", "p2p_rebind", "reintegration",
        ])

    def test_full_checkpoint_restart(self):
        result = self.protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
            recovery_path="CHECKPOINT_RESTART",
            invalidate_iteration_fn=self._make_callback("invalidate"),
            rollback_fn=self._make_callback("rollback"),
            group_repair_fn=self._make_callback("group_repair"),
            checkpoint_restart_fn=self._make_callback("checkpoint_restart"),
            convergence_fn=self._make_callback("convergence"),
            p2p_rebind_fn=self._make_callback("p2p_rebind"),
        )

        self.assertTrue(result.success)
        names = [c[0] for c in self.calls]
        # checkpoint_restart path: no dense_sync or expert_restore
        self.assertIn("checkpoint_restart", names)
        self.assertNotIn("dense_sync", names)
        self.assertNotIn("expert_restore", names)

    def test_no_callbacks_still_succeeds(self):
        """Protocol succeeds even with no callbacks (test/dry-run mode)."""
        result = self.protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
        )
        self.assertTrue(result.success)
        self.assertEqual(result.phase_reached, StageRecoveryPhase.TRAINING_RESUMED)

    def test_protocol_returns_to_idle(self):
        self.protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
        )
        self.assertEqual(self.protocol.phase, StageRecoveryPhase.IDLE)


class TestProtocolErrorHandling(unittest.TestCase):
    """Test error handling in the protocol."""

    def setUp(self):
        self.protocol = StageSafeRecoveryProtocol()

    def test_rollback_failure_recorded(self):
        def bad_rollback(**kwargs):
            raise RuntimeError("rollback failed")

        result = self.protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
            rollback_fn=bad_rollback,
        )

        # Protocol continues despite rollback error
        self.assertEqual(result.phase_reached, StageRecoveryPhase.TRAINING_RESUMED)
        self.assertFalse(result.rollback_success)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("rollback failed", result.errors[0])

    def test_group_repair_failure_recorded(self):
        def bad_repair(**kwargs):
            raise RuntimeError("group repair failed")

        result = self.protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
            group_repair_fn=bad_repair,
        )

        self.assertEqual(len(result.errors), 1)
        self.assertIn("group_repair failed", result.errors[0])

    def test_p2p_rebind_failure_recorded(self):
        def bad_p2p(**kwargs):
            raise RuntimeError("p2p failed")

        result = self.protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
            p2p_rebind_fn=bad_p2p,
        )

        self.assertEqual(len(result.errors), 1)
        self.assertIn("p2p_rebind failed", result.errors[0])
        self.assertFalse(result.p2p_rebound)

    def test_convergence_failure_non_fatal(self):
        def bad_convergence(**kwargs):
            raise RuntimeError("convergence failed")

        result = self.protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
            convergence_fn=bad_convergence,
        )

        # Protocol still reaches TRAINING_RESUMED
        self.assertEqual(result.phase_reached, StageRecoveryPhase.TRAINING_RESUMED)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("convergence failed", result.errors[0])

    def test_multiple_errors_accumulated(self):
        def bad_fn(**kwargs):
            raise RuntimeError("error")

        result = self.protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
            rollback_fn=bad_fn,
            group_repair_fn=bad_fn,
            p2p_rebind_fn=bad_fn,
        )

        self.assertEqual(len(result.errors), 3)


# =====================================================================
# 4. Utility functions
# =====================================================================

class TestComputeNewPPRanks(unittest.TestCase):
    """Test compute_new_pp_ranks."""

    def test_basic_replacement(self):
        result = compute_new_pp_ranks([0, 2, 4, 6], 2, 5)
        self.assertEqual(result, [0, 5, 4, 6])

    def test_first_stage(self):
        result = compute_new_pp_ranks([0, 2, 4, 6], 0, 8)
        self.assertEqual(result, [8, 2, 4, 6])

    def test_last_stage(self):
        result = compute_new_pp_ranks([0, 2, 4, 6], 6, 9)
        self.assertEqual(result, [0, 2, 4, 9])

    def test_failed_rank_not_in_group(self):
        with self.assertRaises(ValueError):
            compute_new_pp_ranks([0, 2, 4, 6], 3, 5)

    def test_single_stage(self):
        result = compute_new_pp_ranks([0], 0, 1)
        self.assertEqual(result, [1])


class TestIdentifyFailedStage(unittest.TestCase):
    """Test identify_failed_stage."""

    def test_found(self):
        self.assertEqual(identify_failed_stage(4, [0, 2, 4, 6]), 2)

    def test_first(self):
        self.assertEqual(identify_failed_stage(0, [0, 2, 4, 6]), 0)

    def test_not_found(self):
        self.assertEqual(identify_failed_stage(3, [0, 2, 4, 6]), -1)


# =====================================================================
# 5. Global singleton
# =====================================================================

class TestGlobalSingleton(unittest.TestCase):
    """Test global singleton management."""

    def setUp(self):
        clear_stage_safe_recovery_protocol()

    def tearDown(self):
        clear_stage_safe_recovery_protocol()

    def test_get_creates(self):
        p = get_stage_safe_recovery_protocol()
        self.assertIsNotNone(p)
        self.assertIsInstance(p, StageSafeRecoveryProtocol)

    def test_get_returns_same(self):
        p1 = get_stage_safe_recovery_protocol()
        p2 = get_stage_safe_recovery_protocol()
        self.assertIs(p1, p2)

    def test_clear(self):
        p1 = get_stage_safe_recovery_protocol()
        clear_stage_safe_recovery_protocol()
        p2 = get_stage_safe_recovery_protocol()
        self.assertIsNot(p1, p2)


# =====================================================================
# 6. Dual-path semantics
# =====================================================================

class TestDualPathSemantics(unittest.TestCase):
    """Test that both recovery paths produce consistent results."""

    def setUp(self):
        self.protocol = StageSafeRecoveryProtocol()

    def test_hybrid_calls_dense_and_expert(self):
        calls = []

        def dense_fn(**kw):
            calls.append("dense")

        def expert_fn(**kw):
            calls.append("expert")

        def ckpt_fn(**kw):
            calls.append("checkpoint")

        result = self.protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
            recovery_path="HYBRID_RECOVERY",
            dense_sync_fn=dense_fn,
            expert_restore_fn=expert_fn,
            checkpoint_restart_fn=ckpt_fn,
        )

        self.assertIn("dense", calls)
        self.assertIn("expert", calls)
        self.assertNotIn("checkpoint", calls)

    def test_checkpoint_calls_only_checkpoint(self):
        calls = []

        def dense_fn(**kw):
            calls.append("dense")

        def expert_fn(**kw):
            calls.append("expert")

        def ckpt_fn(**kw):
            calls.append("checkpoint")

        result = self.protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
            recovery_path="CHECKPOINT_RESTART",
            dense_sync_fn=dense_fn,
            expert_restore_fn=expert_fn,
            checkpoint_restart_fn=ckpt_fn,
        )

        self.assertNotIn("dense", calls)
        self.assertNotIn("expert", calls)
        self.assertIn("checkpoint", calls)

    def test_both_paths_reach_training_resumed(self):
        for path in ["HYBRID_RECOVERY", "CHECKPOINT_RESTART"]:
            p = StageSafeRecoveryProtocol()
            result = p.execute(
                failed_rank=2,
                failed_stage=1,
                replacement_rank=5,
                step=100,
                pp_group_ranks=[0, 2, 4, 6],
                recovery_path=path,
            )
            self.assertTrue(result.success, f"path={path} should succeed")

    def test_both_paths_same_result_fields(self):
        results = {}
        for path in ["HYBRID_RECOVERY", "CHECKPOINT_RESTART"]:
            p = StageSafeRecoveryProtocol()
            r = p.execute(
                failed_rank=2,
                failed_stage=1,
                replacement_rank=5,
                step=100,
                pp_group_ranks=[0, 2, 4, 6],
                recovery_path=path,
            )
            results[path] = r.to_dict()

        # Both should have the same keys
        self.assertEqual(
            set(results["HYBRID_RECOVERY"].keys()),
            set(results["CHECKPOINT_RESTART"].keys()),
        )


# =====================================================================
# 7. History and query API
# =====================================================================

class TestHistoryAndQuery(unittest.TestCase):
    """Test history tracking and query API."""

    def setUp(self):
        self.protocol = StageSafeRecoveryProtocol()

    def test_history_empty_initially(self):
        self.assertEqual(self.protocol.total_recoveries, 0)
        self.assertEqual(self.protocol.total_successes, 0)

    def test_history_tracks_recoveries(self):
        for i in range(3):
            self.protocol.execute(
                failed_rank=i,
                failed_stage=0,
                replacement_rank=i + 10,
                step=i * 100,
                pp_group_ranks=[0, 1, 2, 3],
            )

        self.assertEqual(self.protocol.total_recoveries, 3)
        self.assertEqual(self.protocol.total_successes, 3)
        self.assertEqual(len(self.protocol.history), 3)

    def test_summary(self):
        self.protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
        )
        s = self.protocol.summary()
        self.assertEqual(s["phase"], "IDLE")
        self.assertEqual(s["total_recoveries"], 1)
        self.assertEqual(s["total_successes"], 1)
        self.assertIsNotNone(s["last_result"])

    def test_reset_clears_history(self):
        self.protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
        )
        self.protocol.reset()
        self.assertEqual(self.protocol.total_recoveries, 0)

    def test_elapsed_seconds_positive(self):
        result = self.protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
        )
        self.assertGreaterEqual(result.elapsed_seconds, 0)


# =====================================================================
# 8. RecoveryController PP-aware integration
# =====================================================================

class TestControllerPPIntegration(unittest.TestCase):
    """Test RecoveryController's PP>1 features."""

    def setUp(self):
        from megatron.core.transformer.moe.recovery_controller import (
            RecoveryController,
            RecoveryPhase,
        )
        self.RecoveryPhase = RecoveryPhase
        self.ctrl = RecoveryController()

    def test_stage_safe_recovery_fn_registered(self):
        mock_fn = MagicMock()
        self.ctrl.register_callbacks(stage_safe_recovery_fn=mock_fn)
        self.assertIs(self.ctrl._stage_safe_recovery_fn, mock_fn)

    def test_pipeline_stage_failure_mid_iteration(self):
        """Mid-iteration PP failure → ROLLBACK_PENDING → PENDING_GROUP_REPAIR."""
        self.ctrl.on_pipeline_stage_failure(
            failed_stage=1,
            failed_rank=2,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
            mid_iteration=True,
        )
        self.assertEqual(self.ctrl.phase, self.RecoveryPhase.PENDING_GROUP_REPAIR)
        self.assertTrue(self.ctrl.inflight_microbatches_invalidated)

    def test_pipeline_stage_failure_not_mid_iteration(self):
        """Non-mid-iteration PP failure → PENDING_GROUP_REPAIR directly."""
        self.ctrl.on_pipeline_stage_failure(
            failed_stage=1,
            failed_rank=2,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
            mid_iteration=False,
        )
        self.assertEqual(self.ctrl.phase, self.RecoveryPhase.PENDING_GROUP_REPAIR)

    def test_pipeline_rebinding_phase(self):
        """Full PP>1 recovery goes through PIPELINE_REBINDING."""
        mock_repair = MagicMock()
        self.ctrl.register_callbacks(pipeline_stage_repair_fn=mock_repair)

        # Drive through the state machine
        self.ctrl.on_pipeline_stage_failure(
            failed_stage=1,
            failed_rank=2,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
            mid_iteration=True,
        )
        self.ctrl.on_replacement_assigned(
            failed_rank=2, replacement_rank=5, step=101,
        )
        self.ctrl.on_replacement_ready(failed_rank=2, step=102)
        repaired = self.ctrl.before_iteration(step=103)

        self.assertTrue(repaired)
        mock_repair.assert_called_once()
        # Should be in REINTEGRATED now
        self.assertEqual(self.ctrl.phase, self.RecoveryPhase.REINTEGRATED)

    def test_fault_record_has_pp_fields(self):
        self.ctrl.on_pipeline_stage_failure(
            failed_stage=2,
            failed_rank=4,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
        )
        record = self.ctrl.get_fault_record(4)
        self.assertIsNotNone(record)
        self.assertEqual(record.failed_stage, 2)
        self.assertEqual(record.pp_group_ranks, [0, 2, 4, 6])


# =====================================================================
# 9. End-to-end: protocol + controller consistency
# =====================================================================

class TestEndToEndConsistency(unittest.TestCase):
    """Test that the protocol and controller produce consistent outcomes."""

    def test_protocol_and_controller_same_replacement(self):
        """Both protocol and controller agree on the replacement rank."""
        from megatron.core.transformer.moe.recovery_controller import (
            RecoveryController,
        )

        ctrl = RecoveryController()
        protocol = StageSafeRecoveryProtocol()

        # Controller path
        ctrl.on_pipeline_stage_failure(
            failed_stage=1, failed_rank=2, step=100,
            pp_group_ranks=[0, 2, 4, 6],
        )
        ctrl.on_replacement_assigned(failed_rank=2, replacement_rank=5, step=101)
        ctrl_record = ctrl.get_fault_record(2)

        # Protocol path
        result = protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
        )

        self.assertEqual(ctrl_record.replacement_rank, result.replacement_rank)
        self.assertEqual(ctrl_record.failed_rank, result.failed_rank)

    def test_new_pp_ranks_consistent(self):
        """compute_new_pp_ranks produces correct ranks for controller."""
        pp_ranks = [0, 2, 4, 6]
        new_ranks = compute_new_pp_ranks(pp_ranks, 2, 5)
        self.assertEqual(new_ranks, [0, 5, 4, 6])
        # Stage index preserved
        self.assertEqual(identify_failed_stage(2, pp_ranks), 1)
        self.assertEqual(new_ranks[1], 5)  # Same stage, new rank


# =====================================================================
# 10. Phase transition validation
# =====================================================================

class TestPhaseTransitionValidation(unittest.TestCase):
    """Test that invalid phase transitions are rejected."""

    def test_invalid_transition_raises(self):
        protocol = StageSafeRecoveryProtocol()
        with self.assertRaises(ValueError):
            protocol._transition(StageRecoveryPhase.ROLLBACK_COMPLETED)

    def test_valid_transition_succeeds(self):
        protocol = StageSafeRecoveryProtocol()
        protocol._transition(StageRecoveryPhase.FAILURE_DETECTED)
        self.assertEqual(protocol.phase, StageRecoveryPhase.FAILURE_DETECTED)


# =====================================================================
# 11. Callback argument verification
# =====================================================================

class TestCallbackArguments(unittest.TestCase):
    """Verify that callbacks receive the correct arguments."""

    def test_invalidate_receives_correct_args(self):
        protocol = StageSafeRecoveryProtocol()
        received = {}

        def capture(**kw):
            received.update(kw)

        protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
            invalidate_iteration_fn=capture,
        )

        self.assertEqual(received["step"], 100)
        self.assertEqual(received["failed_rank"], 2)
        self.assertEqual(received["failed_stage"], 1)

    def test_group_repair_receives_pp_ranks(self):
        protocol = StageSafeRecoveryProtocol()
        received = {}

        def capture(**kw):
            received.update(kw)

        protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
            group_repair_fn=capture,
        )

        self.assertEqual(received["pp_group_ranks"], [0, 2, 4, 6])
        self.assertEqual(received["failed_rank"], 2)
        self.assertEqual(received["replacement_rank"], 5)

    def test_p2p_rebind_receives_pp_ranks(self):
        protocol = StageSafeRecoveryProtocol()
        received = {}

        def capture(**kw):
            received.update(kw)

        protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
            p2p_rebind_fn=capture,
        )

        self.assertEqual(received["pp_group_ranks"], [0, 2, 4, 6])

    def test_convergence_receives_path(self):
        protocol = StageSafeRecoveryProtocol()
        received = {}

        def capture(**kw):
            received.update(kw)

        protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
            recovery_path="HYBRID_RECOVERY",
            convergence_fn=capture,
        )

        self.assertEqual(received["path"], "HYBRID_RECOVERY")

    def test_expert_restore_receives_expert_ids(self):
        protocol = StageSafeRecoveryProtocol()
        received = {}

        def capture(**kw):
            received.update(kw)

        protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
            expert_ids=[4, 5, 6, 7],
            recovery_path="HYBRID_RECOVERY",
            expert_restore_fn=capture,
        )

        self.assertEqual(received["expert_ids"], [4, 5, 6, 7])


# =====================================================================
# 12. Multiple sequential recoveries
# =====================================================================

class TestMultipleRecoveries(unittest.TestCase):
    """Test that the protocol can handle multiple sequential recoveries."""

    def test_two_sequential_recoveries(self):
        protocol = StageSafeRecoveryProtocol()

        r1 = protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
        )
        self.assertTrue(r1.success)

        # Second recovery with updated PP ranks
        r2 = protocol.execute(
            failed_rank=4,
            failed_stage=2,
            replacement_rank=7,
            step=200,
            pp_group_ranks=[0, 5, 4, 6],  # rank 2 already replaced by 5
        )
        self.assertTrue(r2.success)

        self.assertEqual(protocol.total_recoveries, 2)
        self.assertEqual(protocol.total_successes, 2)

    def test_recovery_after_failure(self):
        """A failed recovery doesn't prevent subsequent recoveries."""
        protocol = StageSafeRecoveryProtocol()

        def bad_fn(**kw):
            raise RuntimeError("fatal")

        # First recovery fails at rollback
        r1 = protocol.execute(
            failed_rank=2,
            failed_stage=1,
            replacement_rank=5,
            step=100,
            pp_group_ranks=[0, 2, 4, 6],
            rollback_fn=bad_fn,
        )
        self.assertFalse(r1.success)  # has errors

        # Second recovery succeeds
        r2 = protocol.execute(
            failed_rank=4,
            failed_stage=2,
            replacement_rank=7,
            step=200,
            pp_group_ranks=[0, 2, 4, 6],
        )
        self.assertTrue(r2.success)


if __name__ == '__main__':
    unittest.main()

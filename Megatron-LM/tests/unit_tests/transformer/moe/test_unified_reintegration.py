# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# BSR-MoE: Tests for Unified Reintegration (Step 8)

import sys
import time
import unittest
from unittest.mock import MagicMock, patch, call

import torch

import os

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from megatron.core.transformer.moe.unified_reintegration import (
    PostRecoveryConvergence,
    ConvergenceResult,
    RecoveryPath,
    get_post_recovery_convergence,
    clear_post_recovery_convergence,
)


# ===================================================================
# 1. RecoveryPath enum
# ===================================================================

class TestRecoveryPath(unittest.TestCase):

    def test_values(self):
        self.assertEqual(RecoveryPath.CHECKPOINT_RESTART, 1)
        self.assertEqual(RecoveryPath.HYBRID_RECOVERY, 2)

    def test_names(self):
        self.assertEqual(RecoveryPath.CHECKPOINT_RESTART.name, "CHECKPOINT_RESTART")
        self.assertEqual(RecoveryPath.HYBRID_RECOVERY.name, "HYBRID_RECOVERY")


# ===================================================================
# 2. ConvergenceResult
# ===================================================================

class TestConvergenceResult(unittest.TestCase):

    def test_default_success(self):
        r = ConvergenceResult(path=RecoveryPath.HYBRID_RECOVERY)
        self.assertTrue(r.success)
        self.assertEqual(r.errors, [])

    def test_failure_with_errors(self):
        r = ConvergenceResult(path=RecoveryPath.CHECKPOINT_RESTART)
        r.errors.append("something broke")
        self.assertFalse(r.success)

    def test_to_dict(self):
        r = ConvergenceResult(
            path=RecoveryPath.CHECKPOINT_RESTART,
            failed_rank=2, replacement_rank=5, step=100,
            experts_marked_stale=4,
            consistency_verified=True,
            preferential_routing_activated=2,
            elapsed_seconds=0.123456,
        )
        d = r.to_dict()
        self.assertEqual(d["path"], "CHECKPOINT_RESTART")
        self.assertEqual(d["failed_rank"], 2)
        self.assertEqual(d["experts_marked_stale"], 4)
        self.assertTrue(d["success"])
        self.assertAlmostEqual(d["elapsed_seconds"], 0.1235, places=3)


# ===================================================================
# 3. PostRecoveryConvergence — with mock callbacks
# ===================================================================

class TestPostRecoveryConvergenceWithMocks(unittest.TestCase):
    """Test execute() with injected callbacks so no real modules are needed."""

    def _make_convergence(self, config=None):
        return PostRecoveryConvergence(config=config)

    def test_all_steps_called_hybrid(self):
        """All 5 steps execute for hybrid path when enabled."""
        mock_mark = MagicMock(return_value=4)
        mock_verify = MagicMock(return_value=[])
        mock_pref = MagicMock(return_value=3)
        mock_opt = MagicMock(return_value=2)
        mock_two = MagicMock(return_value=True)

        cfg = MagicMock()
        cfg.moe_bsr_preferential_routing = True
        cfg.moe_bsr_expert_opt_restore = True
        cfg.moe_bsr_defer_optimizer_load = True
        cfg.moe_bsr_weights_first_recovery = True

        conv = self._make_convergence(config=cfg)
        result = conv.execute(
            path=RecoveryPath.HYBRID_RECOVERY,
            failed_rank=1, replacement_rank=5,
            expert_ids=[0, 1, 2, 3],
            restored_experts=[(1, 0), (1, 1), (2, 0)],
            step=100,
            mark_stale_fn=mock_mark,
            verify_consistency_fn=mock_verify,
            activate_preferential_fn=mock_pref,
            submit_optimizer_fn=mock_opt,
            drive_two_phase_fn=mock_two,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.experts_marked_stale, 4)
        self.assertTrue(result.consistency_verified)
        self.assertEqual(result.preferential_routing_activated, 3)
        self.assertEqual(result.optimizer_loads_submitted, 2)
        self.assertTrue(result.two_phase_driven)

        mock_mark.assert_called_once_with([0, 1, 2, 3], 100)
        mock_verify.assert_called_once()
        mock_pref.assert_called_once()
        mock_opt.assert_called_once()
        mock_two.assert_called_once()

    def test_all_steps_called_checkpoint_restart(self):
        """Checkpoint restart path: optimizer submission is SKIPPED."""
        mock_mark = MagicMock(return_value=4)
        mock_verify = MagicMock(return_value=[])
        mock_pref = MagicMock(return_value=2)
        mock_opt = MagicMock(return_value=99)  # should NOT be called
        mock_two = MagicMock(return_value=True)

        cfg = MagicMock()
        cfg.moe_bsr_preferential_routing = True
        cfg.moe_bsr_expert_opt_restore = True
        cfg.moe_bsr_defer_optimizer_load = True
        cfg.moe_bsr_weights_first_recovery = True

        conv = self._make_convergence(config=cfg)
        result = conv.execute(
            path=RecoveryPath.CHECKPOINT_RESTART,
            failed_rank=1, replacement_rank=5,
            expert_ids=[0, 1],
            restored_experts=[(1, 0), (1, 1)],
            step=200,
            mark_stale_fn=mock_mark,
            verify_consistency_fn=mock_verify,
            activate_preferential_fn=mock_pref,
            submit_optimizer_fn=mock_opt,
            drive_two_phase_fn=mock_two,
        )

        self.assertTrue(result.success)
        # Optimizer submission skipped for checkpoint restart
        self.assertEqual(result.optimizer_loads_submitted, 0)
        mock_opt.assert_not_called()

    def test_disabled_features_not_called(self):
        """When config flags are False, optional steps are skipped."""
        mock_pref = MagicMock(return_value=1)
        mock_opt = MagicMock(return_value=1)
        mock_two = MagicMock(return_value=True)

        cfg = MagicMock()
        cfg.moe_bsr_preferential_routing = False
        cfg.moe_bsr_expert_opt_restore = False
        cfg.moe_bsr_weights_first_recovery = False

        conv = self._make_convergence(config=cfg)
        result = conv.execute(
            path=RecoveryPath.HYBRID_RECOVERY,
            failed_rank=1, replacement_rank=5,
            expert_ids=[0],
            restored_experts=[(1, 0)],
            step=50,
            mark_stale_fn=MagicMock(return_value=1),
            verify_consistency_fn=MagicMock(return_value=[]),
            activate_preferential_fn=mock_pref,
            submit_optimizer_fn=mock_opt,
            drive_two_phase_fn=mock_two,
        )

        mock_pref.assert_not_called()
        mock_opt.assert_not_called()
        mock_two.assert_not_called()
        self.assertEqual(result.preferential_routing_activated, 0)
        self.assertEqual(result.optimizer_loads_submitted, 0)
        self.assertFalse(result.two_phase_driven)

    def test_empty_expert_ids(self):
        """No expert IDs → mark_stale returns 0, no error."""
        mock_mark = MagicMock()
        conv = self._make_convergence()
        result = conv.execute(
            path=RecoveryPath.HYBRID_RECOVERY,
            failed_rank=1, replacement_rank=5,
            expert_ids=[],
            restored_experts=[],
            step=10,
            mark_stale_fn=mock_mark,
            verify_consistency_fn=MagicMock(return_value=[]),
        )
        self.assertEqual(result.experts_marked_stale, 0)
        mock_mark.assert_not_called()
        self.assertTrue(result.success)

    def test_mark_stale_error_recorded(self):
        """Error in mark_stale is recorded but doesn't abort."""
        def failing_mark(ids, step):
            raise RuntimeError("health manager unavailable")

        conv = self._make_convergence()
        result = conv.execute(
            path=RecoveryPath.HYBRID_RECOVERY,
            failed_rank=1, replacement_rank=5,
            expert_ids=[0, 1],
            restored_experts=[(1, 0)],
            step=10,
            mark_stale_fn=failing_mark,
            verify_consistency_fn=MagicMock(return_value=[]),
        )
        self.assertFalse(result.success)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("mark_stale_runnable failed", result.errors[0])

    def test_consistency_issues_recorded(self):
        """Consistency issues are recorded but not as errors."""
        mock_verify = MagicMock(return_value=["mismatch on expert 3"])

        conv = self._make_convergence()
        result = conv.execute(
            path=RecoveryPath.HYBRID_RECOVERY,
            failed_rank=1, replacement_rank=5,
            expert_ids=[0],
            restored_experts=[],
            step=10,
            mark_stale_fn=MagicMock(return_value=1),
            verify_consistency_fn=mock_verify,
        )
        # Consistency issues are warnings, not errors
        self.assertTrue(result.success)
        self.assertFalse(result.consistency_verified)
        self.assertEqual(result.consistency_issues, ["mismatch on expert 3"])

    def test_elapsed_seconds_tracked(self):
        conv = self._make_convergence()
        result = conv.execute(
            path=RecoveryPath.HYBRID_RECOVERY,
            failed_rank=1, replacement_rank=5,
            expert_ids=[0],
            restored_experts=[],
            step=10,
            mark_stale_fn=MagicMock(return_value=1),
            verify_consistency_fn=MagicMock(return_value=[]),
        )
        self.assertGreaterEqual(result.elapsed_seconds, 0.0)


# ===================================================================
# 4. Both paths produce same interface
# ===================================================================

class TestBothPathsSameInterface(unittest.TestCase):
    """Verify that restart and hybrid paths produce ConvergenceResult
    with the same structure."""

    def test_same_result_fields(self):
        conv = PostRecoveryConvergence()
        mock_mark = MagicMock(return_value=2)
        mock_verify = MagicMock(return_value=[])

        kwargs = dict(
            failed_rank=1, replacement_rank=5,
            expert_ids=[0, 1],
            restored_experts=[(1, 0), (1, 1)],
            step=100,
            mark_stale_fn=mock_mark,
            verify_consistency_fn=mock_verify,
        )

        r1 = conv.execute(path=RecoveryPath.CHECKPOINT_RESTART, **kwargs)
        r2 = conv.execute(path=RecoveryPath.HYBRID_RECOVERY, **kwargs)

        # Both have the same dict keys
        d1 = r1.to_dict()
        d2 = r2.to_dict()
        self.assertEqual(set(d1.keys()), set(d2.keys()))

        # Path field differs
        self.assertEqual(d1["path"], "CHECKPOINT_RESTART")
        self.assertEqual(d2["path"], "HYBRID_RECOVERY")

    def test_both_paths_mark_stale(self):
        """Both paths call mark_stale with the same expert_ids."""
        call_log = []
        def tracking_mark(ids, step):
            call_log.append(("mark_stale", sorted(ids), step))
            return len(ids)

        conv = PostRecoveryConvergence()
        kwargs = dict(
            failed_rank=1, replacement_rank=5,
            expert_ids=[2, 3],
            restored_experts=[(1, 2), (1, 3)],
            step=50,
            mark_stale_fn=tracking_mark,
            verify_consistency_fn=MagicMock(return_value=[]),
        )

        conv.execute(path=RecoveryPath.CHECKPOINT_RESTART, **kwargs)
        conv.execute(path=RecoveryPath.HYBRID_RECOVERY, **kwargs)

        self.assertEqual(len(call_log), 2)
        self.assertEqual(call_log[0], ("mark_stale", [2, 3], 50))
        self.assertEqual(call_log[1], ("mark_stale", [2, 3], 50))


# ===================================================================
# 5. Dispatch topology consistency verification
# ===================================================================

class TestDispatchConsistencyVerification(unittest.TestCase):
    """Test that post-recovery convergence detects inconsistencies."""

    def test_no_issues_means_consistent(self):
        conv = PostRecoveryConvergence()
        result = conv.execute(
            path=RecoveryPath.HYBRID_RECOVERY,
            failed_rank=1, replacement_rank=5,
            expert_ids=[0],
            restored_experts=[],
            step=10,
            mark_stale_fn=MagicMock(return_value=1),
            verify_consistency_fn=lambda: [],
        )
        self.assertTrue(result.consistency_verified)

    def test_issues_detected(self):
        conv = PostRecoveryConvergence()
        result = conv.execute(
            path=RecoveryPath.HYBRID_RECOVERY,
            failed_rank=1, replacement_rank=5,
            expert_ids=[0],
            restored_experts=[],
            step=10,
            mark_stale_fn=MagicMock(return_value=1),
            verify_consistency_fn=lambda: [
                "expert 3 routed to failed rank",
                "health mask stale",
            ],
        )
        self.assertFalse(result.consistency_verified)
        self.assertEqual(len(result.consistency_issues), 2)

    def test_verify_exception_non_fatal(self):
        """Verification exception → issue recorded, not an error."""
        def exploding_verify():
            raise RuntimeError("topology manager not initialized")

        conv = PostRecoveryConvergence()
        result = conv.execute(
            path=RecoveryPath.HYBRID_RECOVERY,
            failed_rank=1, replacement_rank=5,
            expert_ids=[0],
            restored_experts=[],
            step=10,
            mark_stale_fn=MagicMock(return_value=1),
            verify_consistency_fn=exploding_verify,
        )
        # Exception in verification is a consistency issue, not a hard error
        self.assertFalse(result.consistency_verified)
        self.assertTrue(result.success)  # still "success" overall


# ===================================================================
# 6. Two-phase driving logic
# ===================================================================

class TestTwoPhaseDriving(unittest.TestCase):
    """Test that two-phase state machine is correctly driven."""

    def test_checkpoint_restart_drives_to_fully_recovered(self):
        """Checkpoint restart: weights + optimizer loaded inline →
        skip_optimizer_phase called."""
        drive_calls = []

        def mock_drive(experts, step):
            drive_calls.append(("drive", experts, step))
            return True

        cfg = MagicMock()
        cfg.moe_bsr_weights_first_recovery = True
        cfg.moe_bsr_preferential_routing = False
        cfg.moe_bsr_expert_opt_restore = False

        conv = PostRecoveryConvergence(config=cfg)
        result = conv.execute(
            path=RecoveryPath.CHECKPOINT_RESTART,
            failed_rank=1, replacement_rank=5,
            expert_ids=[0, 1],
            restored_experts=[(1, 0), (1, 1)],
            step=100,
            mark_stale_fn=MagicMock(return_value=2),
            verify_consistency_fn=MagicMock(return_value=[]),
            drive_two_phase_fn=mock_drive,
        )
        self.assertTrue(result.two_phase_driven)
        self.assertEqual(len(drive_calls), 1)

    def test_two_phase_disabled_skipped(self):
        cfg = MagicMock()
        cfg.moe_bsr_weights_first_recovery = False
        cfg.moe_bsr_preferential_routing = False
        cfg.moe_bsr_expert_opt_restore = False

        mock_two = MagicMock(return_value=True)
        conv = PostRecoveryConvergence(config=cfg)
        result = conv.execute(
            path=RecoveryPath.HYBRID_RECOVERY,
            failed_rank=1, replacement_rank=5,
            expert_ids=[0],
            restored_experts=[(1, 0)],
            step=50,
            mark_stale_fn=MagicMock(return_value=1),
            verify_consistency_fn=MagicMock(return_value=[]),
            drive_two_phase_fn=mock_two,
        )
        mock_two.assert_not_called()
        self.assertFalse(result.two_phase_driven)


# ===================================================================
# 7. Global singleton
# ===================================================================

class TestGlobalSingleton(unittest.TestCase):

    def setUp(self):
        clear_post_recovery_convergence()

    def tearDown(self):
        clear_post_recovery_convergence()

    def test_get_creates(self):
        conv = get_post_recovery_convergence()
        self.assertIsInstance(conv, PostRecoveryConvergence)

    def test_get_returns_same(self):
        c1 = get_post_recovery_convergence()
        c2 = get_post_recovery_convergence()
        self.assertIs(c1, c2)

    def test_clear(self):
        c1 = get_post_recovery_convergence()
        clear_post_recovery_convergence()
        c2 = get_post_recovery_convergence()
        self.assertIsNot(c1, c2)


# ===================================================================
# 8. RecoveryController convergence callback registration
# ===================================================================

class TestControllerConvergenceCallback(unittest.TestCase):
    """Test that RecoveryController accepts and stores the callback."""

    def test_register_post_recovery_convergence_fn(self):
        from megatron.core.transformer.moe.recovery_controller import (
            RecoveryController,
        )
        ctrl = RecoveryController()
        self.assertIsNone(ctrl._post_recovery_convergence_fn)

        mock_fn = MagicMock()
        ctrl.register_callbacks(post_recovery_convergence_fn=mock_fn)
        self.assertIs(ctrl._post_recovery_convergence_fn, mock_fn)

    def test_callback_not_overwritten_when_none(self):
        from megatron.core.transformer.moe.recovery_controller import (
            RecoveryController,
        )
        ctrl = RecoveryController()
        mock_fn = MagicMock()
        ctrl.register_callbacks(post_recovery_convergence_fn=mock_fn)
        # Second call without the arg should NOT overwrite
        ctrl.register_callbacks(quarantine_fn=MagicMock())
        self.assertIs(ctrl._post_recovery_convergence_fn, mock_fn)


# ===================================================================
# 9. End-to-end: both paths through same convergence
# ===================================================================

class TestEndToEndBothPaths(unittest.TestCase):
    """Simulate both paths calling the same convergence and verify
    that the post-recovery state is identical."""

    def test_restart_and_hybrid_produce_consistent_state(self):
        """Both paths mark stale, verify, and optionally activate pref routing."""
        stale_calls = []
        verify_calls = []

        def mock_mark(ids, step):
            stale_calls.append(sorted(ids))
            return len(ids)

        def mock_verify():
            verify_calls.append(True)
            return []

        cfg = MagicMock()
        cfg.moe_bsr_preferential_routing = False
        cfg.moe_bsr_expert_opt_restore = False
        cfg.moe_bsr_weights_first_recovery = False

        conv = PostRecoveryConvergence(config=cfg)

        # Restart path
        r1 = conv.execute(
            path=RecoveryPath.CHECKPOINT_RESTART,
            failed_rank=2, replacement_rank=7,
            expert_ids=[0, 1, 2, 3],
            restored_experts=[(1, 0), (1, 1), (2, 2), (2, 3)],
            step=500,
            mark_stale_fn=mock_mark,
            verify_consistency_fn=mock_verify,
        )

        # Hybrid path
        r2 = conv.execute(
            path=RecoveryPath.HYBRID_RECOVERY,
            failed_rank=2, replacement_rank=7,
            expert_ids=[0, 1, 2, 3],
            restored_experts=[(1, 0), (1, 1), (2, 2), (2, 3)],
            step=500,
            mark_stale_fn=mock_mark,
            verify_consistency_fn=mock_verify,
        )

        # Both paths called mark_stale with same IDs
        self.assertEqual(stale_calls[0], stale_calls[1])
        # Both paths verified consistency
        self.assertEqual(len(verify_calls), 2)
        # Both succeeded
        self.assertTrue(r1.success)
        self.assertTrue(r2.success)
        # Both marked same number of experts
        self.assertEqual(r1.experts_marked_stale, r2.experts_marked_stale)

    def test_dispatch_never_hits_failed_rank(self):
        """After convergence, verify_consistency should report no issues
        about the failed rank (simulated via mock)."""
        def mock_verify_no_failed_rank():
            # Simulate: dispatch topology has been refreshed, no expert
            # is routed to the failed rank anymore
            return []

        cfg = MagicMock()
        cfg.moe_bsr_preferential_routing = False
        cfg.moe_bsr_expert_opt_restore = False
        cfg.moe_bsr_weights_first_recovery = False

        conv = PostRecoveryConvergence(config=cfg)

        for path in [RecoveryPath.CHECKPOINT_RESTART, RecoveryPath.HYBRID_RECOVERY]:
            result = conv.execute(
                path=path,
                failed_rank=3, replacement_rank=8,
                expert_ids=[0, 1],
                restored_experts=[(1, 0), (1, 1)],
                step=200,
                mark_stale_fn=MagicMock(return_value=2),
                verify_consistency_fn=mock_verify_no_failed_rank,
            )
            self.assertTrue(result.consistency_verified,
                f"Dispatch should not hit failed rank after {path.name}")

    def test_training_can_continue(self):
        """After convergence, no errors → training can continue."""
        cfg = MagicMock()
        cfg.moe_bsr_preferential_routing = False
        cfg.moe_bsr_expert_opt_restore = False
        cfg.moe_bsr_weights_first_recovery = False

        conv = PostRecoveryConvergence(config=cfg)

        for path in [RecoveryPath.CHECKPOINT_RESTART, RecoveryPath.HYBRID_RECOVERY]:
            result = conv.execute(
                path=path,
                failed_rank=1, replacement_rank=5,
                expert_ids=[0, 1, 2, 3],
                restored_experts=[(1, 0), (1, 1), (2, 2), (2, 3)],
                step=1000,
                mark_stale_fn=MagicMock(return_value=4),
                verify_consistency_fn=MagicMock(return_value=[]),
            )
            self.assertTrue(result.success,
                f"Training should be able to continue after {path.name}")
            self.assertEqual(result.errors, [])


if __name__ == "__main__":
    unittest.main()

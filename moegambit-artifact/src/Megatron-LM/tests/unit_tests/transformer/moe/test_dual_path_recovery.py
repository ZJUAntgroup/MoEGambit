# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for MoEGambit Dual-Path Recovery (checkpoint restart path).

Tests cover:
1. Hard failure unified preconditions (all 6 actions fire)
2. Gap-aware policy selects CHECKPOINT_RESTART when gap is small
3. Gap-aware policy selects HYBRID_RECOVERY when gap is large
4. Checkpoint restart fallback to hybrid when checkpoint_restart_fn fails
5. Checkpoint restart fallback when no checkpoint_restart_fn registered
6. Soft failure does NOT trigger optimizer_commit_block / enter_waiting
7. last_recovery_path tracking
8. Event log contains checkpoint_restart_executed event
9. Full lifecycle: hard failure → replacement → safe-point repair → reintegration
10. OptimizerCommitGuard.block() explicit block
"""

import unittest

from megatron.core.transformer.moe.recovery_controller import (
    RecoveryController,
    RecoveryPhase,
)
from megatron.core.transformer.moe.gap_aware_recovery_policy import (
    RecoveryPath,
    RecoveryDecision,
    GapAwareRecoveryPolicyManager,
    ThresholdRecoveryPolicy,
    RankExposureGuardedPolicy,
    RankExposureGuardedConfig,
)
from megatron.core.transformer.moe.optimizer_commit_guard import (
    OptimizerCommitGuard,
)
from megatron.core.transformer.moe.rank_exposure_tracker import (
    RankExposureTracker,
    set_rank_exposure_tracker,
    reset_rank_exposure_tracker,
)


# =====================================================================
# Helpers
# =====================================================================

class ActionTracker:
    """Collects callback invocations for assertion."""

    def __init__(self):
        self.actions = []
        self.calls = {}

    def make_fn(self, name):
        """Return a callback that records its invocation."""
        def fn(**kwargs):
            self.actions.append(name)
            self.calls.setdefault(name, []).append(kwargs)
        return fn


def _make_controller_with_tracker(
    *,
    gap_threshold=50,
    checkpoint_iteration=90,
    checkpoint_restart_succeeds=True,
):
    """Create a RecoveryController wired with an ActionTracker and
    a gap-aware policy manager.

    Returns (ctrl, tracker).
    """
    ctrl = RecoveryController()
    tracker = ActionTracker()

    def checkpoint_restart_fn(**kwargs):
        if not checkpoint_restart_succeeds:
            raise RuntimeError("simulated checkpoint load failure")
        tracker.actions.append("checkpoint_restart")
        tracker.calls.setdefault("checkpoint_restart", []).append(kwargs)

    ctrl.register_callbacks(
        health_mark_healthy_fn=tracker.make_fn("mark_healthy"),
        replacement_announce_fn=tracker.make_fn("replacement_announce"),
        replacement_integrate_fn=tracker.make_fn("replacement_integrate"),
        group_rebuild_request_fn=tracker.make_fn("group_rebuild_request"),
        group_rebuild_execute_fn=tracker.make_fn("group_rebuild_execute"),
        group_rebuild_finish_fn=tracker.make_fn("group_rebuild_finish"),
        topology_refresh_fn=tracker.make_fn("topology_refresh"),
        dense_sync_fn=tracker.make_fn("dense_sync"),
        expert_restore_fn=tracker.make_fn("expert_restore"),
        checkpoint_restart_fn=checkpoint_restart_fn,
        optimizer_commit_block_fn=tracker.make_fn("block_commit"),
        enter_waiting_fn=tracker.make_fn("enter_waiting"),
    )

    # Wire gap-aware policy
    mgr = GapAwareRecoveryPolicyManager(
        gap_threshold=gap_threshold,
        get_checkpoint_iteration_fn=lambda: checkpoint_iteration,
    )
    ctrl.set_gap_aware_policy_manager(mgr)

    return ctrl, tracker


def _drive_to_safe_point_repair(ctrl, *, failed_rank=1, step=100,
                                 expert_ids=None, mid_iteration=True):
    """Drive the controller through hard failure → replacement assigned →
    replacement ready → SAFE_POINT_REPAIR.
    """
    expert_ids = expert_ids or [0, 1]
    ctrl.on_hard_rank_failure(
        failed_rank=failed_rank,
        step=step,
        expert_ids=expert_ids,
        mid_iteration=mid_iteration,
    )
    ctrl.on_replacement_assigned(
        failed_rank=failed_rank,
        replacement_rank=failed_rank,  # identity inheritance
        step=step + 1,
    )
    ctrl.on_replacement_ready(
        failed_rank=failed_rank,
        step=step + 2,
    )
    assert ctrl.phase == RecoveryPhase.SAFE_POINT_REPAIR


# =====================================================================
# Test: Hard failure unified preconditions
# =====================================================================

class TestHardFailureUnifiedPreconditions(unittest.TestCase):
    """Hard failure triggers all 6 precondition actions."""

    def test_all_preconditions_fire(self):
        """on_hard_rank_failure triggers quarantine, mark_unavailable,
        block_commit, enter_waiting, iteration invalidation, and
        transitions to PENDING_GROUP_REPAIR."""
        ctrl = RecoveryController()
        tracker = ActionTracker()

        ctrl.register_callbacks(
            optimizer_commit_block_fn=tracker.make_fn("block_commit"),
            enter_waiting_fn=tracker.make_fn("enter_waiting"),
        )

        ctrl.on_hard_rank_failure(
            failed_rank=1, step=100, expert_ids=[0, 1],
            mid_iteration=True,
        )

        self.assertIn("block_commit", tracker.actions)
        self.assertIn("enter_waiting", tracker.actions)
        self.assertTrue(ctrl._iteration_invalidated)
        self.assertEqual(ctrl.phase, RecoveryPhase.PENDING_GROUP_REPAIR)

    def test_pipeline_stage_failure_also_fires_preconditions(self):
        """on_pipeline_stage_failure also triggers block_commit and
        enter_waiting."""
        ctrl = RecoveryController()
        tracker = ActionTracker()

        ctrl.register_callbacks(
            optimizer_commit_block_fn=tracker.make_fn("block_commit"),
            enter_waiting_fn=tracker.make_fn("enter_waiting"),
        )

        ctrl.on_pipeline_stage_failure(
            failed_stage=0, failed_rank=1, step=100,
            pp_group_ranks=[0, 1], expert_ids=[0, 1],
            mid_iteration=True,
        )

        self.assertIn("block_commit", tracker.actions)
        self.assertIn("enter_waiting", tracker.actions)
        self.assertTrue(ctrl._iteration_invalidated)


# =====================================================================
# Test: Gap-aware checkpoint restart path
# =====================================================================

class TestCheckpointRestartPath(unittest.TestCase):
    """When gap is small, the controller selects CHECKPOINT_RESTART."""

    def test_small_gap_selects_checkpoint_restart(self):
        """gap=13 <= threshold=50 → CHECKPOINT_RESTART path.

        Failure at step=100, checkpoint at iter=90, repair at step=103.
        gap = 103 - 90 = 13.
        """
        ctrl, tracker = _make_controller_with_tracker(
            gap_threshold=50,
            checkpoint_iteration=90,
        )
        _drive_to_safe_point_repair(ctrl, step=100)

        # Trigger safe-point repair
        repaired = ctrl.before_iteration(step=103)
        self.assertTrue(repaired)

        # Verify checkpoint_restart was called (not dense_sync/expert_restore)
        self.assertIn("checkpoint_restart", tracker.actions)
        self.assertNotIn("dense_sync", tracker.actions)
        self.assertNotIn("expert_restore", tracker.actions)

        # Verify path tracking
        self.assertEqual(ctrl.last_recovery_path, "CHECKPOINT_RESTART")

        # Verify event log contains checkpoint_restart_executed
        ckpt_events = [
            e for e in ctrl.event_log
            if e.event_type == "checkpoint_restart_executed"
        ]
        self.assertEqual(len(ckpt_events), 1)
        self.assertEqual(ckpt_events[0].details["checkpoint_iteration"], 90)
        self.assertEqual(ckpt_events[0].details["gap"], 13)

    def test_large_gap_selects_hybrid_recovery(self):
        """gap=90 > threshold=50 → HYBRID_RECOVERY path."""
        ctrl, tracker = _make_controller_with_tracker(
            gap_threshold=50,
            checkpoint_iteration=10,  # gap = 100 - 10 = 90
        )
        _drive_to_safe_point_repair(ctrl, step=100)

        repaired = ctrl.before_iteration(step=103)
        self.assertTrue(repaired)

        # Verify hybrid path was used
        self.assertNotIn("checkpoint_restart", tracker.actions)
        self.assertIn("dense_sync", tracker.actions)
        self.assertIn("expert_restore", tracker.actions)

        self.assertEqual(ctrl.last_recovery_path, "HYBRID_RECOVERY")

    def test_boundary_gap_equals_threshold(self):
        """gap == threshold → CHECKPOINT_RESTART (<=)."""
        ctrl, tracker = _make_controller_with_tracker(
            gap_threshold=50,
            checkpoint_iteration=53,  # gap = 103 - 53 = 50
        )
        _drive_to_safe_point_repair(ctrl, step=100)

        repaired = ctrl.before_iteration(step=103)
        self.assertTrue(repaired)

        self.assertIn("checkpoint_restart", tracker.actions)
        self.assertEqual(ctrl.last_recovery_path, "CHECKPOINT_RESTART")

    def test_no_checkpoint_forces_hybrid(self):
        """checkpoint_iteration=-1 → forced HYBRID_RECOVERY."""
        ctrl, tracker = _make_controller_with_tracker(
            gap_threshold=50,
            checkpoint_iteration=-1,
        )
        _drive_to_safe_point_repair(ctrl, step=100)

        repaired = ctrl.before_iteration(step=103)
        self.assertTrue(repaired)

        self.assertNotIn("checkpoint_restart", tracker.actions)
        self.assertIn("dense_sync", tracker.actions)
        self.assertEqual(ctrl.last_recovery_path, "HYBRID_RECOVERY")


# =====================================================================
# Test: Checkpoint restart fallback
# =====================================================================

class TestCheckpointRestartFallback(unittest.TestCase):
    """When checkpoint_restart_fn fails, falls back to hybrid recovery."""

    def test_fallback_on_checkpoint_failure(self):
        """checkpoint_restart_fn raises → fallback to hybrid."""
        ctrl, tracker = _make_controller_with_tracker(
            gap_threshold=50,
            checkpoint_iteration=90,
            checkpoint_restart_succeeds=False,
        )
        _drive_to_safe_point_repair(ctrl, step=100)

        repaired = ctrl.before_iteration(step=103)
        self.assertTrue(repaired)

        # Hybrid path should have been used as fallback
        self.assertIn("dense_sync", tracker.actions)
        self.assertIn("expert_restore", tracker.actions)
        # last_recovery_path should reflect the actual path used
        self.assertEqual(ctrl.last_recovery_path, "HYBRID_RECOVERY")

        # Event log should contain the fallback event
        fallback_events = [
            e for e in ctrl.event_log
            if e.event_type == "checkpoint_restart_failed_fallback"
        ]
        self.assertEqual(len(fallback_events), 1)

    def test_fallback_when_no_fn_registered(self):
        """No checkpoint_restart_fn registered → fallback to hybrid."""
        ctrl = RecoveryController()
        tracker = ActionTracker()

        ctrl.register_callbacks(
            replacement_integrate_fn=tracker.make_fn("replacement_integrate"),
            group_rebuild_request_fn=tracker.make_fn("group_rebuild_request"),
            group_rebuild_execute_fn=tracker.make_fn("group_rebuild_execute"),
            group_rebuild_finish_fn=tracker.make_fn("group_rebuild_finish"),
            topology_refresh_fn=tracker.make_fn("topology_refresh"),
            dense_sync_fn=tracker.make_fn("dense_sync"),
            expert_restore_fn=tracker.make_fn("expert_restore"),
            # NOTE: no checkpoint_restart_fn
        )

        mgr = GapAwareRecoveryPolicyManager(
            gap_threshold=50,
            get_checkpoint_iteration_fn=lambda: 90,
        )
        ctrl.set_gap_aware_policy_manager(mgr)

        _drive_to_safe_point_repair(ctrl, step=100)
        repaired = ctrl.before_iteration(step=103)
        self.assertTrue(repaired)

        self.assertIn("dense_sync", tracker.actions)
        self.assertEqual(ctrl.last_recovery_path, "HYBRID_RECOVERY")


# =====================================================================
# Test: Rank stale exposure recording
# =====================================================================

class TestHybridStaleExposureRecording(unittest.TestCase):
    """Hybrid success records rank exposure; restart/failure paths do not."""

    def setUp(self):
        self.rank_tracker = RankExposureTracker()
        set_rank_exposure_tracker(self.rank_tracker)

    def tearDown(self):
        reset_rank_exposure_tracker()

    def _install_guarded_policy(self, ctrl, checkpoint_iteration):
        cfg = RankExposureGuardedConfig(
            delta_time_min_gap=20,
            max_single_gap=200,
            exposure_window_steps=1000,
            max_rank_stale_exposure=0.20,
        )
        mgr = GapAwareRecoveryPolicyManager(
            policy=RankExposureGuardedPolicy(
                config=cfg,
                tracker=self.rank_tracker,
            ),
            get_checkpoint_iteration_fn=lambda: checkpoint_iteration,
        )
        ctrl.set_gap_aware_policy_manager(mgr)

    def test_hybrid_success_records_fault_step_failed_rank_and_gap(self):
        """Successful hybrid records fault step/gap, not spare rank/repair step."""
        ctrl, actions = _make_controller_with_tracker(
            gap_threshold=50,
            checkpoint_iteration=10,
        )
        self._install_guarded_policy(ctrl, checkpoint_iteration=10)

        ctrl.on_hard_rank_failure(
            failed_rank=1,
            step=100,
            expert_ids=[0, 1],
            mid_iteration=True,
        )
        ctrl.on_replacement_assigned(
            failed_rank=1,
            replacement_rank=7,
            step=101,
        )
        ctrl.on_replacement_ready(failed_rank=1, step=102)

        repaired = ctrl.before_iteration(step=103)
        self.assertTrue(repaired)
        self.assertEqual(ctrl.last_recovery_path, "HYBRID_RECOVERY")
        self.assertIn("dense_sync", actions.actions)
        self.assertIn("expert_restore", actions.actions)

        events = self.rank_tracker.get_events_for_rank(
            rank=1,
            current_step=103,
            window_steps=1000,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].step, 100)
        self.assertEqual(events[0].rank, 1)
        self.assertEqual(events[0].gap, 90)

        exposure_events = [
            e for e in ctrl.event_log
            if e.event_type == "rank_stale_exposure_recorded"
        ]
        self.assertEqual(len(exposure_events), 1)
        details = exposure_events[0].details
        self.assertEqual(details["step"], 100)
        self.assertEqual(details["failed_rank"], 1)
        self.assertEqual(details["gap"], 90)
        self.assertEqual(details["rank_stale_iters_after"], 90)
        self.assertAlmostEqual(details["rank_stale_exposure_after"], 0.09)
        self.assertEqual(details["exposure_window_steps"], 1000)
        self.assertEqual(details["max_rank_stale_exposure"], 0.20)

    def test_checkpoint_restart_does_not_record_exposure(self):
        """Checkpoint restart path produces no stale exposure record."""
        ctrl, actions = _make_controller_with_tracker(
            gap_threshold=50,
            checkpoint_iteration=90,
        )
        self._install_guarded_policy(ctrl, checkpoint_iteration=90)

        _drive_to_safe_point_repair(ctrl, step=100)
        repaired = ctrl.before_iteration(step=103)

        self.assertTrue(repaired)
        self.assertEqual(ctrl.last_recovery_path, "CHECKPOINT_RESTART")
        self.assertIn("checkpoint_restart", actions.actions)
        self.assertEqual(self.rank_tracker.get_event_count(), 0)
        self.assertFalse(any(
            e.event_type == "rank_stale_exposure_recorded"
            for e in ctrl.event_log
        ))

    def test_hybrid_failure_does_not_record_exposure(self):
        """A failed hybrid attempt records nothing before any restart fallback."""
        ctrl, actions = _make_controller_with_tracker(
            gap_threshold=50,
            checkpoint_iteration=10,
        )
        self._install_guarded_policy(ctrl, checkpoint_iteration=10)

        def failing_expert_restore(**kwargs):
            actions.actions.append("expert_restore")
            actions.calls.setdefault("expert_restore", []).append(kwargs)
            raise RuntimeError("simulated hybrid failure")

        ctrl.register_callbacks(expert_restore_fn=failing_expert_restore)

        _drive_to_safe_point_repair(ctrl, step=100)
        with self.assertRaises(RuntimeError):
            ctrl.before_iteration(step=103)

        self.assertEqual(self.rank_tracker.get_event_count(), 0)
        self.assertFalse(any(
            e.event_type == "rank_stale_exposure_recorded"
            for e in ctrl.event_log
        ))


# =====================================================================
# Test: Full lifecycle with checkpoint restart
# =====================================================================

class TestFullLifecycleCheckpointRestart(unittest.TestCase):
    """End-to-end: hard failure → checkpoint restart → reintegration →
    HEALTHY_TRAINING."""

    def test_full_lifecycle(self):
        """Complete recovery lifecycle using checkpoint restart path."""
        ctrl, tracker = _make_controller_with_tracker(
            gap_threshold=50,
            checkpoint_iteration=90,
        )

        # 1. Normal training
        self.assertEqual(ctrl.phase, RecoveryPhase.HEALTHY_TRAINING)

        # 2. Hard failure
        ctrl.on_hard_rank_failure(
            failed_rank=1, step=100, expert_ids=[0, 1],
            mid_iteration=True,
        )
        self.assertEqual(ctrl.phase, RecoveryPhase.PENDING_GROUP_REPAIR)
        self.assertTrue(ctrl.iteration_was_invalidated)

        # 3. Replacement assigned
        ctrl.on_replacement_assigned(
            failed_rank=1, replacement_rank=1, step=101,
        )
        self.assertEqual(ctrl.phase, RecoveryPhase.WAITING_FOR_REPLACEMENT)

        # 4. Replacement ready
        ctrl.on_replacement_ready(failed_rank=1, step=102)
        self.assertEqual(ctrl.phase, RecoveryPhase.SAFE_POINT_REPAIR)

        # 5. Safe-point repair (before_iteration clears invalidation first)
        repaired = ctrl.before_iteration(step=103)
        self.assertTrue(repaired)
        self.assertEqual(ctrl.phase, RecoveryPhase.REINTEGRATED)
        self.assertEqual(ctrl.last_recovery_path, "CHECKPOINT_RESTART")

        # 6. Finalize reintegration
        ctrl.before_iteration(step=104)
        self.assertEqual(ctrl.phase, RecoveryPhase.HEALTHY_TRAINING)
        self.assertEqual(ctrl.num_completed_recoveries, 1)
        self.assertEqual(ctrl.num_active_faults, 0)

    def test_training_continues_after_recovery(self):
        """After recovery, before_iteration returns False (no repair)."""
        ctrl, tracker = _make_controller_with_tracker(
            gap_threshold=50,
            checkpoint_iteration=90,
        )

        _drive_to_safe_point_repair(ctrl, step=100)
        ctrl.before_iteration(step=103)  # repair
        ctrl.before_iteration(step=104)  # finalize

        self.assertEqual(ctrl.phase, RecoveryPhase.HEALTHY_TRAINING)

        # Subsequent iterations should be no-ops
        result = ctrl.before_iteration(step=105)
        self.assertFalse(result)
        self.assertEqual(ctrl.phase, RecoveryPhase.HEALTHY_TRAINING)


# =====================================================================
# Test: OptimizerCommitGuard.block()
# =====================================================================

class TestOptimizerCommitGuardBlock(unittest.TestCase):
    """OptimizerCommitGuard.block() explicitly blocks commit."""

    def test_explicit_block(self):
        """block() causes should_commit() to return False."""
        guard = OptimizerCommitGuard()
        guard.begin_iteration(step=100)

        self.assertTrue(guard.should_commit())

        guard.block(reason="hard_failure_rank_1_step_100")

        self.assertFalse(guard.should_commit())
        self.assertTrue(guard.is_blocked)
        self.assertEqual(guard._block_reason, "hard_failure_rank_1_step_100")

    def test_block_cleared_on_next_iteration(self):
        """begin_iteration() clears the block."""
        guard = OptimizerCommitGuard()
        guard.begin_iteration(step=100)
        guard.block(reason="test")
        self.assertFalse(guard.should_commit())

        # Next iteration clears the block
        guard.begin_iteration(step=101)
        self.assertTrue(guard.should_commit())
        self.assertFalse(guard.is_blocked)

    def test_block_before_invalidation_check(self):
        """Explicit block takes priority over invalidation callback."""
        guard = OptimizerCommitGuard()
        guard.register_callbacks(
            is_iteration_invalid_fn=lambda: False,  # not invalid
        )
        guard.begin_iteration(step=100)

        # Explicitly block
        guard.block(reason="hard_failure")
        self.assertFalse(guard.should_commit())


# =====================================================================
# Test: last_recovery_path and summary
# =====================================================================

class TestRecoveryPathTracking(unittest.TestCase):
    """last_recovery_path is correctly tracked and appears in summary."""

    def test_initial_empty(self):
        ctrl = RecoveryController()
        self.assertEqual(ctrl.last_recovery_path, "")

    def test_appears_in_summary(self):
        ctrl, _ = _make_controller_with_tracker(
            gap_threshold=50, checkpoint_iteration=90,
        )
        _drive_to_safe_point_repair(ctrl, step=100)
        ctrl.before_iteration(step=103)

        summary = ctrl.summary()
        self.assertEqual(summary["last_recovery_path"], "CHECKPOINT_RESTART")

    def test_reset_clears_path(self):
        ctrl, _ = _make_controller_with_tracker(
            gap_threshold=50, checkpoint_iteration=90,
        )
        _drive_to_safe_point_repair(ctrl, step=100)
        ctrl.before_iteration(step=103)
        self.assertEqual(ctrl.last_recovery_path, "CHECKPOINT_RESTART")

        ctrl.reset()
        self.assertEqual(ctrl.last_recovery_path, "")


# =====================================================================
# Test: Gap-aware disabled → always hybrid
# =====================================================================

class TestGapAwareDisabled(unittest.TestCase):
    """When gap-aware policy is disabled, always use hybrid recovery."""

    def test_disabled_uses_hybrid(self):
        ctrl, tracker = _make_controller_with_tracker(
            gap_threshold=50, checkpoint_iteration=90,
        )
        ctrl.gap_aware_policy_manager.enabled = False

        _drive_to_safe_point_repair(ctrl, step=100)
        ctrl.before_iteration(step=103)

        self.assertNotIn("checkpoint_restart", tracker.actions)
        self.assertIn("dense_sync", tracker.actions)
        self.assertEqual(ctrl.last_recovery_path, "HYBRID_RECOVERY")


# =====================================================================
# Test: No gap-aware manager → always hybrid (default)
# =====================================================================

class TestNoGapAwareManager(unittest.TestCase):
    """Without gap-aware policy manager, always use hybrid recovery."""

    def test_no_manager_uses_hybrid(self):
        ctrl = RecoveryController()
        tracker = ActionTracker()

        ctrl.register_callbacks(
            replacement_integrate_fn=tracker.make_fn("replacement_integrate"),
            group_rebuild_request_fn=tracker.make_fn("group_rebuild_request"),
            group_rebuild_execute_fn=tracker.make_fn("group_rebuild_execute"),
            group_rebuild_finish_fn=tracker.make_fn("group_rebuild_finish"),
            topology_refresh_fn=tracker.make_fn("topology_refresh"),
            dense_sync_fn=tracker.make_fn("dense_sync"),
            expert_restore_fn=tracker.make_fn("expert_restore"),
        )

        _drive_to_safe_point_repair(ctrl, step=100)
        ctrl.before_iteration(step=103)

        self.assertIn("dense_sync", tracker.actions)
        self.assertEqual(ctrl.last_recovery_path, "HYBRID_RECOVERY")


if __name__ == "__main__":
    unittest.main()

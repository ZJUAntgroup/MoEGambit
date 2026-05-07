# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for BSR-MoE Recovery Policy Framework.

Tests cover all 5 policies:
1. RestartAndSparePolicy — always CHECKPOINT_RESTART (except no ckpt)
2. AlwaysHybridPolicy — always HYBRID_RECOVERY
3. FixedGapThresholdPolicy — single gap threshold
4. TwoThresholdPolicy — gap lower + upper bound
5. RankExposureGuardedPolicy — gap bounds + rank stale exposure

Plus:
- RecoveryDecision structure and serialization
- RecoveryPath enum
- GapAwareRecoveryPolicyManager (evaluate, disabled, history, summary)
- Global singleton lifecycle
- initialize_gap_aware_recovery_policy with all policy types
- Backward-compatible aliases
"""

import unittest

from megatron.core.transformer.moe.gap_aware_recovery_policy import (
    # Enum
    RecoveryPath,
    # Decision
    RecoveryDecision,
    # Policies
    RecoveryPolicyBase,
    RestartAndSparePolicy,
    AlwaysHybridPolicy,
    FixedGapThresholdPolicy,
    TwoThresholdPolicy,
    RankExposureGuardedPolicy,
    RankExposureGuardedConfig,
    # Backward-compatible aliases
    ThresholdRecoveryPolicy,
    RankExposureGuardedHybridPolicy,
    # Manager
    GapAwareRecoveryPolicyManager,
    build_recovery_path_chosen_log,
    normalize_decision_reason,
    # Singleton
    get_gap_aware_recovery_policy_manager,
    initialize_gap_aware_recovery_policy,
    clear_gap_aware_recovery_policy,
)
from megatron.core.transformer.moe.rank_exposure_tracker import (
    RankExposureTracker,
    reset_rank_exposure_tracker,
)


# =====================================================================
# Test: RecoveryPath enum
# =====================================================================


class TestRecoveryPathEnum(unittest.TestCase):

    def test_values(self):
        self.assertEqual(RecoveryPath.CHECKPOINT_RESTART.value, "checkpoint_restart")
        self.assertEqual(RecoveryPath.HYBRID_RECOVERY.value, "hybrid_recovery")


# =====================================================================
# Test: RecoveryDecision
# =====================================================================


class TestRecoveryDecision(unittest.TestCase):

    def test_required_fields(self):
        decision = RecoveryDecision(
            path=RecoveryPath.CHECKPOINT_RESTART,
            current_step=200,
            latest_checkpoint_step=150,
            gap=50,
            failed_rank=3,
            reason="gap_below_time_threshold",
            reason_detail="gap=50 < delta_time_min_gap=32",
        )
        self.assertEqual(decision.path, RecoveryPath.CHECKPOINT_RESTART)
        self.assertEqual(decision.current_step, 200)
        self.assertEqual(decision.latest_checkpoint_step, 150)
        self.assertEqual(decision.gap, 50)
        self.assertEqual(decision.failed_rank, 3)
        self.assertEqual(decision.reason, "gap_below_time_threshold")
        self.assertEqual(decision.delta_time_min_gap, -1)  # default
        self.assertEqual(decision.rank_stale_iters_before, 0)  # default

    def test_to_dict(self):
        decision = RecoveryDecision(
            path=RecoveryPath.HYBRID_RECOVERY,
            current_step=500,
            latest_checkpoint_step=100,
            gap=400,
            failed_rank=7,
            reason="within_rank_exposure_safe_region",
            reason_detail="safe",
            delta_time_min_gap=32,
            max_single_gap=192,
            rank_stale_exposure_after=0.001,
        )
        d = decision.to_dict()
        self.assertEqual(d["selected_path"], "hybrid_recovery")
        self.assertEqual(d["current_step"], 500)
        self.assertEqual(d["latest_checkpoint_step"], 100)
        self.assertEqual(d["gap"], 400)
        self.assertEqual(d["failed_rank"], 7)
        self.assertEqual(d["reason"], "within_rank_exposure_safe_region")
        self.assertEqual(d["delta_time_min_gap"], 32)
        self.assertEqual(d["max_single_gap"], 192)
        self.assertIn("timestamp", d)
        self.assertIsInstance(d["metadata"], dict)


# =====================================================================
# Test: structured recovery_path_chosen JSON payload
# =====================================================================


class TestRecoveryPathChosenJson(unittest.TestCase):

    def test_rank_exposure_payload_has_stable_fields(self):
        d = RecoveryDecision(
            path=RecoveryPath.HYBRID_RECOVERY,
            current_step=500,
            latest_checkpoint_step=420,
            gap=80,
            failed_rank=7,
            reason="within_rank_exposure_safe_region",
            reason_detail="safe",
            delta_time_min_gap=32,
            max_single_gap=192,
            exposure_window_steps=20000,
            max_rank_stale_exposure=0.02,
            rank_stale_iters_before=100,
            rank_stale_iters_after=180,
            rank_stale_exposure_before=0.005,
            rank_stale_exposure_after=0.009,
            metadata={
                "policy": "rank_exposure_guarded",
                "estimated_restart_cost": 120.0,
                "estimated_hybrid_cost": 15.0,
                "policy_margin": 105.0,
            },
        )

        payload = build_recovery_path_chosen_log(
            d,
            run_id="run-1",
            policy_type="rank_exposure_guarded",
        )

        self.assertEqual(payload["event"], "recovery_path_chosen")
        self.assertEqual(payload["run_id"], "run-1")
        self.assertEqual(payload["current_step"], 500)
        self.assertEqual(payload["latest_checkpoint_step"], 420)
        self.assertEqual(payload["checkpoint_gap"], 80)
        self.assertEqual(payload["failed_rank"], 7)
        self.assertEqual(payload["policy_type"], "rank_exposure_guarded")
        self.assertEqual(payload["selected_path"], "hybrid_recovery")
        self.assertEqual(
            payload["decision_reason"],
            "within_rank_exposure_safe_region",
        )
        self.assertEqual(payload["rank_stale_iters_before"], 100)
        self.assertEqual(payload["rank_stale_iters_after"], 180)
        self.assertEqual(payload["rank_stale_exposure_before"], 0.005)
        self.assertEqual(payload["rank_stale_exposure_after"], 0.009)
        self.assertEqual(payload["estimated_restart_cost"], 120.0)
        self.assertEqual(payload["estimated_hybrid_cost"], 15.0)
        self.assertEqual(payload["policy_margin"], 105.0)

    def test_fixed_gap_reasons_are_normalized(self):
        restart = RecoveryDecision(
            path=RecoveryPath.CHECKPOINT_RESTART,
            current_step=10,
            latest_checkpoint_step=5,
            gap=5,
            failed_rank=1,
            reason="gap_below_threshold",
            reason_detail="restart",
        )
        hybrid = RecoveryDecision(
            path=RecoveryPath.HYBRID_RECOVERY,
            current_step=100,
            latest_checkpoint_step=5,
            gap=95,
            failed_rank=1,
            reason="gap_at_or_above_threshold",
            reason_detail="hybrid",
        )

        self.assertEqual(normalize_decision_reason(restart), "fixed_gap_restart")
        self.assertEqual(normalize_decision_reason(hybrid), "fixed_gap_hybrid")

    def test_forced_reasons_are_normalized(self):
        forced_restart = RecoveryDecision(
            path=RecoveryPath.CHECKPOINT_RESTART,
            current_step=10,
            latest_checkpoint_step=5,
            gap=5,
            failed_rank=1,
            reason="forced_checkpoint_restart",
            reason_detail="forced",
        )
        forced_hybrid = RecoveryDecision(
            path=RecoveryPath.HYBRID_RECOVERY,
            current_step=10,
            latest_checkpoint_step=-1,
            gap=11,
            failed_rank=1,
            reason="no_checkpoint_available",
            reason_detail="forced",
        )

        self.assertEqual(
            normalize_decision_reason(forced_restart),
            "forced_restart",
        )
        self.assertEqual(
            normalize_decision_reason(forced_hybrid),
            "forced_hybrid",
        )


# =====================================================================
# Test: RestartAndSparePolicy
# =====================================================================


class TestRestartAndSparePolicy(unittest.TestCase):

    def setUp(self):
        self.policy = RestartAndSparePolicy()

    def test_always_restart(self):
        d = self.policy.choose(current_step=500, latest_checkpoint_step=100, failed_rank=3)
        self.assertEqual(d.path, RecoveryPath.CHECKPOINT_RESTART)
        self.assertEqual(d.reason, "always_restart")

    def test_zero_gap_restart(self):
        d = self.policy.choose(current_step=100, latest_checkpoint_step=100, failed_rank=3)
        self.assertEqual(d.path, RecoveryPath.CHECKPOINT_RESTART)

    def test_no_checkpoint_forced_hybrid(self):
        d = self.policy.choose(current_step=50, latest_checkpoint_step=-1, failed_rank=3)
        self.assertEqual(d.path, RecoveryPath.HYBRID_RECOVERY)
        self.assertEqual(d.reason, "no_checkpoint_available")


# =====================================================================
# Test: AlwaysHybridPolicy
# =====================================================================


class TestAlwaysHybridPolicy(unittest.TestCase):

    def setUp(self):
        self.policy = AlwaysHybridPolicy()

    def test_always_hybrid(self):
        d = self.policy.choose(current_step=500, latest_checkpoint_step=100, failed_rank=3)
        self.assertEqual(d.path, RecoveryPath.HYBRID_RECOVERY)
        self.assertEqual(d.reason, "always_hybrid")

    def test_small_gap_still_hybrid(self):
        d = self.policy.choose(current_step=101, latest_checkpoint_step=100, failed_rank=3)
        self.assertEqual(d.path, RecoveryPath.HYBRID_RECOVERY)

    def test_no_checkpoint_still_hybrid(self):
        d = self.policy.choose(current_step=50, latest_checkpoint_step=-1, failed_rank=3)
        self.assertEqual(d.path, RecoveryPath.HYBRID_RECOVERY)


# =====================================================================
# Test: FixedGapThresholdPolicy
# =====================================================================


class TestFixedGapThresholdPolicy(unittest.TestCase):

    def test_small_gap_restart(self):
        policy = FixedGapThresholdPolicy(fixed_gap_threshold=32)
        d = policy.choose(current_step=150, latest_checkpoint_step=130, failed_rank=3)
        self.assertEqual(d.path, RecoveryPath.CHECKPOINT_RESTART)
        self.assertEqual(d.reason, "gap_below_threshold")

    def test_large_gap_hybrid(self):
        policy = FixedGapThresholdPolicy(fixed_gap_threshold=32)
        d = policy.choose(current_step=200, latest_checkpoint_step=100, failed_rank=3)
        self.assertEqual(d.path, RecoveryPath.HYBRID_RECOVERY)
        self.assertEqual(d.reason, "gap_at_or_above_threshold")

    def test_exact_threshold_hybrid(self):
        """gap == threshold → hybrid (gap < threshold is restart)."""
        policy = FixedGapThresholdPolicy(fixed_gap_threshold=32)
        d = policy.choose(current_step=132, latest_checkpoint_step=100, failed_rank=3)
        self.assertEqual(d.gap, 32)
        self.assertEqual(d.path, RecoveryPath.HYBRID_RECOVERY)

    def test_no_checkpoint_forced_hybrid(self):
        policy = FixedGapThresholdPolicy(fixed_gap_threshold=32)
        d = policy.choose(current_step=50, latest_checkpoint_step=-1, failed_rank=3)
        self.assertEqual(d.path, RecoveryPath.HYBRID_RECOVERY)

    def test_negative_threshold_rejected(self):
        with self.assertRaises(ValueError):
            FixedGapThresholdPolicy(fixed_gap_threshold=-1)

    def test_backward_compat_alias(self):
        """ThresholdRecoveryPolicy is an alias for FixedGapThresholdPolicy."""
        self.assertIs(ThresholdRecoveryPolicy, FixedGapThresholdPolicy)


# =====================================================================
# Test: TwoThresholdPolicy
# =====================================================================


class TestTwoThresholdPolicy(unittest.TestCase):

    def setUp(self):
        self.policy = TwoThresholdPolicy(delta_time_min_gap=32, max_single_gap=192)

    def test_gap_below_lower_restart(self):
        d = self.policy.choose(current_step=150, latest_checkpoint_step=130, failed_rank=3)
        self.assertEqual(d.gap, 20)
        self.assertEqual(d.path, RecoveryPath.CHECKPOINT_RESTART)
        self.assertEqual(d.reason, "gap_below_time_threshold")

    def test_gap_above_upper_restart(self):
        d = self.policy.choose(current_step=500, latest_checkpoint_step=100, failed_rank=3)
        self.assertEqual(d.gap, 400)
        self.assertEqual(d.path, RecoveryPath.CHECKPOINT_RESTART)
        self.assertEqual(d.reason, "gap_above_single_gap_threshold")

    def test_gap_in_middle_hybrid(self):
        d = self.policy.choose(current_step=200, latest_checkpoint_step=100, failed_rank=3)
        self.assertEqual(d.gap, 100)
        self.assertEqual(d.path, RecoveryPath.HYBRID_RECOVERY)
        self.assertEqual(d.reason, "within_two_threshold_safe_region")

    def test_gap_at_lower_boundary_restart(self):
        """gap == delta_time_min_gap is NOT < it, so hybrid."""
        d = self.policy.choose(current_step=132, latest_checkpoint_step=100, failed_rank=3)
        self.assertEqual(d.gap, 32)
        # 32 < 32 is False, so not gap_below_time_threshold
        self.assertNotEqual(d.reason, "gap_below_time_threshold")

    def test_gap_at_upper_boundary_hybrid(self):
        """gap == max_single_gap is NOT > it, so hybrid."""
        d = self.policy.choose(current_step=292, latest_checkpoint_step=100, failed_rank=3)
        self.assertEqual(d.gap, 192)
        self.assertEqual(d.path, RecoveryPath.HYBRID_RECOVERY)

    def test_no_checkpoint_forced_hybrid(self):
        d = self.policy.choose(current_step=50, latest_checkpoint_step=-1, failed_rank=3)
        self.assertEqual(d.path, RecoveryPath.HYBRID_RECOVERY)

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            TwoThresholdPolicy(delta_time_min_gap=200, max_single_gap=50)

    def test_negative_delta_rejected(self):
        with self.assertRaises(ValueError):
            TwoThresholdPolicy(delta_time_min_gap=-1, max_single_gap=192)


# =====================================================================
# Test: RankExposureGuardedPolicy
# =====================================================================


class TestRankExposureGuardedPolicy(unittest.TestCase):

    def setUp(self):
        self.tracker = RankExposureTracker()
        self.config = RankExposureGuardedConfig(
            delta_time_min_gap=32,
            max_single_gap=192,
            exposure_window_steps=10000,
            max_rank_stale_exposure=0.05,
        )
        self.policy = RankExposureGuardedPolicy(
            config=self.config,
            tracker=self.tracker,
        )

    def tearDown(self):
        self.tracker.reset()

    # --- Gap boundary tests ---

    def test_gap_below_lower_restart(self):
        d = self.policy.choose(current_step=150, latest_checkpoint_step=130, failed_rank=3)
        self.assertEqual(d.gap, 20)
        self.assertEqual(d.path, RecoveryPath.CHECKPOINT_RESTART)
        self.assertEqual(d.reason, "gap_below_time_threshold")

    def test_gap_above_upper_restart(self):
        d = self.policy.choose(current_step=500, latest_checkpoint_step=100, failed_rank=3)
        self.assertEqual(d.gap, 400)
        self.assertEqual(d.path, RecoveryPath.CHECKPOINT_RESTART)
        self.assertEqual(d.reason, "gap_above_single_gap_threshold")

    def test_gap_in_middle_no_exposure_hybrid(self):
        """Gap in range, no prior events → hybrid."""
        d = self.policy.choose(current_step=200, latest_checkpoint_step=100, failed_rank=3)
        self.assertEqual(d.gap, 100)
        self.assertEqual(d.path, RecoveryPath.HYBRID_RECOVERY)
        self.assertEqual(d.reason, "within_rank_exposure_safe_region")

    def test_no_checkpoint_forced_hybrid(self):
        d = self.policy.choose(current_step=50, latest_checkpoint_step=-1, failed_rank=3)
        self.assertEqual(d.path, RecoveryPath.HYBRID_RECOVERY)

    # --- Rank exposure tests ---

    def test_exposure_exceeded_triggers_restart(self):
        """After recording enough stale iters, exposure exceeds threshold."""
        # Window=10000, max_exposure=0.05
        # After recording stale_iters=400, exposure_before = 400/10000 = 0.04
        # gap=100: stale_after = 500, exposure_after = 0.05 (NOT > 0.05)
        # gap=101: stale_after = 501, exposure_after = 0.0501 > 0.05 → restart
        self.tracker.record_hybrid_recovery(step=100, rank=3, gap=200)
        self.tracker.record_hybrid_recovery(step=300, rank=3, gap=200)
        # stale_iters_before = 400, exposure_before = 0.04

        # gap=100: stale_after=500, exposure_after=0.05 (NOT > 0.05)
        d1 = self.policy.choose(current_step=500, latest_checkpoint_step=400, failed_rank=3)
        self.assertEqual(d1.gap, 100)
        # 500/10000 = 0.05, not > 0.05, so hybrid
        self.assertEqual(d1.path, RecoveryPath.HYBRID_RECOVERY)

        # Now stale_iters = 400 + 100 (from d1) = 500
        # gap=100: stale_after=600, exposure_after=0.06 > 0.05 → restart
        d2 = self.policy.choose(current_step=600, latest_checkpoint_step=500, failed_rank=3)
        self.assertEqual(d2.path, RecoveryPath.CHECKPOINT_RESTART)
        self.assertEqual(d2.reason, "rank_stale_exposure_exceeded")

    def test_exposure_decision_fields(self):
        """Decision includes stale_iters_before/after and exposure fields."""
        self.tracker.record_hybrid_recovery(step=100, rank=3, gap=50)
        # stale_iters_before = 50, exposure_before = 0.005
        d = self.policy.choose(current_step=500, latest_checkpoint_step=400, failed_rank=3)
        # gap=100, in [32, 192]
        self.assertEqual(d.rank_stale_iters_before, 50)
        self.assertEqual(d.rank_stale_iters_after, 150)  # 50 + 100
        self.assertAlmostEqual(d.rank_stale_exposure_before, 0.005)
        self.assertAlmostEqual(d.rank_stale_exposure_after, 0.015)
        self.assertEqual(d.exposure_window_steps, 10000)
        self.assertEqual(d.max_rank_stale_exposure, 0.05)

    def test_hybrid_does_not_record_in_tracker_automatically(self):
        """Choosing hybrid does NOT auto-record in tracker.

        Recording is deferred to the RecoveryController, which calls
        tracker.record_hybrid_recovery() only AFTER hybrid recovery
        succeeds.  The policy's choose() method is a pure decision
        function and must not have side effects on the tracker.
        """
        self.assertEqual(self.tracker.get_event_count(), 0)
        d = self.policy.choose(current_step=200, latest_checkpoint_step=100, failed_rank=3)
        self.assertEqual(d.path, RecoveryPath.HYBRID_RECOVERY)
        # choose() no longer records — the controller does after success
        self.assertEqual(self.tracker.get_event_count(), 0)
        # Simulate post-success recording (what the controller does)
        self.tracker.record_hybrid_recovery(step=d.current_step, rank=d.failed_rank, gap=d.gap)
        self.assertEqual(self.tracker.get_event_count(), 1)

    def test_restart_does_not_record_in_tracker(self):
        """Choosing checkpoint restart does NOT record in tracker.

        Checkpoint restart never introduces stale iterations (all ranks
        reload uniformly), so even the controller-side recording will
        never happen for restart decisions.
        """
        d = self.policy.choose(current_step=150, latest_checkpoint_step=130, failed_rank=3)
        self.assertEqual(d.path, RecoveryPath.CHECKPOINT_RESTART)
        self.assertEqual(self.tracker.get_event_count(), 0)

    def test_different_ranks_independent(self):
        """High exposure on rank 3 doesn't affect rank 5."""
        # Make rank 3 high exposure
        self.tracker.record_hybrid_recovery(step=100, rank=3, gap=300)
        self.tracker.record_hybrid_recovery(step=300, rank=3, gap=300)
        # rank 3 exposure_before = 600/10000 = 0.06 > 0.05

        # gap=100 on rank 3 → restart (exposure exceeded)
        d3 = self.policy.choose(current_step=500, latest_checkpoint_step=400, failed_rank=3)
        self.assertEqual(d3.path, RecoveryPath.CHECKPOINT_RESTART)

        # gap=100 on rank 5 → hybrid (no prior events)
        d5 = self.policy.choose(current_step=500, latest_checkpoint_step=400, failed_rank=5)
        self.assertEqual(d5.path, RecoveryPath.HYBRID_RECOVERY)

    def test_backward_compat_alias(self):
        """RankExposureGuardedHybridPolicy is an alias."""
        self.assertIs(RankExposureGuardedHybridPolicy, RankExposureGuardedPolicy)

    def test_config_validate(self):
        with self.assertRaises(ValueError):
            RankExposureGuardedConfig(delta_time_min_gap=-1).validate()
        with self.assertRaises(ValueError):
            RankExposureGuardedConfig(max_single_gap=10, delta_time_min_gap=20).validate()
        with self.assertRaises(ValueError):
            RankExposureGuardedConfig(exposure_window_steps=0).validate()
        with self.assertRaises(ValueError):
            RankExposureGuardedConfig(max_rank_stale_exposure=0).validate()
        with self.assertRaises(ValueError):
            RankExposureGuardedConfig(max_rank_stale_exposure=1.5).validate()


# =====================================================================
# Test: GapAwareRecoveryPolicyManager
# =====================================================================


class TestGapAwareRecoveryPolicyManager(unittest.TestCase):

    def test_evaluate_with_threshold_policy(self):
        mgr = GapAwareRecoveryPolicyManager(
            gap_threshold=100,
            get_checkpoint_iteration_fn=lambda: 900,
        )
        d = mgr.evaluate(current_iteration=950, failed_rank=1)
        self.assertEqual(d.path, RecoveryPath.CHECKPOINT_RESTART)

    def test_evaluate_with_restart_and_spare(self):
        mgr = GapAwareRecoveryPolicyManager(
            policy=RestartAndSparePolicy(),
            get_checkpoint_iteration_fn=lambda: 100,
        )
        d = mgr.evaluate(current_iteration=500, failed_rank=1)
        self.assertEqual(d.path, RecoveryPath.CHECKPOINT_RESTART)

    def test_evaluate_with_always_hybrid(self):
        mgr = GapAwareRecoveryPolicyManager(
            policy=AlwaysHybridPolicy(),
            get_checkpoint_iteration_fn=lambda: 100,
        )
        d = mgr.evaluate(current_iteration=101, failed_rank=1)
        self.assertEqual(d.path, RecoveryPath.HYBRID_RECOVERY)

    def test_evaluate_with_two_threshold(self):
        mgr = GapAwareRecoveryPolicyManager(
            policy=TwoThresholdPolicy(delta_time_min_gap=32, max_single_gap=192),
            get_checkpoint_iteration_fn=lambda: 100,
        )
        d_small = mgr.evaluate(current_iteration=120, failed_rank=1)  # gap=20
        self.assertEqual(d_small.path, RecoveryPath.CHECKPOINT_RESTART)
        d_mid = mgr.evaluate(current_iteration=200, failed_rank=1)  # gap=100
        self.assertEqual(d_mid.path, RecoveryPath.HYBRID_RECOVERY)
        d_big = mgr.evaluate(current_iteration=500, failed_rank=1)  # gap=400
        self.assertEqual(d_big.path, RecoveryPath.CHECKPOINT_RESTART)

    def test_disabled_always_hybrid(self):
        mgr = GapAwareRecoveryPolicyManager(
            policy=RestartAndSparePolicy(),
            get_checkpoint_iteration_fn=lambda: 100,
        )
        mgr.enabled = False
        d = mgr.evaluate(current_iteration=500, failed_rank=1)
        self.assertEqual(d.path, RecoveryPath.HYBRID_RECOVERY)
        self.assertEqual(d.reason, "gap_aware_disabled")

    def test_no_callback_forced_hybrid(self):
        mgr = GapAwareRecoveryPolicyManager(
            gap_threshold=100,
            get_checkpoint_iteration_fn=None,
        )
        d = mgr.evaluate(current_iteration=100, failed_rank=1)
        self.assertEqual(d.path, RecoveryPath.HYBRID_RECOVERY)

    def test_decision_history(self):
        mgr = GapAwareRecoveryPolicyManager(
            gap_threshold=100,
            get_checkpoint_iteration_fn=lambda: 0,
        )
        mgr.evaluate(current_iteration=50)
        mgr.evaluate(current_iteration=200)
        self.assertEqual(len(mgr.decision_history), 2)

    def test_summary_with_rank_exposure_guarded(self):
        tracker = RankExposureTracker()
        config = RankExposureGuardedConfig()
        policy = RankExposureGuardedPolicy(config=config, tracker=tracker)
        mgr = GapAwareRecoveryPolicyManager(policy=policy)
        s = mgr.summary()
        self.assertEqual(s["policy_type"], "RankExposureGuardedPolicy")
        self.assertIn("policy_config", s)


# =====================================================================
# Test: Global singleton lifecycle
# =====================================================================


class TestGlobalSingleton(unittest.TestCase):

    def setUp(self):
        clear_gap_aware_recovery_policy()
        reset_rank_exposure_tracker()

    def tearDown(self):
        clear_gap_aware_recovery_policy()
        reset_rank_exposure_tracker()

    def test_get_creates_default(self):
        mgr = get_gap_aware_recovery_policy_manager()
        self.assertIsNotNone(mgr)
        self.assertTrue(mgr.enabled)

    def test_initialize_with_policy_type(self):
        mgr = initialize_gap_aware_recovery_policy(
            gap_threshold=32,
            policy_type="fixed_gap_threshold",
            enabled=True,
        )
        self.assertIsInstance(mgr.policy, FixedGapThresholdPolicy)

    def test_initialize_restart_and_spare(self):
        mgr = initialize_gap_aware_recovery_policy(
            policy_type="restart_and_spare",
        )
        self.assertIsInstance(mgr.policy, RestartAndSparePolicy)

    def test_initialize_always_hybrid(self):
        mgr = initialize_gap_aware_recovery_policy(
            policy_type="always_hybrid",
        )
        self.assertIsInstance(mgr.policy, AlwaysHybridPolicy)

    def test_initialize_two_threshold(self):
        mgr = initialize_gap_aware_recovery_policy(
            policy_type="two_threshold",
            delta_time_min_gap=50,
            max_single_gap=300,
        )
        self.assertIsInstance(mgr.policy, TwoThresholdPolicy)
        self.assertEqual(mgr.policy.delta_time_min_gap, 50)
        self.assertEqual(mgr.policy.max_single_gap, 300)

    def test_initialize_rank_exposure_guarded(self):
        mgr = initialize_gap_aware_recovery_policy(
            policy_type="rank_exposure_guarded",
            rank_exposure_config=RankExposureGuardedConfig(
                delta_time_min_gap=64,
                max_single_gap=384,
            ),
        )
        self.assertIsInstance(mgr.policy, RankExposureGuardedPolicy)
        self.assertEqual(mgr.policy.config.delta_time_min_gap, 64)
        self.assertEqual(mgr.policy.config.max_single_gap, 384)

    def test_initialize_backward_compat_threshold(self):
        """'threshold' is a backward-compatible alias."""
        mgr = initialize_gap_aware_recovery_policy(
            gap_threshold=42,
            policy_type="threshold",
        )
        self.assertIsInstance(mgr.policy, FixedGapThresholdPolicy)

    def test_initialize_backward_compat_rank_exposure(self):
        """'rank_exposure_guarded_hybrid' is a backward-compatible alias."""
        mgr = initialize_gap_aware_recovery_policy(
            policy_type="rank_exposure_guarded_hybrid",
        )
        self.assertIsInstance(mgr.policy, RankExposureGuardedPolicy)

    def test_clear_and_recreate(self):
        m1 = get_gap_aware_recovery_policy_manager()
        clear_gap_aware_recovery_policy()
        m2 = get_gap_aware_recovery_policy_manager()
        self.assertIsNot(m1, m2)


if __name__ == "__main__":
    unittest.main()

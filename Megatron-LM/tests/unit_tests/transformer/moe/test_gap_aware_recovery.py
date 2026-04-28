# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for BSR-MoE Gap-Aware Recovery Policy.

Tests cover:
1. ThresholdRecoveryPolicy — small gap → CHECKPOINT_RESTART
2. ThresholdRecoveryPolicy — large gap → HYBRID_RECOVERY
3. ThresholdRecoveryPolicy — no checkpoint (iteration < 0) → forced HYBRID_RECOVERY
4. ThresholdRecoveryPolicy — boundary: gap == threshold → CHECKPOINT_RESTART
5. ThresholdRecoveryPolicy — negative threshold rejected
6. RecoveryDecision serialization (to_dict)
7. GapAwareRecoveryPolicyManager — evaluate with callback
8. GapAwareRecoveryPolicyManager — disabled → always HYBRID_RECOVERY
9. GapAwareRecoveryPolicyManager — decision history tracking
10. GapAwareRecoveryPolicyManager — callback failure → forced HYBRID_RECOVERY
11. Global singleton lifecycle (initialize / get / clear)
"""

import unittest

from megatron.core.transformer.moe.gap_aware_recovery_policy import (
    RecoveryPath,
    RecoveryDecision,
    ThresholdRecoveryPolicy,
    GapAwareRecoveryPolicyManager,
    get_gap_aware_recovery_policy_manager,
    initialize_gap_aware_recovery_policy,
    clear_gap_aware_recovery_policy,
)


# =====================================================================
# Test: ThresholdRecoveryPolicy
# =====================================================================


class TestThresholdRecoveryPolicy(unittest.TestCase):
    """Tests for the v1 threshold-based recovery path selector."""

    def test_small_gap_checkpoint_restart(self):
        """gap <= threshold → CHECKPOINT_RESTART."""
        policy = ThresholdRecoveryPolicy(gap_threshold=100)
        decision = policy.select_path(
            current_iteration=150,
            checkpoint_iteration=100,
        )
        self.assertEqual(decision.path, RecoveryPath.CHECKPOINT_RESTART)
        self.assertEqual(decision.gap, 50)
        self.assertEqual(decision.gap_threshold, 100)
        self.assertIn("checkpoint restart", decision.reason)

    def test_large_gap_hybrid_recovery(self):
        """gap > threshold → HYBRID_RECOVERY."""
        policy = ThresholdRecoveryPolicy(gap_threshold=100)
        decision = policy.select_path(
            current_iteration=500,
            checkpoint_iteration=100,
        )
        self.assertEqual(decision.path, RecoveryPath.HYBRID_RECOVERY)
        self.assertEqual(decision.gap, 400)
        self.assertIn("hybrid recovery", decision.reason)

    def test_no_checkpoint_forced_hybrid(self):
        """checkpoint_iteration < 0 → forced HYBRID_RECOVERY."""
        policy = ThresholdRecoveryPolicy(gap_threshold=100)
        decision = policy.select_path(
            current_iteration=50,
            checkpoint_iteration=-1,
        )
        self.assertEqual(decision.path, RecoveryPath.HYBRID_RECOVERY)
        self.assertIn("no checkpoint available", decision.reason)

    def test_boundary_gap_equals_threshold(self):
        """gap == threshold → CHECKPOINT_RESTART (<=)."""
        policy = ThresholdRecoveryPolicy(gap_threshold=100)
        decision = policy.select_path(
            current_iteration=200,
            checkpoint_iteration=100,
        )
        self.assertEqual(decision.gap, 100)
        self.assertEqual(decision.path, RecoveryPath.CHECKPOINT_RESTART)

    def test_zero_gap(self):
        """gap == 0 → CHECKPOINT_RESTART."""
        policy = ThresholdRecoveryPolicy(gap_threshold=100)
        decision = policy.select_path(
            current_iteration=100,
            checkpoint_iteration=100,
        )
        self.assertEqual(decision.gap, 0)
        self.assertEqual(decision.path, RecoveryPath.CHECKPOINT_RESTART)

    def test_threshold_zero(self):
        """gap_threshold=0: only gap==0 gets CHECKPOINT_RESTART."""
        policy = ThresholdRecoveryPolicy(gap_threshold=0)
        # gap=0 → restart
        d0 = policy.select_path(current_iteration=100, checkpoint_iteration=100)
        self.assertEqual(d0.path, RecoveryPath.CHECKPOINT_RESTART)
        # gap=1 → hybrid
        d1 = policy.select_path(current_iteration=101, checkpoint_iteration=100)
        self.assertEqual(d1.path, RecoveryPath.HYBRID_RECOVERY)

    def test_negative_threshold_rejected(self):
        """gap_threshold < 0 raises ValueError."""
        with self.assertRaises(ValueError):
            ThresholdRecoveryPolicy(gap_threshold=-1)

    def test_metadata_populated(self):
        """Decision metadata includes failed_rank, replacement_rank, etc."""
        policy = ThresholdRecoveryPolicy(gap_threshold=100)
        decision = policy.select_path(
            current_iteration=150,
            checkpoint_iteration=100,
            failed_rank=3,
            replacement_rank=7,
            num_affected_experts=4,
        )
        self.assertEqual(decision.metadata["failed_rank"], 3)
        self.assertEqual(decision.metadata["replacement_rank"], 7)
        self.assertEqual(decision.metadata["num_affected_experts"], 4)
        self.assertEqual(decision.metadata["policy"], "threshold")

    def test_gap_threshold_property(self):
        """gap_threshold property returns configured value."""
        policy = ThresholdRecoveryPolicy(gap_threshold=42)
        self.assertEqual(policy.gap_threshold, 42)


# =====================================================================
# Test: RecoveryDecision
# =====================================================================


class TestRecoveryDecision(unittest.TestCase):
    """Tests for RecoveryDecision dataclass."""

    def test_to_dict(self):
        """to_dict returns JSON-serializable dict with all fields."""
        decision = RecoveryDecision(
            path=RecoveryPath.CHECKPOINT_RESTART,
            current_iteration=200,
            checkpoint_iteration=150,
            gap=50,
            gap_threshold=100,
            reason="test reason",
        )
        d = decision.to_dict()
        self.assertEqual(d["selected_path"], "checkpoint_restart")
        self.assertEqual(d["current_iteration"], 200)
        self.assertEqual(d["checkpoint_iteration"], 150)
        self.assertEqual(d["gap"], 50)
        self.assertEqual(d["gap_threshold"], 100)
        self.assertEqual(d["reason"], "test reason")
        self.assertIn("timestamp", d)
        self.assertIsInstance(d["metadata"], dict)

    def test_to_dict_hybrid(self):
        """to_dict for HYBRID_RECOVERY path."""
        decision = RecoveryDecision(
            path=RecoveryPath.HYBRID_RECOVERY,
            current_iteration=500,
            checkpoint_iteration=100,
            gap=400,
            gap_threshold=100,
            reason="large gap",
        )
        d = decision.to_dict()
        self.assertEqual(d["selected_path"], "hybrid_recovery")


# =====================================================================
# Test: GapAwareRecoveryPolicyManager
# =====================================================================


class TestGapAwareRecoveryPolicyManager(unittest.TestCase):
    """Tests for the policy manager."""

    def test_evaluate_with_callback(self):
        """Manager queries checkpoint iteration via callback."""
        mgr = GapAwareRecoveryPolicyManager(
            gap_threshold=100,
            get_checkpoint_iteration_fn=lambda: 900,
        )
        decision = mgr.evaluate(current_iteration=950, failed_rank=1)
        self.assertEqual(decision.path, RecoveryPath.CHECKPOINT_RESTART)
        self.assertEqual(decision.gap, 50)
        self.assertEqual(decision.checkpoint_iteration, 900)

    def test_evaluate_large_gap(self):
        """Large gap via callback → HYBRID_RECOVERY."""
        mgr = GapAwareRecoveryPolicyManager(
            gap_threshold=100,
            get_checkpoint_iteration_fn=lambda: 500,
        )
        decision = mgr.evaluate(current_iteration=950)
        self.assertEqual(decision.path, RecoveryPath.HYBRID_RECOVERY)
        self.assertEqual(decision.gap, 450)

    def test_disabled_always_hybrid(self):
        """When disabled, always returns HYBRID_RECOVERY."""
        mgr = GapAwareRecoveryPolicyManager(
            gap_threshold=100,
            get_checkpoint_iteration_fn=lambda: 950,  # gap=0, would be restart
        )
        mgr.enabled = False
        decision = mgr.evaluate(current_iteration=950)
        self.assertEqual(decision.path, RecoveryPath.HYBRID_RECOVERY)
        self.assertIn("disabled", decision.reason)

    def test_no_callback_forced_hybrid(self):
        """No checkpoint iteration callback → checkpoint_iteration=-1 → HYBRID."""
        mgr = GapAwareRecoveryPolicyManager(
            gap_threshold=100,
            get_checkpoint_iteration_fn=None,
        )
        decision = mgr.evaluate(current_iteration=100)
        self.assertEqual(decision.path, RecoveryPath.HYBRID_RECOVERY)
        self.assertEqual(decision.checkpoint_iteration, -1)

    def test_callback_failure_forced_hybrid(self):
        """Callback raises exception → checkpoint_iteration=-1 → HYBRID."""
        def failing_fn():
            raise RuntimeError("disk error")

        mgr = GapAwareRecoveryPolicyManager(
            gap_threshold=100,
            get_checkpoint_iteration_fn=failing_fn,
        )
        decision = mgr.evaluate(current_iteration=100)
        self.assertEqual(decision.path, RecoveryPath.HYBRID_RECOVERY)
        self.assertEqual(decision.checkpoint_iteration, -1)

    def test_decision_history(self):
        """Decisions are recorded in history."""
        mgr = GapAwareRecoveryPolicyManager(
            gap_threshold=100,
            get_checkpoint_iteration_fn=lambda: 0,
        )
        mgr.evaluate(current_iteration=50)
        mgr.evaluate(current_iteration=200)
        mgr.evaluate(current_iteration=500)

        history = mgr.decision_history
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0].path, RecoveryPath.CHECKPOINT_RESTART)
        self.assertEqual(history[1].path, RecoveryPath.HYBRID_RECOVERY)
        self.assertEqual(history[2].path, RecoveryPath.HYBRID_RECOVERY)

    def test_last_decision(self):
        """last_decision returns the most recent decision."""
        mgr = GapAwareRecoveryPolicyManager(
            gap_threshold=100,
            get_checkpoint_iteration_fn=lambda: 0,
        )
        self.assertIsNone(mgr.last_decision)
        mgr.evaluate(current_iteration=50)
        self.assertIsNotNone(mgr.last_decision)
        self.assertEqual(mgr.last_decision.current_iteration, 50)

    def test_set_checkpoint_iteration_fn(self):
        """set_checkpoint_iteration_fn replaces the callback."""
        mgr = GapAwareRecoveryPolicyManager(
            gap_threshold=100,
            get_checkpoint_iteration_fn=lambda: 0,
        )
        # gap=200 → hybrid
        d1 = mgr.evaluate(current_iteration=200)
        self.assertEqual(d1.path, RecoveryPath.HYBRID_RECOVERY)

        # Replace callback: now checkpoint is at 190
        mgr.set_checkpoint_iteration_fn(lambda: 190)
        # gap=10 → restart
        d2 = mgr.evaluate(current_iteration=200)
        self.assertEqual(d2.path, RecoveryPath.CHECKPOINT_RESTART)

    def test_summary(self):
        """summary() returns correct structure."""
        mgr = GapAwareRecoveryPolicyManager(gap_threshold=42)
        s = mgr.summary()
        self.assertTrue(s["enabled"])
        self.assertEqual(s["policy_type"], "ThresholdRecoveryPolicy")
        self.assertEqual(s["gap_threshold"], 42)
        self.assertEqual(s["num_decisions"], 0)
        self.assertIsNone(s["last_decision"])

    def test_reset(self):
        """reset() clears decision history."""
        mgr = GapAwareRecoveryPolicyManager(
            gap_threshold=100,
            get_checkpoint_iteration_fn=lambda: 0,
        )
        mgr.evaluate(current_iteration=50)
        self.assertEqual(len(mgr.decision_history), 1)
        mgr.reset()
        self.assertEqual(len(mgr.decision_history), 0)

    def test_custom_policy(self):
        """Custom policy overrides default ThresholdRecoveryPolicy."""
        class AlwaysHybridPolicy(ThresholdRecoveryPolicy):
            def select_path(self, **kwargs):
                return RecoveryDecision(
                    path=RecoveryPath.HYBRID_RECOVERY,
                    current_iteration=kwargs.get("current_iteration", 0),
                    checkpoint_iteration=kwargs.get("checkpoint_iteration", 0),
                    gap=0,
                    gap_threshold=0,
                    reason="always hybrid",
                )

        mgr = GapAwareRecoveryPolicyManager(
            policy=AlwaysHybridPolicy(gap_threshold=0),
            get_checkpoint_iteration_fn=lambda: 999,
        )
        # Even with gap=1, custom policy returns hybrid
        decision = mgr.evaluate(current_iteration=1000)
        self.assertEqual(decision.path, RecoveryPath.HYBRID_RECOVERY)


# =====================================================================
# Test: Global singleton lifecycle
# =====================================================================


class TestGlobalSingleton(unittest.TestCase):
    """Tests for global singleton management functions."""

    def setUp(self):
        clear_gap_aware_recovery_policy()

    def tearDown(self):
        clear_gap_aware_recovery_policy()

    def test_get_creates_default(self):
        """get_gap_aware_recovery_policy_manager creates a default manager."""
        mgr = get_gap_aware_recovery_policy_manager()
        self.assertIsNotNone(mgr)
        self.assertTrue(mgr.enabled)

    def test_get_returns_same_instance(self):
        """Repeated calls return the same singleton."""
        m1 = get_gap_aware_recovery_policy_manager()
        m2 = get_gap_aware_recovery_policy_manager()
        self.assertIs(m1, m2)

    def test_initialize_configures_singleton(self):
        """initialize_gap_aware_recovery_policy sets threshold and enabled."""
        mgr = initialize_gap_aware_recovery_policy(
            gap_threshold=42,
            enabled=True,
            get_checkpoint_iteration_fn=lambda: 100,
        )
        self.assertEqual(mgr.policy.gap_threshold, 42)
        self.assertTrue(mgr.enabled)
        # Verify it's the global singleton
        self.assertIs(mgr, get_gap_aware_recovery_policy_manager())

    def test_initialize_disabled(self):
        """initialize with enabled=False."""
        mgr = initialize_gap_aware_recovery_policy(
            gap_threshold=100,
            enabled=False,
        )
        self.assertFalse(mgr.enabled)
        decision = mgr.evaluate(current_iteration=50)
        self.assertEqual(decision.path, RecoveryPath.HYBRID_RECOVERY)

    def test_clear_and_recreate(self):
        """clear resets the singleton; next get creates a new one."""
        m1 = get_gap_aware_recovery_policy_manager()
        clear_gap_aware_recovery_policy()
        m2 = get_gap_aware_recovery_policy_manager()
        self.assertIsNot(m1, m2)


# =====================================================================
# Test: RecoveryPath enum
# =====================================================================


class TestRecoveryPathEnum(unittest.TestCase):
    """Tests for RecoveryPath enum values."""

    def test_values(self):
        self.assertEqual(RecoveryPath.CHECKPOINT_RESTART.value, "checkpoint_restart")
        self.assertEqual(RecoveryPath.HYBRID_RECOVERY.value, "hybrid_recovery")

    def test_name(self):
        self.assertEqual(RecoveryPath.CHECKPOINT_RESTART.name, "CHECKPOINT_RESTART")
        self.assertEqual(RecoveryPath.HYBRID_RECOVERY.name, "HYBRID_RECOVERY")


if __name__ == "__main__":
    unittest.main()

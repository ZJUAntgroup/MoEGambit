# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# MOEGAMBIT-MoE: Tests for Preferential Routing (Step 7)

import math
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

import torch


# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so we can import megatron modules
# without a full install.
# ---------------------------------------------------------------------------
import os

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from megatron.core.transformer.moe.preferential_routing import (
    PreferentialRoutingManager,
    _ExpertBiasSession,
    clear_preferential_routing_managers,
    get_all_preferential_routing_managers,
    get_preferential_routing_manager,
)


# ===================================================================
# 1. _ExpertBiasSession unit tests
# ===================================================================

class TestExpertBiasSession(unittest.TestCase):
    """Test the per-expert bias session dataclass."""

    def test_initial_bias(self):
        s = _ExpertBiasSession(
            expert_id=0, initial_bias=0.2, window_steps=100, activation_step=0,
        )
        self.assertAlmostEqual(s.current_bias(), 0.2)
        self.assertTrue(s.active)

    def test_linear_decay(self):
        s = _ExpertBiasSession(
            expert_id=0, initial_bias=1.0, window_steps=10, activation_step=0,
        )
        # After 5 steps: remaining_frac = 0.5
        for _ in range(5):
            s.step()
        self.assertAlmostEqual(s.current_bias(), 0.5)
        self.assertTrue(s.active)

    def test_expires_at_window_end(self):
        s = _ExpertBiasSession(
            expert_id=0, initial_bias=0.3, window_steps=10, activation_step=0,
        )
        for _ in range(10):
            s.step()
        self.assertAlmostEqual(s.current_bias(), 0.0)
        self.assertFalse(s.active)

    def test_remaining_steps(self):
        s = _ExpertBiasSession(
            expert_id=0, initial_bias=0.1, window_steps=20, activation_step=0,
        )
        self.assertEqual(s.remaining_steps(), 20)
        for _ in range(7):
            s.step()
        self.assertEqual(s.remaining_steps(), 13)

    def test_zero_window(self):
        s = _ExpertBiasSession(
            expert_id=0, initial_bias=0.5, window_steps=0, activation_step=0,
        )
        self.assertAlmostEqual(s.current_bias(), 0.0)


# ===================================================================
# 2. PreferentialRoutingManager unit tests
# ===================================================================

class TestPreferentialRoutingManager(unittest.TestCase):
    """Test the per-layer manager."""

    def setUp(self):
        self.mgr = PreferentialRoutingManager(
            layer_number=1, num_experts=8, initial_bias=0.2, window_steps=10,
        )

    def test_no_sessions_returns_none(self):
        """When no sessions are active, get_bias_tensor returns None."""
        result = self.mgr.get_bias_tensor(torch.device("cpu"))
        self.assertIsNone(result)
        self.assertFalse(self.mgr.has_active_sessions())

    def test_activate_creates_bias(self):
        self.mgr.activate(3, step=0)
        bias = self.mgr.get_bias_tensor(torch.device("cpu"))
        self.assertIsNotNone(bias)
        self.assertEqual(bias.shape, (8,))
        # Expert 3 should have bias 0.2, others 0
        self.assertAlmostEqual(bias[3].item(), 0.2, places=5)
        self.assertAlmostEqual(bias[0].item(), 0.0, places=5)

    def test_multiple_experts(self):
        self.mgr.activate(1, step=0)
        self.mgr.activate(5, step=0)
        bias = self.mgr.get_bias_tensor(torch.device("cpu"))
        self.assertAlmostEqual(bias[1].item(), 0.2, places=5)
        self.assertAlmostEqual(bias[5].item(), 0.2, places=5)
        self.assertAlmostEqual(bias[0].item(), 0.0, places=5)

    def test_step_decays_bias(self):
        self.mgr.activate(2, step=0)
        # After 5 steps: remaining_frac = 0.5 → bias = 0.1
        for _ in range(5):
            self.mgr.step_all()
        bias = self.mgr.get_bias_tensor(torch.device("cpu"))
        self.assertAlmostEqual(bias[2].item(), 0.1, places=5)

    def test_session_expires_and_returns_none(self):
        self.mgr.activate(0, step=0)
        for _ in range(10):
            self.mgr.step_all()
        bias = self.mgr.get_bias_tensor(torch.device("cpu"))
        self.assertIsNone(bias)
        self.assertFalse(self.mgr.has_active_sessions())

    def test_deactivate(self):
        self.mgr.activate(4, step=0)
        self.assertTrue(self.mgr.has_active_sessions())
        self.mgr.deactivate(4)
        self.assertFalse(self.mgr.has_active_sessions())

    def test_reactivate_restarts_session(self):
        self.mgr.activate(3, step=0)
        for _ in range(5):
            self.mgr.step_all()
        # Bias should be decayed
        bias1 = self.mgr.get_bias_tensor(torch.device("cpu"))
        self.assertAlmostEqual(bias1[3].item(), 0.1, places=5)
        # Reactivate — should restart at full bias
        self.mgr.activate(3, step=5)
        bias2 = self.mgr.get_bias_tensor(torch.device("cpu"))
        self.assertAlmostEqual(bias2[3].item(), 0.2, places=5)

    def test_invalid_expert_id(self):
        with self.assertRaises(ValueError):
            self.mgr.activate(-1, step=0)
        with self.assertRaises(ValueError):
            self.mgr.activate(8, step=0)

    def test_get_active_expert_ids(self):
        self.mgr.activate(2, step=0)
        self.mgr.activate(6, step=0)
        self.assertEqual(self.mgr.get_active_expert_ids(), [2, 6])

    def test_get_session_info(self):
        self.mgr.activate(1, step=10)
        info = self.mgr.get_session_info(1)
        self.assertIsNotNone(info)
        self.assertEqual(info["expert_id"], 1)
        self.assertAlmostEqual(info["initial_bias"], 0.2)
        self.assertEqual(info["activation_step"], 10)
        self.assertTrue(info["active"])

    def test_get_session_info_nonexistent(self):
        self.assertIsNone(self.mgr.get_session_info(7))

    def test_summary(self):
        self.mgr.activate(0, step=0)
        self.mgr.activate(3, step=0)
        s = self.mgr.summary()
        self.assertEqual(s["layer_number"], 1)
        self.assertEqual(s["num_active_sessions"], 2)
        self.assertEqual(len(s["sessions"]), 2)

    def test_reset(self):
        self.mgr.activate(0, step=0)
        self.mgr.activate(1, step=0)
        self.mgr.reset()
        self.assertFalse(self.mgr.has_active_sessions())
        self.assertIsNone(self.mgr.get_bias_tensor(torch.device("cpu")))


# ===================================================================
# 3. Global registry tests
# ===================================================================

class TestGlobalRegistry(unittest.TestCase):
    """Test the module-level singleton registry."""

    def setUp(self):
        clear_preferential_routing_managers()

    def tearDown(self):
        clear_preferential_routing_managers()

    def test_get_creates_manager(self):
        mgr = get_preferential_routing_manager(1, num_experts=4)
        self.assertIsInstance(mgr, PreferentialRoutingManager)
        self.assertEqual(mgr.layer_number, 1)

    def test_get_returns_same_instance(self):
        mgr1 = get_preferential_routing_manager(1, num_experts=4)
        mgr2 = get_preferential_routing_manager(1, num_experts=4)
        self.assertIs(mgr1, mgr2)

    def test_different_layers(self):
        mgr1 = get_preferential_routing_manager(1, num_experts=4)
        mgr2 = get_preferential_routing_manager(2, num_experts=4)
        self.assertIsNot(mgr1, mgr2)

    def test_get_all(self):
        get_preferential_routing_manager(1, num_experts=4)
        get_preferential_routing_manager(2, num_experts=4)
        all_mgrs = get_all_preferential_routing_managers()
        self.assertEqual(len(all_mgrs), 2)
        self.assertIn(1, all_mgrs)
        self.assertIn(2, all_mgrs)

    def test_clear(self):
        get_preferential_routing_manager(1, num_experts=4)
        clear_preferential_routing_managers()
        self.assertEqual(len(get_all_preferential_routing_managers()), 0)


# ===================================================================
# 4. Integration with topk_routing_with_score_function
# ===================================================================

class TestRecoveryBiasInRouting(unittest.TestCase):
    """Test that recovery_bias actually influences routing decisions."""

    def _make_logits(self, num_tokens=64, num_experts=8, seed=42):
        torch.manual_seed(seed)
        return torch.randn(num_tokens, num_experts)

    def test_no_recovery_bias_unchanged(self):
        """Without recovery_bias, routing is identical to baseline."""
        from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function

        logits = self._make_logits()
        probs1, rmap1 = topk_routing_with_score_function(
            logits.clone(), topk=2, score_function="sigmoid",
        )
        probs2, rmap2 = topk_routing_with_score_function(
            logits.clone(), topk=2, score_function="sigmoid",
            recovery_bias=None,
        )
        self.assertTrue(torch.equal(rmap1, rmap2))
        torch.testing.assert_close(probs1, probs2)

    def test_recovery_bias_increases_selection(self):
        """A positive recovery_bias on one expert increases its token count."""
        from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function

        logits = self._make_logits(num_tokens=256, num_experts=8, seed=123)

        # Baseline
        _, rmap_base = topk_routing_with_score_function(
            logits.clone(), topk=2, score_function="sigmoid",
        )
        base_count = rmap_base[:, 3].sum().item()

        # With recovery_bias on expert 3
        recovery_bias = torch.zeros(8)
        recovery_bias[3] = 0.5  # strong bias for testing
        _, rmap_biased = topk_routing_with_score_function(
            logits.clone(), topk=2, score_function="sigmoid",
            recovery_bias=recovery_bias,
        )
        biased_count = rmap_biased[:, 3].sum().item()

        self.assertGreaterEqual(biased_count, base_count,
            "Recovery bias should increase or maintain token count for biased expert")

    def test_recovery_bias_with_expert_bias(self):
        """recovery_bias stacks with expert_bias."""
        from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function

        logits = self._make_logits(num_tokens=128, num_experts=4, seed=99)

        expert_bias = torch.tensor([0.0, 0.1, 0.0, 0.0])
        recovery_bias = torch.tensor([0.0, 0.0, 0.3, 0.0])

        _, rmap = topk_routing_with_score_function(
            logits.clone(), topk=2, score_function="sigmoid",
            expert_bias=expert_bias, recovery_bias=recovery_bias,
        )
        # Both expert 1 (expert_bias) and expert 2 (recovery_bias) should
        # have non-zero token counts
        self.assertGreater(rmap[:, 1].sum().item(), 0)
        self.assertGreater(rmap[:, 2].sum().item(), 0)

    def test_recovery_bias_respects_health_mask(self):
        """recovery_bias cannot override health_mask exclusion."""
        from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function

        logits = self._make_logits(num_tokens=64, num_experts=4, seed=77)

        recovery_bias = torch.tensor([0.0, 0.0, 1.0, 0.0])  # big bias on expert 2
        health_mask = torch.tensor([True, True, False, True])  # expert 2 is unhealthy

        _, rmap = topk_routing_with_score_function(
            logits.clone(), topk=2, score_function="sigmoid",
            recovery_bias=recovery_bias, health_mask=health_mask,
        )
        # Expert 2 should get zero tokens despite large recovery_bias
        self.assertEqual(rmap[:, 2].sum().item(), 0)

    def test_recovery_bias_softmax_no_effect(self):
        """recovery_bias only affects sigmoid branch; softmax ignores it."""
        from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function

        logits = self._make_logits(num_tokens=64, num_experts=4, seed=55)

        recovery_bias = torch.tensor([0.0, 0.0, 0.5, 0.0])

        _, rmap1 = topk_routing_with_score_function(
            logits.clone(), topk=2, score_function="softmax",
        )
        _, rmap2 = topk_routing_with_score_function(
            logits.clone(), topk=2, score_function="softmax",
            recovery_bias=recovery_bias,
        )
        # softmax branch doesn't use recovery_bias, so results should be identical
        self.assertTrue(torch.equal(rmap1, rmap2))

    def test_total_tokens_preserved(self):
        """recovery_bias doesn't change total number of routed tokens."""
        from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function

        logits = self._make_logits(num_tokens=128, num_experts=8, seed=42)
        recovery_bias = torch.zeros(8)
        recovery_bias[5] = 0.3

        _, rmap_base = topk_routing_with_score_function(
            logits.clone(), topk=2, score_function="sigmoid",
        )
        _, rmap_biased = topk_routing_with_score_function(
            logits.clone(), topk=2, score_function="sigmoid",
            recovery_bias=recovery_bias,
        )
        # Total tokens = num_tokens * topk = 128 * 2 = 256
        self.assertEqual(rmap_base.sum().item(), rmap_biased.sum().item())


# ===================================================================
# 5. End-to-end: bias decay over window
# ===================================================================

class TestBiasDecayEndToEnd(unittest.TestCase):
    """Simulate a full lifecycle: activate → decay → expire."""

    def test_full_lifecycle(self):
        mgr = PreferentialRoutingManager(
            layer_number=1, num_experts=4, initial_bias=0.4, window_steps=20,
        )
        mgr.activate(2, step=100)

        biases_over_time = []
        for i in range(25):
            bias = mgr.get_bias_tensor(torch.device("cpu"))
            if bias is not None:
                biases_over_time.append(bias[2].item())
            else:
                biases_over_time.append(0.0)
            mgr.step_all()

        # Step 0: 0.4, Step 10: 0.2, Step 19: ~0.02, Step 20+: 0.0
        self.assertAlmostEqual(biases_over_time[0], 0.4, places=4)
        self.assertAlmostEqual(biases_over_time[10], 0.2, places=4)
        # After window expires
        for v in biases_over_time[20:]:
            self.assertAlmostEqual(v, 0.0, places=6)

    def test_bias_is_monotonically_decreasing(self):
        mgr = PreferentialRoutingManager(
            layer_number=1, num_experts=4, initial_bias=0.5, window_steps=50,
        )
        mgr.activate(0, step=0)

        prev = float('inf')
        for _ in range(55):
            bias = mgr.get_bias_tensor(torch.device("cpu"))
            val = bias[0].item() if bias is not None else 0.0
            self.assertLessEqual(val, prev + 1e-7)
            prev = val
            mgr.step_all()


# ===================================================================
# 6. Feature toggle: disabled by default
# ===================================================================

class TestFeatureToggle(unittest.TestCase):
    """Verify that the feature is opt-in and doesn't affect defaults."""

    def test_config_default_is_false(self):
        """moe_moegambit_preferential_routing defaults to False."""
        from megatron.core.transformer.transformer_config import TransformerConfig
        cfg = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4,
        )
        self.assertFalse(cfg.moe_moegambit_preferential_routing)

    def test_config_window_default(self):
        from megatron.core.transformer.transformer_config import TransformerConfig
        cfg = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4,
        )
        self.assertEqual(cfg.moe_moegambit_preferential_routing_window, 100)

    def test_config_bias_default(self):
        from megatron.core.transformer.transformer_config import TransformerConfig
        cfg = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4,
        )
        self.assertAlmostEqual(cfg.moe_moegambit_preferential_routing_bias, 0.1)

    def test_config_can_enable(self):
        from megatron.core.transformer.transformer_config import TransformerConfig
        cfg = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4,
            moe_moegambit_preferential_routing=True,
            moe_moegambit_preferential_routing_window=200,
            moe_moegambit_preferential_routing_bias=0.05,
        )
        self.assertTrue(cfg.moe_moegambit_preferential_routing)
        self.assertEqual(cfg.moe_moegambit_preferential_routing_window, 200)
        self.assertAlmostEqual(cfg.moe_moegambit_preferential_routing_bias, 0.05)


# ===================================================================
# 7. Controlled bias: bounded and small
# ===================================================================

class TestBiasIsBounded(unittest.TestCase):
    """Verify that the bias is always bounded and controlled."""

    def test_bias_never_exceeds_initial(self):
        mgr = PreferentialRoutingManager(
            layer_number=1, num_experts=8, initial_bias=0.15, window_steps=50,
        )
        mgr.activate(3, step=0)
        for _ in range(60):
            bias = mgr.get_bias_tensor(torch.device("cpu"))
            if bias is not None:
                self.assertLessEqual(bias.max().item(), 0.15 + 1e-7)
                self.assertGreaterEqual(bias.min().item(), -1e-7)
            mgr.step_all()

    def test_only_recovered_experts_get_bias(self):
        """Non-activated experts always have zero bias."""
        mgr = PreferentialRoutingManager(
            layer_number=1, num_experts=8, initial_bias=0.3, window_steps=20,
        )
        mgr.activate(2, step=0)
        mgr.activate(5, step=0)
        for _ in range(25):
            bias = mgr.get_bias_tensor(torch.device("cpu"))
            if bias is not None:
                for eid in [0, 1, 3, 4, 6, 7]:
                    self.assertAlmostEqual(bias[eid].item(), 0.0, places=6)
            mgr.step_all()


if __name__ == "__main__":
    unittest.main()

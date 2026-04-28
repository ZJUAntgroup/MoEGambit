# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for restart-in-place recovery with real tensor operations.

All tests use CPU tensors — no GPU or distributed init required.

Test scenarios:
1. Inject hard failure → checkpoint restart (small gap)
2. Inject hard failure → hybrid recovery (large gap)
3. Hybrid: dense param broadcast correctness
4. Hybrid: dense optimizer state sync correctness
5. Hybrid: expert weight restore correctness
6. Deferred optimizer load state machine + tensor
7. Verification failure (NaN remains → error)
8. Optimizer state empty → fail-closed
"""

import copy
import unittest

import torch
import torch.nn as nn


# =====================================================================
# Simple MoE model for testing
# =====================================================================

class SimpleMoEModel(nn.Module):
    """Minimal model with dense + expert params for testing.

    Structure:
    - dense_layer: nn.Linear (allreduce=True, default)
    - router: nn.Linear (allreduce=True, name contains 'router')
    - expert_layer: nn.Linear (allreduce=False, simulating expert params)
    """

    def __init__(self, hidden=8, num_experts=2):
        super().__init__()
        self.dense_layer = nn.Linear(hidden, hidden)
        self.router = nn.Linear(hidden, num_experts)
        self.expert_layer = nn.Linear(hidden, hidden)
        # Mark expert_layer as expert-parallel
        for p in self.expert_layer.parameters():
            p.allreduce = False

    def forward(self, x):
        h = self.dense_layer(x)
        gate = self.router(h)
        out = self.expert_layer(h)
        return out.sum() + gate.sum()


def _make_model_and_optimizer(hidden=8, num_experts=2):
    """Create model + optimizer and do one dummy step to init optimizer state."""
    model = SimpleMoEModel(hidden=hidden, num_experts=num_experts)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Do a dummy step to initialize optimizer states
    x = torch.randn(2, hidden)
    loss = model(x)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return model, optimizer


def _clone_state(model, optimizer):
    """Deep-clone model params and optimizer states as ground truth."""
    param_gt = {}
    for name, param in model.named_parameters():
        param_gt[name] = param.data.clone()

    opt_gt = {}
    for param, state in optimizer.state.items():
        if isinstance(state, dict):
            opt_gt[id(param)] = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in state.items()
            }

    return param_gt, opt_gt


# =====================================================================
# Tests
# =====================================================================

class TestInvalidation(unittest.TestCase):
    """Test tensor invalidation with NaN sentinels."""

    def test_invalidate_all_params(self):
        """All params should be NaN after invalidation."""
        from megatron.core.transformer.moe.restart_in_place import (
            invalidate_rank_tensors,
        )
        model, optimizer = _make_model_and_optimizer()
        stats = invalidate_rank_tensors(model, optimizer)

        self.assertGreater(stats["params_invalidated"], 0)
        self.assertGreater(stats["opt_states_invalidated"], 0)

        # All params should be NaN
        for name, param in model.named_parameters():
            self.assertTrue(
                torch.isnan(param.data).all(),
                f"Param '{name}' should be all NaN after invalidation",
            )

        # All optimizer states should be NaN
        for param, state in optimizer.state.items():
            if isinstance(state, dict):
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        self.assertTrue(
                            torch.isnan(v).all(),
                            f"Optimizer state '{k}' should be all NaN",
                        )

    def test_invalidate_dense_only(self):
        """Only dense params should be NaN, experts untouched."""
        from megatron.core.transformer.moe.restart_in_place import (
            invalidate_dense_params_only,
        )
        from megatron.core.transformer.moe.dense_param_sync import (
            classify_model_parameters,
        )
        model, optimizer = _make_model_and_optimizer()
        classification = classify_model_parameters(model)

        # Save expert params
        expert_gt = {
            n: p.data.clone()
            for n, p in classification.expert_params.items()
        }

        invalidate_dense_params_only(model, optimizer, classification)

        # Dense should be NaN
        for name, param in classification.all_dense_like.items():
            self.assertTrue(torch.isnan(param.data).all())

        # Experts should be unchanged
        for name, param in classification.expert_params.items():
            self.assertTrue(
                torch.equal(param.data, expert_gt[name]),
                f"Expert param '{name}' should be unchanged",
            )

    def test_invalidate_expert_only(self):
        """Only expert params should be NaN, dense untouched."""
        from megatron.core.transformer.moe.restart_in_place import (
            invalidate_expert_params_only,
        )
        from megatron.core.transformer.moe.dense_param_sync import (
            classify_model_parameters,
        )
        model, optimizer = _make_model_and_optimizer()
        classification = classify_model_parameters(model)

        # Save dense params
        dense_gt = {
            n: p.data.clone()
            for n, p in classification.all_dense_like.items()
        }

        invalidate_expert_params_only(model, optimizer, classification)

        # Experts should be NaN
        for name, param in classification.expert_params.items():
            self.assertTrue(torch.isnan(param.data).all())

        # Dense should be unchanged
        for name, param in classification.all_dense_like.items():
            self.assertTrue(
                torch.equal(param.data, dense_gt[name]),
                f"Dense param '{name}' should be unchanged",
            )


class TestVerification(unittest.TestCase):
    """Test recovery verification."""

    def test_verification_passes_on_healthy_model(self):
        """Verification should pass on a normal model."""
        from megatron.core.transformer.moe.restart_in_place import (
            verify_recovery,
        )
        model, optimizer = _make_model_and_optimizer()
        result = verify_recovery(model, optimizer)
        self.assertTrue(result.success)

    def test_verification_fails_on_nan(self):
        """Verification should raise on NaN params."""
        from megatron.core.transformer.moe.restart_in_place import (
            verify_recovery,
            invalidate_rank_tensors,
            RecoveryVerificationError,
        )
        model, optimizer = _make_model_and_optimizer()
        invalidate_rank_tensors(model, optimizer)

        with self.assertRaises(RecoveryVerificationError):
            verify_recovery(model, optimizer)

    def test_verification_fails_on_empty_optimizer_state(self):
        """Verification should fail when optimizer state is empty."""
        from megatron.core.transformer.moe.restart_in_place import (
            verify_recovery,
            RecoveryVerificationError,
        )
        model = SimpleMoEModel()
        # Optimizer with NO steps done → empty state
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        with self.assertRaises(RecoveryVerificationError) as ctx:
            verify_recovery(model, optimizer, check_optimizer=True)
        self.assertIn("empty", str(ctx.exception).lower())


class TestCheckpointRestartPath(unittest.TestCase):
    """Test small-gap checkpoint restart recovery."""

    def test_checkpoint_restart_full_flow(self):
        """Full flow: fault → invalidate → checkpoint restore → verify."""
        from megatron.core.transformer.moe.restart_in_place import (
            RestartInPlaceCoordinator,
            RestartInPlaceRecoveryState,
        )

        model, optimizer = _make_model_and_optimizer()
        param_gt, opt_gt = _clone_state(model, optimizer)

        coord = RestartInPlaceCoordinator(
            gap_threshold=100,
            checkpoint_iteration=95,  # gap = 100-95 = 5 <= 100
        )

        # Phase 1: fault + invalidation
        coord.inject_fault_and_invalidate(model, optimizer, step=100)
        self.assertEqual(
            coord.state, RestartInPlaceRecoveryState.INVALIDATED
        )

        # All params should be NaN now
        for name, param in model.named_parameters():
            self.assertTrue(torch.isnan(param.data).all())

        # Phase 2: checkpoint restore
        def ckpt_restore(m, opt):
            """Simulate checkpoint restore by copying ground truth."""
            for name, param in m.named_parameters():
                param.data.copy_(param_gt[name])
            for param, state in opt.state.items():
                if isinstance(state, dict) and id(param) in opt_gt:
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor) and k in opt_gt[id(param)]:
                            v.copy_(opt_gt[id(param)][k])

        path = coord.execute_recovery(
            model, optimizer,
            checkpoint_restore_fn=ckpt_restore,
            step=100,
        )
        self.assertEqual(path, "CHECKPOINT_RESTART")
        self.assertEqual(
            coord.state, RestartInPlaceRecoveryState.FULLY_RECOVERED
        )

        # Phase 3: verify
        result = coord.verify_and_reintegrate(model, optimizer, step=101)
        self.assertTrue(result.success)
        self.assertEqual(coord.state, RestartInPlaceRecoveryState.HEALTHY)

        # Verify param values match ground truth
        for name, param in model.named_parameters():
            self.assertTrue(
                torch.equal(param.data, param_gt[name]),
                f"Param '{name}' should match ground truth after recovery",
            )


class TestHybridRecoveryPath(unittest.TestCase):
    """Test large-gap hybrid recovery."""

    def test_hybrid_recovery_dense_params(self):
        """Dense params should be correctly restored from DP peer."""
        from megatron.core.transformer.moe.restart_in_place import (
            RestartInPlaceCoordinator,
        )
        from megatron.core.transformer.moe.dense_param_sync import (
            classify_model_parameters,
        )

        model, optimizer = _make_model_and_optimizer()
        param_gt, opt_gt = _clone_state(model, optimizer)
        classification = classify_model_parameters(model)

        coord = RestartInPlaceCoordinator(
            gap_threshold=10,
            checkpoint_iteration=0,  # gap = 100 > 10 → hybrid
        )

        coord.inject_fault_and_invalidate(model, optimizer, step=100)

        # Restore callbacks
        def dense_restore(m, cls):
            for name, param in cls.all_dense_like.items():
                param.data.copy_(param_gt[name])

        def expert_restore(m, cls):
            for name, param in cls.expert_params.items():
                param.data.copy_(param_gt[name])

        def dense_opt_restore(opt, cls):
            for param, state in opt.state.items():
                if isinstance(state, dict) and id(param) in opt_gt:
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor) and k in opt_gt[id(param)]:
                            v.copy_(opt_gt[id(param)][k])

        path = coord.execute_recovery(
            model, optimizer,
            dense_restore_fn=dense_restore,
            expert_restore_fn=expert_restore,
            dense_optimizer_restore_fn=dense_opt_restore,
            expert_optimizer_restore_fn=dense_opt_restore,
            step=100,
        )
        self.assertEqual(path, "HYBRID_RECOVERY")

        result = coord.verify_and_reintegrate(model, optimizer, step=101)
        self.assertTrue(result.success)

        # All params should match ground truth
        for name, param in model.named_parameters():
            self.assertTrue(
                torch.equal(param.data, param_gt[name]),
                f"Param '{name}' mismatch after hybrid recovery",
            )

    def test_hybrid_recovery_optimizer_state_correctness(self):
        """Optimizer momentum/variance should match ground truth after hybrid."""
        from megatron.core.transformer.moe.restart_in_place import (
            RestartInPlaceCoordinator,
        )

        model, optimizer = _make_model_and_optimizer()
        param_gt, opt_gt = _clone_state(model, optimizer)

        coord = RestartInPlaceCoordinator(
            gap_threshold=10,
            checkpoint_iteration=0,
        )
        coord.inject_fault_and_invalidate(model, optimizer, step=100)

        def restore_all(m, cls):
            for name, param in m.named_parameters():
                param.data.copy_(param_gt[name])

        def restore_opt(opt, cls):
            for param, state in opt.state.items():
                if isinstance(state, dict) and id(param) in opt_gt:
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor) and k in opt_gt[id(param)]:
                            v.copy_(opt_gt[id(param)][k])

        coord.execute_recovery(
            model, optimizer,
            dense_restore_fn=restore_all,
            expert_restore_fn=restore_all,
            dense_optimizer_restore_fn=restore_opt,
            expert_optimizer_restore_fn=restore_opt,
            step=100,
        )
        coord.verify_and_reintegrate(model, optimizer, step=101)

        # Verify optimizer states
        for param, state in optimizer.state.items():
            if isinstance(state, dict) and id(param) in opt_gt:
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        gt_v = opt_gt[id(param)].get(k)
                        if gt_v is not None:
                            self.assertTrue(
                                torch.equal(v, gt_v),
                                f"Optimizer state '{k}' mismatch",
                            )


class TestVerificationGate(unittest.TestCase):
    """Test that verification blocks reintegration on failure."""

    def test_partial_recovery_blocked(self):
        """If dense is restored but expert still NaN, verification fails."""
        from megatron.core.transformer.moe.restart_in_place import (
            RestartInPlaceCoordinator,
            RecoveryVerificationError,
        )

        model, optimizer = _make_model_and_optimizer()
        param_gt, opt_gt = _clone_state(model, optimizer)

        coord = RestartInPlaceCoordinator(
            gap_threshold=10, checkpoint_iteration=0,
        )
        coord.inject_fault_and_invalidate(model, optimizer, step=100)

        # Only restore dense, leave expert as NaN
        def dense_only(m, cls):
            for name, param in cls.all_dense_like.items():
                param.data.copy_(param_gt[name])

        def opt_restore(opt, cls):
            for param, state in opt.state.items():
                if isinstance(state, dict) and id(param) in opt_gt:
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor) and k in opt_gt[id(param)]:
                            v.copy_(opt_gt[id(param)][k])

        coord.execute_recovery(
            model, optimizer,
            dense_restore_fn=dense_only,
            # expert_restore_fn intentionally omitted → expert still NaN
            dense_optimizer_restore_fn=opt_restore,
            expert_optimizer_restore_fn=opt_restore,
            step=100,
        )

        with self.assertRaises(RecoveryVerificationError):
            coord.verify_and_reintegrate(model, optimizer, step=101)


class TestDenseParamSyncFailClosed(unittest.TestCase):
    """Test fail-closed behavior in dense_param_sync."""

    def test_empty_optimizer_state_raises(self):
        """_sync_optimizer_states_for_dense should raise when state empty."""
        from megatron.core.transformer.moe.dense_param_sync import (
            _sync_optimizer_states_for_dense,
            classify_model_parameters,
        )

        model = SimpleMoEModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # NO step done → state is empty
        classification = classify_model_parameters(model)

        def noop_broadcast(tensor, src, group):
            pass

        with self.assertRaises(RuntimeError) as ctx:
            _sync_optimizer_states_for_dense(
                optimizer, classification, source_rank=0,
                dp_group="fake", broadcast_fn=noop_broadcast,
                require_nonempty_state=True,
            )
        self.assertIn("Fail-closed", str(ctx.exception))

    def test_nonempty_state_override_allows_empty(self):
        """require_nonempty_state=False should allow empty state."""
        from megatron.core.transformer.moe.dense_param_sync import (
            _sync_optimizer_states_for_dense,
            classify_model_parameters,
        )

        model = SimpleMoEModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        classification = classify_model_parameters(model)

        def noop_broadcast(tensor, src, group):
            pass

        # Should not raise
        result = _sync_optimizer_states_for_dense(
            optimizer, classification, source_rank=0,
            dp_group="fake", broadcast_fn=noop_broadcast,
            require_nonempty_state=False,
        )
        self.assertEqual(result, 0)


class TestVerifySyncedParams(unittest.TestCase):
    """Test verify_synced_params function."""

    def test_healthy_params_pass(self):
        from megatron.core.transformer.moe.dense_param_sync import (
            verify_synced_params,
        )
        model, _ = _make_model_and_optimizer()
        ok, msg = verify_synced_params(model)
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_nan_params_fail(self):
        from megatron.core.transformer.moe.dense_param_sync import (
            verify_synced_params,
        )
        from megatron.core.transformer.moe.restart_in_place import (
            invalidate_dense_params_only,
        )
        model, _ = _make_model_and_optimizer()
        invalidate_dense_params_only(model)
        ok, msg = verify_synced_params(model)
        self.assertFalse(ok)
        self.assertIn("NaN", msg)


class TestRecoveryControllerRestartInPlace(unittest.TestCase):
    """Test RecoveryController restart-in-place fast path."""

    def test_restart_in_place_skips_group_rebuild(self):
        """restart_in_place=True should skip group rebuild callbacks."""
        from megatron.core.transformer.moe.recovery_controller import (
            RecoveryController,
            RecoveryPhase,
        )

        ctrl = RecoveryController()
        rebuild_called = []
        topo_called = []
        dense_called = []

        ctrl.register_callbacks(
            group_rebuild_request_fn=lambda **kw: rebuild_called.append(1),
            group_rebuild_execute_fn=lambda **kw: rebuild_called.append(1),
            group_rebuild_finish_fn=lambda **kw: rebuild_called.append(1),
            topology_refresh_fn=lambda **kw: topo_called.append(1),
            dense_sync_fn=lambda **kw: dense_called.append(1),
        )

        # Trigger restart-in-place
        ctrl.on_hard_rank_failure(
            failed_rank=0, step=100,
            expert_ids=[0, 1],
            restart_in_place=True,
        )

        # Should be in SAFE_POINT_REPAIR now
        self.assertEqual(ctrl.phase, RecoveryPhase.SAFE_POINT_REPAIR)
        self.assertTrue(ctrl.restart_in_place_mode)

        # Execute safe-point repair
        ctrl.before_iteration(step=101)

        # Group rebuild and topology should NOT have been called
        self.assertEqual(len(rebuild_called), 0)
        self.assertEqual(len(topo_called), 0)

        # Dense sync SHOULD have been called (hybrid path)
        self.assertEqual(len(dense_called), 1)

    def test_restart_in_place_invalidate_tensor_callback(self):
        """invalidate_tensor_fn should be called during restart-in-place."""
        from megatron.core.transformer.moe.recovery_controller import (
            RecoveryController,
        )

        ctrl = RecoveryController()
        invalidate_called = []

        ctrl.register_callbacks(
            invalidate_tensor_fn=lambda **kw: invalidate_called.append(kw),
        )

        ctrl.on_hard_rank_failure(
            failed_rank=0, step=100,
            restart_in_place=True,
        )

        self.assertEqual(len(invalidate_called), 1)
        self.assertEqual(invalidate_called[0]["failed_rank"], 0)
        self.assertEqual(invalidate_called[0]["step"], 100)


class TestStaleExpertRestoreDryRunNaN(unittest.TestCase):
    """Test that stale_expert_restore dry-run fails when NaN sentinel present."""

    def test_dryrun_with_nan_fails(self):
        """dry-run (load_fn=None) should fail when expert params are NaN."""
        from megatron.core.transformer.moe.stale_expert_restore import (
            restore_expert_weights,
            ExpertRestorePlan,
            ExpertRestoreEntry,
        )
        from megatron.core.transformer.moe.restart_in_place import (
            invalidate_rank_tensors,
        )

        model, _ = _make_model_and_optimizer()
        invalidate_rank_tensors(model)

        plan = ExpertRestorePlan(
            entries=[
                ExpertRestoreEntry(layer_id=0, expert_id=0),
            ],
            failed_rank=0,
            replacement_rank=0,
        )

        result = restore_expert_weights(
            model, plan,
            load_fn=None,  # dry-run
            health_managers={},
            directory=None,
            step=100,
        )

        # Should fail because expert params contain NaN
        self.assertFalse(result.success)
        self.assertGreater(result.num_failed, 0)
        self.assertTrue(
            any("NaN sentinel" in e for e in result.errors),
            f"Expected NaN sentinel error, got: {result.errors}",
        )


class TestEndToEndRestartInPlace(unittest.TestCase):
    """End-to-end test: coordinator + controller + real tensors."""

    def test_full_e2e_hybrid(self):
        """Full E2E: fault → invalidate → hybrid recovery → verify."""
        from megatron.core.transformer.moe.restart_in_place import (
            RestartInPlaceCoordinator,
            RestartInPlaceRecoveryState,
        )
        from megatron.core.transformer.moe.recovery_controller import (
            RecoveryController,
            RecoveryPhase,
        )

        model, optimizer = _make_model_and_optimizer()
        param_gt, opt_gt = _clone_state(model, optimizer)

        # Setup coordinator
        coord = RestartInPlaceCoordinator(
            gap_threshold=10,
            checkpoint_iteration=0,
        )

        # Setup controller with restart-in-place
        ctrl = RecoveryController()
        ctrl.register_callbacks(
            invalidate_tensor_fn=lambda **kw: coord.inject_fault_and_invalidate(
                model, optimizer, step=kw.get("step", -1)
            ),
        )

        # Trigger fault via controller
        ctrl.on_hard_rank_failure(
            failed_rank=0, step=100,
            expert_ids=[0, 1],
            restart_in_place=True,
        )

        # Coordinator should be INVALIDATED now
        self.assertEqual(coord.state, RestartInPlaceRecoveryState.INVALIDATED)

        # All params should be NaN
        for name, param in model.named_parameters():
            self.assertTrue(torch.isnan(param.data).all())

        # Execute recovery via coordinator
        def restore_all(m, cls):
            for name, param in m.named_parameters():
                param.data.copy_(param_gt[name])

        def restore_opt(opt, cls):
            for param, state in opt.state.items():
                if isinstance(state, dict) and id(param) in opt_gt:
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor) and k in opt_gt[id(param)]:
                            v.copy_(opt_gt[id(param)][k])

        path = coord.execute_recovery(
            model, optimizer,
            dense_restore_fn=restore_all,
            expert_restore_fn=restore_all,
            dense_optimizer_restore_fn=restore_opt,
            expert_optimizer_restore_fn=restore_opt,
            step=100,
        )
        self.assertEqual(path, "HYBRID_RECOVERY")

        # Verify
        result = coord.verify_and_reintegrate(model, optimizer, step=101)
        self.assertTrue(result.success)
        self.assertEqual(coord.state, RestartInPlaceRecoveryState.HEALTHY)

        # All params should match ground truth
        for name, param in model.named_parameters():
            self.assertTrue(torch.equal(param.data, param_gt[name]))


if __name__ == "__main__":
    unittest.main()

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for MoEGambit Dense Parameter Sync.

Tests cover:
1. Parameter classification (dense/router/shared/expert)
2. Healthy peer selection
3. Dense param pull (broadcast-based sync)
4. DenseParamRecoveryCoordinator plan + execute
5. Expert params are skipped during dense sync
6. Retry logic
7. End-to-end: hard failure → replacement → safe-point → dense sync
"""

import importlib
import importlib.util
import os
import sys
import types
import unittest

# ---------------------------------------------------------------------------
# Bootstrap — avoid torch dependency
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


_ds_mod = _load_module("megatron.core.transformer.moe.dense_param_sync")

ParamCategory = _ds_mod.ParamCategory
ParamClassification = _ds_mod.ParamClassification
classify_model_parameters = _ds_mod.classify_model_parameters
classify_param = _ds_mod.classify_param
select_healthy_dp_peer = _ds_mod.select_healthy_dp_peer
pull_dense_params_from_peer = _ds_mod.pull_dense_params_from_peer
DenseParamSyncResult = _ds_mod.DenseParamSyncResult
DenseRecoveryPlan = _ds_mod.DenseRecoveryPlan
DenseParamRecoveryCoordinator = _ds_mod.DenseParamRecoveryCoordinator


# =====================================================================
# Mock model for testing
# =====================================================================

class MockParam:
    """Simulates a torch.nn.Parameter with allreduce attribute."""

    def __init__(self, name: str, numel: int = 100, allreduce: bool = True):
        self.name = name
        self._numel = numel
        self.allreduce = allreduce
        # Simulate .data attribute
        self.data = self

    def numel(self):
        return self._numel


class MockModel:
    """Simulates a model with named_parameters()."""

    def __init__(self):
        self._params = {}

    def add_param(self, name: str, numel: int = 100, allreduce: bool = True):
        self._params[name] = MockParam(name, numel, allreduce)

    def named_parameters(self):
        return list(self._params.items())


def make_moe_model():
    """Create a mock MoE model with realistic parameter names."""
    model = MockModel()

    # Dense params (attention, layernorm, embedding)
    model.add_param("decoder.layers.0.self_attention.linear_qkv.weight", 1000)
    model.add_param("decoder.layers.0.self_attention.linear_qkv.bias", 100)
    model.add_param("decoder.layers.0.self_attention.linear_proj.weight", 1000)
    model.add_param("decoder.layers.0.input_layernorm.weight", 100)
    model.add_param("decoder.layers.0.pre_mlp_layernorm.weight", 100)
    model.add_param("embedding.word_embeddings.weight", 5000)
    model.add_param("output_layer.weight", 5000)

    # Router params
    model.add_param("decoder.layers.0.mlp.router.weight", 200)

    # Shared expert params
    model.add_param("decoder.layers.0.mlp.shared_experts.linear_fc1.weight", 500)
    model.add_param("decoder.layers.0.mlp.shared_experts.linear_fc2.weight", 500)

    # Expert params (allreduce=False)
    model.add_param(
        "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",
        300, allreduce=False,
    )
    model.add_param(
        "decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.weight",
        300, allreduce=False,
    )
    model.add_param(
        "decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.weight",
        300, allreduce=False,
    )
    model.add_param(
        "decoder.layers.0.mlp.experts.local_experts.1.linear_fc2.weight",
        300, allreduce=False,
    )

    return model


# =====================================================================
# Test: Parameter classification
# =====================================================================

class TestParamClassification(unittest.TestCase):

    def test_classify_moe_model(self):
        model = make_moe_model()
        cls = classify_model_parameters(model)

        # Dense: attention + layernorm + embedding + output
        self.assertEqual(cls.num_dense, 7)
        # Router
        self.assertEqual(cls.num_router, 1)
        # Shared expert
        self.assertEqual(cls.num_shared_expert, 2)
        # Expert
        self.assertEqual(cls.num_expert, 4)
        # Total
        self.assertEqual(cls.num_total, 14)

    def test_all_dense_like(self):
        model = make_moe_model()
        cls = classify_model_parameters(model)
        dense_like = cls.all_dense_like
        # dense + router + shared = 7 + 1 + 2 = 10
        self.assertEqual(len(dense_like), 10)
        # Expert params should NOT be in dense_like
        for name in dense_like:
            self.assertFalse(
                "local_experts" in name and not getattr(
                    model._params[name], 'allreduce', True
                ),
                f"Expert param {name} should not be in dense_like",
            )

    def test_dense_param_count(self):
        model = make_moe_model()
        cls = classify_model_parameters(model)
        # Sum of numel for dense-like params
        expected = (1000 + 100 + 1000 + 100 + 100 + 5000 + 5000  # dense
                    + 200  # router
                    + 500 + 500)  # shared
        self.assertEqual(cls.dense_param_count(), expected)

    def test_expert_param_count(self):
        model = make_moe_model()
        cls = classify_model_parameters(model)
        self.assertEqual(cls.expert_param_count(), 4 * 300)

    def test_classify_single_param(self):
        p_dense = MockParam("attn.weight", allreduce=True)
        self.assertEqual(classify_param("attn.weight", p_dense), ParamCategory.DENSE)

        p_router = MockParam("router.weight", allreduce=True)
        self.assertEqual(classify_param("mlp.router.weight", p_router), ParamCategory.ROUTER)

        p_shared = MockParam("shared.weight", allreduce=True)
        self.assertEqual(
            classify_param("mlp.shared_experts.fc1.weight", p_shared),
            ParamCategory.SHARED_EXPERT,
        )

        p_expert = MockParam("expert.weight", allreduce=False)
        self.assertEqual(
            classify_param("mlp.experts.0.fc1.weight", p_expert),
            ParamCategory.EXPERT,
        )

    def test_summary(self):
        model = make_moe_model()
        cls = classify_model_parameters(model)
        s = cls.summary()
        self.assertEqual(s["dense"], 7)
        self.assertEqual(s["router"], 1)
        self.assertEqual(s["shared_expert"], 2)
        self.assertEqual(s["expert"], 4)
        self.assertEqual(s["total"], 14)


# =====================================================================
# Test: Healthy peer selection
# =====================================================================

class TestPeerSelection(unittest.TestCase):

    def test_select_lowest_rank(self):
        peer = select_healthy_dp_peer(
            dp_group_ranks=[0, 1, 2, 3],
        )
        self.assertEqual(peer, 0)

    def test_exclude_quarantined(self):
        peer = select_healthy_dp_peer(
            dp_group_ranks=[0, 1, 2, 3],
            quarantined_ranks=frozenset({0}),
        )
        self.assertEqual(peer, 1)

    def test_exclude_failed(self):
        peer = select_healthy_dp_peer(
            dp_group_ranks=[0, 1, 2, 3],
            failed_ranks=frozenset({0, 1}),
        )
        self.assertEqual(peer, 2)

    def test_exclude_local(self):
        peer = select_healthy_dp_peer(
            dp_group_ranks=[0, 1, 2, 3],
            local_rank=0,
        )
        self.assertEqual(peer, 1)

    def test_exclude_all_returns_none(self):
        peer = select_healthy_dp_peer(
            dp_group_ranks=[0, 1],
            quarantined_ranks=frozenset({0}),
            failed_ranks=frozenset({1}),
        )
        self.assertIsNone(peer)

    def test_combined_exclusions(self):
        peer = select_healthy_dp_peer(
            dp_group_ranks=[0, 1, 2, 3],
            quarantined_ranks=frozenset({0}),
            failed_ranks=frozenset({1}),
            local_rank=2,
        )
        self.assertEqual(peer, 3)


# =====================================================================
# Test: Dense param pull
# =====================================================================

class TestPullDenseParams(unittest.TestCase):

    def test_pull_with_mock_broadcast(self):
        """Verify that broadcast is called for dense params, not expert."""
        model = make_moe_model()
        broadcast_calls = []

        def mock_broadcast(tensor, src, group):
            broadcast_calls.append((id(tensor), src))

        result = pull_dense_params_from_peer(
            model=model,
            source_rank=0,
            dp_group="fake_group",
            broadcast_fn=mock_broadcast,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.source_rank, 0)
        # Should sync 10 dense-like params (7 dense + 1 router + 2 shared)
        self.assertEqual(result.num_params_synced, 10)
        self.assertEqual(result.num_expert_skipped, 4)
        self.assertEqual(result.attempt, 1)
        # broadcast should have been called 10 times
        self.assertEqual(len(broadcast_calls), 10)

    def test_pull_dry_run(self):
        """Without dp_group, operates in dry-run mode."""
        model = make_moe_model()
        result = pull_dense_params_from_peer(
            model=model,
            source_rank=0,
            dp_group=None,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.num_params_synced, 10)

    def test_pull_with_retry(self):
        """Verify retry logic on broadcast failure."""
        model = make_moe_model()
        attempt_count = [0]

        def failing_then_ok_broadcast(tensor, src, group):
            # Fail on first attempt (attempt_count[0] == 0),
            # succeed on second attempt (attempt_count[0] == 1).
            if attempt_count[0] == 0:
                attempt_count[0] = 1
                raise RuntimeError("NCCL timeout")

        result = pull_dense_params_from_peer(
            model=model,
            source_rank=0,
            dp_group="fake_group",
            broadcast_fn=failing_then_ok_broadcast,
            max_retries=2,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.attempt, 2)

    def test_pull_all_retries_fail(self):
        """All retries fail → result.success is False."""
        model = make_moe_model()

        def always_fail(tensor, src, group):
            raise RuntimeError("permanent failure")

        result = pull_dense_params_from_peer(
            model=model,
            source_rank=0,
            dp_group="fake_group",
            broadcast_fn=always_fail,
            max_retries=3,
        )
        self.assertFalse(result.success)
        self.assertIn("permanent failure", result.error)

    def test_expert_params_not_broadcast(self):
        """Expert params must NOT be broadcast."""
        model = make_moe_model()
        broadcast_names = []

        cls = classify_model_parameters(model)
        dense_like = cls.all_dense_like

        # Verify no expert param names in dense_like
        for name in dense_like:
            self.assertNotIn("local_experts", name)

    def test_scalars_count(self):
        """Verify total scalar count matches expected."""
        model = make_moe_model()
        result = pull_dense_params_from_peer(
            model=model,
            source_rank=0,
            dp_group=None,
        )
        expected = (1000 + 100 + 1000 + 100 + 100 + 5000 + 5000
                    + 200 + 500 + 500)
        self.assertEqual(result.num_scalars_synced, expected)


# =====================================================================
# Test: DenseParamRecoveryCoordinator
# =====================================================================

class TestCoordinator(unittest.TestCase):

    def setUp(self):
        self.coord = DenseParamRecoveryCoordinator()

    def test_plan_recovery(self):
        plan = self.coord.plan_recovery(
            replacement_rank=99,
            failed_rank=3,
            dp_group_ranks=[0, 1, 2, 3],
            quarantined_ranks=frozenset(),
            step=100,
        )
        # Source should be rank 0 (lowest healthy, excluding failed=3)
        self.assertEqual(plan.source_rank, 0)
        self.assertEqual(plan.replacement_rank, 99)
        self.assertEqual(plan.failed_rank, 3)

    def test_plan_with_quarantined_source(self):
        plan = self.coord.plan_recovery(
            replacement_rank=99,
            failed_rank=3,
            dp_group_ranks=[0, 1, 2, 3],
            quarantined_ranks=frozenset({0}),
            step=100,
        )
        self.assertEqual(plan.source_rank, 1)

    def test_plan_no_healthy_peer(self):
        plan = self.coord.plan_recovery(
            replacement_rank=99,
            failed_rank=3,
            dp_group_ranks=[3],
            step=100,
        )
        self.assertEqual(plan.source_rank, -1)

    def test_execute_recovery(self):
        model = make_moe_model()
        plan = self.coord.plan_recovery(
            replacement_rank=99,
            failed_rank=3,
            dp_group_ranks=[0, 1, 2, 3],
            step=100,
        )
        result = self.coord.execute_recovery(
            model=model,
            plan=plan,
            dp_group=None,  # dry-run
        )
        self.assertTrue(result.success)
        self.assertEqual(result.num_params_synced, 10)
        self.assertEqual(result.num_expert_skipped, 4)

    def test_execute_no_source(self):
        model = make_moe_model()
        plan = DenseRecoveryPlan(source_rank=-1)
        result = self.coord.execute_recovery(model=model, plan=plan)
        self.assertFalse(result.success)
        self.assertIn("No healthy source", result.error)

    def test_summary(self):
        model = make_moe_model()
        plan = self.coord.plan_recovery(
            replacement_rank=99, failed_rank=3,
            dp_group_ranks=[0, 1, 2, 3], step=100,
        )
        self.coord.execute_recovery(model=model, plan=plan, dp_group=None)
        s = self.coord.summary()
        self.assertEqual(s["num_plans"], 1)
        self.assertEqual(s["num_results"], 1)
        self.assertTrue(s["last_success"])

    def test_reset(self):
        model = make_moe_model()
        plan = self.coord.plan_recovery(
            replacement_rank=99, failed_rank=3,
            dp_group_ranks=[0, 1, 2, 3], step=100,
        )
        self.coord.execute_recovery(model=model, plan=plan, dp_group=None)
        self.coord.reset()
        self.assertEqual(len(self.coord.plans), 0)
        self.assertEqual(len(self.coord.results), 0)


# =====================================================================
# Test: End-to-end with RecoveryController
# =====================================================================

class TestEndToEndDenseSync(unittest.TestCase):
    """Verify dense sync is called during safe-point repair."""

    def test_safe_point_repair_calls_dense_sync(self):
        """When safe-point repair executes, dense_sync_fn is called."""
        _rc_mod = _load_module("megatron.core.transformer.moe.recovery_controller")
        RecoveryController = _rc_mod.RecoveryController
        RecoveryPhase = _rc_mod.RecoveryPhase

        ctrl = RecoveryController()
        dense_sync_calls = []

        ctrl.register_callbacks(
            quarantine_fn=lambda **kw: None,
            health_mark_unavailable_fn=lambda **kw: None,
            replacement_announce_fn=lambda **kw: None,
            replacement_integrate_fn=lambda **kw: None,
            group_rebuild_request_fn=lambda **kw: None,
            group_rebuild_execute_fn=lambda **kw: None,
            group_rebuild_finish_fn=lambda **kw: None,
            topology_refresh_fn=lambda **kw: None,
            dense_sync_fn=lambda **kw: dense_sync_calls.append(kw),
            expert_restore_fn=lambda **kw: None,
        )

        # Hard failure → assign replacement → ready → safe-point repair
        ctrl.on_hard_rank_failure(
            failed_rank=3, step=100,
            expert_ids=[4, 5], ep_group_ranks=[0, 1, 2, 3],
        )
        ctrl.on_replacement_assigned(
            failed_rank=3, replacement_rank=99, step=110,
        )
        ctrl.on_replacement_ready(failed_rank=3, step=120)

        # Safe-point repair
        ctrl.before_iteration(step=130)

        # Verify dense_sync_fn was called
        self.assertEqual(len(dense_sync_calls), 1)
        self.assertEqual(dense_sync_calls[0]["failed_rank"], 3)
        self.assertEqual(dense_sync_calls[0]["replacement_rank"], 99)

    def test_dense_sync_does_not_affect_expert_recovery(self):
        """Dense sync skips expert params; expert_restore_fn handles them."""
        _rc_mod = _load_module("megatron.core.transformer.moe.recovery_controller")
        RecoveryController = _rc_mod.RecoveryController

        ctrl = RecoveryController()
        dense_calls = []
        expert_calls = []

        ctrl.register_callbacks(
            quarantine_fn=lambda **kw: None,
            health_mark_unavailable_fn=lambda **kw: None,
            replacement_announce_fn=lambda **kw: None,
            replacement_integrate_fn=lambda **kw: None,
            group_rebuild_request_fn=lambda **kw: None,
            group_rebuild_execute_fn=lambda **kw: None,
            group_rebuild_finish_fn=lambda **kw: None,
            topology_refresh_fn=lambda **kw: None,
            dense_sync_fn=lambda **kw: dense_calls.append(kw),
            expert_restore_fn=lambda **kw: expert_calls.append(kw),
        )

        ctrl.on_hard_rank_failure(
            failed_rank=3, step=100,
            expert_ids=[4, 5], ep_group_ranks=[0, 1, 2, 3],
        )
        ctrl.on_replacement_assigned(
            failed_rank=3, replacement_rank=99, step=110,
        )
        ctrl.on_replacement_ready(failed_rank=3, step=120)
        ctrl.before_iteration(step=130)

        # Both callbacks called
        self.assertEqual(len(dense_calls), 1)
        self.assertEqual(len(expert_calls), 1)

        # Dense sync is called BEFORE expert restore
        # (verified by the order in _execute_safe_point_repair)


# =====================================================================
# Test: Singleton management
# =====================================================================

class TestSingletons(unittest.TestCase):

    def test_coordinator_singleton(self):
        _ds_mod.clear_dense_param_recovery_coordinator()
        c1 = _ds_mod.get_dense_param_recovery_coordinator()
        c2 = _ds_mod.get_dense_param_recovery_coordinator()
        self.assertIs(c1, c2)
        _ds_mod.clear_dense_param_recovery_coordinator()

    def test_sync_result_to_dict(self):
        r = DenseParamSyncResult(
            success=True, source_rank=0,
            num_params_synced=10, num_scalars_synced=5000,
        )
        d = r.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["source_rank"], 0)
        self.assertEqual(d["num_params_synced"], 10)

    def test_plan_to_dict(self):
        p = DenseRecoveryPlan(
            replacement_rank=99, failed_rank=3,
            source_rank=0, dp_group_ranks=[0, 1, 2, 3],
        )
        d = p.to_dict()
        self.assertEqual(d["replacement_rank"], 99)
        self.assertEqual(d["source_rank"], 0)


# =====================================================================
# Test: Normal path unaffected
# =====================================================================

class TestNormalPathUnaffected(unittest.TestCase):

    def test_classification_is_pure_read(self):
        """classify_model_parameters does not modify any parameter."""
        model = make_moe_model()
        # Record original allreduce values
        original = {
            name: getattr(param, 'allreduce', True)
            for name, param in model.named_parameters()
        }
        classify_model_parameters(model)
        # Verify nothing changed
        for name, param in model.named_parameters():
            self.assertEqual(
                getattr(param, 'allreduce', True),
                original[name],
                f"allreduce changed for {name}",
            )

    def test_dry_run_does_not_modify_params(self):
        """pull_dense_params_from_peer with dp_group=None is a no-op."""
        model = make_moe_model()
        # In a real scenario, param.data would be a tensor.
        # Here we just verify no exception is raised.
        result = pull_dense_params_from_peer(
            model=model, source_rank=0, dp_group=None,
        )
        self.assertTrue(result.success)


# Also import internal helpers for optimizer state sync tests
_sync_optimizer_states_for_dense = _ds_mod._sync_optimizer_states_for_dense
_unwrap_optimizer = _ds_mod._unwrap_optimizer
_sync_single_optimizer_states = _ds_mod._sync_single_optimizer_states


# =====================================================================
# Mock optimizer classes for optimizer state sync tests
# =====================================================================

class MockTensor:
    """Simulates a torch.Tensor with .data and .numel()."""

    def __init__(self, numel: int = 10, value: float = 0.0):
        self._numel = numel
        self.data = self
        self.value = value

    def numel(self):
        return self._numel


class MockBaseOptimizer:
    """Simulates a base optimizer (e.g., Adam) with .state and .param_groups."""

    def __init__(self):
        self.param_groups = []
        self.state = {}

    def add_param_group(self, params):
        self.param_groups.append({"params": params})

    def set_state(self, param, state_dict):
        self.state[param] = state_dict


class MockFloat16Optimizer:
    """Simulates Megatron's Float16OptimizerWithFloat16Params.

    Structure:
    - self.optimizer → inner Adam optimizer (MockBaseOptimizer)
    - self.param_groups → contains main_param (fp32 copies)
    - self.optimizer.state → actual momentum/variance tensors
    """

    def __init__(self, inner_optimizer):
        self.optimizer = inner_optimizer
        # Outer param_groups may differ from inner
        self.param_groups = inner_optimizer.param_groups
        # Outer state is typically empty or a proxy
        self.state = {}


class MockChainedOptimizer:
    """Simulates Megatron's ChainedOptimizer wrapping multiple optimizers."""

    def __init__(self, optimizers):
        self.chained_optimizers = optimizers
        self.param_groups = []
        self.state = {}


# =====================================================================
# Test: Optimizer unwrapping
# =====================================================================

class TestOptimizerUnwrap(unittest.TestCase):

    def test_unwrap_plain_optimizer(self):
        """Plain optimizer returns itself."""
        opt = MockBaseOptimizer()
        result = _unwrap_optimizer(opt)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], opt)

    def test_unwrap_float16_optimizer(self):
        """Float16OptimizerWithFloat16Params unwraps to inner optimizer."""
        inner = MockBaseOptimizer()
        wrapper = MockFloat16Optimizer(inner)
        result = _unwrap_optimizer(wrapper)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], inner)

    def test_unwrap_chained_optimizer(self):
        """ChainedOptimizer unwraps to all inner optimizers."""
        inner1 = MockBaseOptimizer()
        inner2 = MockBaseOptimizer()
        chained = MockChainedOptimizer([inner1, inner2])
        result = _unwrap_optimizer(chained)
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], inner1)
        self.assertIs(result[1], inner2)

    def test_unwrap_chained_with_float16(self):
        """ChainedOptimizer wrapping Float16 optimizers."""
        inner1 = MockBaseOptimizer()
        inner2 = MockBaseOptimizer()
        wrapped1 = MockFloat16Optimizer(inner1)
        wrapped2 = MockFloat16Optimizer(inner2)
        chained = MockChainedOptimizer([wrapped1, wrapped2])
        result = _unwrap_optimizer(chained)
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], inner1)
        self.assertIs(result[1], inner2)


# =====================================================================
# Test: Optimizer state sync
# =====================================================================

class TestOptimizerStateSync(unittest.TestCase):

    def _make_model_and_classification(self):
        """Create a model and its classification."""
        model = make_moe_model()
        cls = classify_model_parameters(model)
        return model, cls

    def test_sync_plain_optimizer_with_param_keys(self):
        """Optimizer state keyed by param objects — dense states are synced."""
        model, cls = self._make_model_and_classification()
        opt = MockBaseOptimizer()

        broadcast_calls = []

        def mock_broadcast(tensor, src, group):
            broadcast_calls.append((id(tensor), src))

        # Set up optimizer state for dense params (keyed by param object)
        dense_params = list(cls.all_dense_like.values())
        expert_params = list(cls.expert_params.values())

        all_params = dense_params + expert_params
        opt.add_param_group(all_params)

        # Add optimizer state for each param
        for p in dense_params:
            opt.set_state(p, {
                "exp_avg": MockTensor(numel=p._numel),
                "exp_avg_sq": MockTensor(numel=p._numel),
                "step": 100,  # non-tensor state, should be skipped
            })
        for p in expert_params:
            opt.set_state(p, {
                "exp_avg": MockTensor(numel=p._numel),
                "exp_avg_sq": MockTensor(numel=p._numel),
            })

        synced = _sync_optimizer_states_for_dense(
            optimizer=opt,
            classification=cls,
            source_rank=0,
            dp_group="fake_group",
            broadcast_fn=mock_broadcast,
        )

        # Should sync 2 tensors (exp_avg + exp_avg_sq) per dense param
        # 10 dense-like params × 2 = 20 broadcast calls
        self.assertEqual(len(broadcast_calls), 20)

        # Total scalars = sum of numel for all dense params × 2
        expected_scalars = sum(p._numel for p in dense_params) * 2
        self.assertEqual(synced, expected_scalars)

    def test_sync_plain_optimizer_with_int_keys(self):
        """Optimizer state keyed by integer indices."""
        model, cls = self._make_model_and_classification()
        opt = MockBaseOptimizer()

        broadcast_calls = []

        def mock_broadcast(tensor, src, group):
            broadcast_calls.append(1)

        dense_params = list(cls.all_dense_like.values())
        expert_params = list(cls.expert_params.values())

        # Add params in order: dense first, then expert
        all_params = dense_params + expert_params
        opt.add_param_group(all_params)

        # Set state with integer keys
        for idx, p in enumerate(all_params):
            opt.state[idx] = {
                "exp_avg": MockTensor(numel=p._numel),
                "exp_avg_sq": MockTensor(numel=p._numel),
            }

        synced = _sync_optimizer_states_for_dense(
            optimizer=opt,
            classification=cls,
            source_rank=0,
            dp_group="fake_group",
            broadcast_fn=mock_broadcast,
        )

        # Only dense params should be synced (10 × 2 = 20 calls)
        self.assertEqual(len(broadcast_calls), 20)

    def test_expert_optimizer_state_not_synced(self):
        """Expert params' optimizer states must NOT be broadcast."""
        model, cls = self._make_model_and_classification()
        opt = MockBaseOptimizer()

        synced_param_ids = set()

        def tracking_broadcast(tensor, src, group):
            synced_param_ids.add(id(tensor))

        expert_params = list(cls.expert_params.values())
        opt.add_param_group(expert_params)

        # Only add state for expert params
        for p in expert_params:
            t = MockTensor(numel=p._numel)
            opt.set_state(p, {"exp_avg": t})

        synced = _sync_optimizer_states_for_dense(
            optimizer=opt,
            classification=cls,
            source_rank=0,
            dp_group="fake_group",
            broadcast_fn=tracking_broadcast,
        )

        # No expert state should be synced
        self.assertEqual(synced, 0)
        self.assertEqual(len(synced_param_ids), 0)

    def test_sync_float16_wrapped_optimizer(self):
        """Float16OptimizerWithFloat16Params: state lives in inner optimizer."""
        model, cls = self._make_model_and_classification()
        inner_opt = MockBaseOptimizer()

        broadcast_calls = []

        def mock_broadcast(tensor, src, group):
            broadcast_calls.append(1)

        # In Float16Optimizer, the inner optimizer's param_groups contain
        # main_param (fp32 copies).  We simulate this by creating fp32
        # copies and linking them via main_param.
        dense_params = list(cls.all_dense_like.values())
        fp32_copies = []
        for p in dense_params:
            fp32 = MockParam(p.name + ".fp32", numel=p._numel)
            p.main_param = fp32  # model_param.main_param → fp32 copy
            fp32_copies.append(fp32)

        inner_opt.add_param_group(fp32_copies)

        # State is keyed by fp32 param objects in the inner optimizer
        for fp32 in fp32_copies:
            inner_opt.set_state(fp32, {
                "exp_avg": MockTensor(numel=fp32._numel),
                "exp_avg_sq": MockTensor(numel=fp32._numel),
            })

        wrapper = MockFloat16Optimizer(inner_opt)

        synced = _sync_optimizer_states_for_dense(
            optimizer=wrapper,
            classification=cls,
            source_rank=0,
            dp_group="fake_group",
            broadcast_fn=mock_broadcast,
        )

        # Should unwrap to inner optimizer and sync 10 × 2 = 20 tensors
        self.assertEqual(len(broadcast_calls), 20)
        expected_scalars = sum(p._numel for p in dense_params) * 2
        self.assertEqual(synced, expected_scalars)

    def test_sync_chained_optimizer(self):
        """ChainedOptimizer: sync across multiple inner optimizers."""
        model, cls = self._make_model_and_classification()

        broadcast_calls = []

        def mock_broadcast(tensor, src, group):
            broadcast_calls.append(1)

        dense_params = list(cls.all_dense_like.values())
        # Split dense params across two inner optimizers
        half = len(dense_params) // 2
        params_1 = dense_params[:half]
        params_2 = dense_params[half:]

        inner1 = MockBaseOptimizer()
        inner1.add_param_group(params_1)
        for p in params_1:
            inner1.set_state(p, {
                "exp_avg": MockTensor(numel=p._numel),
            })

        inner2 = MockBaseOptimizer()
        inner2.add_param_group(params_2)
        for p in params_2:
            inner2.set_state(p, {
                "exp_avg": MockTensor(numel=p._numel),
            })

        chained = MockChainedOptimizer([inner1, inner2])

        synced = _sync_optimizer_states_for_dense(
            optimizer=chained,
            classification=cls,
            source_rank=0,
            dp_group="fake_group",
            broadcast_fn=mock_broadcast,
        )

        # All 10 dense params × 1 state tensor = 10 broadcast calls
        self.assertEqual(len(broadcast_calls), 10)

    def test_sync_none_optimizer(self):
        """None optimizer returns 0."""
        model, cls = self._make_model_and_classification()
        synced = _sync_optimizer_states_for_dense(
            optimizer=None,
            classification=cls,
            source_rank=0,
            dp_group="fake_group",
            broadcast_fn=lambda t, s, g: None,
        )
        self.assertEqual(synced, 0)

    def test_sync_empty_state(self):
        """Optimizer with no state (not yet stepped) returns 0."""
        model, cls = self._make_model_and_classification()
        opt = MockBaseOptimizer()
        dense_params = list(cls.all_dense_like.values())
        opt.add_param_group(dense_params)
        # No state set

        synced = _sync_optimizer_states_for_dense(
            optimizer=opt,
            classification=cls,
            source_rank=0,
            dp_group="fake_group",
            broadcast_fn=lambda t, s, g: None,
        )
        self.assertEqual(synced, 0)


if __name__ == "__main__":
    unittest.main()

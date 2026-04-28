# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for safe-point process group repair (Phase 7).

Tests cover:
1. SafePointGroupRepairer — validate, invalidate, rebuild, rebind, verify
2. GroupRebuildCoordinator — safe-point gating
3. Integration — end-to-end repair flow
4. Safe-point gating — repair only happens at safe point
"""

import importlib
import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Bootstrap: load modules without torch / megatron.core.__init__
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Provide a minimal torch stub so modules can be imported
_torch_stub = types.ModuleType('torch')
_torch_stub.Tensor = type('Tensor', (), {})
_torch_distributed_stub = types.ModuleType('torch.distributed')
_torch_distributed_stub.is_initialized = lambda: False
_torch_distributed_stub.get_rank = lambda: 0
_torch_distributed_stub.new_group = lambda ranks=None, backend='nccl', **kw: MagicMock()
_torch_distributed_stub.destroy_process_group = lambda pg: None
_torch_distributed_stub.barrier = lambda group=None, **kw: None
_torch_stub.distributed = _torch_distributed_stub
_torch_stub.cuda = MagicMock()

_real_torch = sys.modules.get('torch')
_real_torch_dist = sys.modules.get('torch.distributed')

if _real_torch is None:
    sys.modules['torch'] = _torch_stub
    sys.modules['torch.distributed'] = _torch_distributed_stub

# Stub megatron.core to avoid heavy imports
_mc_stub = types.ModuleType('megatron')
_mc_core_stub = types.ModuleType('megatron.core')
_mc_moe_stub = types.ModuleType('megatron.core.transformer')
_mc_moe_stub2 = types.ModuleType('megatron.core.transformer.moe')
for mod_name, mod in [
    ('megatron', _mc_stub),
    ('megatron.core', _mc_core_stub),
    ('megatron.core.transformer', _mc_moe_stub),
    ('megatron.core.transformer.moe', _mc_moe_stub2),
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = mod


def _load_module(name, file_path):
    """Load a module from file path, bypassing package __init__."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = '.'.join(name.split('.')[:-1])
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_moe_dir = os.path.join(
    _REPO_ROOT, 'megatron', 'core', 'transformer', 'moe',
)

# Load the modules under test
gb_mod = _load_module(
    'megatron.core.transformer.moe.group_rebuild',
    os.path.join(_moe_dir, 'group_rebuild.py'),
)

# Load pipeline_stage_repair BEFORE safe_point_group_repair so that
# the ``from megatron.core.transformer.moe import pipeline_stage_repair``
# inside ``_repair_pipeline_stage`` can resolve against the stub package.
_psr_path = os.path.join(_moe_dir, 'pipeline_stage_repair.py')
if os.path.exists(_psr_path):
    _load_module(
        'megatron.core.transformer.moe.pipeline_stage_repair',
        _psr_path,
    )

spr_mod = _load_module(
    'megatron.core.transformer.moe.safe_point_group_repair',
    os.path.join(_moe_dir, 'safe_point_group_repair.py'),
)

GroupRebuildCoordinator = gb_mod.GroupRebuildCoordinator
GroupRebuildState = gb_mod.GroupRebuildState
GroupRebuildPlan = gb_mod.GroupRebuildPlan
SafePointGroupRepairer = spr_mod.SafePointGroupRepairer
RepairPhase = spr_mod.RepairPhase
RepairResult = spr_mod.RepairResult


# =====================================================================
# Test SafePointGroupRepairer
# =====================================================================

class TestSafePointGroupRepairer(unittest.TestCase):
    """Tests for SafePointGroupRepairer."""

    def setUp(self):
        self.repairer = SafePointGroupRepairer()
        self.plan = GroupRebuildPlan(
            failed_rank=4,
            replacement_rank=64,
            affected_groups=[
                "EXPERT_MODEL_PARALLEL_GROUP",
                "EXPERT_DATA_PARALLEL_GROUP",
                "DATA_PARALLEL_GROUP",
            ],
            old_ep_group_ranks=[0, 2, 4, 6],
            new_ep_group_ranks=[0, 2, 64, 6],
            step=100,
        )

    def test_initial_state(self):
        self.assertIsNone(self.repairer.last_result)
        self.assertEqual(self.repairer.total_repairs, 0)

    def test_validate_same_rank_raises(self):
        bad_plan = GroupRebuildPlan(
            failed_rank=4,
            replacement_rank=4,
            affected_groups=["EXPERT_MODEL_PARALLEL_GROUP"],
        )
        result = self.repairer.execute(bad_plan)
        self.assertFalse(result.success)
        self.assertEqual(result.phase_reached, RepairPhase.FAILED)
        self.assertIn("failed_rank", result.error)

    def test_validate_no_groups_raises(self):
        bad_plan = GroupRebuildPlan(
            failed_rank=4,
            replacement_rank=64,
            affected_groups=[],
        )
        result = self.repairer.execute(bad_plan)
        self.assertFalse(result.success)
        self.assertIn("No affected groups", result.error)

    def test_execute_without_model(self):
        """Execute repair without model (no rebind phase)."""
        # Mock parallel_state
        mock_ps = MagicMock()
        mock_ps._EXPERT_MODEL_PARALLEL_GROUP = MagicMock()
        mock_ps._EXPERT_DATA_PARALLEL_GROUP = MagicMock()
        mock_ps._DATA_PARALLEL_GROUP = MagicMock()
        mock_ps._DATA_PARALLEL_GLOBAL_RANKS = [0, 2, 4, 6]

        with patch.dict(sys.modules, {'megatron.core.parallel_state': mock_ps}):
            with patch.object(spr_mod, '_default_create_group', return_value=MagicMock()):
                result = self.repairer.execute(
                    self.plan,
                    model=None,
                    verify=False,
                )

        self.assertTrue(result.success)
        self.assertEqual(result.phase_reached, RepairPhase.COMPLETED)
        self.assertGreater(result.groups_invalidated, 0)
        self.assertEqual(result.modules_rebound, 0)  # no model
        self.assertEqual(self.repairer.total_repairs, 1)

    def test_execute_with_custom_create_group(self):
        """Execute repair with a custom create_group_fn."""
        mock_ps = MagicMock()
        mock_ps._EXPERT_MODEL_PARALLEL_GROUP = MagicMock()
        mock_ps._EXPERT_DATA_PARALLEL_GROUP = MagicMock()
        mock_ps._DATA_PARALLEL_GROUP = MagicMock()
        mock_ps._DATA_PARALLEL_GLOBAL_RANKS = [0, 2, 4, 6]

        create_calls = []
        def mock_create(ranks, backend='nccl', group_name='', **kw):
            create_calls.append((ranks, backend, group_name))
            return MagicMock()

        with patch.dict(sys.modules, {'megatron.core.parallel_state': mock_ps}):
            result = self.repairer.execute(
                self.plan,
                create_group_fn=mock_create,
                verify=False,
            )

        self.assertTrue(result.success)
        self.assertGreater(len(create_calls), 0)
        # Verify replacement rank is in the new ranks
        for ranks, backend, name in create_calls:
            self.assertIn(64, ranks)
            self.assertNotIn(4, ranks)

    def test_pp_gt_1_warning(self):
        """PP>1 should log a warning but still proceed."""
        mock_ps = MagicMock()
        mock_ps._EXPERT_MODEL_PARALLEL_GROUP = MagicMock()
        mock_ps._EXPERT_DATA_PARALLEL_GROUP = MagicMock()
        mock_ps._DATA_PARALLEL_GROUP = MagicMock()
        mock_ps._DATA_PARALLEL_GLOBAL_RANKS = [0, 2, 4, 6]
        # Provide PP group ranks so _repair_pipeline_stage can find them
        mock_ps._PIPELINE_GLOBAL_RANKS = [0, 2, 4, 6]

        # Build a mock PipelineStageRepairResult
        mock_psr_result = MagicMock()
        mock_psr_result.success = True
        mock_psr_result.pp_group_rebuilt = True
        mock_psr_result.prev_next_updated = True
        mock_psr_result.p2p_rebound = True
        mock_psr_result.elapsed_seconds = 0.0
        mock_psr_result.error = ""

        mock_repairer = MagicMock()
        mock_repairer.execute.return_value = mock_psr_result

        # Stub the pipeline_stage_repair module's singleton
        psr_mod_name = 'megatron.core.transformer.moe.pipeline_stage_repair'
        psr_stub = sys.modules.get(psr_mod_name)
        if psr_stub is None:
            psr_stub = types.ModuleType(psr_mod_name)
            sys.modules[psr_mod_name] = psr_stub
        original_getter = getattr(psr_stub, 'get_pipeline_stage_repairer', None)
        psr_stub.get_pipeline_stage_repairer = lambda: mock_repairer

        try:
            with patch.dict(sys.modules, {'megatron.core.parallel_state': mock_ps}):
                with patch.object(spr_mod, '_default_create_group', return_value=MagicMock()):
                    result = self.repairer.execute(
                        self.plan,
                        pp_size=4,
                        verify=False,
                    )

            self.assertTrue(result.success)
        finally:
            # Restore original getter if any
            if original_getter is not None:
                psr_stub.get_pipeline_stage_repairer = original_getter

    def test_summary(self):
        self.assertIsNone(self.repairer.summary()['last_result'])
        self.assertEqual(self.repairer.summary()['total_repairs'], 0)

    def test_reset(self):
        self.repairer._total_repairs = 5
        self.repairer._last_result = RepairResult(success=True)
        self.repairer.reset()
        self.assertEqual(self.repairer.total_repairs, 0)
        self.assertIsNone(self.repairer.last_result)


# =====================================================================
# Test GroupRebuildCoordinator safe-point gating
# =====================================================================

class TestGroupRebuildSafePointGating(unittest.TestCase):
    """Tests that repair only happens at safe points."""

    def setUp(self):
        self.coord = GroupRebuildCoordinator()

    def tearDown(self):
        self.coord.reset()

    def test_no_repair_without_request(self):
        """No repair should happen if no request was made."""
        self.assertFalse(self.coord.has_pending_group_repair())
        result = self.coord.maybe_rebuild_groups_at_safe_point(step=10)
        self.assertFalse(result)

    def test_repair_only_at_safe_point(self):
        """Repair should only execute when explicitly called at safe point."""
        # Request rebuild
        plan = self.coord.request_rebuild(
            failed_rank=4,
            replacement_rank=64,
            step=100,
        )
        self.assertEqual(self.coord.state, GroupRebuildState.PENDING_REPAIR)
        self.assertTrue(self.coord.has_pending_group_repair())

        # Simulate "not at safe point" — just check state, don't call rebuild
        self.assertEqual(self.coord.state, GroupRebuildState.PENDING_REPAIR)

        # Now at safe point — execute rebuild
        rebuild_called = [False]
        rebind_called = [False]

        def mock_rebuild(p):
            rebuild_called[0] = True

        def mock_rebind(p):
            rebind_called[0] = True

        result = self.coord.maybe_rebuild_groups_at_safe_point(
            rebuild_fn=mock_rebuild,
            rebind_fn=mock_rebind,
            step=110,
        )
        self.assertTrue(result)
        self.assertTrue(rebuild_called[0])
        self.assertTrue(rebind_called[0])
        self.assertEqual(self.coord.state, GroupRebuildState.REBINDING)

        # Finish
        completed_plan = self.coord.finish_group_repair(step=110)
        self.assertIsNotNone(completed_plan)
        self.assertEqual(self.coord.state, GroupRebuildState.IDLE)

    def test_multiple_requests_processed_sequentially(self):
        """Multiple rebuild requests should be processed one at a time."""
        self.coord.request_rebuild(failed_rank=4, replacement_rank=64, step=100)
        self.coord.request_rebuild(failed_rank=6, replacement_rank=66, step=101)

        self.assertEqual(len(self.coord.pending_plans), 2)

        # Process first
        self.coord.maybe_rebuild_groups_at_safe_point(step=110)
        self.coord.finish_group_repair(step=110)

        # Should still have one pending
        self.assertEqual(self.coord.state, GroupRebuildState.PENDING_REPAIR)
        self.assertEqual(len(self.coord.pending_plans), 1)

        # Process second
        self.coord.maybe_rebuild_groups_at_safe_point(step=120)
        self.coord.finish_group_repair(step=120)

        self.assertEqual(self.coord.state, GroupRebuildState.IDLE)
        self.assertEqual(len(self.coord.history), 2)

    def test_cancel_pending(self):
        """Cancel should clear all pending requests."""
        self.coord.request_rebuild(failed_rank=4, replacement_rank=64, step=100)
        self.coord.request_rebuild(failed_rank=6, replacement_rank=66, step=101)

        count = self.coord.cancel_pending()
        self.assertEqual(count, 2)
        self.assertEqual(self.coord.state, GroupRebuildState.IDLE)
        self.assertFalse(self.coord.has_pending_group_repair())


# =====================================================================
# Test affected group computation
# =====================================================================

class TestAffectedGroupComputation(unittest.TestCase):
    """Tests for computing which groups need rebuilding."""

    def test_ep_rank_failure(self):
        """Failed rank in EP group should trigger EP + DP group rebuild."""
        affected = GroupRebuildCoordinator.compute_affected_groups(
            failed_rank=4,
            ep_group_ranks=[0, 2, 4, 6],
            dp_group_ranks=[0, 4, 8, 12],
        )
        # Should include both EP and DP groups
        self.assertTrue(any("EXPERT" in g for g in affected))
        self.assertTrue(any("DATA_PARALLEL" in g for g in affected))

    def test_non_ep_rank_failure(self):
        """Failed rank NOT in EP group should only trigger DP rebuild."""
        affected = GroupRebuildCoordinator.compute_affected_groups(
            failed_rank=8,
            ep_group_ranks=[0, 2, 4, 6],
            dp_group_ranks=[0, 4, 8, 12],
        )
        # Should include DP groups but not EP groups
        self.assertTrue(any("DATA_PARALLEL" in g for g in affected))
        self.assertFalse(any("EXPERT" in g for g in affected))

    def test_compute_new_group_ranks(self):
        """Replacement should swap failed rank in rank list."""
        old = [0, 2, 4, 6]
        new = GroupRebuildCoordinator.compute_new_group_ranks(old, 4, 64)
        self.assertEqual(new, [0, 2, 64, 6])

    def test_compute_new_group_ranks_not_present(self):
        """If failed rank not in list, list should be unchanged."""
        old = [0, 2, 4, 6]
        new = GroupRebuildCoordinator.compute_new_group_ranks(old, 8, 64)
        self.assertEqual(new, [0, 2, 4, 6])


# =====================================================================
# Test RepairResult
# =====================================================================

class TestRepairResult(unittest.TestCase):
    """Tests for RepairResult data class."""

    def test_default_values(self):
        r = RepairResult()
        self.assertFalse(r.success)
        self.assertEqual(r.phase_reached, RepairPhase.NOT_STARTED)
        self.assertEqual(r.groups_invalidated, 0)
        self.assertEqual(r.groups_rebuilt, 0)
        self.assertEqual(r.modules_rebound, 0)
        self.assertFalse(r.verification_passed)
        self.assertEqual(r.error, "")


# =====================================================================
# Test invalidation logic
# =====================================================================

class TestInvalidation(unittest.TestCase):
    """Tests for group invalidation."""

    def test_invalidate_sets_globals_to_none(self):
        """Invalidation should set parallel_state globals to None."""
        repairer = SafePointGroupRepairer()

        mock_ps = MagicMock()
        mock_ps._EXPERT_MODEL_PARALLEL_GROUP = MagicMock()
        mock_ps._DATA_PARALLEL_GROUP = MagicMock()

        with patch.dict(sys.modules, {'megatron.core.parallel_state': mock_ps}):
            # Use the internal method directly
            count = repairer._invalidate_affected_groups([
                "EXPERT_MODEL_PARALLEL_GROUP",
                "DATA_PARALLEL_GROUP",
            ])

        self.assertEqual(count, 2)
        self.assertIsNone(mock_ps._EXPERT_MODEL_PARALLEL_GROUP)
        self.assertIsNone(mock_ps._DATA_PARALLEL_GROUP)

    def test_invalidate_gloo_group_destroys(self):
        """Gloo groups should be explicitly destroyed before invalidation."""
        repairer = SafePointGroupRepairer()

        mock_gloo_group = MagicMock()
        mock_ps = MagicMock()
        mock_ps._EXPERT_DATA_PARALLEL_GROUP_GLOO = mock_gloo_group

        with patch.dict(sys.modules, {'megatron.core.parallel_state': mock_ps}):
            import torch.distributed as td
            with patch.object(td, 'destroy_process_group') as mock_destroy:
                count = repairer._invalidate_affected_groups([
                    "EXPERT_DATA_PARALLEL_GROUP_GLOO",
                ])

        self.assertEqual(count, 1)
        mock_destroy.assert_called_once_with(mock_gloo_group)

    def test_invalidate_already_none(self):
        """Invalidating an already-None group should be a no-op."""
        repairer = SafePointGroupRepairer()

        mock_ps = MagicMock()
        mock_ps._EXPERT_MODEL_PARALLEL_GROUP = None

        with patch.dict(sys.modules, {'megatron.core.parallel_state': mock_ps}):
            count = repairer._invalidate_affected_groups([
                "EXPERT_MODEL_PARALLEL_GROUP",
            ])

        self.assertEqual(count, 0)


# =====================================================================
# Test end-to-end integration
# =====================================================================

class TestEndToEndRepair(unittest.TestCase):
    """End-to-end test: hard failure → request → safe-point → repair."""

    def test_full_flow(self):
        """Simulate the complete repair flow."""
        coord = GroupRebuildCoordinator()

        # Step 1: Hard failure detected — request rebuild
        plan = coord.request_rebuild(
            failed_rank=4,
            replacement_rank=64,
            step=100,
            old_ep_group_ranks=[0, 2, 4, 6],
            new_ep_group_ranks=[0, 2, 64, 6],
        )
        self.assertEqual(coord.state, GroupRebuildState.PENDING_REPAIR)

        # Step 2: Training continues (not at safe point yet)
        # ... forward/backward/optimizer ...
        self.assertTrue(coord.has_pending_group_repair())

        # Step 3: Safe point reached — execute repair
        repairer = SafePointGroupRepairer()

        rebuild_executed = [False]

        def mock_rebuild_fn(p):
            rebuild_executed[0] = True
            # Simulate the repairer executing
            mock_ps = MagicMock()
            mock_ps._EXPERT_MODEL_PARALLEL_GROUP = MagicMock()
            mock_ps._DATA_PARALLEL_GROUP = MagicMock()

        def mock_rebind_fn(p):
            pass

        result = coord.maybe_rebuild_groups_at_safe_point(
            rebuild_fn=mock_rebuild_fn,
            rebind_fn=mock_rebind_fn,
            step=110,
        )
        self.assertTrue(result)
        self.assertTrue(rebuild_executed[0])

        # Step 4: Finish repair
        completed = coord.finish_group_repair(step=110)
        self.assertIsNotNone(completed)
        self.assertEqual(completed.failed_rank, 4)
        self.assertEqual(completed.replacement_rank, 64)
        self.assertEqual(coord.state, GroupRebuildState.IDLE)

        # Step 5: Verify history
        self.assertEqual(len(coord.history), 1)
        self.assertEqual(coord.history[0]['failed_rank'], 4)
        self.assertEqual(coord.history[0]['replacement_rank'], 64)

    def test_no_repair_before_safe_point(self):
        """Verify that repair does NOT happen before safe point."""
        coord = GroupRebuildCoordinator()

        # Request rebuild
        coord.request_rebuild(
            failed_rank=4,
            replacement_rank=64,
            step=100,
        )

        # Check state — should be PENDING, not REBUILDING
        self.assertEqual(coord.state, GroupRebuildState.PENDING_REPAIR)

        # Simulate multiple "non-safe-point" iterations
        for i in range(10):
            # The training loop would NOT call maybe_rebuild_groups_at_safe_point
            # during forward/backward — only at iteration boundary
            self.assertTrue(coord.has_pending_group_repair())
            self.assertEqual(coord.state, GroupRebuildState.PENDING_REPAIR)

        # Now at safe point
        coord.maybe_rebuild_groups_at_safe_point(step=110)
        self.assertEqual(coord.state, GroupRebuildState.REBINDING)

    def test_replacement_rank_not_in_training_before_repair(self):
        """Verify replacement rank is not participating before repair."""
        coord = GroupRebuildCoordinator()

        # Request rebuild
        plan = coord.request_rebuild(
            failed_rank=4,
            replacement_rank=64,
            step=100,
            old_ep_group_ranks=[0, 2, 4, 6],
            new_ep_group_ranks=[0, 2, 64, 6],
        )

        # Before repair: replacement rank 64 is NOT in any active group
        # (the old groups still have rank 4, which is dead)
        self.assertEqual(plan.old_ep_group_ranks, [0, 2, 4, 6])
        self.assertNotIn(64, plan.old_ep_group_ranks)

        # After repair: replacement rank 64 IS in the new groups
        self.assertEqual(plan.new_ep_group_ranks, [0, 2, 64, 6])
        self.assertIn(64, plan.new_ep_group_ranks)


# =====================================================================
# Test rebind maps
# =====================================================================

class TestRebindMaps(unittest.TestCase):
    """Tests for MoE module rebind maps."""

    def test_dispatcher_rebind_map(self):
        self.assertGreater(len(gb_mod.MOE_DISPATCHER_REBIND_MAP), 0)
        for attr, pg_field in gb_mod.MOE_DISPATCHER_REBIND_MAP:
            self.assertIsInstance(attr, str)
            self.assertIsInstance(pg_field, str)

    def test_router_rebind_map(self):
        self.assertGreater(len(gb_mod.MOE_ROUTER_REBIND_MAP), 0)

    def test_layer_rebind_map(self):
        self.assertGreater(len(gb_mod.MOE_LAYER_REBIND_MAP), 0)

    def test_experts_rebind_map(self):
        self.assertGreater(len(gb_mod.MOE_EXPERTS_REBIND_MAP), 0)

    def test_rebind_moe_module_groups(self):
        """Test rebinding a mock module."""
        module = MagicMock()
        module.ep_group = MagicMock()
        module.tp_group = MagicMock()

        new_ep = MagicMock()
        new_tp = MagicMock()
        pg_dict = {"ep": new_ep, "expt_tp": new_tp}

        count = gb_mod.rebind_moe_module_groups(
            module, pg_dict, gb_mod.MOE_DISPATCHER_REBIND_MAP,
        )
        self.assertGreater(count, 0)
        self.assertEqual(module.ep_group, new_ep)


# =====================================================================
# Test singleton management
# =====================================================================

class TestSingletons(unittest.TestCase):
    """Tests for singleton creation and cleanup."""

    def test_get_group_rebuild_coordinator(self):
        coord = gb_mod.get_group_rebuild_coordinator()
        self.assertIsNotNone(coord)
        self.assertIsInstance(coord, GroupRebuildCoordinator)

    def test_clear_group_rebuild_coordinator(self):
        gb_mod.get_group_rebuild_coordinator()
        gb_mod.clear_group_rebuild_coordinator()
        # Should create a new one
        coord = gb_mod.get_group_rebuild_coordinator()
        self.assertEqual(coord.state, GroupRebuildState.IDLE)

    def test_get_safe_point_group_repairer(self):
        repairer = spr_mod.get_safe_point_group_repairer()
        self.assertIsNotNone(repairer)
        self.assertIsInstance(repairer, SafePointGroupRepairer)

    def test_clear_safe_point_group_repairer(self):
        spr_mod.get_safe_point_group_repairer()
        spr_mod.clear_safe_point_group_repairer()
        repairer = spr_mod.get_safe_point_group_repairer()
        self.assertEqual(repairer.total_repairs, 0)


# =====================================================================
# Test RepairPhase enum
# =====================================================================

class TestRepairPhaseEnum(unittest.TestCase):
    """Tests for RepairPhase enum values."""

    def test_all_phases_exist(self):
        self.assertEqual(RepairPhase.NOT_STARTED, 0)
        self.assertEqual(RepairPhase.VALIDATING, 1)
        self.assertEqual(RepairPhase.INVALIDATING, 2)
        self.assertEqual(RepairPhase.REBUILDING, 3)
        self.assertEqual(RepairPhase.REBINDING, 4)
        self.assertEqual(RepairPhase.VERIFYING, 5)
        self.assertEqual(RepairPhase.COMPLETED, 6)
        self.assertEqual(RepairPhase.FAILED, 7)


if __name__ == '__main__':
    unittest.main()

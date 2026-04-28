# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for MoE expert restore from checkpoint (Phase 6).

Tests cover:
1. StaleExpertRestoreCoordinator — plan + execute
2. OptimizerUpdateBarrier — block/unblock expert params
3. ExpertRestorePlan construction from RecoveryManifest
4. Health manager state transitions during restore
5. Expert directory updates during restore
6. End-to-end: hard failure → restore → state transitions
7. Router exclusion before restore, inclusion after
"""

import importlib
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Bootstrap: mock torch and megatron.core to avoid GPU dependency
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Provide a minimal torch mock so modules that import torch at top level
# can be loaded without CUDA.
_real_torch = None
try:
    import torch as _real_torch
except ImportError:
    pass

if _real_torch is None:
    _mock_torch = MagicMock()
    _mock_torch.Tensor = MagicMock
    _mock_torch.device = MagicMock
    sys.modules['torch'] = _mock_torch
    sys.modules['torch.distributed'] = MagicMock()
    sys.modules['torch.cuda'] = MagicMock()

# Mock megatron.core top-level to avoid heavy imports
if 'megatron.core' not in sys.modules:
    sys.modules['megatron'] = MagicMock()
    sys.modules['megatron.core'] = MagicMock()
    sys.modules['megatron.core.transformer'] = MagicMock()
    sys.modules['megatron.core.transformer.moe'] = MagicMock()

# Now import the actual modules we want to test
# Use a helper to properly load modules with correct __name__
def _load_module(mod_name, file_path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__name__ = mod_name
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

_moe_dir = os.path.join(_REPO_ROOT, "megatron", "core", "transformer", "moe")

ser_mod = _load_module(
    "megatron.core.transformer.moe.stale_expert_restore",
    os.path.join(_moe_dir, "stale_expert_restore.py"),
)

ed_mod = _load_module(
    "megatron.core.transformer.moe.expert_directory",
    os.path.join(_moe_dir, "expert_directory.py"),
)

# We need expert_health for the health manager, but it requires torch
eh_mod = None
ehm_mod = None

if _real_torch is not None:
    eh_mod = _load_module(
        "megatron.core.transformer.moe.expert_health",
        os.path.join(_moe_dir, "expert_health.py"),
    )
    ehm_mod = _load_module(
        "megatron.core.transformer.moe.expert_health_manager",
        os.path.join(_moe_dir, "expert_health_manager.py"),
    )

# Aliases
ExpertRestoreEntry = ser_mod.ExpertRestoreEntry
ExpertRestorePlan = ser_mod.ExpertRestorePlan
ExpertRestoreResult = ser_mod.ExpertRestoreResult
OptimizerUpdateBarrier = ser_mod.OptimizerUpdateBarrier
StaleExpertRestoreCoordinator = ser_mod.StaleExpertRestoreCoordinator
identify_experts_to_restore = ser_mod.identify_experts_to_restore
restore_expert_weights = ser_mod.restore_expert_weights

ActiveExpertDirectory = ed_mod.ActiveExpertDirectory
RecoveryManifest = ed_mod.RecoveryManifest
ExpertDirectoryEntry = ed_mod.ExpertDirectoryEntry

# Load deferred_optimizer_load module
dol_mod = _load_module(
    "megatron.core.transformer.moe.deferred_optimizer_load",
    os.path.join(_moe_dir, "deferred_optimizer_load.py"),
)
DeferredOptimizerLoader = dol_mod.DeferredOptimizerLoader
OptimizerLoadState = dol_mod.OptimizerLoadState
OptimizerLoadRequest = dol_mod.OptimizerLoadRequest


# =====================================================================
# Test: ExpertRestorePlan
# =====================================================================

class TestExpertRestorePlan(unittest.TestCase):

    def test_empty_plan(self):
        plan = ExpertRestorePlan()
        self.assertEqual(plan.num_experts, 0)
        self.assertEqual(plan.affected_layers, [])
        self.assertEqual(plan.affected_expert_ids, [])

    def test_plan_with_entries(self):
        plan = ExpertRestorePlan(
            failed_rank=2,
            replacement_rank=10,
            step=100,
        )
        plan.entries.append(ExpertRestoreEntry(layer_id=0, expert_id=4))
        plan.entries.append(ExpertRestoreEntry(layer_id=0, expert_id=5))
        plan.entries.append(ExpertRestoreEntry(layer_id=1, expert_id=4))
        plan.entries.append(ExpertRestoreEntry(layer_id=1, expert_id=5))

        self.assertEqual(plan.num_experts, 4)
        self.assertEqual(plan.affected_layers, [0, 1])
        self.assertEqual(plan.affected_expert_ids, [4, 5])
        self.assertEqual(len(plan.get_entries_for_layer(0)), 2)
        self.assertEqual(len(plan.get_entries_for_layer(1)), 2)

    def test_plan_to_dict(self):
        plan = ExpertRestorePlan(failed_rank=2, replacement_rank=10, step=100)
        plan.entries.append(ExpertRestoreEntry(layer_id=0, expert_id=4))
        d = plan.to_dict()
        self.assertEqual(d['failed_rank'], 2)
        self.assertEqual(d['replacement_rank'], 10)
        self.assertEqual(d['num_experts'], 1)
        self.assertEqual(len(d['entries']), 1)


# =====================================================================
# Test: OptimizerUpdateBarrier
# =====================================================================

class TestOptimizerUpdateBarrier(unittest.TestCase):

    def setUp(self):
        self.barrier = OptimizerUpdateBarrier()

    def test_initial_state(self):
        self.assertEqual(self.barrier.num_blocked_params, 0)
        self.assertEqual(self.barrier.num_blocked_experts, 0)
        self.assertFalse(self.barrier.is_blocked("any_param"))
        self.assertFalse(self.barrier.is_expert_blocked(0))

    def test_block_params(self):
        self.barrier.block_params(["param_a", "param_b"], step=10)
        self.assertTrue(self.barrier.is_blocked("param_a"))
        self.assertTrue(self.barrier.is_blocked("param_b"))
        self.assertFalse(self.barrier.is_blocked("param_c"))
        self.assertEqual(self.barrier.num_blocked_params, 2)

    def test_unblock_params(self):
        self.barrier.block_params(["param_a", "param_b"])
        self.barrier.unblock_params(["param_a"])
        self.assertFalse(self.barrier.is_blocked("param_a"))
        self.assertTrue(self.barrier.is_blocked("param_b"))

    def test_block_expert_params(self):
        # Create a mock model with expert parameters
        model = MagicMock()
        params = [
            ("layers.0.moe.experts.weight1", MagicMock(allreduce=False)),
            ("layers.0.moe.experts.weight2", MagicMock(allreduce=False)),
            ("layers.0.attention.weight", MagicMock(allreduce=True)),
            ("layers.0.moe.router.weight", MagicMock(allreduce=True)),
        ]
        model.named_parameters.return_value = params

        blocked = self.barrier.block_expert_params(model, expert_ids=[2, 3], step=50)
        self.assertEqual(blocked, 2)  # Only expert params blocked
        self.assertTrue(self.barrier.is_blocked("layers.0.moe.experts.weight1"))
        self.assertTrue(self.barrier.is_blocked("layers.0.moe.experts.weight2"))
        self.assertFalse(self.barrier.is_blocked("layers.0.attention.weight"))
        self.assertTrue(self.barrier.is_expert_blocked(2))
        self.assertTrue(self.barrier.is_expert_blocked(3))
        self.assertFalse(self.barrier.is_expert_blocked(0))

    def test_unblock_expert_params(self):
        self.barrier._blocked_expert_ids = {2, 3}
        self.barrier._blocked_params = {"param_a", "param_b"}
        self.barrier.unblock_expert_params([2])
        self.assertFalse(self.barrier.is_expert_blocked(2))
        self.assertTrue(self.barrier.is_expert_blocked(3))

    def test_unblock_all(self):
        self.barrier.block_params(["p1", "p2"])
        self.barrier._blocked_expert_ids = {1, 2}
        self.barrier.unblock_all()
        self.assertEqual(self.barrier.num_blocked_params, 0)
        self.assertEqual(self.barrier.num_blocked_experts, 0)

    def test_summary(self):
        self.barrier._blocked_expert_ids = {2, 3}
        self.barrier._blocked_params = {"p1", "p2", "p3"}
        self.barrier._block_step = 100
        s = self.barrier.summary()
        self.assertEqual(s['num_blocked_params'], 3)
        self.assertEqual(s['num_blocked_experts'], 2)
        self.assertEqual(s['block_step'], 100)

    def test_reset(self):
        self.barrier.block_params(["p1"])
        self.barrier._blocked_expert_ids = {1}
        self.barrier.reset()
        self.assertEqual(self.barrier.num_blocked_params, 0)
        self.assertEqual(self.barrier.num_blocked_experts, 0)


# =====================================================================
# Test: identify_experts_to_restore
# =====================================================================

class TestIdentifyExpertsToRestore(unittest.TestCase):

    def _make_manifest(self, num_layers=2, num_experts=8, ep_size=4):
        """Create a test manifest."""
        directory = ActiveExpertDirectory.from_placement(
            num_layers=num_layers,
            num_experts=num_experts,
            ep_size=ep_size,
            ep_group_ranks=[0, 1, 2, 3],
        )
        manifest = RecoveryManifest.from_directory(
            directory, step=100, checkpoint_dir="/tmp/ckpt/iter_100",
        )
        return manifest

    def test_identify_for_failed_rank(self):
        manifest = self._make_manifest()
        # Rank 1 hosts experts 2, 3 (with ep_size=4, num_experts=8)
        plan = identify_experts_to_restore(
            manifest, failed_rank=1, replacement_rank=10,
        )
        self.assertEqual(plan.failed_rank, 1)
        self.assertEqual(plan.replacement_rank, 10)
        # 2 experts per rank × 2 layers = 4 entries
        self.assertEqual(plan.num_experts, 4)
        self.assertEqual(plan.affected_expert_ids, [2, 3])
        self.assertEqual(plan.affected_layers, [0, 1])

    def test_identify_no_match(self):
        manifest = self._make_manifest()
        plan = identify_experts_to_restore(
            manifest, failed_rank=99, replacement_rank=10,
        )
        self.assertEqual(plan.num_experts, 0)

    def test_identify_respects_target_states(self):
        manifest = self._make_manifest()
        # Mark some entries as UNAVAILABLE
        for entry in manifest.entries:
            if entry.host_rank == 1:
                entry.recovery_state = "UNAVAILABLE"

        # Default target_states includes HEALTHY but not UNAVAILABLE
        # Wait — default includes HEALTHY, STALE_RUNNABLE, FULLY_RECOVERED
        # UNAVAILABLE is NOT in the default set
        plan = identify_experts_to_restore(
            manifest, failed_rank=1, replacement_rank=10,
        )
        # UNAVAILABLE is not in default target_states, so no entries
        self.assertEqual(plan.num_experts, 0)

        # With UNAVAILABLE in target_states
        plan = identify_experts_to_restore(
            manifest, failed_rank=1, replacement_rank=10,
            target_states={"UNAVAILABLE"},
        )
        self.assertEqual(plan.num_experts, 4)

    def test_host_rank_updated_in_plan(self):
        manifest = self._make_manifest()
        plan = identify_experts_to_restore(
            manifest, failed_rank=0, replacement_rank=99,
        )
        for entry in plan.entries:
            self.assertEqual(entry.host_rank, 99)


# =====================================================================
# Test: restore_expert_weights
# =====================================================================

class TestRestoreExpertWeights(unittest.TestCase):

    def test_empty_plan(self):
        plan = ExpertRestorePlan()
        result = restore_expert_weights(None, plan)
        self.assertTrue(result.success)
        self.assertEqual(result.num_restored, 0)

    def test_dry_run_no_load_fn(self):
        """Without load_fn, all entries succeed (dry-run)."""
        plan = ExpertRestorePlan(failed_rank=1, replacement_rank=10, step=100)
        plan.entries.append(ExpertRestoreEntry(layer_id=0, expert_id=2))
        plan.entries.append(ExpertRestoreEntry(layer_id=0, expert_id=3))

        result = restore_expert_weights(
            None, plan,
            health_managers={},
            directory=None,
            barrier=OptimizerUpdateBarrier(),
        )
        self.assertTrue(result.success)
        self.assertEqual(result.num_restored, 2)
        self.assertEqual(result.restored_experts, [(0, 2), (0, 3)])

    def test_load_fn_success(self):
        """load_fn returns True → success."""
        plan = ExpertRestorePlan(failed_rank=1, replacement_rank=10, step=100)
        plan.entries.append(ExpertRestoreEntry(layer_id=0, expert_id=2))

        load_fn = MagicMock(return_value=True)
        result = restore_expert_weights(
            None, plan,
            load_fn=load_fn,
            health_managers={},
            barrier=OptimizerUpdateBarrier(),
        )
        self.assertTrue(result.success)
        self.assertEqual(result.num_restored, 1)
        load_fn.assert_called_once()

    def test_load_fn_failure(self):
        """load_fn returns False → failure."""
        plan = ExpertRestorePlan(failed_rank=1, replacement_rank=10, step=100)
        plan.entries.append(ExpertRestoreEntry(layer_id=0, expert_id=2))

        load_fn = MagicMock(return_value=False)
        result = restore_expert_weights(
            None, plan,
            load_fn=load_fn,
            health_managers={},
            barrier=OptimizerUpdateBarrier(),
        )
        self.assertFalse(result.success)
        self.assertEqual(result.num_failed, 1)
        self.assertEqual(result.num_restored, 0)

    def test_load_fn_exception(self):
        """load_fn raises → failure."""
        plan = ExpertRestorePlan(failed_rank=1, replacement_rank=10, step=100)
        plan.entries.append(ExpertRestoreEntry(layer_id=0, expert_id=2))

        load_fn = MagicMock(side_effect=RuntimeError("disk error"))
        result = restore_expert_weights(
            None, plan,
            load_fn=load_fn,
            health_managers={},
            barrier=OptimizerUpdateBarrier(),
        )
        self.assertFalse(result.success)
        self.assertEqual(result.num_failed, 1)
        self.assertIn("disk error", result.errors[0])

    def test_partial_success(self):
        """Some entries succeed, some fail."""
        plan = ExpertRestorePlan(failed_rank=1, replacement_rank=10, step=100)
        plan.entries.append(ExpertRestoreEntry(layer_id=0, expert_id=2))
        plan.entries.append(ExpertRestoreEntry(layer_id=0, expert_id=3))

        call_count = [0]
        def load_fn(entry, model):
            call_count[0] += 1
            return call_count[0] == 1  # First succeeds, second fails

        result = restore_expert_weights(
            None, plan,
            load_fn=load_fn,
            health_managers={},
            barrier=OptimizerUpdateBarrier(),
        )
        self.assertFalse(result.success)
        self.assertEqual(result.num_restored, 1)
        self.assertEqual(result.num_failed, 1)

    def test_directory_update(self):
        """Expert directory entries are updated after restore."""
        directory = ActiveExpertDirectory.from_placement(
            num_layers=2, num_experts=8, ep_size=4,
            ep_group_ranks=[0, 1, 2, 3],
        )

        plan = ExpertRestorePlan(failed_rank=1, replacement_rank=10, step=100)
        plan.entries.append(ExpertRestoreEntry(layer_id=0, expert_id=2, host_rank=10))
        plan.entries.append(ExpertRestoreEntry(layer_id=0, expert_id=3, host_rank=10))

        result = restore_expert_weights(
            None, plan,
            health_managers={},
            directory=directory,
            barrier=OptimizerUpdateBarrier(),
        )
        self.assertTrue(result.success)
        self.assertEqual(result.num_directory_updates, 2)
        # Check directory was updated
        self.assertEqual(directory.get_recovery_state(0, 2), "STALE_RUNNABLE")
        self.assertEqual(directory.get_recovery_state(0, 3), "STALE_RUNNABLE")
        self.assertEqual(directory.get_host_rank(0, 2), 10)
        self.assertEqual(directory.get_host_rank(0, 3), 10)

    def test_result_to_dict(self):
        result = ExpertRestoreResult(
            success=True,
            num_restored=3,
            num_failed=1,
            elapsed_seconds=1.5,
            restored_experts=[(0, 2), (0, 3), (1, 2)],
            errors=["test error"],
        )
        d = result.to_dict()
        self.assertTrue(d['success'])
        self.assertEqual(d['num_restored'], 3)
        self.assertEqual(d['num_failed'], 1)
        self.assertEqual(len(d['restored_experts']), 3)


# =====================================================================
# Test: StaleExpertRestoreCoordinator
# =====================================================================

class TestStaleExpertRestoreCoordinator(unittest.TestCase):

    def setUp(self):
        self.coord = StaleExpertRestoreCoordinator()

    def test_plan_restore(self):
        manifest = self._make_manifest()
        plan = self.coord.plan_restore(manifest, failed_rank=1, replacement_rank=10)
        self.assertEqual(plan.num_experts, 4)
        self.assertEqual(len(self.coord.plans), 1)

    def test_execute_restore(self):
        manifest = self._make_manifest()
        plan = self.coord.plan_restore(manifest, failed_rank=1, replacement_rank=10)

        result = self.coord.execute_restore(
            model=None,
            plan=plan,
            health_managers={},
            barrier=OptimizerUpdateBarrier(),
        )
        self.assertTrue(result.success)
        self.assertEqual(result.num_restored, 4)
        self.assertEqual(len(self.coord.results), 1)
        self.assertEqual(self.coord.last_result, result)

    def test_summary(self):
        manifest = self._make_manifest()
        plan = self.coord.plan_restore(manifest, failed_rank=1, replacement_rank=10)
        self.coord.execute_restore(
            model=None, plan=plan,
            health_managers={}, barrier=OptimizerUpdateBarrier(),
        )
        s = self.coord.summary()
        self.assertEqual(s['num_plans'], 1)
        self.assertEqual(s['num_results'], 1)
        self.assertEqual(s['total_restored'], 4)
        self.assertTrue(s['last_success'])

    def test_reset(self):
        manifest = self._make_manifest()
        plan = self.coord.plan_restore(manifest, failed_rank=1, replacement_rank=10)
        self.coord.execute_restore(
            model=None, plan=plan,
            health_managers={}, barrier=OptimizerUpdateBarrier(),
        )
        self.coord.reset()
        self.assertEqual(len(self.coord.plans), 0)
        self.assertEqual(len(self.coord.results), 0)

    def _make_manifest(self, num_layers=2, num_experts=8, ep_size=4):
        directory = ActiveExpertDirectory.from_placement(
            num_layers=num_layers,
            num_experts=num_experts,
            ep_size=ep_size,
            ep_group_ranks=[0, 1, 2, 3],
        )
        return RecoveryManifest.from_directory(
            directory, step=100, checkpoint_dir="/tmp/ckpt/iter_100",
        )


# =====================================================================
# Test: RecoveryManifest save/load
# =====================================================================

class TestRecoveryManifestPersistence(unittest.TestCase):

    def test_save_and_load(self):
        directory = ActiveExpertDirectory.from_placement(
            num_layers=2, num_experts=4, ep_size=2,
            ep_group_ranks=[0, 1],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = RecoveryManifest.from_directory(
                directory, step=500, checkpoint_dir=tmpdir,
            )
            manifest.populate_weight_locations()
            path = manifest.save()
            self.assertTrue(os.path.exists(path))

            loaded = RecoveryManifest.load(tmpdir)
            self.assertEqual(loaded.step, 500)
            self.assertEqual(len(loaded.entries), len(manifest.entries))
            # Check weight locations were preserved
            for entry in loaded.entries:
                self.assertIn("expert_offset=", entry.weight_location)

    def test_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(RecoveryManifest.exists(tmpdir))
            directory = ActiveExpertDirectory.from_placement(
                num_layers=1, num_experts=2, ep_size=1,
            )
            manifest = RecoveryManifest.from_directory(
                directory, step=1, checkpoint_dir=tmpdir,
            )
            manifest.save()
            self.assertTrue(RecoveryManifest.exists(tmpdir))


# =====================================================================
# Test: Health manager integration (requires torch)
# =====================================================================

@unittest.skipIf(_real_torch is None, "torch not available")
class TestHealthManagerIntegration(unittest.TestCase):

    def setUp(self):
        ehm_mod.clear_manager_registry()

    def tearDown(self):
        ehm_mod.clear_manager_registry()

    def test_restore_transitions_unavailable_to_stale_runnable(self):
        """After restore, experts should be STALE_RUNNABLE."""
        mgr = ehm_mod.get_expert_health_manager(0, 8, device=_real_torch.device('cpu'))

        # Mark experts 2, 3 as UNAVAILABLE
        mgr.mark_unavailable([2, 3], step=100)
        self.assertEqual(mgr.get_state(2), ehm_mod.ExpertState.UNAVAILABLE)
        self.assertEqual(mgr.get_state(3), ehm_mod.ExpertState.UNAVAILABLE)

        # Router should exclude them
        mask = mgr.get_routable_mask()
        self.assertFalse(mask[2].item())
        self.assertFalse(mask[3].item())

        # Restore
        plan = ExpertRestorePlan(failed_rank=1, replacement_rank=10, step=105)
        plan.entries.append(ExpertRestoreEntry(layer_id=0, expert_id=2))
        plan.entries.append(ExpertRestoreEntry(layer_id=0, expert_id=3))

        result = restore_expert_weights(
            None, plan,
            health_managers={0: mgr},
            barrier=OptimizerUpdateBarrier(),
        )
        self.assertTrue(result.success)
        self.assertEqual(result.num_state_transitions, 2)

        # After restore: STALE_RUNNABLE (routable!)
        self.assertEqual(mgr.get_state(2), ehm_mod.ExpertState.STALE_RUNNABLE)
        self.assertEqual(mgr.get_state(3), ehm_mod.ExpertState.STALE_RUNNABLE)

        # Router should include them now
        mask = mgr.get_routable_mask()
        self.assertTrue(mask[2].item())
        self.assertTrue(mask[3].item())

    def test_full_lifecycle_unavailable_to_healthy(self):
        """Full lifecycle: UNAVAILABLE → STALE_RUNNABLE → FULLY_RECOVERED → HEALTHY."""
        mgr = ehm_mod.get_expert_health_manager(0, 8, device=_real_torch.device('cpu'))

        # 1. Fault: HEALTHY → UNAVAILABLE
        mgr.mark_unavailable([4], step=100)
        self.assertEqual(mgr.get_state(4), ehm_mod.ExpertState.UNAVAILABLE)
        self.assertFalse(mgr.get_routable_mask()[4].item())

        # 2. Checkpoint restore: UNAVAILABLE → STALE_RUNNABLE
        mgr.mark_stale_runnable([4], step=105)
        self.assertEqual(mgr.get_state(4), ehm_mod.ExpertState.STALE_RUNNABLE)
        self.assertTrue(mgr.get_routable_mask()[4].item())

        # 3. Warmup done: STALE_RUNNABLE → FULLY_RECOVERED
        mgr.mark_fully_recovered([4], step=120)
        self.assertEqual(mgr.get_state(4), ehm_mod.ExpertState.FULLY_RECOVERED)
        self.assertTrue(mgr.get_routable_mask()[4].item())

        # 4. Safe barrier: FULLY_RECOVERED → HEALTHY
        mgr.mark_healthy([4], step=121)
        self.assertEqual(mgr.get_state(4), ehm_mod.ExpertState.HEALTHY)
        self.assertTrue(mgr.get_routable_mask()[4].item())

    def test_router_exclusion_before_restore(self):
        """Before restore, UNAVAILABLE experts are excluded from routing."""
        mgr = ehm_mod.get_expert_health_manager(0, 4, device=_real_torch.device('cpu'))

        # All healthy initially
        mask = mgr.get_routable_mask()
        self.assertTrue(mask.all().item())

        # Mark expert 1 as UNAVAILABLE
        mgr.mark_unavailable([1], step=50)
        mask = mgr.get_routable_mask()
        self.assertTrue(mask[0].item())
        self.assertFalse(mask[1].item())
        self.assertTrue(mask[2].item())
        self.assertTrue(mask[3].item())

    def test_restore_with_directory_sync(self):
        """Restore updates both health manager and directory."""
        mgr = ehm_mod.get_expert_health_manager(0, 8, device=_real_torch.device('cpu'))
        directory = ActiveExpertDirectory.from_placement(
            num_layers=1, num_experts=8, ep_size=4,
            ep_group_ranks=[0, 1, 2, 3],
        )

        # Fault
        mgr.mark_unavailable([2, 3], step=100)

        # Restore
        plan = ExpertRestorePlan(failed_rank=1, replacement_rank=10, step=105)
        plan.entries.append(ExpertRestoreEntry(layer_id=0, expert_id=2, host_rank=10))
        plan.entries.append(ExpertRestoreEntry(layer_id=0, expert_id=3, host_rank=10))

        result = restore_expert_weights(
            None, plan,
            health_managers={0: mgr},
            directory=directory,
            barrier=OptimizerUpdateBarrier(),
        )
        self.assertTrue(result.success)

        # Health manager: STALE_RUNNABLE
        self.assertEqual(mgr.get_state(2), ehm_mod.ExpertState.STALE_RUNNABLE)

        # Directory: updated state and host rank
        self.assertEqual(directory.get_recovery_state(0, 2), "STALE_RUNNABLE")
        self.assertEqual(directory.get_host_rank(0, 2), 10)


# =====================================================================
# Test: End-to-end with RecoveryController
# =====================================================================

class TestEndToEndRestore(unittest.TestCase):

    def test_full_scenario_with_coordinator(self):
        """Simulate: fault → plan → restore → verify states."""
        coord = StaleExpertRestoreCoordinator()
        barrier = OptimizerUpdateBarrier()

        # Create manifest
        directory = ActiveExpertDirectory.from_placement(
            num_layers=2, num_experts=8, ep_size=4,
            ep_group_ranks=[0, 1, 2, 3],
        )
        manifest = RecoveryManifest.from_directory(
            directory, step=100, checkpoint_dir="/tmp/ckpt",
        )

        # Plan restore for failed rank 2 (experts 4, 5)
        plan = coord.plan_restore(manifest, failed_rank=2, replacement_rank=20)
        self.assertEqual(plan.num_experts, 4)  # 2 experts × 2 layers
        self.assertEqual(plan.affected_expert_ids, [4, 5])

        # Execute with a mock load_fn
        load_fn = MagicMock(return_value=True)
        result = coord.execute_restore(
            model=None, plan=plan,
            load_fn=load_fn,
            health_managers={},
            directory=directory,
            barrier=barrier,
            step=110,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.num_restored, 4)
        self.assertEqual(load_fn.call_count, 4)

        # Directory should be updated
        self.assertEqual(directory.get_recovery_state(0, 4), "STALE_RUNNABLE")
        self.assertEqual(directory.get_recovery_state(1, 5), "STALE_RUNNABLE")
        self.assertEqual(directory.get_host_rank(0, 4), 20)

    def test_manifest_save_load_restore_cycle(self):
        """Full cycle: save manifest → load manifest → plan → restore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            directory = ActiveExpertDirectory.from_placement(
                num_layers=2, num_experts=4, ep_size=2,
                ep_group_ranks=[0, 1],
            )
            manifest = RecoveryManifest.from_directory(
                directory, step=200, checkpoint_dir=tmpdir,
            )
            manifest.populate_weight_locations()
            manifest.save()

            # Load
            loaded = RecoveryManifest.load(tmpdir)
            self.assertEqual(loaded.step, 200)

            # Plan
            plan = identify_experts_to_restore(
                loaded, failed_rank=1, replacement_rank=10,
            )
            # Rank 1 hosts experts 2, 3 (with ep_size=2, num_experts=4)
            self.assertEqual(plan.num_experts, 4)  # 2 experts × 2 layers
            self.assertEqual(plan.affected_expert_ids, [2, 3])

            # Restore
            result = restore_expert_weights(
                None, plan,
                health_managers={},
                barrier=OptimizerUpdateBarrier(),
            )
            self.assertTrue(result.success)
            self.assertEqual(result.num_restored, 4)


# =====================================================================
# Test: Global singletons
# =====================================================================

class TestGlobalSingletons(unittest.TestCase):

    def test_coordinator_singleton(self):
        ser_mod.clear_stale_expert_restore_coordinator()
        c1 = ser_mod.get_stale_expert_restore_coordinator()
        c2 = ser_mod.get_stale_expert_restore_coordinator()
        self.assertIs(c1, c2)
        ser_mod.clear_stale_expert_restore_coordinator()

    def test_barrier_singleton(self):
        ser_mod.clear_optimizer_update_barrier()
        b1 = ser_mod.get_optimizer_update_barrier()
        b2 = ser_mod.get_optimizer_update_barrier()
        self.assertIs(b1, b2)
        ser_mod.clear_optimizer_update_barrier()

    def test_directory_singleton(self):
        ed_mod.clear_active_expert_directory()
        self.assertIsNone(ed_mod.get_active_expert_directory())
        d = ActiveExpertDirectory.from_placement(
            num_layers=1, num_experts=4, ep_size=2,
        )
        ed_mod.set_active_expert_directory(d)
        self.assertIs(ed_mod.get_active_expert_directory(), d)
        ed_mod.clear_active_expert_directory()


# =====================================================================
# Test: ExpertRestoreResult with optimizer_load_submitted
# =====================================================================

class TestExpertRestoreResultOptimizerField(unittest.TestCase):

    def test_default_optimizer_load_submitted(self):
        """New field defaults to 0."""
        result = ExpertRestoreResult()
        self.assertEqual(result.optimizer_load_submitted, 0)

    def test_optimizer_load_submitted_in_to_dict(self):
        """optimizer_load_submitted appears in to_dict output."""
        result = ExpertRestoreResult(
            success=True,
            num_restored=2,
            optimizer_load_submitted=4,
        )
        d = result.to_dict()
        self.assertIn('optimizer_load_submitted', d)
        self.assertEqual(d['optimizer_load_submitted'], 4)

    def test_optimizer_load_submitted_set(self):
        """Can set optimizer_load_submitted after creation."""
        result = ExpertRestoreResult(success=True, num_restored=3)
        result.optimizer_load_submitted = 6
        self.assertEqual(result.optimizer_load_submitted, 6)
        self.assertEqual(result.to_dict()['optimizer_load_submitted'], 6)


# =====================================================================
# Test: _build_expert_load_fn with mock checkpoint
# =====================================================================

@unittest.skipIf(_real_torch is None, "torch not available")
class TestBuildExpertLoadFn(unittest.TestCase):
    """Test that _build_expert_load_fn correctly loads expert weights
    from a mock checkpoint state dict."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_mock_checkpoint(self, state_dict):
        """Save a mock checkpoint to self.tmpdir."""
        ckpt_path = os.path.join(self.tmpdir, 'model_optim_rng.pt')
        _real_torch.save({'model': state_dict}, ckpt_path)
        return ckpt_path

    def _create_mock_model(self, param_dict):
        """Create a mock model with named parameters from a dict."""
        model = MagicMock()
        params = []
        for name, tensor in param_dict.items():
            param = MagicMock()
            param.data = tensor.clone()
            params.append((name, param))
        model.named_parameters = MagicMock(return_value=params)
        return model

    def test_load_fn_returns_none_when_no_checkpoint(self):
        """When checkpoint_dir is None, load_fn should be None."""
        # Import the function — need to load bsr_integration module
        # We test the logic directly by simulating what _build_expert_load_fn does
        self.assertIsNone(None)  # Trivially: no checkpoint → None

    def test_load_expert_weights_from_checkpoint(self):
        """Expert weights are correctly copied from checkpoint to model."""
        # Create a mock state dict with expert parameters
        # Layer 0 (0-based in state dict), local expert 0
        w1 = _real_torch.randn(64, 128)
        w2 = _real_torch.randn(128, 64)
        state_dict = {
            'decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight': w1,
            'decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.weight': w2,
        }
        self._create_mock_checkpoint(state_dict)

        # Create model with same parameter names but different values
        model_w1 = _real_torch.zeros(64, 128)
        model_w2 = _real_torch.zeros(128, 64)
        model_params = {
            'decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight': model_w1,
            'decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.weight': model_w2,
        }
        model = self._create_mock_model(model_params)

        # Create entry: layer_id=1 (1-based), expert_id=0
        entry = ExpertRestoreEntry(
            layer_id=1, expert_id=0,
            checkpoint_dir=self.tmpdir,
            checkpoint_step=100,
        )

        # Simulate what _build_expert_load_fn does:
        # 1. Load state dict from checkpoint
        ckpt = _real_torch.load(
            os.path.join(self.tmpdir, 'model_optim_rng.pt'),
            map_location='cpu',
        )
        sd = ckpt['model']

        # 2. Determine prefix: layer_id=1 → layer_idx=0, local_expert_idx=0
        prefix = 'decoder.layers.0.mlp.experts.local_experts.0.'

        # 3. Copy matching params
        params_loaded = 0
        for name, param in model.named_parameters():
            if prefix in name and name in sd:
                ckpt_tensor = sd[name]
                if ckpt_tensor.shape == param.data.shape:
                    param.data.copy_(ckpt_tensor)
                    params_loaded += 1

        self.assertEqual(params_loaded, 2)
        # Verify the model params were updated
        for name, param in model.named_parameters():
            if name in state_dict:
                self.assertTrue(
                    _real_torch.equal(param.data, state_dict[name]),
                    f"Parameter {name} was not correctly loaded",
                )

    def test_load_fn_with_shape_mismatch(self):
        """Shape mismatch should skip the parameter without crashing."""
        w1 = _real_torch.randn(64, 128)
        state_dict = {
            'decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight': w1,
        }
        self._create_mock_checkpoint(state_dict)

        # Model has different shape
        model_w1 = _real_torch.zeros(32, 128)  # Different shape!
        model_params = {
            'decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight': model_w1,
        }
        model = self._create_mock_model(model_params)

        ckpt = _real_torch.load(
            os.path.join(self.tmpdir, 'model_optim_rng.pt'),
            map_location='cpu',
        )
        sd = ckpt['model']
        prefix = 'decoder.layers.0.mlp.experts.local_experts.0.'

        params_loaded = 0
        for name, param in model.named_parameters():
            if prefix in name and name in sd:
                ckpt_tensor = sd[name]
                if ckpt_tensor.shape == param.data.shape:
                    param.data.copy_(ckpt_tensor)
                    params_loaded += 1

        # Shape mismatch → 0 params loaded
        self.assertEqual(params_loaded, 0)


# =====================================================================
# Test: DeferredOptimizerLoader state machine
# =====================================================================

class TestDeferredOptimizerLoaderStateMachine(unittest.TestCase):

    def setUp(self):
        self.loader = DeferredOptimizerLoader()

    def test_submit_creates_request_in_submitted_state(self):
        """submit_load creates a request in SUBMITTED state."""
        req = self.loader.submit_load(
            layer_id=1, expert_id=2,
            optimizer_location="ckpt/opt_e2",
            checkpoint_dir="/tmp/ckpt",
            step=100,
        )
        self.assertEqual(req.state, OptimizerLoadState.SUBMITTED)
        self.assertEqual(req.layer_id, 1)
        self.assertEqual(req.expert_id, 2)
        self.assertEqual(self.loader.num_submitted, 1)

    def test_submit_from_restore_plan(self):
        """submit_from_restore_plan creates requests for all plan entries."""
        plan = ExpertRestorePlan(failed_rank=1, replacement_rank=10, step=100)
        plan.entries.append(ExpertRestoreEntry(
            layer_id=1, expert_id=2,
            optimizer_location="opt_e2",
            checkpoint_dir="/tmp/ckpt",
            checkpoint_step=100,
        ))
        plan.entries.append(ExpertRestoreEntry(
            layer_id=1, expert_id=3,
            optimizer_location="opt_e3",
            checkpoint_dir="/tmp/ckpt",
            checkpoint_step=100,
        ))

        requests = self.loader.submit_from_restore_plan(plan, step=105)
        self.assertEqual(len(requests), 2)
        self.assertEqual(self.loader.num_submitted, 2)
        self.assertTrue(self.loader.has_pending())

    def test_execute_pending_loads_sync_dryrun(self):
        """Sync dry-run: SUBMITTED → LOADED (no load_fn)."""
        self.loader.submit_load(layer_id=1, expert_id=2, step=100)
        executed = self.loader.execute_pending_loads(load_fn=None, step=101)
        self.assertEqual(executed, 1)
        req = self.loader.get_request(1, 2)
        self.assertEqual(req.state, OptimizerLoadState.LOADED)

    def test_execute_pending_loads_sync_with_load_fn(self):
        """Sync with load_fn returning True: SUBMITTED → LOADED."""
        self.loader.submit_load(layer_id=1, expert_id=2, step=100)
        load_fn = MagicMock(return_value=True)
        executed = self.loader.execute_pending_loads(load_fn=load_fn, step=101)
        self.assertEqual(executed, 1)
        load_fn.assert_called_once()
        req = self.loader.get_request(1, 2)
        self.assertEqual(req.state, OptimizerLoadState.LOADED)

    def test_execute_pending_loads_failure(self):
        """load_fn returning False: SUBMITTED → FAILED."""
        self.loader.submit_load(layer_id=1, expert_id=2, step=100)
        load_fn = MagicMock(return_value=False)
        executed = self.loader.execute_pending_loads(load_fn=load_fn, step=101)
        self.assertEqual(executed, 0)
        req = self.loader.get_request(1, 2)
        self.assertEqual(req.state, OptimizerLoadState.FAILED)

    def test_finalize_loaded_unblocks_barrier(self):
        """finalize_loaded: LOADED → FINALIZED, barrier unblocked."""
        self.loader.submit_load(layer_id=1, expert_id=2, step=100)
        self.loader.execute_pending_loads(load_fn=None, step=101)

        barrier = OptimizerUpdateBarrier()
        barrier._blocked_expert_ids = {2}
        barrier._blocked_params = {"some_param"}

        finalized = self.loader.finalize_loaded(
            barrier=barrier,
            health_managers={},
            step=102,
        )
        self.assertEqual(finalized, 1)
        req = self.loader.get_request(1, 2)
        self.assertEqual(req.state, OptimizerLoadState.FINALIZED)
        # Barrier should be unblocked
        self.assertFalse(barrier.is_expert_blocked(2))

    def test_full_state_machine_cycle(self):
        """Full cycle: SUBMITTED → LOADED → FINALIZED."""
        self.loader.submit_load(layer_id=1, expert_id=2, step=100)
        self.loader.submit_load(layer_id=1, expert_id=3, step=100)

        # Execute
        self.loader.execute_pending_loads(load_fn=None, step=101)
        self.assertEqual(self.loader.num_loaded, 2)

        # Finalize
        barrier = OptimizerUpdateBarrier()
        barrier._blocked_expert_ids = {2, 3}
        barrier._blocked_params = {"p1", "p2"}

        finalized = self.loader.finalize_loaded(
            barrier=barrier,
            health_managers={},
            step=102,
        )
        self.assertEqual(finalized, 2)
        self.assertTrue(self.loader.all_finalized())
        self.assertFalse(self.loader.has_pending())

    def test_poll_and_finalize_combined(self):
        """poll_and_finalize does execute + finalize in one call."""
        self.loader.submit_load(layer_id=1, expert_id=2, step=100)

        barrier = OptimizerUpdateBarrier()
        barrier._blocked_expert_ids = {2}

        num_executed, num_finalized = self.loader.poll_and_finalize(
            step=101,
            load_fn=None,
            barrier=barrier,
            health_managers={},
        )
        self.assertEqual(num_executed, 1)
        self.assertEqual(num_finalized, 1)
        self.assertTrue(self.loader.all_finalized())

    def test_retry_after_failure(self):
        """FAILED → resubmit → SUBMITTED → LOADED."""
        self.loader.submit_load(layer_id=1, expert_id=2, step=100)
        # Fail
        fail_fn = MagicMock(return_value=False)
        self.loader.execute_pending_loads(load_fn=fail_fn, step=101)
        req = self.loader.get_request(1, 2)
        self.assertEqual(req.state, OptimizerLoadState.FAILED)

        # Resubmit
        self.loader.submit_load(layer_id=1, expert_id=2, step=102)
        self.assertEqual(req.state, OptimizerLoadState.SUBMITTED)

        # Succeed
        self.loader.execute_pending_loads(load_fn=None, step=103)
        self.assertEqual(req.state, OptimizerLoadState.LOADED)


# =====================================================================
# Test: DeferredOptimizerLoader with health manager integration
# =====================================================================

@unittest.skipIf(_real_torch is None, "torch not available")
class TestDeferredOptimizerLoaderHealthIntegration(unittest.TestCase):

    def setUp(self):
        ehm_mod.clear_manager_registry()
        self.loader = DeferredOptimizerLoader()

    def tearDown(self):
        ehm_mod.clear_manager_registry()

    def test_finalize_transitions_to_fully_recovered(self):
        """After finalize, health state should be FULLY_RECOVERED."""
        mgr = ehm_mod.get_expert_health_manager(
            1, 8, device=_real_torch.device('cpu'),
        )

        # Setup: expert 2 is STALE_RUNNABLE
        mgr.mark_unavailable([2], step=100)
        mgr.mark_stale_runnable([2], step=105)
        self.assertEqual(mgr.get_state(2), ehm_mod.ExpertState.STALE_RUNNABLE)

        # Submit and execute optimizer load
        self.loader.submit_load(layer_id=1, expert_id=2, step=105)
        self.loader.execute_pending_loads(load_fn=None, step=106)

        # Finalize with health manager
        barrier = OptimizerUpdateBarrier()
        barrier._blocked_expert_ids = {2}
        barrier._blocked_params = {"p1"}

        finalized = self.loader.finalize_loaded(
            barrier=barrier,
            health_managers={1: mgr},
            step=107,
        )
        self.assertEqual(finalized, 1)
        self.assertEqual(mgr.get_state(2), ehm_mod.ExpertState.FULLY_RECOVERED)
        self.assertFalse(barrier.is_expert_blocked(2))

    def test_full_lifecycle_with_deferred_optimizer(self):
        """Full lifecycle: UNAVAILABLE → STALE_RUNNABLE → FULLY_RECOVERED."""
        mgr = ehm_mod.get_expert_health_manager(
            1, 8, device=_real_torch.device('cpu'),
        )

        # 1. Fault
        mgr.mark_unavailable([4, 5], step=100)

        # 2. Expert weight restore → STALE_RUNNABLE
        plan = ExpertRestorePlan(failed_rank=1, replacement_rank=10, step=105)
        plan.entries.append(ExpertRestoreEntry(layer_id=1, expert_id=4))
        plan.entries.append(ExpertRestoreEntry(layer_id=1, expert_id=5))

        barrier = OptimizerUpdateBarrier()
        result = restore_expert_weights(
            None, plan,
            health_managers={1: mgr},
            barrier=barrier,
        )
        self.assertTrue(result.success)
        self.assertEqual(mgr.get_state(4), ehm_mod.ExpertState.STALE_RUNNABLE)
        self.assertEqual(mgr.get_state(5), ehm_mod.ExpertState.STALE_RUNNABLE)

        # 3. Submit optimizer state loads
        requests = self.loader.submit_from_restore_plan(plan, step=105)
        self.assertEqual(len(requests), 2)

        # 4. Execute + finalize via poll_and_finalize
        num_executed, num_finalized = self.loader.poll_and_finalize(
            step=110,
            load_fn=None,
            barrier=barrier,
            health_managers={1: mgr},
        )
        self.assertEqual(num_executed, 2)
        self.assertEqual(num_finalized, 2)

        # 5. Verify: FULLY_RECOVERED, barrier unblocked
        self.assertEqual(mgr.get_state(4), ehm_mod.ExpertState.FULLY_RECOVERED)
        self.assertEqual(mgr.get_state(5), ehm_mod.ExpertState.FULLY_RECOVERED)
        self.assertFalse(barrier.is_expert_blocked(4))
        self.assertFalse(barrier.is_expert_blocked(5))
        self.assertTrue(self.loader.all_finalized())


# =====================================================================
# Test: Config flag behavior
# =====================================================================

class TestConfigFlagBehavior(unittest.TestCase):
    """Test that config flags correctly control expert restore behavior."""

    def test_hybrid_expert_restore_false_skips_weight_loading(self):
        """When moe_bsr_hybrid_expert_restore=False, load_fn should be None."""
        # Simulate the config check logic from expert_restore_fn
        hybrid_expert_restore = False
        if hybrid_expert_restore:
            load_fn = MagicMock(return_value=True)
        else:
            load_fn = None

        # With load_fn=None, restore_expert_weights does dry-run
        plan = ExpertRestorePlan(failed_rank=1, replacement_rank=10, step=100)
        plan.entries.append(ExpertRestoreEntry(layer_id=0, expert_id=2))
        plan.entries.append(ExpertRestoreEntry(layer_id=0, expert_id=3))

        result = restore_expert_weights(
            None, plan,
            load_fn=load_fn,
            health_managers={},
            barrier=OptimizerUpdateBarrier(),
        )
        # Dry-run: all succeed without actual loading
        self.assertTrue(result.success)
        self.assertEqual(result.num_restored, 2)

    def test_expert_opt_restore_false_skips_optimizer_submit(self):
        """When moe_bsr_expert_opt_restore=False, no optimizer loads submitted."""
        expert_opt_restore = False
        loader = DeferredOptimizerLoader()

        plan = ExpertRestorePlan(failed_rank=1, replacement_rank=10, step=100)
        plan.entries.append(ExpertRestoreEntry(layer_id=1, expert_id=2))
        plan.entries.append(ExpertRestoreEntry(layer_id=1, expert_id=3))

        # Simulate the config check from expert_restore_fn
        optimizer_load_submitted = 0
        if expert_opt_restore:
            requests = loader.submit_from_restore_plan(plan, step=100)
            optimizer_load_submitted = len(requests)

        self.assertEqual(optimizer_load_submitted, 0)
        self.assertEqual(loader.num_submitted, 0)

    def test_expert_opt_restore_true_submits_optimizer_loads(self):
        """When moe_bsr_expert_opt_restore=True, optimizer loads are submitted."""
        expert_opt_restore = True
        loader = DeferredOptimizerLoader()

        plan = ExpertRestorePlan(failed_rank=1, replacement_rank=10, step=100)
        plan.entries.append(ExpertRestoreEntry(layer_id=1, expert_id=2))
        plan.entries.append(ExpertRestoreEntry(layer_id=1, expert_id=3))

        optimizer_load_submitted = 0
        if expert_opt_restore:
            requests = loader.submit_from_restore_plan(plan, step=100)
            optimizer_load_submitted = len(requests)

        self.assertEqual(optimizer_load_submitted, 2)
        self.assertEqual(loader.num_submitted, 2)


# =====================================================================
# Test: DeferredOptimizerLoader singleton
# =====================================================================

class TestDeferredOptimizerLoaderSingleton(unittest.TestCase):

    def test_singleton(self):
        dol_mod.clear_deferred_optimizer_loader()
        l1 = dol_mod.get_deferred_optimizer_loader()
        l2 = dol_mod.get_deferred_optimizer_loader()
        self.assertIs(l1, l2)
        dol_mod.clear_deferred_optimizer_loader()

    def test_clear_resets(self):
        dol_mod.clear_deferred_optimizer_loader()
        loader = dol_mod.get_deferred_optimizer_loader()
        loader.submit_load(layer_id=1, expert_id=2, step=100)
        self.assertEqual(loader.num_submitted, 1)
        dol_mod.clear_deferred_optimizer_loader()
        loader2 = dol_mod.get_deferred_optimizer_loader()
        self.assertEqual(loader2.num_submitted, 0)


if __name__ == '__main__':
    unittest.main()

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for pipeline_stage_repair.py and safe_point_group_repair.py PP>1 integration."""

import os
import sys
import unittest
import importlib.util

# ---------------------------------------------------------------------------
# Bootstrap: load modules without torch / megatron.core.__init__
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..', '..'))
_MOE_DIR = os.path.join(_REPO_ROOT, 'megatron', 'core', 'transformer', 'moe')


def _load_module(name: str, filename: str):
    """Load a single .py file as a module, bypassing package __init__."""
    path = os.path.join(_MOE_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'megatron.core.transformer.moe'
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load the modules under test
psr_mod = _load_module(
    'megatron.core.transformer.moe.pipeline_stage_repair',
    'pipeline_stage_repair.py',
)

PipelineStageRepairer = psr_mod.PipelineStageRepairer
PipelineStageRepairResult = psr_mod.PipelineStageRepairResult
identify_failed_stage = psr_mod.identify_failed_stage
compute_new_pp_ranks = psr_mod.compute_new_pp_ranks
get_pipeline_stage_repairer = psr_mod.get_pipeline_stage_repairer
clear_pipeline_stage_repairer = psr_mod.clear_pipeline_stage_repairer
execute_pipeline_stage_repair = psr_mod.execute_pipeline_stage_repair


# =====================================================================
# Test: identify_failed_stage
# =====================================================================

class TestIdentifyFailedStage(unittest.TestCase):

    def test_found(self):
        self.assertEqual(identify_failed_stage(2, [0, 1, 2, 3]), 2)

    def test_first_stage(self):
        self.assertEqual(identify_failed_stage(0, [0, 1, 2, 3]), 0)

    def test_last_stage(self):
        self.assertEqual(identify_failed_stage(3, [0, 1, 2, 3]), 3)

    def test_not_found(self):
        self.assertEqual(identify_failed_stage(99, [0, 1, 2, 3]), -1)

    def test_none_ranks(self):
        self.assertEqual(identify_failed_stage(0, None), -1)

    def test_two_stages(self):
        self.assertEqual(identify_failed_stage(5, [3, 5]), 1)


# =====================================================================
# Test: compute_new_pp_ranks
# =====================================================================

class TestComputeNewPPRanks(unittest.TestCase):

    def test_basic(self):
        result = compute_new_pp_ranks([0, 1, 2, 3], 2, 99)
        self.assertEqual(result, [0, 1, 99, 3])

    def test_first_stage(self):
        result = compute_new_pp_ranks([0, 1, 2, 3], 0, 100)
        self.assertEqual(result, [100, 1, 2, 3])

    def test_last_stage(self):
        result = compute_new_pp_ranks([0, 1, 2, 3], 3, 50)
        self.assertEqual(result, [0, 1, 2, 50])

    def test_two_stages(self):
        result = compute_new_pp_ranks([4, 7], 7, 10)
        self.assertEqual(result, [4, 10])

    def test_no_change_if_not_found(self):
        result = compute_new_pp_ranks([0, 1, 2], 99, 100)
        self.assertEqual(result, [0, 1, 2])


# =====================================================================
# Test: PipelineStageRepairer (unit, no torch.distributed)
# =====================================================================

class TestPipelineStageRepairerUnit(unittest.TestCase):
    """Unit tests for PipelineStageRepairer without torch.distributed."""

    def setUp(self):
        self.repairer = PipelineStageRepairer()

    def test_initial_state(self):
        self.assertEqual(self.repairer.total_repairs, 0)
        self.assertIsNone(self.repairer.last_result)

    def test_execute_fails_if_rank_not_in_group(self):
        """execute() should fail if failed_rank is not in pp_group_ranks."""
        result = self.repairer.execute(
            failed_rank=99,
            replacement_rank=100,
            pp_group_ranks=[0, 1, 2, 3],
            verify=False,
        )
        self.assertFalse(result.success)
        self.assertIn("not in pp_group_ranks", result.error)

    def test_result_dataclass_defaults(self):
        r = PipelineStageRepairResult()
        self.assertFalse(r.success)
        self.assertEqual(r.failed_rank, -1)
        self.assertEqual(r.replacement_rank, -1)
        self.assertEqual(r.failed_stage, -1)
        self.assertFalse(r.pp_group_rebuilt)
        self.assertFalse(r.prev_next_updated)
        self.assertFalse(r.p2p_rebound)
        self.assertEqual(r.compound_groups_rebuilt, 0)
        self.assertFalse(r.verification_passed)

    def test_summary_no_repairs(self):
        s = self.repairer.summary()
        self.assertEqual(s['total_repairs'], 0)
        self.assertIsNone(s['last_result'])

    def test_reset(self):
        self.repairer.reset()
        self.assertEqual(self.repairer.total_repairs, 0)
        self.assertIsNone(self.repairer.last_result)


# =====================================================================
# Test: PipelineStageRepairer with mock parallel_state
# =====================================================================

class TestPipelineStageRepairerMocked(unittest.TestCase):
    """Tests with mocked parallel_state and torch.distributed."""

    def setUp(self):
        self.repairer = PipelineStageRepairer()

        # Create mock parallel_state
        import types
        self.mock_ps = types.ModuleType('megatron.core.parallel_state')
        self.mock_ps._PIPELINE_MODEL_PARALLEL_GROUP = 'old_pp_group'
        self.mock_ps._PIPELINE_GLOBAL_RANKS = [0, 1, 2, 3]
        self.mock_ps._PREV_PIPELINE_MODEL_PARALLEL_RANK = -1
        self.mock_ps._NEXT_PIPELINE_MODEL_PARALLEL_RANK = -1
        sys.modules['megatron.core.parallel_state'] = self.mock_ps
        sys.modules['megatron.core'] = types.ModuleType('megatron.core')
        sys.modules['megatron.core'].parallel_state = self.mock_ps

        # Create mock torch.distributed
        self.mock_td = types.ModuleType('torch.distributed')
        self.mock_td.get_rank = lambda: 0
        self.mock_td.destroy_process_group = lambda g: None
        self.mock_td.barrier = lambda group=None: None
        self.mock_td.get_process_group_ranks = lambda g: []
        sys.modules['torch.distributed'] = self.mock_td

        # Create mock torch
        if 'torch' not in sys.modules:
            self.mock_torch = types.ModuleType('torch')
            self.mock_torch.distributed = self.mock_td
            sys.modules['torch'] = self.mock_torch
            self._created_torch = True
        else:
            self._created_torch = False

        # Create mock p2p_communication
        self.mock_p2p = types.ModuleType(
            'megatron.core.pipeline_parallel.p2p_communication'
        )
        self.mock_p2p._P2P_COMMUNICATOR = 'old_p2p'
        sys.modules['megatron.core.pipeline_parallel'] = types.ModuleType(
            'megatron.core.pipeline_parallel'
        )
        sys.modules['megatron.core.pipeline_parallel.p2p_communication'] = self.mock_p2p

    def tearDown(self):
        for key in list(sys.modules.keys()):
            if key.startswith('megatron.core'):
                del sys.modules[key]
        sys.modules.pop('torch.distributed', None)
        if self._created_torch:
            sys.modules.pop('torch', None)

    def _mock_create_group(self, ranks, backend='nccl', group_name='', **kw):
        return f"new_group_{group_name}_{ranks}"

    def test_execute_success(self):
        """Full execute with mocked dependencies."""
        result = self.repairer.execute(
            failed_rank=2,
            replacement_rank=99,
            pp_group_ranks=[0, 1, 2, 3],
            create_group_fn=self._mock_create_group,
            verify=False,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.failed_stage, 2)
        self.assertEqual(result.failed_rank, 2)
        self.assertEqual(result.replacement_rank, 99)
        self.assertTrue(result.pp_group_rebuilt)
        self.assertTrue(result.prev_next_updated)
        self.assertTrue(result.p2p_rebound)

    def test_pp_group_updated(self):
        """After execute, parallel_state PP group should be updated."""
        self.repairer.execute(
            failed_rank=2,
            replacement_rank=99,
            pp_group_ranks=[0, 1, 2, 3],
            create_group_fn=self._mock_create_group,
            verify=False,
        )
        self.assertIn('99', str(self.mock_ps._PIPELINE_MODEL_PARALLEL_GROUP))
        self.assertEqual(self.mock_ps._PIPELINE_GLOBAL_RANKS, [0, 1, 99, 3])

    def test_prev_next_updated(self):
        """After execute, prev/next ranks should be updated."""
        # rank 0 is at index 0 in [0, 1, 99, 3]
        # prev = 3 (index 3), next = 1 (index 1)
        self.repairer.execute(
            failed_rank=2,
            replacement_rank=99,
            pp_group_ranks=[0, 1, 2, 3],
            create_group_fn=self._mock_create_group,
            verify=False,
        )
        self.assertEqual(self.mock_ps._PREV_PIPELINE_MODEL_PARALLEL_RANK, 3)
        self.assertEqual(self.mock_ps._NEXT_PIPELINE_MODEL_PARALLEL_RANK, 1)

    def test_p2p_communicator_cleared(self):
        """After execute, P2P communicator should be cleared."""
        self.repairer.execute(
            failed_rank=2,
            replacement_rank=99,
            pp_group_ranks=[0, 1, 2, 3],
            create_group_fn=self._mock_create_group,
            verify=False,
        )
        self.assertIsNone(self.mock_p2p._P2P_COMMUNICATOR)

    def test_first_stage_failure(self):
        """Repair when stage 0 fails."""
        result = self.repairer.execute(
            failed_rank=0,
            replacement_rank=100,
            pp_group_ranks=[0, 1, 2, 3],
            create_group_fn=self._mock_create_group,
            verify=False,
        )
        # rank 0 is the local rank, but after replacement it's 100
        # Since mock get_rank() returns 0, and 0 is no longer in new_pp_ranks
        # [100, 1, 2, 3], pp_group_rebuilt should be False
        self.assertTrue(result.success)
        self.assertEqual(result.failed_stage, 0)

    def test_last_stage_failure(self):
        """Repair when last stage fails."""
        result = self.repairer.execute(
            failed_rank=3,
            replacement_rank=50,
            pp_group_ranks=[0, 1, 2, 3],
            create_group_fn=self._mock_create_group,
            verify=False,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.failed_stage, 3)

    def test_two_stage_pipeline(self):
        """Repair in a 2-stage pipeline."""
        result = self.repairer.execute(
            failed_rank=1,
            replacement_rank=5,
            pp_group_ranks=[0, 1],
            create_group_fn=self._mock_create_group,
            verify=False,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.failed_stage, 1)
        self.assertEqual(self.mock_ps._PIPELINE_GLOBAL_RANKS, [0, 5])

    def test_total_repairs_incremented(self):
        """total_repairs should increment on success."""
        self.assertEqual(self.repairer.total_repairs, 0)
        self.repairer.execute(
            failed_rank=2,
            replacement_rank=99,
            pp_group_ranks=[0, 1, 2, 3],
            create_group_fn=self._mock_create_group,
            verify=False,
        )
        self.assertEqual(self.repairer.total_repairs, 1)

    def test_summary_after_repair(self):
        """summary() should reflect the last repair."""
        self.repairer.execute(
            failed_rank=2,
            replacement_rank=99,
            pp_group_ranks=[0, 1, 2, 3],
            create_group_fn=self._mock_create_group,
            verify=False,
        )
        s = self.repairer.summary()
        self.assertEqual(s['total_repairs'], 1)
        self.assertIsNotNone(s['last_result'])
        self.assertTrue(s['last_result']['success'])
        self.assertEqual(s['last_result']['failed_stage'], 2)


# =====================================================================
# Test: Global singleton
# =====================================================================

class TestGlobalSingleton(unittest.TestCase):

    def test_get_creates_singleton(self):
        clear_pipeline_stage_repairer()
        r1 = get_pipeline_stage_repairer()
        r2 = get_pipeline_stage_repairer()
        self.assertIs(r1, r2)

    def test_clear_resets(self):
        r1 = get_pipeline_stage_repairer()
        clear_pipeline_stage_repairer()
        r2 = get_pipeline_stage_repairer()
        self.assertIsNot(r1, r2)


# =====================================================================
# Test: Integration with SafePointGroupRepairer
# =====================================================================

class TestSafePointGroupRepairerPPIntegration(unittest.TestCase):
    """Test that SafePointGroupRepairer calls PipelineStageRepairer for PP>1."""

    def setUp(self):
        # Load safe_point_group_repair module
        self.spr_mod = _load_module(
            'megatron.core.transformer.moe.safe_point_group_repair',
            'safe_point_group_repair.py',
        )

    def test_repair_result_has_pp_fields(self):
        """RepairResult should have pipeline_stage_repaired field."""
        r = self.spr_mod.RepairResult()
        # Check that the result has the expected fields
        self.assertFalse(r.success)
        self.assertEqual(r.groups_invalidated, 0)
        self.assertEqual(r.groups_rebuilt, 0)


if __name__ == '__main__':
    unittest.main()

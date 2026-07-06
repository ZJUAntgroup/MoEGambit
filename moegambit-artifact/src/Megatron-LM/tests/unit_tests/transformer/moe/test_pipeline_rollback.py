# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for MoEGambit Pipeline Rollback Coordinator (PP > 1)."""

import os
import sys
import importlib
import importlib.util
import unittest

# ---------------------------------------------------------------------------
# Bootstrap: load target modules without importing megatron.core.__init__
# (which requires torch).
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


pr_mod = _load_module(
    'megatron.core.transformer.moe.pipeline_rollback',
    'pipeline_rollback.py',
)

PipelineRollbackCoordinator = pr_mod.PipelineRollbackCoordinator
PipelineRollbackState = pr_mod.PipelineRollbackState
PipelineFailureInfo = pr_mod.PipelineFailureInfo
PipelineRollbackResult = pr_mod.PipelineRollbackResult
pipeline_safe_rollback = pr_mod.pipeline_safe_rollback
get_pipeline_rollback_coordinator = pr_mod.get_pipeline_rollback_coordinator
clear_pipeline_rollback_coordinator = pr_mod.clear_pipeline_rollback_coordinator


class TestPipelineRollbackState(unittest.TestCase):
    """Test PipelineRollbackState enum."""

    def test_states_exist(self):
        self.assertEqual(PipelineRollbackState.NORMAL.value, 0)
        self.assertEqual(PipelineRollbackState.FAILURE_DETECTED.value, 1)
        self.assertEqual(PipelineRollbackState.ROLLBACK_IN_PROGRESS.value, 2)
        self.assertEqual(PipelineRollbackState.AWAITING_REPLAY.value, 3)
        self.assertEqual(PipelineRollbackState.REPLAY_IN_PROGRESS.value, 4)


class TestPipelineFailureInfo(unittest.TestCase):
    """Test PipelineFailureInfo dataclass."""

    def test_defaults(self):
        info = PipelineFailureInfo()
        self.assertEqual(info.failed_stage, -1)
        self.assertEqual(info.failed_rank, -1)
        self.assertEqual(info.step, -1)
        self.assertEqual(info.num_microbatches_completed, 0)
        self.assertEqual(info.total_microbatches, 0)
        self.assertEqual(info.failure_phase, "")
        self.assertEqual(info.reason, "")
        self.assertTrue(info.nccl_healthy)

    def test_custom_values(self):
        info = PipelineFailureInfo(
            failed_stage=2,
            failed_rank=5,
            step=100,
            num_microbatches_completed=3,
            total_microbatches=8,
            failure_phase="forward",
            reason="NCCL timeout",
            nccl_healthy=False,
        )
        self.assertEqual(info.failed_stage, 2)
        self.assertEqual(info.failed_rank, 5)
        self.assertEqual(info.step, 100)
        self.assertEqual(info.num_microbatches_completed, 3)
        self.assertEqual(info.total_microbatches, 8)
        self.assertEqual(info.failure_phase, "forward")
        self.assertEqual(info.reason, "NCCL timeout")
        self.assertFalse(info.nccl_healthy)


class TestPipelineRollbackCoordinator(unittest.TestCase):
    """Test PipelineRollbackCoordinator."""

    def setUp(self):
        self.coord = PipelineRollbackCoordinator()

    def test_initial_state(self):
        self.assertEqual(self.coord.state, PipelineRollbackState.NORMAL)
        self.assertFalse(self.coord.is_in_rollback)
        self.assertFalse(self.coord.is_awaiting_replay)
        self.assertFalse(self.coord.is_in_replay)
        self.assertEqual(self.coord.pp_size, 1)
        self.assertEqual(self.coord.pp_rank, 0)
        self.assertEqual(self.coord.total_rollbacks, 0)
        self.assertEqual(self.coord.total_replays, 0)
        self.assertIsNone(self.coord.current_failure)

    def test_begin_iteration(self):
        self.coord.begin_iteration(step=10, pp_rank=2, pp_size=4, num_microbatches=8)
        self.assertEqual(self.coord.pp_rank, 2)
        self.assertEqual(self.coord.pp_size, 4)

    def test_on_pipeline_failure(self):
        self.coord.begin_iteration(step=50, pp_rank=1, pp_size=4, num_microbatches=8)
        info = self.coord.on_pipeline_failure(
            failed_stage=1,
            failed_rank=3,
            step=50,
            num_microbatches_completed=3,
            failure_phase="forward",
            reason="NCCL error",
            nccl_healthy=False,
        )
        self.assertEqual(self.coord.state, PipelineRollbackState.FAILURE_DETECTED)
        self.assertTrue(self.coord.is_in_rollback)
        self.assertIsNotNone(self.coord.current_failure)
        self.assertEqual(info.failed_stage, 1)
        self.assertEqual(info.failed_rank, 3)
        self.assertEqual(info.step, 50)
        self.assertEqual(info.failure_phase, "forward")
        self.assertFalse(info.nccl_healthy)

    def test_initiate_rollback_without_failure(self):
        """Cannot rollback without a failure."""
        result = self.coord.initiate_rollback()
        self.assertFalse(result.success)
        self.assertIn("Cannot initiate rollback", result.error)

    def test_full_rollback_cycle(self):
        """Test complete rollback → replay cycle."""
        self.coord.begin_iteration(step=100, pp_rank=0, pp_size=4, num_microbatches=8)

        # Failure
        self.coord.on_pipeline_failure(
            failed_stage=2,
            failed_rank=5,
            step=100,
            failure_phase="backward",
            reason="P2P timeout",
        )
        self.assertEqual(self.coord.state, PipelineRollbackState.FAILURE_DETECTED)

        # Rollback
        result = self.coord.initiate_rollback()
        self.assertTrue(result.success)
        self.assertEqual(self.coord.state, PipelineRollbackState.AWAITING_REPLAY)
        self.assertEqual(self.coord.total_rollbacks, 1)

        # Begin replay
        ok = self.coord.begin_replay()
        self.assertTrue(ok)
        self.assertEqual(self.coord.state, PipelineRollbackState.REPLAY_IN_PROGRESS)
        self.assertTrue(self.coord.is_in_replay)

        # Complete replay
        self.coord.complete_replay(success=True)
        self.assertEqual(self.coord.state, PipelineRollbackState.NORMAL)
        self.assertEqual(self.coord.total_replay_successes, 1)
        self.assertIsNone(self.coord.current_failure)

    def test_rollback_with_sync_fn(self):
        """Test rollback with custom sync function."""
        self.coord.begin_iteration(step=200, pp_rank=1, pp_size=4, num_microbatches=4)
        self.coord.on_pipeline_failure(
            failed_stage=0, failed_rank=0, step=200,
            failure_phase="forward", reason="rank exit",
        )

        sync_called = [False]
        def sync_fn():
            sync_called[0] = True
            return True

        result = self.coord.initiate_rollback(sync_fn=sync_fn)
        self.assertTrue(result.success)
        self.assertTrue(sync_called[0])
        self.assertTrue(result.stages_synchronized)

    def test_rollback_with_sync_fn_failure(self):
        """Test rollback when sync function fails (expected with dead NCCL)."""
        self.coord.begin_iteration(step=300, pp_rank=0, pp_size=2, num_microbatches=4)
        self.coord.on_pipeline_failure(
            failed_stage=1, failed_rank=1, step=300,
            failure_phase="forward", reason="NCCL dead",
            nccl_healthy=False,
        )

        def sync_fn():
            raise RuntimeError("NCCL is dead")

        result = self.coord.initiate_rollback(sync_fn=sync_fn)
        self.assertTrue(result.success)  # Rollback still succeeds
        self.assertFalse(result.stages_synchronized)
        self.assertFalse(result.nccl_recovered)

    def test_rollback_with_clear_grad_fn(self):
        """Test rollback with custom grad clear function."""
        self.coord.begin_iteration(step=400, pp_rank=0, pp_size=2, num_microbatches=4)
        self.coord.on_pipeline_failure(
            failed_stage=0, failed_rank=0, step=400,
            failure_phase="backward", reason="error",
        )

        cleared = [False]
        def clear_grad_fn():
            cleared[0] = True

        result = self.coord.initiate_rollback(clear_grad_fn=clear_grad_fn)
        self.assertTrue(result.success)
        self.assertTrue(cleared[0])
        self.assertTrue(result.grad_buffers_cleared)

    def test_failed_replay_goes_back_to_awaiting(self):
        """Test that a failed replay transitions back to AWAITING_REPLAY."""
        self.coord.begin_iteration(step=500, pp_rank=0, pp_size=2, num_microbatches=4)
        self.coord.on_pipeline_failure(
            failed_stage=0, failed_rank=0, step=500,
            failure_phase="forward", reason="error",
        )
        self.coord.initiate_rollback()

        self.coord.begin_replay()
        self.assertEqual(self.coord.state, PipelineRollbackState.REPLAY_IN_PROGRESS)

        self.coord.complete_replay(success=False)
        self.assertEqual(self.coord.state, PipelineRollbackState.AWAITING_REPLAY)
        self.assertEqual(self.coord.total_replay_successes, 0)

    def test_begin_iteration_transitions_awaiting_to_replay(self):
        """Test that begin_iteration auto-transitions AWAITING_REPLAY."""
        self.coord.begin_iteration(step=600, pp_rank=0, pp_size=2, num_microbatches=4)
        self.coord.on_pipeline_failure(
            failed_stage=0, failed_rank=0, step=600,
            failure_phase="forward", reason="error",
        )
        self.coord.initiate_rollback()
        self.assertEqual(self.coord.state, PipelineRollbackState.AWAITING_REPLAY)

        # Next iteration auto-transitions
        self.coord.begin_iteration(step=600, pp_rank=0, pp_size=2, num_microbatches=4)
        self.assertEqual(self.coord.state, PipelineRollbackState.REPLAY_IN_PROGRESS)

    def test_verify_pipeline_clean(self):
        """Test pipeline cleanup verification."""
        self.coord.begin_iteration(step=700, pp_rank=0, pp_size=2, num_microbatches=4)
        self.coord.on_pipeline_failure(
            failed_stage=0, failed_rank=0, step=700,
            failure_phase="forward", reason="error",
        )
        self.coord.initiate_rollback()

        issues = self.coord.verify_pipeline_clean()
        self.assertEqual(len(issues), 0)  # AWAITING_REPLAY is valid

    def test_verify_pipeline_clean_bad_state(self):
        """Test pipeline cleanup verification in bad state."""
        self.coord.begin_iteration(step=800, pp_rank=0, pp_size=2, num_microbatches=4)
        self.coord.on_pipeline_failure(
            failed_stage=0, failed_rank=0, step=800,
            failure_phase="forward", reason="error",
        )
        # FAILURE_DETECTED is not valid for replay
        issues = self.coord.verify_pipeline_clean()
        self.assertEqual(len(issues), 1)
        self.assertIn("Unexpected state", issues[0])

    def test_callbacks(self):
        """Test rollback and replay callbacks."""
        rollback_results = []
        replay_results = []

        self.coord.register_callbacks(
            on_rollback_fn=lambda r: rollback_results.append(r),
            on_replay_complete_fn=lambda s: replay_results.append(s),
        )

        self.coord.begin_iteration(step=900, pp_rank=0, pp_size=2, num_microbatches=4)
        self.coord.on_pipeline_failure(
            failed_stage=0, failed_rank=0, step=900,
            failure_phase="forward", reason="error",
        )
        self.coord.initiate_rollback()
        self.assertEqual(len(rollback_results), 1)
        self.assertTrue(rollback_results[0].success)

        self.coord.begin_replay()
        self.coord.complete_replay(success=True)
        self.assertEqual(len(replay_results), 1)
        self.assertTrue(replay_results[0])

    def test_failure_history(self):
        """Test failure history tracking."""
        for i in range(5):
            self.coord.begin_iteration(step=i, pp_rank=0, pp_size=2, num_microbatches=4)
            self.coord.on_pipeline_failure(
                failed_stage=0, failed_rank=0, step=i,
                failure_phase="forward", reason=f"error_{i}",
            )
            self.coord.initiate_rollback()
            self.coord.begin_replay()
            self.coord.complete_replay(success=True)

        self.assertEqual(self.coord.total_rollbacks, 5)
        self.assertEqual(self.coord.total_replay_successes, 5)

    def test_summary(self):
        """Test summary output."""
        self.coord.begin_iteration(step=1000, pp_rank=1, pp_size=4, num_microbatches=8)
        summary = self.coord.summary()
        self.assertEqual(summary['state'], 'NORMAL')
        self.assertEqual(summary['pp_rank'], 1)
        self.assertEqual(summary['pp_size'], 4)
        self.assertEqual(summary['current_step'], 1000)
        self.assertEqual(summary['total_rollbacks'], 0)
        self.assertIsNone(summary['current_failure'])

    def test_summary_with_failure(self):
        """Test summary with active failure."""
        self.coord.begin_iteration(step=1100, pp_rank=2, pp_size=4, num_microbatches=8)
        self.coord.on_pipeline_failure(
            failed_stage=3, failed_rank=7, step=1100,
            failure_phase="backward", reason="timeout",
        )
        summary = self.coord.summary()
        self.assertEqual(summary['state'], 'FAILURE_DETECTED')
        self.assertIsNotNone(summary['current_failure'])
        self.assertEqual(summary['current_failure']['stage'], 3)
        self.assertEqual(summary['current_failure']['rank'], 7)

    def test_reset(self):
        """Test full reset."""
        self.coord.begin_iteration(step=1200, pp_rank=1, pp_size=4, num_microbatches=8)
        self.coord.on_pipeline_failure(
            failed_stage=0, failed_rank=0, step=1200,
            failure_phase="forward", reason="error",
        )
        self.coord.initiate_rollback()

        self.coord.reset()
        self.assertEqual(self.coord.state, PipelineRollbackState.NORMAL)
        self.assertEqual(self.coord.total_rollbacks, 0)
        self.assertIsNone(self.coord.current_failure)
        self.assertEqual(self.coord.pp_rank, 0)
        self.assertEqual(self.coord.pp_size, 1)


class TestPipelineSafeRollback(unittest.TestCase):
    """Test the convenience pipeline_safe_rollback function."""

    def setUp(self):
        clear_pipeline_rollback_coordinator()
        # Pre-initialize the coordinator with iteration context
        coord = get_pipeline_rollback_coordinator()
        coord.begin_iteration(step=42, pp_rank=0, pp_size=4, num_microbatches=8)

    def tearDown(self):
        clear_pipeline_rollback_coordinator()

    def test_one_shot_rollback(self):
        result = pipeline_safe_rollback(
            failed_stage=1,
            failed_rank=3,
            step=42,
            reason="NCCL timeout",
            nccl_healthy=False,
        )
        self.assertTrue(result.success)
        self.assertFalse(result.nccl_recovered)

        coord = get_pipeline_rollback_coordinator()
        self.assertEqual(coord.state, PipelineRollbackState.AWAITING_REPLAY)
        self.assertEqual(coord.total_rollbacks, 1)


class TestGlobalSingleton(unittest.TestCase):
    """Test global singleton management."""

    def setUp(self):
        clear_pipeline_rollback_coordinator()

    def tearDown(self):
        clear_pipeline_rollback_coordinator()

    def test_get_creates_singleton(self):
        coord = get_pipeline_rollback_coordinator()
        self.assertIsNotNone(coord)
        coord2 = get_pipeline_rollback_coordinator()
        self.assertIs(coord, coord2)

    def test_clear_resets(self):
        coord = get_pipeline_rollback_coordinator()
        coord.begin_iteration(step=1, pp_rank=0, pp_size=2, num_microbatches=4)
        coord.on_pipeline_failure(
            failed_stage=0, failed_rank=0, step=1,
            failure_phase="forward", reason="error",
        )
        clear_pipeline_rollback_coordinator()
        coord2 = get_pipeline_rollback_coordinator()
        self.assertIsNot(coord, coord2)
        self.assertEqual(coord2.state, PipelineRollbackState.NORMAL)


class TestMultiStageScenario(unittest.TestCase):
    """Test multi-stage pipeline scenarios."""

    def test_pp4_forward_failure_stage2(self):
        """Simulate PP=4, failure at stage 2 during forward."""
        # Each stage has its own coordinator in practice,
        # but we test the logic for a single stage's perspective.
        coord = PipelineRollbackCoordinator()
        coord.begin_iteration(step=50, pp_rank=2, pp_size=4, num_microbatches=8)

        # Stage 2 detects failure during forward of microbatch 3
        info = coord.on_pipeline_failure(
            failed_stage=2,
            failed_rank=6,  # rank 6 is stage 2
            step=50,
            num_microbatches_completed=3,
            failure_phase="forward",
            reason="NCCL error on recv from stage 1",
            nccl_healthy=False,
        )
        self.assertEqual(info.num_microbatches_completed, 3)
        self.assertEqual(info.total_microbatches, 8)

        # Rollback
        result = coord.initiate_rollback()
        self.assertTrue(result.success)
        self.assertFalse(result.nccl_recovered)

        # Replay
        coord.begin_iteration(step=50, pp_rank=2, pp_size=4, num_microbatches=8)
        self.assertTrue(coord.is_in_replay)

        coord.complete_replay(success=True)
        self.assertEqual(coord.state, PipelineRollbackState.NORMAL)

    def test_pp4_backward_failure_stage0(self):
        """Simulate PP=4, failure at stage 0 during backward."""
        coord = PipelineRollbackCoordinator()
        coord.begin_iteration(step=75, pp_rank=0, pp_size=4, num_microbatches=8)

        # Stage 0 detects failure during backward (cooldown phase)
        coord.on_pipeline_failure(
            failed_stage=0,
            failed_rank=0,
            step=75,
            num_microbatches_completed=6,
            failure_phase="backward",
            reason="NCCL error on recv gradient from stage 1",
        )

        result = coord.initiate_rollback()
        self.assertTrue(result.success)

        # Verify pipeline is clean
        issues = coord.verify_pipeline_clean()
        self.assertEqual(len(issues), 0)

        # Replay
        coord.begin_replay()
        coord.complete_replay(success=True)
        self.assertEqual(coord.state, PipelineRollbackState.NORMAL)

    def test_pp2_multiple_failures(self):
        """Simulate PP=2 with multiple consecutive failures."""
        coord = PipelineRollbackCoordinator()

        for attempt in range(3):
            coord.begin_iteration(step=100, pp_rank=0, pp_size=2, num_microbatches=4)

            if attempt < 2:
                # Failure
                coord.on_pipeline_failure(
                    failed_stage=1, failed_rank=1, step=100,
                    failure_phase="forward",
                    reason=f"attempt_{attempt}",
                )
                coord.initiate_rollback()

                if attempt < 1:
                    # First replay fails
                    coord.begin_replay()
                    coord.complete_replay(success=False)
            else:
                # Third attempt succeeds (state was AWAITING_REPLAY,
                # begin_iteration auto-transitions to REPLAY_IN_PROGRESS)
                self.assertTrue(coord.is_in_replay)
                coord.complete_replay(success=True)

        self.assertEqual(coord.state, PipelineRollbackState.NORMAL)
        self.assertEqual(coord.total_rollbacks, 2)
        self.assertEqual(coord.total_replay_successes, 1)


if __name__ == '__main__':
    unittest.main()

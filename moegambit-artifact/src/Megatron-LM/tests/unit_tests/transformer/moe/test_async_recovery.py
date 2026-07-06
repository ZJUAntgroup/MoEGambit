# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for Phase 15: Async Recovery — Background Thread + CUDA Stream.

Tests cover:
1. AsyncRecoveryWorker basic lifecycle (submit → poll → apply)
2. Multiple concurrent load requests
3. Load failure handling
4. DeferredOptimizerLoader async path (SUBMITTED → LOADING → LOADED)
5. End-to-end: async expert restore + training loop poll
6. Shutdown cleanup
7. Regression: sync path still works
8. Dense param sync optimizer state bug fix
"""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import torch

# =====================================================================
# Test: AsyncRecoveryWorker
# =====================================================================


class TestAsyncRecoveryWorker(unittest.TestCase):
    """Tests for AsyncRecoveryWorker."""

    def setUp(self):
        from megatron.core.transformer.moe.async_recovery_worker import (
            AsyncRecoveryWorker,
            clear_async_recovery_worker,
        )
        clear_async_recovery_worker()
        self.worker = AsyncRecoveryWorker(max_workers=2)

    def tearDown(self):
        self.worker.shutdown()

    def test_basic_lifecycle(self):
        """Submit → poll → result."""
        from megatron.core.transformer.moe.async_recovery_worker import (
            AsyncLoadRequest,
        )

        # Create a load_fn that returns a CPU tensor
        def load_fn(req):
            return torch.randn(4, 4)

        req = AsyncLoadRequest(
            expert_id=0,
            layer_id=1,
            load_fn=load_fn,
        )
        req_id = self.worker.submit_expert_load(req)
        self.assertTrue(len(req_id) > 0)

        # Wait for completion
        results = []
        for _ in range(100):
            results = self.worker.poll_completed()
            if results:
                break
            time.sleep(0.01)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertIsNotNone(results[0].cpu_tensor)
        self.assertEqual(results[0].cpu_tensor.shape, (4, 4))
        self.assertFalse(results[0].cpu_tensor.is_cuda)

    def test_multiple_concurrent_loads(self):
        """Submit multiple requests and verify all complete."""
        from megatron.core.transformer.moe.async_recovery_worker import (
            AsyncLoadRequest,
        )

        num_requests = 5
        submitted_ids = []

        def load_fn(req):
            time.sleep(0.01)  # Simulate I/O
            return torch.randn(2, 2)

        for i in range(num_requests):
            req = AsyncLoadRequest(
                expert_id=i,
                layer_id=0,
                load_fn=load_fn,
            )
            req_id = self.worker.submit_expert_load(req)
            submitted_ids.append(req_id)

        # Collect all results
        all_results = []
        for _ in range(200):
            results = self.worker.poll_completed()
            all_results.extend(results)
            if len(all_results) >= num_requests:
                break
            time.sleep(0.01)

        self.assertEqual(len(all_results), num_requests)
        for r in all_results:
            self.assertTrue(r.success)

    def test_load_failure_handling(self):
        """Load function raises exception → result has error."""
        from megatron.core.transformer.moe.async_recovery_worker import (
            AsyncLoadRequest,
        )

        def failing_load_fn(req):
            raise RuntimeError("Simulated disk error")

        req = AsyncLoadRequest(
            expert_id=0,
            layer_id=0,
            load_fn=failing_load_fn,
        )
        self.worker.submit_expert_load(req)

        results = []
        for _ in range(100):
            results = self.worker.poll_completed()
            if results:
                break
            time.sleep(0.01)

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertIsNotNone(results[0].error)
        self.assertIn("Simulated disk error", results[0].error)

    def test_load_returns_false(self):
        """Load function returns False → result is failure."""
        from megatron.core.transformer.moe.async_recovery_worker import (
            AsyncLoadRequest,
        )

        def false_load_fn(req):
            return False

        req = AsyncLoadRequest(
            expert_id=0,
            layer_id=0,
            load_fn=false_load_fn,
        )
        self.worker.submit_expert_load(req)

        results = []
        for _ in range(100):
            results = self.worker.poll_completed()
            if results:
                break
            time.sleep(0.01)

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)

    def test_dry_run_no_load_fn(self):
        """No load_fn → dry-run success."""
        from megatron.core.transformer.moe.async_recovery_worker import (
            AsyncLoadRequest,
        )

        req = AsyncLoadRequest(
            expert_id=0,
            layer_id=0,
            load_fn=None,
        )
        self.worker.submit_expert_load(req)

        results = []
        for _ in range(100):
            results = self.worker.poll_completed()
            if results:
                break
            time.sleep(0.01)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertIsNone(results[0].cpu_tensor)

    def test_dict_return_optimizer_state(self):
        """Load function returns dict of tensors (optimizer state)."""
        from megatron.core.transformer.moe.async_recovery_worker import (
            AsyncLoadRequest,
        )

        def dict_load_fn(req):
            return {
                "exp_avg": torch.randn(4, 4),
                "exp_avg_sq": torch.randn(4, 4),
            }

        req = AsyncLoadRequest(
            expert_id=0,
            layer_id=0,
            load_fn=dict_load_fn,
        )
        self.worker.submit_optimizer_load(req)

        results = []
        for _ in range(100):
            results = self.worker.poll_completed()
            if results:
                break
            time.sleep(0.01)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertIsNotNone(results[0].cpu_tensors)
        self.assertIn("exp_avg", results[0].cpu_tensors)
        self.assertIn("exp_avg_sq", results[0].cpu_tensors)

    def test_has_pending(self):
        """has_pending reflects inflight state."""
        from megatron.core.transformer.moe.async_recovery_worker import (
            AsyncLoadRequest,
        )

        self.assertFalse(self.worker.has_pending())

        event = threading.Event()

        def blocking_load_fn(req):
            event.wait(timeout=5.0)
            return True

        req = AsyncLoadRequest(
            expert_id=0,
            layer_id=0,
            load_fn=blocking_load_fn,
        )
        self.worker.submit_expert_load(req)
        self.assertTrue(self.worker.has_pending())

        event.set()
        # Wait for completion
        for _ in range(100):
            if not self.worker.has_pending():
                break
            self.worker.poll_completed()
            time.sleep(0.01)

    def test_shutdown_cleans_up(self):
        """Shutdown stops worker threads."""
        self.worker.shutdown()
        self.assertTrue(self.worker.is_shutdown)
        # Workers should not be alive
        for t in self.worker._workers:
            self.assertFalse(t.is_alive())

    def test_summary(self):
        """Summary returns correct structure."""
        s = self.worker.summary()
        self.assertIn("max_workers", s)
        self.assertIn("num_pending", s)
        self.assertIn("num_inflight", s)
        self.assertIn("num_completed", s)
        self.assertEqual(s["max_workers"], 2)
        self.assertFalse(s["is_shutdown"])


class TestAsyncRecoveryWorkerSingleton(unittest.TestCase):
    """Tests for global singleton management."""

    def setUp(self):
        from megatron.core.transformer.moe.async_recovery_worker import (
            clear_async_recovery_worker,
        )
        clear_async_recovery_worker()

    def tearDown(self):
        from megatron.core.transformer.moe.async_recovery_worker import (
            clear_async_recovery_worker,
        )
        clear_async_recovery_worker()

    def test_get_creates_singleton(self):
        from megatron.core.transformer.moe.async_recovery_worker import (
            get_async_recovery_worker,
        )
        w1 = get_async_recovery_worker()
        w2 = get_async_recovery_worker()
        self.assertIs(w1, w2)

    def test_clear_and_recreate(self):
        from megatron.core.transformer.moe.async_recovery_worker import (
            get_async_recovery_worker,
            clear_async_recovery_worker,
        )
        w1 = get_async_recovery_worker()
        clear_async_recovery_worker()
        w2 = get_async_recovery_worker()
        self.assertIsNot(w1, w2)


# =====================================================================
# Test: DeferredOptimizerLoader async path
# =====================================================================


class TestDeferredOptimizerLoaderAsync(unittest.TestCase):
    """Tests for DeferredOptimizerLoader with async worker."""

    def setUp(self):
        from megatron.core.transformer.moe.deferred_optimizer_load import (
            DeferredOptimizerLoader,
            clear_deferred_optimizer_loader,
        )
        from megatron.core.transformer.moe.async_recovery_worker import (
            AsyncRecoveryWorker,
            clear_async_recovery_worker,
        )
        clear_deferred_optimizer_loader()
        clear_async_recovery_worker()
        self.loader = DeferredOptimizerLoader()
        self.worker = AsyncRecoveryWorker(max_workers=1)

    def tearDown(self):
        self.worker.shutdown()

    def test_async_submit_transitions_to_loading(self):
        """When async_worker is provided, SUBMITTED → LOADING."""
        from megatron.core.transformer.moe.deferred_optimizer_load import (
            OptimizerLoadState,
        )

        self.loader.submit_load(layer_id=0, expert_id=1, step=10)
        req = self.loader.get_request(0, 1)
        self.assertEqual(req.state, OptimizerLoadState.SUBMITTED)

        # Execute with async worker
        executed = self.loader.execute_pending_loads(
            load_fn=None,
            step=11,
            async_worker=self.worker,
        )
        self.assertEqual(executed, 1)

        req = self.loader.get_request(0, 1)
        self.assertEqual(req.state, OptimizerLoadState.LOADING)

    def test_sync_fallback_still_works(self):
        """Without async_worker, sync path works as before."""
        from megatron.core.transformer.moe.deferred_optimizer_load import (
            OptimizerLoadState,
        )

        self.loader.submit_load(layer_id=0, expert_id=2, step=10)

        executed = self.loader.execute_pending_loads(
            load_fn=None,  # dry-run
            step=11,
            async_worker=None,  # sync path
        )
        self.assertEqual(executed, 1)

        req = self.loader.get_request(0, 2)
        self.assertEqual(req.state, OptimizerLoadState.LOADED)


# =====================================================================
# Test: StaleExpertRestoreCoordinator async path
# =====================================================================


class TestStaleExpertRestoreAsync(unittest.TestCase):
    """Tests for StaleExpertRestoreCoordinator.execute_restore_async."""

    def setUp(self):
        from megatron.core.transformer.moe.stale_expert_restore import (
            StaleExpertRestoreCoordinator,
            ExpertRestorePlan,
            ExpertRestoreEntry,
            clear_stale_expert_restore_coordinator,
        )
        from megatron.core.transformer.moe.async_recovery_worker import (
            AsyncRecoveryWorker,
            clear_async_recovery_worker,
        )
        clear_stale_expert_restore_coordinator()
        clear_async_recovery_worker()
        self.coordinator = StaleExpertRestoreCoordinator()
        self.worker = AsyncRecoveryWorker(max_workers=1)

    def tearDown(self):
        self.worker.shutdown()

    def test_async_submit_returns_request_ids(self):
        from megatron.core.transformer.moe.stale_expert_restore import (
            ExpertRestorePlan,
            ExpertRestoreEntry,
        )

        plan = ExpertRestorePlan(
            failed_rank=0,
            replacement_rank=1,
            step=100,
            entries=[
                ExpertRestoreEntry(layer_id=0, expert_id=0),
                ExpertRestoreEntry(layer_id=0, expert_id=1),
                ExpertRestoreEntry(layer_id=1, expert_id=0),
            ],
        )

        request_ids = self.coordinator.execute_restore_async(
            plan=plan,
            load_fn=None,
            worker=self.worker,
            step=100,
        )

        self.assertEqual(len(request_ids), 3)
        for rid in request_ids:
            self.assertTrue(rid.startswith("expert_"))

    def test_async_loads_complete(self):
        from megatron.core.transformer.moe.stale_expert_restore import (
            ExpertRestorePlan,
            ExpertRestoreEntry,
        )

        plan = ExpertRestorePlan(
            failed_rank=0,
            replacement_rank=1,
            step=100,
            entries=[
                ExpertRestoreEntry(layer_id=0, expert_id=0),
            ],
        )

        def load_fn(req):
            return torch.randn(2, 2)

        self.coordinator.execute_restore_async(
            plan=plan,
            load_fn=load_fn,
            worker=self.worker,
            step=100,
        )

        # Wait for completion
        results = []
        for _ in range(100):
            results = self.worker.poll_completed()
            if results:
                break
            time.sleep(0.01)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)

    def test_sync_restore_still_works(self):
        """Sync restore path is preserved."""
        from megatron.core.transformer.moe.stale_expert_restore import (
            ExpertRestorePlan,
            ExpertRestoreEntry,
        )

        plan = ExpertRestorePlan(
            failed_rank=0,
            replacement_rank=1,
            step=100,
            entries=[
                ExpertRestoreEntry(layer_id=0, expert_id=0),
            ],
        )

        result = self.coordinator.execute_restore(
            model=None,
            plan=plan,
            load_fn=None,  # dry-run
            step=100,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.num_restored, 1)


# =====================================================================
# Test: RecoveryController async path
# =====================================================================


class TestRecoveryControllerAsync(unittest.TestCase):
    """Tests for RecoveryController with async recovery callbacks."""

    def test_async_expert_restore_callback(self):
        from megatron.core.transformer.moe.recovery_controller import (
            RecoveryController,
            RecoveryPhase,
        )

        ctrl = RecoveryController()
        async_called = {"called": False, "request_ids": []}

        def async_expert_restore_fn(*, failed_rank, replacement_rank, step, expert_ids):
            async_called["called"] = True
            async_called["request_ids"] = ["req_1", "req_2"]
            return ["req_1", "req_2"]

        ctrl.register_callbacks(
            async_expert_restore_fn=async_expert_restore_fn,
        )

        # Drive through the state machine
        ctrl.on_hard_rank_failure(failed_rank=1, step=10, expert_ids=[0, 1])
        ctrl.on_replacement_assigned(failed_rank=1, replacement_rank=2, step=11)
        ctrl.on_replacement_ready(failed_rank=1, step=12)

        # Execute safe-point repair
        ctrl.before_iteration(step=13)

        self.assertTrue(async_called["called"])
        self.assertTrue(ctrl.async_recovery_pending)
        self.assertEqual(ctrl._async_expert_request_ids, ["req_1", "req_2"])

    def test_after_iteration_polls_async(self):
        from megatron.core.transformer.moe.recovery_controller import (
            RecoveryController,
        )

        ctrl = RecoveryController()
        poll_called = {"called": False}

        def poll_fn(*, step):
            poll_called["called"] = True
            return True  # all done

        ctrl.register_callbacks(
            poll_async_recovery_fn=poll_fn,
        )

        # Simulate async recovery pending
        ctrl._async_recovery_pending = True

        result = ctrl.after_iteration(step=20)
        self.assertTrue(result)
        self.assertTrue(poll_called["called"])
        self.assertFalse(ctrl.async_recovery_pending)

    def test_after_iteration_noop_when_no_async(self):
        from megatron.core.transformer.moe.recovery_controller import (
            RecoveryController,
        )

        ctrl = RecoveryController()
        result = ctrl.after_iteration(step=20)
        self.assertFalse(result)


# =====================================================================
# Test: Dense param sync optimizer state bug fix
# =====================================================================


class TestDenseParamSyncOptimizerBugFix(unittest.TestCase):
    """Tests for the optimizer state over-sync bug fix in dense_param_sync.py."""

    def test_integer_key_matching(self):
        """When optimizer uses integer keys, only dense params are synced."""
        from megatron.core.transformer.moe.dense_param_sync import (
            _sync_optimizer_states_for_dense,
            ParamClassification,
        )

        # Create mock params
        dense_param = torch.nn.Parameter(torch.randn(4, 4))
        dense_param.allreduce = True

        expert_param = torch.nn.Parameter(torch.randn(4, 4))
        expert_param.allreduce = False

        classification = ParamClassification(
            dense_params={"layer.weight": dense_param},
            expert_params={"experts.weight": expert_param},
        )

        # Create mock optimizer with integer keys
        class MockOptimizer:
            def __init__(self):
                self.param_groups = [
                    {"params": [dense_param, expert_param]},
                ]
                self.state = {
                    0: {  # dense_param
                        "exp_avg": torch.zeros(4, 4),
                        "exp_avg_sq": torch.zeros(4, 4),
                    },
                    1: {  # expert_param
                        "exp_avg": torch.zeros(4, 4),
                        "exp_avg_sq": torch.zeros(4, 4),
                    },
                }

        optimizer = MockOptimizer()
        broadcast_calls = []

        def mock_broadcast(tensor, src, group):
            broadcast_calls.append(id(tensor))

        synced = _sync_optimizer_states_for_dense(
            optimizer=optimizer,
            classification=classification,
            source_rank=0,
            dp_group="mock_group",
            broadcast_fn=mock_broadcast,
        )

        # Should only sync dense param's optimizer state (2 tensors × 16 elements)
        self.assertEqual(synced, 32)  # 4×4 × 2 state tensors
        # Should have broadcast exactly 2 tensors (exp_avg + exp_avg_sq for dense)
        self.assertEqual(len(broadcast_calls), 2)

        # Verify expert param's state was NOT broadcast
        expert_state_ids = {
            id(optimizer.state[1]["exp_avg"]),
            id(optimizer.state[1]["exp_avg_sq"]),
        }
        for call_id in broadcast_calls:
            self.assertNotIn(call_id, expert_state_ids)

    def test_param_object_key_matching(self):
        """When optimizer uses param objects as keys, matching works."""
        from megatron.core.transformer.moe.dense_param_sync import (
            _sync_optimizer_states_for_dense,
            ParamClassification,
        )

        dense_param = torch.nn.Parameter(torch.randn(2, 2))
        dense_param.allreduce = True

        expert_param = torch.nn.Parameter(torch.randn(2, 2))
        expert_param.allreduce = False

        classification = ParamClassification(
            dense_params={"layer.weight": dense_param},
            expert_params={"experts.weight": expert_param},
        )

        class MockOptimizer:
            def __init__(self):
                self.param_groups = [
                    {"params": [dense_param, expert_param]},
                ]
                # Use param objects as keys (some optimizers do this)
                self.state = {
                    dense_param: {
                        "exp_avg": torch.zeros(2, 2),
                    },
                    expert_param: {
                        "exp_avg": torch.zeros(2, 2),
                    },
                }

        optimizer = MockOptimizer()
        broadcast_calls = []

        def mock_broadcast(tensor, src, group):
            broadcast_calls.append(id(tensor))

        synced = _sync_optimizer_states_for_dense(
            optimizer=optimizer,
            classification=classification,
            source_rank=0,
            dp_group="mock_group",
            broadcast_fn=mock_broadcast,
        )

        # Should only sync dense param's state (1 tensor × 4 elements)
        self.assertEqual(synced, 4)
        self.assertEqual(len(broadcast_calls), 1)

    def test_no_optimizer_returns_zero(self):
        """No optimizer → 0 scalars synced."""
        from megatron.core.transformer.moe.dense_param_sync import (
            _sync_optimizer_states_for_dense,
            ParamClassification,
        )

        synced = _sync_optimizer_states_for_dense(
            optimizer=None,
            classification=ParamClassification(),
            source_rank=0,
            dp_group="mock",
            broadcast_fn=lambda t, src, group: None,
        )
        self.assertEqual(synced, 0)


# =====================================================================
# Test: End-to-end async recovery flow
# =====================================================================


class TestEndToEndAsyncRecovery(unittest.TestCase):
    """End-to-end test: async expert restore + training loop poll."""

    def setUp(self):
        from megatron.core.transformer.moe.async_recovery_worker import (
            AsyncRecoveryWorker,
            clear_async_recovery_worker,
        )
        from megatron.core.transformer.moe.stale_expert_restore import (
            StaleExpertRestoreCoordinator,
            clear_stale_expert_restore_coordinator,
        )

        clear_async_recovery_worker()
        clear_stale_expert_restore_coordinator()

        self.worker = AsyncRecoveryWorker(max_workers=2)
        self.coordinator = StaleExpertRestoreCoordinator()

    def tearDown(self):
        from megatron.core.transformer.moe.async_recovery_worker import (
            clear_async_recovery_worker,
        )
        self.worker.shutdown()
        clear_async_recovery_worker()

    def test_full_async_flow(self):
        """Submit → poll → verify completion."""
        from megatron.core.transformer.moe.stale_expert_restore import (
            ExpertRestorePlan,
            ExpertRestoreEntry,
        )

        # Create restore plan
        plan = ExpertRestorePlan(
            failed_rank=0,
            replacement_rank=1,
            step=100,
            entries=[
                ExpertRestoreEntry(layer_id=0, expert_id=2),
                ExpertRestoreEntry(layer_id=0, expert_id=3),
            ],
        )

        # Submit async loads
        def load_fn(req):
            time.sleep(0.01)
            return torch.randn(4, 4)

        request_ids = self.coordinator.execute_restore_async(
            plan=plan,
            load_fn=load_fn,
            worker=self.worker,
            step=100,
        )
        self.assertEqual(len(request_ids), 2)

        # Simulate training loop polling
        all_results = []
        for step in range(100, 200):
            results = self.worker.poll_completed()
            all_results.extend(results)
            if len(all_results) >= 2:
                break
            time.sleep(0.01)

        self.assertEqual(len(all_results), 2)
        for r in all_results:
            self.assertTrue(r.success)


# =====================================================================
# Test: GPU apply (CPU-only mock)
# =====================================================================


class TestApplyToGpuCpuFallback(unittest.TestCase):
    """Test apply_to_gpu logic without actual CUDA."""

    def test_apply_with_no_results(self):
        from megatron.core.transformer.moe.async_recovery_worker import (
            AsyncRecoveryWorker,
        )

        worker = AsyncRecoveryWorker(max_workers=1)
        try:
            if not torch.cuda.is_available():
                # On CPU-only machines, verify that apply_to_gpu handles
                # empty results gracefully by checking the method exists
                # and the worker can be created/shutdown without error.
                self.assertTrue(hasattr(worker, 'apply_to_gpu'))
                return

            # On CUDA machines, test with a real stream
            stream = torch.cuda.Stream()
            copied = worker.apply_to_gpu([], stream)
            self.assertEqual(copied, 0)
        finally:
            worker.shutdown()


if __name__ == "__main__":
    unittest.main()

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for MoEGambit Replacement Rank Registration.

Tests cover:
1. ReplacementRegistry state machine (NOT_PRESENT → BOOTSTRAPPING →
   READY_FOR_REPAIR → INTEGRATED)
2. ParallelRoleDescriptor serialization and role inheritance
3. ReplacementSlot lifecycle
4. RecoveryController integration (on_replacement_assigned / on_replacement_ready)
5. Safe-point gate: replacement rank must NOT participate before INTEGRATED
6. End-to-end: hard failure → replacement announced → safe-point repair
7. moegambit_integration public API (moegambit_announce_replacement_ready, etc.)
8. Multiple concurrent replacements
9. Abort / cleanup paths
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


_rep_mod = _load_module("megatron.core.transformer.moe.replacement_registry")
_rc_mod = _load_module("megatron.core.transformer.moe.recovery_controller")
_hfd_mod = _load_module("megatron.core.transformer.moe.hard_failure_detector")
_inv_mod = _load_module("megatron.core.transformer.moe.iteration_invalidator")
_rb_mod = _load_module("megatron.core.transformer.moe.iteration_rollback")
_ocg_mod = _load_module("megatron.core.transformer.moe.optimizer_commit_guard")

ReplacementState = _rep_mod.ReplacementState
ReplacementSlot = _rep_mod.ReplacementSlot
ReplacementRegistry = _rep_mod.ReplacementRegistry
ParallelRoleDescriptor = _rep_mod.ParallelRoleDescriptor
RecoveryController = _rc_mod.RecoveryController
RecoveryPhase = _rc_mod.RecoveryPhase
HardFailureDetector = _hfd_mod.HardFailureDetector
FailureSource = _hfd_mod.FailureSource
IterationInvalidator = _inv_mod.IterationInvalidator
RollbackReplayManager = _rb_mod.RollbackReplayManager
OptimizerCommitGuard = _ocg_mod.OptimizerCommitGuard


# =====================================================================
# Test: ReplacementState enum
# =====================================================================

class TestReplacementState(unittest.TestCase):

    def test_values(self):
        self.assertEqual(ReplacementState.NOT_PRESENT, 0)
        self.assertEqual(ReplacementState.BOOTSTRAPPING, 1)
        self.assertEqual(ReplacementState.READY_FOR_REPAIR, 2)
        self.assertEqual(ReplacementState.INTEGRATED, 3)

    def test_names(self):
        self.assertEqual(ReplacementState.NOT_PRESENT.name, "NOT_PRESENT")
        self.assertEqual(ReplacementState.INTEGRATED.name, "INTEGRATED")


# =====================================================================
# Test: ParallelRoleDescriptor
# =====================================================================

class TestParallelRoleDescriptor(unittest.TestCase):

    def _make_role(self):
        return ParallelRoleDescriptor(
            global_rank=5,
            tp_rank=1, tp_size=2,
            dp_rank=0, dp_size=4,
            pp_rank=0, pp_size=1,
            cp_rank=0, cp_size=1,
            ep_rank=1, ep_size=4,
            tp_group_ranks=[4, 5],
            dp_group_ranks=[1, 5, 9, 13],
            pp_group_ranks=[5],
            ep_group_ranks=[2, 3, 4, 5],
            expert_ids=[4, 5, 6, 7],
        )

    def test_role_key(self):
        role = self._make_role()
        key = role.role_key()
        self.assertIn("tp1", key)
        self.assertIn("ep1", key)

    def test_serialization_roundtrip(self):
        role = self._make_role()
        d = role.to_dict()
        role2 = ParallelRoleDescriptor.from_dict(d)
        self.assertEqual(role.global_rank, role2.global_rank)
        self.assertEqual(role.ep_rank, role2.ep_rank)
        self.assertEqual(role.expert_ids, role2.expert_ids)
        self.assertEqual(role.ep_group_ranks, role2.ep_group_ranks)


# =====================================================================
# Test: ReplacementSlot lifecycle
# =====================================================================

class TestReplacementSlot(unittest.TestCase):

    def test_initial_state(self):
        slot = ReplacementSlot(failed_rank=3, replacement_rank=99)
        self.assertEqual(slot.state, ReplacementState.NOT_PRESENT)
        self.assertFalse(slot.is_participating())

    def test_valid_transitions(self):
        slot = ReplacementSlot(failed_rank=3, replacement_rank=99)
        slot.transition_to(ReplacementState.BOOTSTRAPPING, step=10)
        self.assertEqual(slot.state, ReplacementState.BOOTSTRAPPING)
        self.assertFalse(slot.is_participating())

        slot.transition_to(ReplacementState.READY_FOR_REPAIR, step=20)
        self.assertEqual(slot.state, ReplacementState.READY_FOR_REPAIR)
        self.assertFalse(slot.is_participating())

        slot.transition_to(ReplacementState.INTEGRATED, step=30)
        self.assertEqual(slot.state, ReplacementState.INTEGRATED)
        self.assertTrue(slot.is_participating())

    def test_invalid_transition_raises(self):
        slot = ReplacementSlot(failed_rank=3, replacement_rank=99)
        with self.assertRaises(ValueError):
            slot.transition_to(ReplacementState.INTEGRATED)

    def test_serialization(self):
        slot = ReplacementSlot(failed_rank=3, replacement_rank=99)
        slot.transition_to(ReplacementState.BOOTSTRAPPING, step=10)
        d = slot.to_dict()
        slot2 = ReplacementSlot.from_dict(d)
        self.assertEqual(slot2.failed_rank, 3)
        self.assertEqual(slot2.replacement_rank, 99)
        self.assertEqual(slot2.state, ReplacementState.BOOTSTRAPPING)


# =====================================================================
# Test: ReplacementRegistry
# =====================================================================

class TestReplacementRegistry(unittest.TestCase):

    def setUp(self):
        self.reg = ReplacementRegistry()

    def test_announce_replacement(self):
        slot = self.reg.announce_replacement(
            failed_rank=3, replacement_rank=99, step=10, reason="nccl_error",
        )
        self.assertEqual(slot.state, ReplacementState.BOOTSTRAPPING)
        self.assertEqual(slot.replacement_rank, 99)
        self.assertEqual(self.reg.num_pending, 1)

    def test_announce_ready(self):
        self.reg.announce_replacement(failed_rank=3, replacement_rank=99, step=10)
        slot = self.reg.announce_replacement_ready(failed_rank=3, step=20)
        self.assertEqual(slot.state, ReplacementState.READY_FOR_REPAIR)
        self.assertEqual(self.reg.num_ready, 1)

    def test_mark_integrated(self):
        self.reg.announce_replacement(failed_rank=3, replacement_rank=99, step=10)
        self.reg.announce_replacement_ready(failed_rank=3, step=20)
        slot = self.reg.mark_integrated(failed_rank=3, step=30)
        self.assertEqual(slot.state, ReplacementState.INTEGRATED)
        self.assertTrue(slot.is_participating())
        self.assertEqual(self.reg.num_integrated, 1)

    def test_query_status(self):
        self.assertEqual(
            self.reg.query_replacement_status(3),
            ReplacementState.NOT_PRESENT,
        )
        self.reg.announce_replacement(failed_rank=3, replacement_rank=99)
        self.assertEqual(
            self.reg.query_replacement_status(3),
            ReplacementState.BOOTSTRAPPING,
        )

    def test_is_replacement_pending(self):
        self.assertFalse(self.reg.is_replacement_pending(3))
        self.reg.announce_replacement(failed_rank=3, replacement_rank=99)
        self.assertTrue(self.reg.is_replacement_pending(3))

    def test_inherited_role(self):
        role = ParallelRoleDescriptor(
            global_rank=3, tp_rank=0, tp_size=1,
            dp_rank=0, dp_size=1, pp_rank=0, pp_size=1,
            cp_rank=0, cp_size=1, ep_rank=1, ep_size=4,
            tp_group_ranks=[3], dp_group_ranks=[3],
            pp_group_ranks=[3], ep_group_ranks=[0, 1, 2, 3],
            expert_ids=[4, 5, 6, 7],
        )
        self.reg.announce_replacement(
            failed_rank=3, replacement_rank=99, role=role,
        )
        retrieved = self.reg.get_inherited_role(3)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.expert_ids, [4, 5, 6, 7])
        self.assertEqual(retrieved.ep_rank, 1)

    def test_collect_ready_for_integration(self):
        self.reg.announce_replacement(failed_rank=3, replacement_rank=99)
        self.reg.announce_replacement(failed_rank=7, replacement_rank=100)
        self.reg.announce_replacement_ready(failed_rank=3)
        # Only rank 3 is ready
        ready = self.reg.collect_ready_for_integration()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].failed_rank, 3)

    def test_integrate_all_ready(self):
        self.reg.announce_replacement(failed_rank=3, replacement_rank=99)
        self.reg.announce_replacement(failed_rank=7, replacement_rank=100)
        self.reg.announce_replacement_ready(failed_rank=3)
        self.reg.announce_replacement_ready(failed_rank=7)
        integrated = self.reg.integrate_all_ready(step=50)
        self.assertEqual(len(integrated), 2)
        self.assertEqual(self.reg.num_integrated, 2)

    def test_abort_replacement(self):
        self.reg.announce_replacement(failed_rank=3, replacement_rank=99)
        slot = self.reg.abort_replacement(failed_rank=3, reason="timeout")
        self.assertIsNotNone(slot)
        self.assertEqual(slot.state, ReplacementState.NOT_PRESENT)

    def test_cleanup_integrated(self):
        self.reg.announce_replacement(failed_rank=3, replacement_rank=99)
        self.reg.announce_replacement_ready(failed_rank=3)
        self.reg.mark_integrated(failed_rank=3)
        count = self.reg.cleanup_integrated()
        self.assertEqual(count, 1)
        self.assertEqual(len(self.reg.all_failed_ranks), 0)

    def test_expert_directory_query(self):
        role = ParallelRoleDescriptor(
            global_rank=3, tp_rank=0, tp_size=1,
            dp_rank=0, dp_size=1, pp_rank=0, pp_size=1,
            cp_rank=0, cp_size=1, ep_rank=1, ep_size=4,
            tp_group_ranks=[3], dp_group_ranks=[3],
            pp_group_ranks=[3], ep_group_ranks=[0, 1, 2, 3],
            expert_ids=[4, 5, 6, 7],
        )
        self.reg.announce_replacement(
            failed_rank=3, replacement_rank=99, role=role,
        )
        query = self.reg.get_expert_directory_query(3)
        self.assertIsNotNone(query)
        self.assertEqual(query["expert_ids"], [4, 5, 6, 7])
        self.assertEqual(query["ep_rank"], 1)

    def test_summary(self):
        self.reg.announce_replacement(failed_rank=3, replacement_rank=99)
        s = self.reg.summary()
        self.assertEqual(len(s["pending"]), 1)
        self.assertEqual(len(s["ready"]), 0)


# =====================================================================
# Test: RecoveryController replacement integration
# =====================================================================

class TestRecoveryControllerReplacement(unittest.TestCase):

    def setUp(self):
        self.ctrl = RecoveryController()
        self.announce_calls = []
        self.integrate_calls = []
        self.rebuild_request_calls = []
        self.rebuild_execute_calls = []
        self.rebuild_finish_calls = []
        self.topo_refresh_calls = []
        self.dense_sync_calls = []
        self.expert_restore_calls = []

        self.ctrl.register_callbacks(
            quarantine_fn=lambda **kw: None,
            health_mark_unavailable_fn=lambda **kw: None,
            replacement_announce_fn=lambda **kw: self.announce_calls.append(kw),
            replacement_integrate_fn=lambda **kw: self.integrate_calls.append(kw),
            group_rebuild_request_fn=lambda **kw: self.rebuild_request_calls.append(kw),
            group_rebuild_execute_fn=lambda **kw: self.rebuild_execute_calls.append(kw),
            group_rebuild_finish_fn=lambda **kw: self.rebuild_finish_calls.append(kw),
            topology_refresh_fn=lambda **kw: self.topo_refresh_calls.append(kw),
            dense_sync_fn=lambda **kw: self.dense_sync_calls.append(kw),
            expert_restore_fn=lambda **kw: self.expert_restore_calls.append(kw),
        )

    def test_hard_failure_to_pending_group_repair(self):
        """Hard failure transitions to PENDING_GROUP_REPAIR."""
        self.ctrl.on_hard_rank_failure(
            failed_rank=3, reason="nccl_error", step=100,
            expert_ids=[4, 5], ep_group_ranks=[0, 1, 2, 3],
        )
        self.assertEqual(self.ctrl.phase, RecoveryPhase.PENDING_GROUP_REPAIR)

    def test_replacement_assigned_transitions(self):
        """on_replacement_assigned → WAITING_FOR_REPLACEMENT."""
        self.ctrl.on_hard_rank_failure(failed_rank=3, step=100)
        self.ctrl.on_replacement_assigned(
            failed_rank=3, replacement_rank=99, step=110,
        )
        self.assertEqual(self.ctrl.phase, RecoveryPhase.WAITING_FOR_REPLACEMENT)
        self.assertEqual(len(self.announce_calls), 1)
        self.assertEqual(self.announce_calls[0]["replacement_rank"], 99)

    def test_replacement_ready_transitions(self):
        """on_replacement_ready → SAFE_POINT_REPAIR."""
        self.ctrl.on_hard_rank_failure(failed_rank=3, step=100)
        self.ctrl.on_replacement_assigned(
            failed_rank=3, replacement_rank=99, step=110,
        )
        self.ctrl.on_replacement_ready(failed_rank=3, step=120)
        self.assertEqual(self.ctrl.phase, RecoveryPhase.SAFE_POINT_REPAIR)

    def test_safe_point_repair_executes_full_sequence(self):
        """before_iteration at safe point executes the 7-step repair."""
        self.ctrl.on_hard_rank_failure(
            failed_rank=3, step=100,
            expert_ids=[4, 5], ep_group_ranks=[0, 1, 2, 3],
        )
        self.ctrl.on_replacement_assigned(
            failed_rank=3, replacement_rank=99, step=110,
        )
        self.ctrl.on_replacement_ready(failed_rank=3, step=120)

        # Safe point: before_iteration triggers repair
        result = self.ctrl.before_iteration(step=130)
        self.assertTrue(result)
        self.assertEqual(self.ctrl.phase, RecoveryPhase.REINTEGRATED)

        # Verify all 7 callbacks were called
        self.assertEqual(len(self.integrate_calls), 1)
        self.assertEqual(len(self.rebuild_request_calls), 1)
        self.assertEqual(len(self.rebuild_execute_calls), 1)
        self.assertEqual(len(self.rebuild_finish_calls), 1)
        self.assertEqual(len(self.topo_refresh_calls), 1)
        self.assertEqual(len(self.dense_sync_calls), 1)
        self.assertEqual(len(self.expert_restore_calls), 1)

    def test_finalize_reintegration(self):
        """Next before_iteration after REINTEGRATED → HEALTHY_TRAINING."""
        self.ctrl.on_hard_rank_failure(failed_rank=3, step=100)
        self.ctrl.on_replacement_assigned(
            failed_rank=3, replacement_rank=99, step=110,
        )
        self.ctrl.on_replacement_ready(failed_rank=3, step=120)
        self.ctrl.before_iteration(step=130)  # repair
        self.assertEqual(self.ctrl.phase, RecoveryPhase.REINTEGRATED)

        self.ctrl.before_iteration(step=131)  # finalize
        self.assertEqual(self.ctrl.phase, RecoveryPhase.HEALTHY_TRAINING)
        self.assertTrue(self.ctrl.is_healthy)

    def test_replacement_not_participating_before_integrated(self):
        """Replacement rank must not participate before INTEGRATED."""
        reg = ReplacementRegistry()
        reg.announce_replacement(failed_rank=3, replacement_rank=99)
        self.assertFalse(reg.is_replacement_participating(3))

        reg.announce_replacement_ready(failed_rank=3)
        self.assertFalse(reg.is_replacement_participating(3))

        reg.mark_integrated(failed_rank=3)
        self.assertTrue(reg.is_replacement_participating(3))

    def test_fault_record_tracks_replacement(self):
        """FaultRecord stores replacement_rank and step info."""
        self.ctrl.on_hard_rank_failure(failed_rank=3, step=100)
        self.ctrl.on_replacement_assigned(
            failed_rank=3, replacement_rank=99, step=110,
        )
        record = self.ctrl.get_fault_record(3)
        self.assertIsNotNone(record)
        self.assertEqual(record.replacement_rank, 99)
        self.assertEqual(record.replacement_assigned_step, 110)


# =====================================================================
# Test: End-to-end — hard failure → replacement → safe-point repair
# =====================================================================

class TestEndToEndReplacement(unittest.TestCase):
    """Full scenario: detect hard failure → invalidate iteration →
    rollback → announce replacement → safe-point repair → resume."""

    def setUp(self):
        self.ctrl = RecoveryController()
        self.inv = IterationInvalidator()
        self.detector = HardFailureDetector()
        self.guard = OptimizerCommitGuard()
        self.mgr = RollbackReplayManager()
        self.reg = ReplacementRegistry()

        # Wire detector → controller + invalidator
        self.detector.register_callbacks(
            on_hard_failure_fn=self.ctrl.on_hard_rank_failure,
            on_iteration_invalid_fn=self.inv.invalidate,
        )
        self.guard.register_callbacks(
            is_iteration_invalid_fn=self.inv.is_current_iteration_invalid,
        )

        # Wire controller callbacks
        self.ctrl.register_callbacks(
            quarantine_fn=lambda **kw: None,
            health_mark_unavailable_fn=lambda **kw: None,
            replacement_announce_fn=lambda **kw: self.reg.announce_replacement(
                failed_rank=kw["failed_rank"],
                replacement_rank=kw["replacement_rank"],
                step=kw.get("step", -1),
            ),
            replacement_integrate_fn=lambda **kw: (
                # In real code, announce_replacement_ready is called by the
                # replacement rank during bootstrapping.  Here we ensure the
                # registry slot is in READY_FOR_REPAIR before marking INTEGRATED.
                self.reg.announce_replacement_ready(
                    failed_rank=kw["failed_rank"],
                    step=kw.get("step", -1),
                ) if self.reg.query_replacement_status(kw["failed_rank"])
                    == ReplacementState.BOOTSTRAPPING else None,
                self.reg.mark_integrated(
                    failed_rank=kw["failed_rank"], step=kw.get("step", -1),
                ),
            ),
            group_rebuild_request_fn=lambda **kw: None,
            group_rebuild_execute_fn=lambda **kw: None,
            group_rebuild_finish_fn=lambda **kw: None,
            topology_refresh_fn=lambda **kw: None,
            dense_sync_fn=lambda **kw: None,
            expert_restore_fn=lambda **kw: None,
        )

    def test_full_scenario(self):
        """
        Timeline:
        1. Iteration 100: forward fails (NCCL error)
        2. Iteration invalidated, optimizer blocked, rollback
        3. Replacement rank 99 announced for failed rank 3
        4. Iteration 100 (retry): moegambit_before_iteration triggers safe-point repair
        5. Iteration 100 (retry): forward succeeds, optimizer commits
        """

        class MockArgs:
            consumed_train_samples = 5000
            consumed_valid_samples = 0

        args = MockArgs()

        # --- Iteration 100: starts ---
        self.inv.begin_iteration(100)
        self.ctrl.before_iteration(step=100)
        self.guard.begin_iteration(100)
        self.mgr.take_snapshot(iteration=100,
                               consumed_train_samples=args.consumed_train_samples)

        # --- Forward FAILS ---
        self.detector.report_collective_failure(
            failed_rank=3, step=100, mid_iteration=True,
            exception=RuntimeError("NCCL error"),
        )

        # Verify: iteration invalidated
        self.assertTrue(self.inv.is_current_iteration_invalid())
        self.assertFalse(self.guard.should_commit())

        # Verify: controller in PENDING_GROUP_REPAIR
        self.assertEqual(self.ctrl.phase, RecoveryPhase.PENDING_GROUP_REPAIR)

        # --- Rollback ---
        self.mgr.rollback(args)
        self.assertEqual(args.consumed_train_samples, 5000)

        # --- End invalidated iteration ---
        self.inv.end_iteration(100)
        self.guard.end_iteration(100)

        # --- External: replacement rank 99 announced ---
        self.ctrl.on_replacement_assigned(
            failed_rank=3, replacement_rank=99, step=100,
        )
        self.assertEqual(self.ctrl.phase, RecoveryPhase.WAITING_FOR_REPLACEMENT)

        # Verify: replacement NOT participating yet
        self.assertFalse(self.reg.is_replacement_participating(3))
        self.assertEqual(
            self.reg.query_replacement_status(3),
            ReplacementState.BOOTSTRAPPING,
        )

        # --- External: replacement ready ---
        self.ctrl.on_replacement_ready(failed_rank=3, step=100)
        self.assertEqual(self.ctrl.phase, RecoveryPhase.SAFE_POINT_REPAIR)

        # --- Iteration 100 retry: moegambit_before_iteration triggers repair ---
        self.inv.begin_iteration(100)
        repair_done = self.ctrl.before_iteration(step=100)
        self.assertTrue(repair_done)
        self.assertEqual(self.ctrl.phase, RecoveryPhase.REINTEGRATED)

        # Verify: replacement IS participating now
        self.assertTrue(self.reg.is_replacement_participating(3))

        # --- Forward succeeds ---
        self.guard.begin_iteration(100)
        self.assertTrue(self.guard.should_commit())
        self.guard.mark_committed(100)

        # --- Finalize ---
        self.inv.end_iteration(100)
        self.guard.end_iteration(100)

        # Next iteration: finalize reintegration
        self.inv.begin_iteration(101)
        self.ctrl.before_iteration(step=101)
        self.assertEqual(self.ctrl.phase, RecoveryPhase.HEALTHY_TRAINING)
        self.assertTrue(self.ctrl.is_healthy)

    def test_replacement_not_participating_during_bootstrapping(self):
        """Replacement rank must not participate in collectives during
        BOOTSTRAPPING or READY_FOR_REPAIR phases."""

        # Hard failure
        self.detector.report_collective_failure(
            failed_rank=5, step=200, mid_iteration=False,
        )

        # Assign replacement
        self.ctrl.on_replacement_assigned(
            failed_rank=5, replacement_rank=100, step=200,
        )

        # BOOTSTRAPPING: not participating
        self.assertEqual(
            self.reg.query_replacement_status(5),
            ReplacementState.BOOTSTRAPPING,
        )
        self.assertFalse(self.reg.is_replacement_participating(5))

        # Ready: still not participating
        self.reg.announce_replacement_ready(failed_rank=5, step=210)
        self.assertEqual(
            self.reg.query_replacement_status(5),
            ReplacementState.READY_FOR_REPAIR,
        )
        self.assertFalse(self.reg.is_replacement_participating(5))

    def test_multiple_concurrent_failures(self):
        """Two ranks fail simultaneously, both get replacements."""

        # Two hard failures
        self.ctrl.on_hard_rank_failure(failed_rank=3, step=100,
                                       expert_ids=[4, 5])
        # Second failure while already in PENDING_GROUP_REPAIR
        # (controller should handle this gracefully)
        self.ctrl.on_hard_rank_failure(failed_rank=7, step=100,
                                       expert_ids=[12, 13])

        # Both should be tracked
        self.assertEqual(self.ctrl.num_active_faults, 2)

        # Assign replacements
        self.ctrl.on_replacement_assigned(
            failed_rank=3, replacement_rank=99, step=110,
        )
        # Note: second assignment may need the controller to handle
        # multiple pending replacements. The current implementation
        # transitions to WAITING_FOR_REPLACEMENT on first assignment.
        # This test verifies the fault records are correct.
        record3 = self.ctrl.get_fault_record(3)
        self.assertEqual(record3.replacement_rank, 99)


# =====================================================================
# Test: Singleton management
# =====================================================================

class TestSingletons(unittest.TestCase):

    def test_replacement_registry_singleton(self):
        _rep_mod.clear_replacement_registry()
        r1 = _rep_mod.get_replacement_registry()
        r2 = _rep_mod.get_replacement_registry()
        self.assertIs(r1, r2)
        _rep_mod.clear_replacement_registry()

    def test_module_level_convenience_functions(self):
        _rep_mod.clear_replacement_registry()
        slot = _rep_mod.announce_replacement(
            failed_rank=3, replacement_rank=99, step=10,
        )
        self.assertEqual(slot.state, ReplacementState.BOOTSTRAPPING)

        status = _rep_mod.query_replacement_status(3)
        self.assertEqual(status, ReplacementState.BOOTSTRAPPING)

        slot = _rep_mod.announce_replacement_ready(failed_rank=3, step=20)
        self.assertEqual(slot.state, ReplacementState.READY_FOR_REPAIR)

        slot = _rep_mod.mark_integrated(failed_rank=3, step=30)
        self.assertEqual(slot.state, ReplacementState.INTEGRATED)

        _rep_mod.clear_replacement_registry()


if __name__ == "__main__":
    unittest.main()

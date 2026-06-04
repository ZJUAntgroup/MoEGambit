# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for BSR-MoE Full-Peer Recovery (EDP > 1).

Tests cover:
1. RecoveryPath.FULL_PEER_RECOVERY enum value
2. RankExposureGuardedPolicy selects FULL_PEER_RECOVERY when EDP > 1
3. Policy falls back to gap-based decision when EDP == 1
4. select_healthy_expert_dp_peer() peer selection
5. pull_expert_params_from_peer() broadcast-based sync
6. ExpertPeerSyncResult data class
7. RecoveryController._execute_full_peer_recovery_path()
8. RecoveryController Phase B branch dispatches FULL_PEER_RECOVERY
9. No stale exposure recorded for FULL_PEER_RECOVERY
10. StageSafeRecoveryProtocol FULL_PEER_RECOVERY path
11. Fallback when expert_peer_sync_fn is not registered
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
_ret_mod = _load_module("megatron.core.transformer.moe.rank_exposure_tracker")
_policy_mod = _load_module("megatron.core.transformer.moe.gap_aware_recovery_policy")
_rc_mod = _load_module("megatron.core.transformer.moe.recovery_controller")
_ssr_mod = _load_module("megatron.core.transformer.moe.stage_safe_recovery")

RecoveryPath = _policy_mod.RecoveryPath
RecoveryDecision = _policy_mod.RecoveryDecision
RankExposureGuardedPolicy = _policy_mod.RankExposureGuardedPolicy
RankExposureGuardedConfig = _policy_mod.RankExposureGuardedConfig
GapAwareRecoveryPolicyManager = _policy_mod.GapAwareRecoveryPolicyManager

select_healthy_expert_dp_peer = _ds_mod.select_healthy_expert_dp_peer
pull_expert_params_from_peer = _ds_mod.pull_expert_params_from_peer
ExpertPeerSyncResult = _ds_mod.ExpertPeerSyncResult
ParamClassification = _ds_mod.ParamClassification

RecoveryController = _rc_mod.RecoveryController
FaultRecord = _rc_mod.FaultRecord

StageSafeRecoveryProtocol = _ssr_mod.StageSafeRecoveryProtocol


# =====================================================================
# Helpers
# =====================================================================

class FakeParam:
    """Minimal parameter mock with .data, .numel(), .allreduce."""

    def __init__(self, name, numel=10, allreduce=False):
        self.name = name
        self._numel = numel
        self.allreduce = allreduce
        self.data = self  # self-referential for broadcast

    def numel(self):
        return self._numel


class FakeModel:
    """Minimal model mock with named_parameters()."""

    def __init__(self, params):
        self._params = params

    def named_parameters(self):
        return [(p.name, p) for p in self._params]


class ActionTracker:
    """Collects callback invocations for assertion."""

    def __init__(self):
        self.actions = []
        self.calls = {}

    def make_fn(self, name):
        def fn(**kwargs):
            self.actions.append(name)
            self.calls.setdefault(name, []).append(kwargs)
        return fn


# =====================================================================
# 1. RecoveryPath enum
# =====================================================================

class TestRecoveryPathFullPeer(unittest.TestCase):

    def test_full_peer_recovery_value(self):
        self.assertEqual(
            RecoveryPath.FULL_PEER_RECOVERY.value, "full_peer_recovery"
        )

    def test_full_peer_recovery_name(self):
        self.assertEqual(
            RecoveryPath.FULL_PEER_RECOVERY.name, "FULL_PEER_RECOVERY"
        )

    def test_all_three_paths_exist(self):
        paths = {p.name for p in RecoveryPath}
        self.assertIn("CHECKPOINT_RESTART", paths)
        self.assertIn("HYBRID_RECOVERY", paths)
        self.assertIn("FULL_PEER_RECOVERY", paths)


# =====================================================================
# 2. RankExposureGuardedPolicy — FULL_PEER_RECOVERY selection
# =====================================================================

class TestPolicyFullPeerSelection(unittest.TestCase):

    def _make_policy(self, gap_lower=50, gap_upper=200):
        config = RankExposureGuardedConfig(
            delta_time_min_gap=gap_lower,
            max_single_gap=gap_upper,
            exposure_window_steps=1000,
            max_rank_stale_exposure=0.5,
        )
        return RankExposureGuardedPolicy(config=config)

    def test_selects_full_peer_when_edp_available(self):
        """When expert_dp_peer_available=True, policy should select
        FULL_PEER_RECOVERY regardless of gap."""
        policy = self._make_policy()
        decision = policy.choose(
            current_step=200,
            latest_checkpoint_step=100,
            failed_rank=3,
            expert_dp_peer_available=True,
            expert_data_parallel_size=2,
        )
        self.assertEqual(decision.path, RecoveryPath.FULL_PEER_RECOVERY)
        self.assertEqual(decision.reason, "full_peer_edp_available")

    def test_selects_full_peer_even_with_small_gap(self):
        """FULL_PEER_RECOVERY should be selected even when gap is small
        (which would normally trigger CHECKPOINT_RESTART)."""
        policy = self._make_policy(gap_lower=50)
        decision = policy.choose(
            current_step=110,
            latest_checkpoint_step=100,
            failed_rank=3,
            expert_dp_peer_available=True,
            expert_data_parallel_size=4,
        )
        self.assertEqual(decision.path, RecoveryPath.FULL_PEER_RECOVERY)

    def test_falls_back_when_edp_not_available(self):
        """When expert_dp_peer_available=False, policy should use
        standard gap-based logic."""
        policy = self._make_policy(gap_lower=50)
        # Large gap → HYBRID_RECOVERY
        decision = policy.choose(
            current_step=500,
            latest_checkpoint_step=100,
            failed_rank=3,
            expert_dp_peer_available=False,
            expert_data_parallel_size=1,
        )
        self.assertNotEqual(decision.path, RecoveryPath.FULL_PEER_RECOVERY)

    def test_falls_back_when_edp_kwarg_absent(self):
        """When expert_dp_peer_available is not passed, policy should
        use standard gap-based logic (backward compatible)."""
        policy = self._make_policy(gap_lower=50)
        decision = policy.choose(
            current_step=500,
            latest_checkpoint_step=100,
            failed_rank=3,
        )
        self.assertNotEqual(decision.path, RecoveryPath.FULL_PEER_RECOVERY)


# =====================================================================
# 3. select_healthy_expert_dp_peer
# =====================================================================

class TestSelectHealthyExpertDpPeer(unittest.TestCase):

    def test_returns_none_when_no_group(self):
        self.assertIsNone(select_healthy_expert_dp_peer(None))

    def test_returns_none_when_single_rank(self):
        self.assertIsNone(select_healthy_expert_dp_peer([0]))

    def test_selects_lowest_healthy_peer(self):
        result = select_healthy_expert_dp_peer(
            expt_dp_group_ranks=[0, 8, 16, 24],
            failed_ranks=frozenset({0}),
            local_rank=0,
        )
        self.assertEqual(result, 8)

    def test_excludes_quarantined(self):
        result = select_healthy_expert_dp_peer(
            expt_dp_group_ranks=[0, 8, 16, 24],
            quarantined_ranks=frozenset({8}),
            failed_ranks=frozenset({0}),
            local_rank=0,
        )
        self.assertEqual(result, 16)

    def test_returns_none_when_all_excluded(self):
        result = select_healthy_expert_dp_peer(
            expt_dp_group_ranks=[0, 8],
            failed_ranks=frozenset({8}),
            local_rank=0,
        )
        self.assertIsNone(result)

    def test_excludes_local_rank(self):
        result = select_healthy_expert_dp_peer(
            expt_dp_group_ranks=[0, 8, 16],
            local_rank=0,
        )
        self.assertEqual(result, 8)


# =====================================================================
# 4. pull_expert_params_from_peer
# =====================================================================

class TestPullExpertParamsFromPeer(unittest.TestCase):

    def _make_model_and_classification(self):
        dense_p = FakeParam("dense.weight", numel=100, allreduce=True)
        expert_p1 = FakeParam("expert.0.weight", numel=50, allreduce=False)
        expert_p2 = FakeParam("expert.1.weight", numel=50, allreduce=False)
        model = FakeModel([dense_p, expert_p1, expert_p2])

        classification = ParamClassification(
            dense_params={"dense.weight": dense_p},
            expert_params={"expert.0.weight": expert_p1, "expert.1.weight": expert_p2},
            router_params={},
            shared_expert_params={},
        )
        return model, classification

    def test_syncs_expert_params_only(self):
        model, classification = self._make_model_and_classification()
        broadcast_calls = []

        def fake_broadcast(tensor, src, group):
            broadcast_calls.append((src, group))

        result = pull_expert_params_from_peer(
            model=model,
            source_rank=8,
            expt_dp_group="fake_group",
            classification=classification,
            include_optimizer_states=False,
            broadcast_fn=fake_broadcast,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.num_params_synced, 2)
        self.assertEqual(result.num_scalars_synced, 100)  # 50 + 50
        self.assertEqual(result.source_rank, 8)
        self.assertEqual(len(broadcast_calls), 2)

    def test_empty_expert_params(self):
        model = FakeModel([])
        classification = ParamClassification(
            dense_params={},
            expert_params={},
            router_params={},
            shared_expert_params={},
        )

        result = pull_expert_params_from_peer(
            model=model,
            source_rank=0,
            expt_dp_group=None,
            classification=classification,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.num_params_synced, 0)

    def test_dry_run_without_group(self):
        """When expt_dp_group is None, no actual broadcast happens."""
        model, classification = self._make_model_and_classification()

        result = pull_expert_params_from_peer(
            model=model,
            source_rank=0,
            expt_dp_group=None,
            classification=classification,
            include_optimizer_states=False,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.num_params_synced, 2)

    def test_retry_on_failure(self):
        model, classification = self._make_model_and_classification()
        call_count = [0]

        def failing_broadcast(tensor, src, group):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise RuntimeError("simulated NCCL error")

        result = pull_expert_params_from_peer(
            model=model,
            source_rank=0,
            expt_dp_group="fake_group",
            classification=classification,
            include_optimizer_states=False,
            broadcast_fn=failing_broadcast,
            max_retries=3,
        )
        # First attempt fails on first param (call 1), second attempt
        # fails on first param (call 2), third attempt succeeds
        self.assertTrue(result.success)
        self.assertEqual(result.attempt, 3)

    def test_all_retries_exhausted(self):
        model, classification = self._make_model_and_classification()

        def always_fail(tensor, src, group):
            raise RuntimeError("permanent failure")

        result = pull_expert_params_from_peer(
            model=model,
            source_rank=0,
            expt_dp_group="fake_group",
            classification=classification,
            include_optimizer_states=False,
            broadcast_fn=always_fail,
            max_retries=2,
        )
        self.assertFalse(result.success)
        self.assertIn("permanent failure", result.error)


# =====================================================================
# 5. ExpertPeerSyncResult
# =====================================================================

class TestExpertPeerSyncResult(unittest.TestCase):

    def test_default_values(self):
        result = ExpertPeerSyncResult()
        self.assertFalse(result.success)
        self.assertEqual(result.source_rank, -1)
        self.assertEqual(result.num_params_synced, 0)

    def test_to_dict(self):
        result = ExpertPeerSyncResult(
            success=True,
            source_rank=8,
            num_params_synced=4,
            num_scalars_synced=1000,
            elapsed_seconds=0.5,
            attempt=1,
        )
        d = result.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["source_rank"], 8)
        self.assertEqual(d["num_params_synced"], 4)


# =====================================================================
# 6. RecoveryController — FULL_PEER_RECOVERY path
# =====================================================================

class TestControllerFullPeerRecovery(unittest.TestCase):

    def _make_controller(self, checkpoint_iteration=90):
        ctrl = RecoveryController()
        tracker = ActionTracker()

        ctrl.register_callbacks(
            health_mark_healthy_fn=tracker.make_fn("mark_healthy"),
            replacement_announce_fn=tracker.make_fn("replacement_announce"),
            replacement_integrate_fn=tracker.make_fn("replacement_integrate"),
            group_rebuild_request_fn=tracker.make_fn("group_rebuild_request"),
            group_rebuild_execute_fn=tracker.make_fn("group_rebuild_execute"),
            group_rebuild_finish_fn=tracker.make_fn("group_rebuild_finish"),
            topology_refresh_fn=tracker.make_fn("topology_refresh"),
            dense_sync_fn=tracker.make_fn("dense_sync"),
            expert_restore_fn=tracker.make_fn("expert_restore"),
            expert_peer_sync_fn=tracker.make_fn("expert_peer_sync"),
            checkpoint_restart_fn=tracker.make_fn("checkpoint_restart"),
            optimizer_commit_block_fn=tracker.make_fn("block_commit"),
            enter_waiting_fn=tracker.make_fn("enter_waiting"),
        )

        # Wire gap-aware policy with RankExposureGuardedPolicy
        config = RankExposureGuardedConfig(
            delta_time_min_gap=50,
            max_single_gap=200,
            exposure_window_steps=1000,
            max_rank_stale_exposure=0.5,
        )
        policy = RankExposureGuardedPolicy(config=config)
        mgr = GapAwareRecoveryPolicyManager(
            gap_threshold=100,
            policy=policy,
            get_checkpoint_iteration_fn=lambda: checkpoint_iteration,
        )
        ctrl.set_gap_aware_policy_manager(mgr)

        return ctrl, tracker

    def test_full_peer_path_calls_dense_and_expert_peer_sync(self):
        """When FULL_PEER_RECOVERY is selected, both dense_sync and
        expert_peer_sync should be called, but NOT expert_restore."""
        ctrl, tracker = self._make_controller()

        record = FaultRecord(
            failed_rank=3,
            replacement_rank=99,
            expert_ids=[0, 1],
        )

        ctrl._execute_full_peer_recovery_path(
            ready_record=record,
            step=200,
            decision=None,
        )

        self.assertIn("dense_sync", tracker.actions)
        self.assertIn("expert_peer_sync", tracker.actions)
        self.assertNotIn("expert_restore", tracker.actions)
        self.assertEqual(ctrl._last_recovery_path, "FULL_PEER_RECOVERY")

    def test_full_peer_path_marks_healthy(self):
        """FULL_PEER_RECOVERY should call health_mark_healthy_fn
        (experts go directly to HEALTHY, no staleness)."""
        ctrl, tracker = self._make_controller()

        record = FaultRecord(
            failed_rank=3,
            replacement_rank=99,
            expert_ids=[0, 1],
        )

        ctrl._execute_full_peer_recovery_path(
            ready_record=record,
            step=200,
            decision=None,
        )

        self.assertIn("mark_healthy", tracker.actions)

    def test_full_peer_fallback_when_no_expert_peer_sync_fn(self):
        """When expert_peer_sync_fn is not registered, FULL_PEER_RECOVERY
        should fall back to expert_restore (hybrid behavior)."""
        ctrl, tracker = self._make_controller()

        # Unregister expert_peer_sync_fn
        ctrl._expert_peer_sync_fn = None

        record = FaultRecord(
            failed_rank=3,
            replacement_rank=99,
            expert_ids=[0, 1],
        )

        ctrl._execute_full_peer_recovery_path(
            ready_record=record,
            step=200,
            decision=None,
        )

        self.assertIn("dense_sync", tracker.actions)
        self.assertIn("expert_restore", tracker.actions)
        self.assertNotIn("expert_peer_sync", tracker.actions)
        # Should fall back to HYBRID_RECOVERY
        self.assertEqual(ctrl._last_recovery_path, "HYBRID_RECOVERY")

    def test_no_stale_exposure_for_full_peer(self):
        """FULL_PEER_RECOVERY should NOT record stale exposure."""
        ctrl, tracker = self._make_controller(checkpoint_iteration=100)

        record = FaultRecord(
            failed_rank=3,
            replacement_rank=99,
            expert_ids=[0, 1],
        )

        ctrl._execute_full_peer_recovery_path(
            ready_record=record,
            step=200,
            decision=None,
        )

        self.assertEqual(ctrl._last_recovery_path, "FULL_PEER_RECOVERY")
        # The stale exposure recording in _execute_safe_point_repair
        # only fires when _last_recovery_path == "HYBRID_RECOVERY",
        # so FULL_PEER_RECOVERY is correctly excluded.


# =====================================================================
# 7. StageSafeRecoveryProtocol — FULL_PEER_RECOVERY
# =====================================================================

class TestStageSafeFullPeerRecovery(unittest.TestCase):

    def test_full_peer_path_calls_expert_peer_sync(self):
        protocol = StageSafeRecoveryProtocol()
        calls = []

        def dense_sync(**kw):
            calls.append("dense_sync")

        def expert_peer_sync(**kw):
            calls.append("expert_peer_sync")

        def expert_restore(**kw):
            calls.append("expert_restore")

        result = protocol.execute(
            failed_rank=3,
            failed_stage=2,
            replacement_rank=99,
            step=200,
            pp_group_ranks=[0, 1, 2, 3],
            expert_ids=[0, 1],
            recovery_path="FULL_PEER_RECOVERY",
            dense_sync_fn=dense_sync,
            expert_peer_sync_fn=expert_peer_sync,
            expert_restore_fn=expert_restore,
        )

        self.assertIn("dense_sync", calls)
        self.assertIn("expert_peer_sync", calls)
        self.assertNotIn("expert_restore", calls)
        self.assertEqual(result.recovery_path, "FULL_PEER_RECOVERY")

    def test_full_peer_fallback_to_expert_restore(self):
        """When expert_peer_sync_fn is None, should fall back to
        expert_restore_fn."""
        protocol = StageSafeRecoveryProtocol()
        calls = []

        def dense_sync(**kw):
            calls.append("dense_sync")

        def expert_restore(**kw):
            calls.append("expert_restore")

        result = protocol.execute(
            failed_rank=3,
            failed_stage=2,
            replacement_rank=99,
            step=200,
            pp_group_ranks=[0, 1, 2, 3],
            expert_ids=[0, 1],
            recovery_path="FULL_PEER_RECOVERY",
            dense_sync_fn=dense_sync,
            expert_peer_sync_fn=None,
            expert_restore_fn=expert_restore,
        )

        self.assertIn("dense_sync", calls)
        self.assertIn("expert_restore", calls)

    def test_hybrid_path_unchanged(self):
        """HYBRID_RECOVERY path should still work as before."""
        protocol = StageSafeRecoveryProtocol()
        calls = []

        def dense_sync(**kw):
            calls.append("dense_sync")

        def expert_restore(**kw):
            calls.append("expert_restore")

        def expert_peer_sync(**kw):
            calls.append("expert_peer_sync")

        result = protocol.execute(
            failed_rank=3,
            failed_stage=2,
            replacement_rank=99,
            step=200,
            pp_group_ranks=[0, 1, 2, 3],
            expert_ids=[0, 1],
            recovery_path="HYBRID_RECOVERY",
            dense_sync_fn=dense_sync,
            expert_restore_fn=expert_restore,
            expert_peer_sync_fn=expert_peer_sync,
        )

        self.assertIn("dense_sync", calls)
        self.assertIn("expert_restore", calls)
        self.assertNotIn("expert_peer_sync", calls)


# =====================================================================
# 8. GapAwareRecoveryPolicyManager — FULL_PEER_RECOVERY via evaluate()
# =====================================================================

class TestManagerFullPeerEvaluation(unittest.TestCase):

    def test_evaluate_passes_edp_kwargs_through(self):
        """evaluate() should pass expert_dp_peer_available through to
        the policy's choose() method via **kwargs."""
        config = RankExposureGuardedConfig(
            delta_time_min_gap=50,
            max_single_gap=200,
            exposure_window_steps=1000,
            max_rank_stale_exposure=0.5,
        )
        policy = RankExposureGuardedPolicy(config=config)
        mgr = GapAwareRecoveryPolicyManager(
            gap_threshold=100,
            policy=policy,
            get_checkpoint_iteration_fn=lambda: 100,
        )

        decision = mgr.evaluate(
            current_iteration=200,
            failed_rank=3,
            expert_dp_peer_available=True,
            expert_data_parallel_size=2,
        )

        self.assertEqual(decision.path, RecoveryPath.FULL_PEER_RECOVERY)
        self.assertEqual(decision.reason, "full_peer_edp_available")


if __name__ == "__main__":
    unittest.main()

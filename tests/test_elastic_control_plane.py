import ast
import os
import tempfile
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from elastic_watcher import ElasticWatcher, _PHASE_ORDER


class _Connection:
    def __init__(self):
        self.messages = []

    def sendall(self, data):
        self.messages.append(data.decode())


def _args(fault_dir, training_nnodes=2, nproc_per_node=2):
    return Namespace(
        port=20200,
        training_nnodes=training_nnodes,
        nproc_per_node=nproc_per_node,
        master_addr="10.0.0.1",
        master_port="20123",
        fault_dir=fault_dir,
        heartbeat_timeout=30.0,
        startup_heartbeat_timeout=600.0,
        forward_heartbeat_timeout=180.0,
        checkpoint_heartbeat_timeout=900.0,
        disconnect_grace_timeout=15.0,
        fallback_relaunch=True,
        fallback_exit_code=75,
        fallback_restart_standby=True,
        fault_inject_step=17,
        fault_inject_node=0,
        fault_inject_local_rank=1,
    )


def _manifest(node_rank, attempt=0, master_port="20123", digest="same"):
    return {
        "attempt": attempt,
        "node_rank": node_rank,
        "nnodes": 2,
        "nproc_per_node": 2,
        "world_size": 4,
        "master_addr": "10.0.0.1",
        "master_port": master_port,
        "command_sha256": digest,
    }


def _record(watcher, node_rank, manifest, connection):
    with watcher.lock:
        return watcher._record_startup_manifest_locked(
            node_rank, manifest, connection
        )


def test_startup_requires_a_matching_launcher_quorum():
    with tempfile.TemporaryDirectory() as fault_dir:
        watcher = ElasticWatcher(_args(fault_dir))
        connections = [_Connection(), _Connection()]

        assert _record(watcher, 0, _manifest(0), connections[0]) is None
        action = _record(watcher, 1, _manifest(1), connections[1])
        assert action[:2] == ("release", 0)

        watcher._complete_startup_action(*action)
        assert watcher.startup_released_attempt == 0
        assert all('"startup_release"' in conn.messages[-1] for conn in connections)


def test_startup_rejects_a_mismatched_master_port():
    with tempfile.TemporaryDirectory() as fault_dir:
        watcher = ElasticWatcher(_args(fault_dir))
        connection = _Connection()

        action = _record(
            watcher,
            0,
            _manifest(0, master_port="29999"),
            connection,
        )
        assert action[0] == "reject"
        assert "master_port" in action[3]
        assert watcher.startup_released_attempt == -1


def test_fallback_stays_latched_until_the_next_attempt_quorum():
    with tempfile.TemporaryDirectory() as fault_dir:
        watcher = ElasticWatcher(_args(fault_dir))
        standby_restarts = []
        watcher._start_standby_worker = lambda: standby_restarts.append("started")
        connections = [_Connection(), _Connection()]

        _record(watcher, 0, _manifest(0), connections[0])
        watcher._complete_startup_action(
            *_record(watcher, 1, _manifest(1), connections[1])
        )

        watcher.recovery_in_progress = True
        watcher.recovery_epoch = 1
        watcher.failed_node = 0
        watcher.killed_local_rank = 1
        watcher.node_connections = {0: connections[0], 1: connections[1]}
        watcher._initiate_fallback_relaunch("test")

        assert watcher.fallback_initiated
        assert not watcher.recovery_in_progress
        assert standby_restarts == []

        watcher._handle_fault(1, killed_local_rank=0, failure_scope="software")
        assert watcher.recovery_epoch == 1

        retry_connections = [_Connection(), _Connection()]
        assert (
            _record(watcher, 0, _manifest(0, attempt=1), retry_connections[0])
            is None
        )
        watcher._complete_startup_action(
            *_record(watcher, 1, _manifest(1, attempt=1), retry_connections[1])
        )

        assert watcher.startup_released_attempt == 1
        assert not watcher.fallback_initiated
        assert standby_restarts == ["started"]


def test_active_attempt_heartbeats_are_ignored_during_fallback_shutdown():
    with tempfile.TemporaryDirectory() as fault_dir:
        watcher = ElasticWatcher(_args(fault_dir))
        connections = [_Connection(), _Connection()]

        _record(watcher, 0, _manifest(0), connections[0])
        watcher._complete_startup_action(
            *_record(watcher, 1, _manifest(1), connections[1])
        )
        watcher.fallback_initiated = True

        assert _record(watcher, 0, _manifest(0), connections[0]) is None
        assert 0 not in watcher.startup_rejections


def test_ordinal_barrier_rejects_asymmetric_group_init_modes():
    with tempfile.TemporaryDirectory() as fault_dir:
        watcher = ElasticWatcher(_args(fault_dir))
        base = {
            "group_desc": "EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP",
            "group_backend": "default",
            "group_use_local_synchronization": False,
            "group_c10d_count": 329,
            "group_size": 2,
            "group_ranks": [0, 1],
            "group_ordinal": 329,
            "barrier_stage": "enter",
        }

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                watcher._wait_for_ordinal_barrier,
                "group-329",
                0,
                2,
                1.0,
                {**base, "group_init_mode": "lazy"},
            )
            time.sleep(0.01)
            second = executor.submit(
                watcher._wait_for_ordinal_barrier,
                "group-329",
                1,
                2,
                1.0,
                {**base, "group_init_mode": "eager"},
            )

            assert first.result()[-1] is False
            assert second.result()[-1] is False


def test_ordinal_barrier_rejects_different_c10d_generations():
    with tempfile.TemporaryDirectory() as fault_dir:
        watcher = ElasticWatcher(_args(fault_dir))
        base = {
            "group_desc": "EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP",
            "group_backend": "default",
            "group_init_mode": "lazy",
            "group_use_local_synchronization": False,
            "group_size": 2,
            "group_ranks": [0, 1],
            "group_ordinal": 329,
            "barrier_stage": "enter",
        }

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                watcher._wait_for_ordinal_barrier,
                "group-329-generation",
                0,
                2,
                1.0,
                {**base, "group_c10d_count": 329},
            )
            time.sleep(0.01)
            second = executor.submit(
                watcher._wait_for_ordinal_barrier,
                "group-329-generation",
                1,
                2,
                1.0,
                {**base, "group_c10d_count": 330},
            )

            assert first.result()[-1] is False
            assert second.result()[-1] is False


def test_ordinal_barrier_stall_fail_fast_is_disabled_by_default():
    with tempfile.TemporaryDirectory() as fault_dir:
        watcher = ElasticWatcher(_args(fault_dir))
        watcher.ordinal_barriers["group-329-exit"] = {
            "arrived": {rank for rank in range(64) if rank != 1},
            "min_count": 64,
            "created": 100.0,
            "last_progress": 100.0,
        }

        assert watcher.recovery_stall_timeout == 0.0
        assert watcher._stalled_ordinal_barrier_locked(now=1000.0) is None


def test_ordinal_barrier_stall_is_detected_when_explicitly_enabled():
    with tempfile.TemporaryDirectory() as fault_dir:
        watcher = ElasticWatcher(_args(fault_dir))
        watcher.recovery_stall_timeout = 70.0
        watcher.ordinal_barriers["group-329-exit"] = {
            "arrived": {rank for rank in range(64) if rank != 1},
            "meta_by_rank": {
                0: {
                    "group_desc": "EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP",
                    "barrier_stage": "exit",
                }
            },
            "min_count": 64,
            "created": 100.0,
            "last_progress": 100.0,
        }

        assert watcher._stalled_ordinal_barrier_locked(now=169.9) is None
        stalled = watcher._stalled_ordinal_barrier_locked(now=170.0)
        assert stalled["barrier_id"] == "group-329-exit"
        assert stalled["count"] == 63
        assert stalled["missing"] == [1]


def test_replacement_env_carries_recovery_nccl_transport_contract():
    with tempfile.TemporaryDirectory() as fault_dir:
        watcher = ElasticWatcher(_args(fault_dir))
        watcher.recovery_epoch = 1
        with patch.dict(
            os.environ,
            {
                "NODE_RANK": "2",
                "ELASTIC_RECOVERY_NCCL_SOCKET_ONLY": "1",
                "ELASTIC_RECOVERY_NCCL_DEBUG": "INFO",
                "ELASTIC_RECOVERY_NCCL_SOCKET_IFNAME": "=eth9",
            },
            clear=False,
        ):
            env = watcher._build_spare_env(0, 1, resume_iteration=18)

        assert env["ELASTIC_RECOVERY_NCCL_SOCKET_ONLY"] == "1"
        assert env["ELASTIC_RECOVERY_NCCL_DEBUG"] == "INFO"
        assert env["ELASTIC_RECOVERY_NCCL_SOCKET_IFNAME"] == "=eth9"


def test_replacement_env_drops_inherited_torchelastic_store_namespace():
    with tempfile.TemporaryDirectory() as fault_dir:
        watcher = ElasticWatcher(_args(fault_dir))
        watcher.recovery_epoch = 1
        with patch.dict(
            os.environ,
            {
                "NODE_RANK": "2",
                "TORCHELASTIC_USE_AGENT_STORE": "True",
                "TORCHELASTIC_RESTART_COUNT": "7",
                "TORCHELASTIC_RUN_ID": "stale-run",
            },
            clear=False,
        ):
            env = watcher._build_spare_env(0, 1, resume_iteration=18)

        assert "TORCHELASTIC_USE_AGENT_STORE" not in env
        assert "TORCHELASTIC_RESTART_COUNT" not in env
        assert "TORCHELASTIC_RUN_ID" not in env


def test_replacement_reports_ready_before_blocking_rebuild_store_connect():
    initialize_path = (
        Path(__file__).parents[1]
        / "Megatron-LM"
        / "megatron"
        / "training"
        / "initialize.py"
    )
    tree = ast.parse(initialize_path.read_text())
    initialize_distributed = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_initialize_distributed"
    )

    init_pg_start_line = None
    rebuild_store_line = None
    for node in ast.walk(initialize_distributed):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if (
            node.func.id == "_elastic_report_phase_safely"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "init_pg_start"
        ):
            init_pg_start_line = node.lineno
        elif node.func.id == "elastic_create_rebuild_store":
            rebuild_store_line = node.lineno

    assert _PHASE_ORDER["init_pg_start"] < _PHASE_ORDER["rebuild_store_ready"]
    assert _PHASE_ORDER["rebuild_store_ready"] < _PHASE_ORDER["pg_ready"]
    assert init_pg_start_line is not None
    assert rebuild_store_line is not None
    assert init_pg_start_line < rebuild_store_line


def test_optimizer_rebind_classifies_non_distributed_dense_and_expert_groups():
    elastic_client_path = (
        Path(__file__).parents[1]
        / "Megatron-LM"
        / "megatron"
        / "training"
        / "elastic_client.py"
    )
    tree = ast.parse(elastic_client_path.read_text())
    classifier_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_elastic_classify_optimizer_param_groups"
    )
    namespace = {}
    classifier_module = ast.Module(body=[classifier_node], type_ignores=[])
    exec(compile(classifier_module, "classifier", "exec"), namespace)
    classify = namespace["_elastic_classify_optimizer_param_groups"]

    class _BaseOptimizer:
        def __init__(self, expert_flags):
            self.param_groups = [
                {"is_expert_parallel": expert_flag} for expert_flag in expert_flags
            ]

    class _MegatronOptimizer:
        def __init__(self, expert_flags):
            self.optimizer = _BaseOptimizer(expert_flags)

    assert classify(_MegatronOptimizer([False, False])) == "dense"
    assert classify(_MegatronOptimizer([True, True])) == "expert"
    assert classify(_MegatronOptimizer([])) is None
    try:
        classify(_MegatronOptimizer([False, True]))
    except RuntimeError as exc:
        assert "both dense and expert" in str(exc)
    else:
        raise AssertionError("mixed dense/expert optimizer must be rejected")

    assert _PHASE_ORDER["forward_backward_done"] < _PHASE_ORDER[
        "optimizer_pg_contract_ready"
    ]
    assert _PHASE_ORDER["optimizer_pg_contract_ready"] < _PHASE_ORDER[
        "optimizer_step_start"
    ]

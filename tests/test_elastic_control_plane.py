import ast
import json
import os
import tempfile
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional
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
        recovery_abort_exit_code=76,
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


def test_disabled_fallback_aborts_all_launchers_without_reset_or_relaunch():
    with tempfile.TemporaryDirectory() as fault_dir:
        args = _args(fault_dir)
        args.fallback_relaunch = False
        watcher = ElasticWatcher(args)
        connections = [_Connection(), _Connection()]
        watcher.recovery_in_progress = True
        watcher.recovery_epoch = 3
        watcher.failed_node = 0
        watcher.killed_local_rank = 1
        watcher.node_connections = {0: connections[0], 1: connections[1]}

        watcher._initiate_fallback_relaunch("ordinal_barrier_timeout", {"count": 3})

        assert watcher.fallback_initiated
        assert watcher.recovery_in_progress
        assert not watcher.running
        assert watcher.exit_code == 76
        requests = [json.loads(conn.messages[-1]) for conn in connections]
        assert all(request["action"] == "abort_training" for request in requests)
        assert all(request["exit_code"] == 76 for request in requests)
        assert all("resume_iteration" not in request for request in requests)
        manifest = json.loads(Path(watcher.recovery_abort_file).read_text())
        assert manifest["reason"] == "ordinal_barrier_timeout"
        assert manifest["action"] == "abort_training"


def test_recovery_abort_prefers_launcher_control_connections():
    with tempfile.TemporaryDirectory() as fault_dir:
        args = _args(fault_dir)
        args.fallback_relaunch = False
        watcher = ElasticWatcher(args)
        worker_connections = [_Connection(), _Connection()]
        launcher_connections = [_Connection(), _Connection()]
        watcher.recovery_in_progress = True
        watcher.node_connections = {
            0: worker_connections[0],
            1: worker_connections[1],
        }
        watcher.launcher_connections = {
            0: launcher_connections[0],
            1: launcher_connections[1],
        }

        watcher._initiate_fallback_relaunch("test_abort")

        assert all(not conn.messages for conn in worker_connections)
        assert all(
            '"action": "abort_training"' in conn.messages[-1]
            for conn in launcher_connections
        )


def test_recovery_abort_releases_pending_ordinal_barriers():
    with tempfile.TemporaryDirectory() as fault_dir:
        args = _args(fault_dir)
        args.fallback_relaunch = False
        watcher = ElasticWatcher(args)
        watcher.recovery_in_progress = True

        with ThreadPoolExecutor(max_workers=1) as executor:
            waiting = executor.submit(
                watcher._wait_for_ordinal_barrier,
                "pending-recovery-barrier",
                0,
                2,
                30.0,
                {"group_ranks": [0, 1]},
            )
            deadline = time.time() + 1.0
            while "pending-recovery-barrier" not in watcher.ordinal_barriers:
                assert time.time() < deadline
                time.sleep(0.01)

            watcher._initiate_fallback_relaunch("test_abort")

            count, missing, arrived, manifest_ok = waiting.result(timeout=1.0)
            assert (count, missing, arrived, manifest_ok) == (1, [1], [0], True)


def test_hotspare_fail_fast_is_terminal_and_cannot_relaunch():
    script = (Path(__file__).parents[1] / "test_hotspare_replace.sh").read_text()

    assert 'ELASTIC_RECOVERY_NCCL_DEBUG:-WARN' in script
    assert "export ELASTIC_RECOVERY_STALL_TIMEOUT_SECONDS=70" in script
    assert "export ELASTIC_MOE_FIRST_COLLECTIVE_FAIL_FAST=1" in script
    assert "export ELASTIC_MOE_FIRST_COLLECTIVE_TIMEOUT=70" in script
    assert "export ELASTIC_FALLBACK_RELAUNCH=0" in script
    assert "export ELASTIC_FALLBACK_RESTART_STANDBY=0" in script
    assert "HOTSPARE_MAX_RETRIES=0" in script


def test_post_rebuild_communicators_follow_the_real_training_order():
    root = Path(__file__).parents[1]
    sources = (
        (root / "test_hotspare_replace.sh").read_text(),
        (root / "run_spare_single_rank.sh").read_text(),
        (root / "elastic_watcher.py").read_text(),
        (root / "Megatron-LM/megatron/training/training.py").read_text(),
        (root / "Megatron-LM/megatron/training/elastic_client.py").read_text(),
        (
            root
            / "Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py"
        ).read_text(),
    )

    for source in sources:
        assert "ELASTIC_POST_REBUILD_COMM_WARMUP" not in source
        assert "ELASTIC_MOE_FIRST_COLLECTIVE_WARMUP" not in source
        assert "elastic_warmup_post_rebuild_communicators" not in source


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


def test_ordinal_barrier_rejects_different_rerun_state_contracts():
    with tempfile.TemporaryDirectory() as fault_dir:
        watcher = ElasticWatcher(_args(fault_dir))
        base = {
            "group_desc": "RERUN_STATE_MACHINE",
            "group_backend": "control_plane",
            "group_size": 2,
            "group_ranks": [0, 1],
            "barrier_stage": "pre_first_recovered_step",
        }
        canonical = {
            "mode": "validate_results",
            "state": 0,
            "state_name": "NOT_RUNNING_YET",
            "current_iteration": 19,
        }

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                watcher._wait_for_ordinal_barrier,
                "rerun-state-19",
                0,
                2,
                1.0,
                {**base, "state_contract": canonical},
            )
            time.sleep(0.01)
            second = executor.submit(
                watcher._wait_for_ordinal_barrier,
                "rerun-state-19",
                1,
                2,
                1.0,
                {
                    **base,
                    "state_contract": {
                        **canonical,
                        "state": 1,
                        "state_name": "INITIAL_RUN",
                    },
                },
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


def test_planned_fault_prearms_the_exact_replacement_runtime():
    class _Proc:
        pid = 1234
        stdout = None

        @staticmethod
        def poll():
            return None

    captured = {}

    def _popen(command, env, stdout, stderr):
        captured.update(command=command, env=dict(env), stdout=stdout, stderr=stderr)
        return _Proc()

    with tempfile.TemporaryDirectory() as fault_dir:
        watcher = ElasticWatcher(_args(fault_dir))
        with (
            patch.dict(
                os.environ,
                {
                    "NODE_RANK": "2",
                    "ELASTIC_PREARM_PLANNED_SPARE": "1",
                    "ELASTIC_STANDBY_STORE_TIMEOUT_MINUTES": "999",
                },
                clear=False,
            ),
            patch("elastic_watcher.subprocess.Popen", side_effect=_popen),
            patch("elastic_watcher.threading.Thread.start", return_value=None),
        ):
            watcher._start_standby_worker()

        env = captured["env"]
        assert env["ELASTIC_PREARMED_STANDBY"] == "1"
        assert "ELASTIC_STANDBY_MODE" not in env
        assert env["RANK"] == "1"
        assert env["CUDA_VISIBLE_DEVICES"] == "1"
        assert env["ELASTIC_RECOVERY_EPOCH"] == "1"
        assert env["MASTER_PORT"] == "20124"
        assert env["ELASTIC_REBUILD_TIMEOUT_MINUTES"] == "999"
        assert env["ELASTIC_SELECTIVE_GROUP_REBUILD"] == "1"
        assert watcher.standby_prearmed
        assert watcher.standby_prearmed_epoch == 1
        spare_script = (Path(__file__).parents[1] / "run_spare_single_rank.sh").read_text()
        assert '[ "${ELASTIC_PREARMED_STANDBY:-0}" = "1" ]' in spare_script


def test_future_prearmed_phase_is_the_only_future_epoch_message_accepted():
    with tempfile.TemporaryDirectory() as fault_dir:
        watcher = ElasticWatcher(_args(fault_dir))
        prearmed = {
            "recovery_epoch": 1,
            "standby_prearmed": True,
        }
        assert not watcher._is_stale_recovery_message(prearmed, "recovery_phase")
        assert watcher._is_stale_recovery_message(
            {"recovery_epoch": 1}, "recovery_phase"
        )
        assert watcher._is_stale_recovery_message(prearmed, "ready_to_rebuild")


def test_prearmed_assignment_refreshes_only_dynamic_recovery_metadata():
    elastic_client_path = (
        Path(__file__).parents[1]
        / "Megatron-LM"
        / "megatron"
        / "training"
        / "elastic_client.py"
    )
    tree = ast.parse(elastic_client_path.read_text())
    refresh_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "elastic_refresh_prearmed_standby_assignment"
    )
    namespace = {
        "json": json,
        "os": os,
        "time": time,
        "logger": type("_Logger", (), {"warning": lambda *args, **kwargs: None})(),
        "_PREARMED_STANDBY_RUNTIME": {
            "summary": {"te_status": "imported"},
            "cache": {},
        },
    }
    module = ast.Module(body=[refresh_node], type_ignores=[])
    exec(compile(module, str(elastic_client_path), "exec"), namespace)

    with tempfile.TemporaryDirectory() as directory:
        assignment_path = Path(directory) / "assignment.json"
        assignment_path.write_text(
            json.dumps(
                {
                    "RANK": "1",
                    "WORLD_SIZE": "4",
                    "LOCAL_RANK": "0",
                    "MASTER_ADDR": "10.0.0.1",
                    "MASTER_PORT": "20124",
                    "CUDA_VISIBLE_DEVICES": "1",
                    "ELASTIC_RESUME_ITERATION": "18",
                    "ELASTIC_RECOVERY_DESCRIPTOR_SHA256": "digest",
                }
            )
        )
        with patch.dict(
            os.environ,
            {
                "ELASTIC_PREARMED_STANDBY": "1",
                "ELASTIC_SPARE_ASSIGNMENT_FILE": str(assignment_path),
                "RANK": "1",
                "WORLD_SIZE": "4",
                "LOCAL_RANK": "0",
                "MASTER_ADDR": "10.0.0.1",
                "MASTER_PORT": "20124",
                "CUDA_VISIBLE_DEVICES": "1",
            },
            clear=False,
        ):
            result = namespace[
                "elastic_refresh_prearmed_standby_assignment"
            ]()
            assert result["warm_runtime"]["te_status"] == "imported"
            assert os.environ["ELASTIC_RESUME_ITERATION"] == "18"
            assert os.environ["ELASTIC_STANDBY_ACTIVATED"] == "1"


def test_selective_rebuild_detaches_and_restores_healthy_c10d_group():
    parallel_state_path = (
        Path(__file__).parents[1]
        / "Megatron-LM"
        / "megatron"
        / "core"
        / "parallel_state.py"
    )
    tree = ast.parse(parallel_state_path.read_text())
    function_names = {
        "_elastic_snapshot_registered_group",
        "_elastic_detach_registered_group",
        "_elastic_restore_registered_group",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]

    group = object()

    class _World:
        pg_map = {group: ("nccl", "store")}
        pg_names = {group: "7"}
        pg_group_ranks = {group: {0: 0, 2: 1}}
        pg_backend_config = {group: "cuda:nccl"}
        pg_to_tag = {group: "ptd:7"}
        tags_to_pg = {"ptd:7": [group], "": [group]}
        pg_coalesce_state = {}

    class _C10d:
        _world = _World()
        registered = {}

        @classmethod
        def _register_process_group(cls, name, process_group):
            cls.registered[name] = process_group

    class _Distributed:
        distributed_c10d = _C10d

    class _Torch:
        distributed = _Distributed

    namespace = {"torch": _Torch}
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(parallel_state_path), "exec"), namespace)

    snapshot = namespace["_elastic_snapshot_registered_group"](group)
    namespace["_elastic_detach_registered_group"](snapshot)
    assert group not in _World.pg_map
    assert group not in _World.pg_names
    assert group not in _World.tags_to_pg.get("ptd:7", [])

    restored = namespace["_elastic_restore_registered_group"](
        snapshot, "elastic_retained_1_0"
    )
    assert restored is group
    assert _World.pg_map[group] == ("nccl", "store")
    assert _World.pg_names[group] == "elastic_retained_1_0"
    assert _C10d.registered["elastic_retained_1_0"] is group


def test_selective_group_retention_precedes_world_teardown():
    elastic_client_path = (
        Path(__file__).parents[1]
        / "Megatron-LM"
        / "megatron"
        / "training"
        / "elastic_client.py"
    )
    source = elastic_client_path.read_text()
    tree = ast.parse(source)
    rebuild_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "elastic_do_rebuild"
    )
    rebuild_source = ast.get_source_segment(source, rebuild_node)
    retain = rebuild_source.index("prepare_elastic_selective_group_rebuild(")
    teardown = rebuild_source.index("_elastic_destroy_process_group_generation(")
    assert retain < teardown

    parallel_state_source = (
        Path(__file__).parents[1]
        / "Megatron-LM"
        / "megatron"
        / "core"
        / "parallel_state.py"
    ).read_text()
    parallel_state_tree = ast.parse(parallel_state_source)
    create_group_node = next(
        node
        for node in parallel_state_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_group"
    )
    create_group_source = ast.get_source_segment(
        parallel_state_source, create_group_node
    )
    reuse_branch = create_group_source.index("if reuse_group:")
    new_group = create_group_source.index("torch.distributed.new_group(**kwargs)")
    assert reuse_branch < new_group


def test_selective_finalize_keeps_auxiliary_group_in_next_manifest():
    parallel_state_path = (
        Path(__file__).parents[1]
        / "Megatron-LM"
        / "megatron"
        / "core"
        / "parallel_state.py"
    )
    tree = ast.parse(parallel_state_path.read_text())
    finalize = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "finalize_elastic_selective_group_rebuild"
    )
    signature = ("AUX", (0, 2), "nccl", False)
    group = object()
    namespace = {
        "os": os,
        "logger": type("_Logger", (), {"warning": lambda *args, **kwargs: None})(),
        "_ELASTIC_MPU_GROUP_SPECS": [],
        "_ELASTIC_RETAINED_MPU_GROUPS": {signature: [{"group": group}]},
        "_ELASTIC_SELECTIVE_REBUILD_ACTIVE": True,
        "_ELASTIC_SELECTIVE_REBUILD_RANK": 1,
        "_ELASTIC_SELECTIVE_REBUILD_STATS": {
            "retained": 1,
            "reused": 0,
            "rebuilt": 0,
            "skipped_nonmember": 0,
        },
        "_global_process_group_list": [None],
        "_elastic_restore_registered_group": lambda snapshot, alias: snapshot["group"],
    }
    module = ast.Module(body=[finalize], type_ignores=[])
    exec(compile(module, str(parallel_state_path), "exec"), namespace)

    result = namespace["finalize_elastic_selective_group_rebuild"]()

    assert result["reused"] == 1
    assert namespace["_ELASTIC_MPU_GROUP_SPECS"] == [
        {
            "signature": signature,
            "ranks": (0, 2),
            "backend": "nccl",
            "group_desc": "AUX",
            "use_local_synchronization": False,
            "group": group,
        }
    ]


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
    assert _PHASE_ORDER["optimizer_step_done"] < _PHASE_ORDER[
        "train_step_finalize_done"
    ]
    assert _PHASE_ORDER["train_step_finalize_done"] < _PHASE_ORDER[
        "training_log_start"
    ]
    assert _PHASE_ORDER["training_log_start"] < _PHASE_ORDER["training_log_done"]
    assert _PHASE_ORDER["training_log_done"] < _PHASE_ORDER[
        "post_step_callbacks_start"
    ]
    assert _PHASE_ORDER["post_step_callbacks_done"] < _PHASE_ORDER[
        "checkpoint_exit_start"
    ]
    assert _PHASE_ORDER["checkpoint_exit_done"] < _PHASE_ORDER[
        "post_rebuild_commit_ready"
    ]


def test_aux_loss_reduction_skips_single_rank_process_groups():
    moe_utils_path = (
        Path(__file__).parents[1]
        / "Megatron-LM"
        / "megatron"
        / "core"
        / "transformer"
        / "moe"
        / "moe_utils.py"
    )
    tree = ast.parse(moe_utils_path.read_text())
    reducer_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "reduce_aux_losses_tracker_across_ranks"
    )

    class _Group:
        def __init__(self, size):
            self._size = size

        def size(self):
            return self._size

    class _Distributed:
        class ReduceOp:
            AVG = "avg"

        def __init__(self):
            self.calls = []

        def all_reduce(self, values, group, op=None):
            self.calls.append((values, group, op))

    class _Torch:
        distributed = _Distributed()

    class _ParallelState:
        pp_group = _Group(1)
        dp_group = _Group(1)

        @classmethod
        def get_pipeline_model_parallel_group(cls):
            return cls.pp_group

        @classmethod
        def get_data_parallel_group(cls, with_context_parallel=False):
            assert with_context_parallel
            return cls.dp_group

    tracker = {
        "aux": {
            "values": object(),
            "reduce_group": _Group(1),
            "avg_group": _Group(1),
        }
    }
    namespace = {
        "List": List,
        "Optional": Optional,
        "torch": _Torch,
        "parallel_state": _ParallelState,
        "get_moe_layer_wise_logging_tracker": lambda: tracker,
    }
    reducer_module = ast.Module(body=[reducer_node], type_ignores=[])
    exec(compile(reducer_module, "moe_utils", "exec"), namespace)
    reduce_aux_losses = namespace["reduce_aux_losses_tracker_across_ranks"]

    reduce_aux_losses(["aux"])
    assert _Torch.distributed.calls == []

    _ParallelState.pp_group = _Group(2)
    _ParallelState.dp_group = _Group(2)
    tracker["aux"]["reduce_group"] = _Group(2)
    tracker["aux"]["avg_group"] = _Group(2)
    reduce_aux_losses(["aux"])
    assert len(_Torch.distributed.calls) == 4


def test_recovery_contract_is_committed_at_train_loop_boundary():
    training_path = (
        Path(__file__).parents[1]
        / "Megatron-LM"
        / "megatron"
        / "training"
        / "training.py"
    )
    source = training_path.read_text()
    tree = ast.parse(source)
    train_step_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_step"
    )
    train_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    train_step_source = ast.get_source_segment(source, train_step_node)
    train_source = ast.get_source_segment(source, train_node)

    assert '"post_rebuild_step_complete"' not in train_step_source
    training_log_call = train_source.index("report_memory_flag = training_log(")
    training_log_done = train_source.index('"training_log_done"', training_log_call)
    callbacks_call = train_source.index(
        "post_training_step_callbacks(", training_log_done
    )
    checkpoint_call = train_source.index(
        "should_exit = checkpoint_and_decide_exit(", callbacks_call
    )
    recovery_commit = train_source.index(
        "elastic_commit_post_rebuild_iteration(args.curr_iteration)", checkpoint_call
    )
    assert (
        training_log_call
        < training_log_done
        < callbacks_call
        < checkpoint_call
        < recovery_commit
    )


def test_external_recovery_resets_megatron_rerun_state_on_every_rank():
    root = Path(__file__).parents[1]
    rerun_path = (
        root
        / "Megatron-LM"
        / "megatron"
        / "core"
        / "rerun_state_machine.py"
    )
    elastic_client_path = (
        root
        / "Megatron-LM"
        / "megatron"
        / "training"
        / "elastic_client.py"
    )
    rerun_source = rerun_path.read_text()
    rerun_tree = ast.parse(rerun_source)
    rerun_class = next(
        node
        for node in rerun_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RerunStateMachine"
    )
    reset_node = next(
        node
        for node in rerun_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "reset_after_external_recovery"
    )
    reset_source = ast.get_source_segment(rerun_source, reset_node)
    assert "self.state = RerunState.NOT_RUNNING_YET" in reset_source
    assert "self.current_iteration = int(current_iteration)" in reset_source
    assert "self.rerun_requested = False" in reset_source
    assert "self.data_iterator_checkpoints = None" in reset_source
    assert '"first_iteration_complete": self.first_iteration_complete' in reset_source

    elastic_source = elastic_client_path.read_text()
    elastic_tree = ast.parse(elastic_source)
    assert elastic_source.count("_elastic_reset_rerun_state_machine(") == 3
    assert 'elastic_report_recovery_phase("rerun_state_reset"' in elastic_source
    assert 'group_desc="RERUN_STATE_MACHINE"' in elastic_source
    assert "state_contract=canonical" in elastic_source
    assert "_elastic_validate_rerun_state_machine(iteration" in elastic_source
    assert '"train_start_iteration": train_start_iteration' in elastic_source
    validator_node = next(
        node
        for node in elastic_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_elastic_validate_rerun_state_machine"
    )
    validator_source = ast.get_source_segment(elastic_source, validator_node)
    assert "state_contract=canonical" in validator_source
    assert "state_contract=local" not in validator_source
    assert _PHASE_ORDER["state_contract_ready"] < _PHASE_ORDER["rerun_state_reset"]
    assert _PHASE_ORDER["rerun_state_reset"] < _PHASE_ORDER["train_ready"]
    assert _PHASE_ORDER["post_rebuild_iteration_ready"] < _PHASE_ORDER[
        "rerun_state_contract_start"
    ]
    assert _PHASE_ORDER["rerun_state_contract_ready"] < _PHASE_ORDER[
        "forward_backward_start"
    ]
    training_path = (
        root
        / "Megatron-LM"
        / "megatron"
        / "training"
        / "training.py"
    )
    training_source = training_path.read_text()
    assert 'getattr(args, "elastic_train_start_iteration", iteration)' in training_source
    capture = elastic_source.index(
        "args.elastic_train_start_iteration = int(args.iteration)"
    )
    align = elastic_source.index(
        "aligned_iteration = elastic_align_resume_state(", capture
    )
    assert capture < align

    contract_node = next(
        node
        for node in elastic_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_elastic_train_start_contract"
    )
    contract_module = ast.Module(body=[contract_node], type_ignores=[])
    ast.fix_missing_locations(contract_module)
    namespace = {}
    exec(compile(contract_module, str(elastic_client_path), "exec"), namespace)
    contract = namespace["_elastic_train_start_contract"]

    # A survivor started at iteration 0 while its replacement loaded the
    # iteration-10 checkpoint.  At recovery iteration 18 they select the same
    # next-step branches and therefore must not mismatch the global manifest.
    assert contract(18, 0, True) == contract(18, 10, True)
    # Keep detecting a real branch divergence, such as only the replacement
    # being due to update process-group timeouts after checkpoint startup.
    assert contract(11, 0, True) != contract(11, 10, True)


def test_post_rebuild_commit_uses_one_barrier_after_first_recovered_iteration():
    elastic_client_path = (
        Path(__file__).parents[1]
        / "Megatron-LM"
        / "megatron"
        / "training"
        / "elastic_client.py"
    )
    source = elastic_client_path.read_text()
    tree = ast.parse(source)
    commit_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "elastic_commit_post_rebuild_iteration"
    )
    commit_source = ast.get_source_segment(source, commit_node)

    ready_report = commit_source.index("elastic_report_recovery_phase(commit_phase")
    ready_wait = commit_source.index(
        "elastic_wait_for_recovery_phase_count(commit_phase"
    )
    trace_clear = commit_source.index("elastic_clear_post_rebuild_trace()")
    assert ready_report < ready_wait < trace_clear
    assert "post_rebuild_step_complete" not in commit_source
    assert "stabilization" not in commit_source
    assert "post_rebuild_stabilization" not in source
    assert not any(phase.startswith("stabilization_") for phase in _PHASE_ORDER)

    validator_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "elastic_validate_optimizer_process_groups"
    )
    validator_source = ast.get_source_segment(source, validator_node)
    effective_phase = validator_source.index(
        'ready_phase = "optimizer_pg_contract_ready"'
    )
    phase_wait = validator_source.index(
        "elastic_wait_for_recovery_phase_count(\n        ready_phase"
    )
    assert effective_phase < phase_wait


def test_disabled_rebuild_warmup_skips_the_global_control_barrier():
    elastic_client_path = (
        Path(__file__).parents[1]
        / "Megatron-LM"
        / "megatron"
        / "training"
        / "elastic_client.py"
    )
    source = elastic_client_path.read_text()
    tree = ast.parse(source)
    warmup_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_elastic_warmup_rebuild_communicators"
    )
    warmup_source = ast.get_source_segment(source, warmup_node)

    disabled_guard = warmup_source.index("if not selected_group_names:")
    disabled_return = warmup_source.index("return", disabled_guard)
    barrier_report = warmup_source.index(
        'elastic_report_recovery_phase("comm_warmup_start")'
    )
    assert disabled_guard < disabled_return < barrier_report


def test_moe_first_collective_fail_fast_is_one_shot_per_recovery_step():
    dispatcher_path = (
        Path(__file__).parents[1]
        / "Megatron-LM"
        / "megatron"
        / "core"
        / "transformer"
        / "moe"
        / "token_dispatcher.py"
    )
    source = dispatcher_path.read_text()
    tree = ast.parse(source)
    dispatcher = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MoETokenDispatcher"
    )
    gather_node = next(
        node
        for node in dispatcher.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_elastic_gather_first_dim_ready_aligned"
    )
    gather_source = ast.get_source_segment(source, gather_node)

    fast_path = gather_source.index(
        "if check_key in _ELASTIC_MOE_FIRST_COLLECTIVE_CHECKS"
    )
    stream_sync = gather_source.index("torch.cuda.current_stream().synchronize()")
    fail_fast = gather_source.index("_elastic_gather_first_dim_fail_fast(")
    mark_checked = gather_source.index(
        "_ELASTIC_MOE_FIRST_COLLECTIVE_CHECKS.add(check_key)"
    )
    assert fast_path < stream_sync < fail_fast < mark_checked


def test_post_rebuild_detail_trace_uses_node_representatives():
    elastic_client_path = (
        Path(__file__).parents[1]
        / "Megatron-LM"
        / "megatron"
        / "training"
        / "elastic_client.py"
    )
    source = elastic_client_path.read_text()
    tree = ast.parse(source)
    trace_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "elastic_trace_post_rebuild_phase"
    )
    trace_source = ast.get_source_segment(source, trace_node)

    first_step_validation = trace_source.index(
        'os.environ.get("ELASTIC_RECOVERY_STATE") == "post_rebuild_trace"'
    )
    representative_guard = trace_source.index(
        "if local_rank != 0 and not is_rebuild_mode()"
    )
    report = trace_source.index("elastic_report_recovery_phase(phase")
    assert first_step_validation < representative_guard < report


def test_pipeline_p2p_diagnostics_are_recovery_gated_and_non_blocking():
    p2p_path = (
        Path(__file__).parents[1]
        / "Megatron-LM"
        / "megatron"
        / "core"
        / "pipeline_parallel"
        / "p2p_communication.py"
    )
    source = p2p_path.read_text()
    tree = ast.parse(source)
    communicator = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "P2PCommunicator"
    )
    trace_node = next(
        node
        for node in communicator.body
        if isinstance(node, ast.FunctionDef) and node.name == "_elastic_trace_p2p_once"
    )
    communicate_node = next(
        node
        for node in communicator.body
        if isinstance(node, ast.FunctionDef) and node.name == "_communicate"
    )
    trace_source = ast.get_source_segment(source, trace_node)
    communicate_source = ast.get_source_segment(source, communicate_node)

    assert "elastic_is_post_rebuild_trace_active()" in trace_source
    assert "elastic_wait_for" not in trace_source
    start_trace = communicate_source.index('"pipeline_p2p_start"')
    p2p_call = communicate_source.index("p2p_reqs = p2p_func(")
    returned_trace = communicate_source.index('"pipeline_p2p_returned"', p2p_call)
    assert start_trace < p2p_call < returned_trace

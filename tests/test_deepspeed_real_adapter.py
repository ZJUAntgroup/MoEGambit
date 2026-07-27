import ast
import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deepspeed_adapter"))


def test_real_workload_contains_both_supported_topologies():
    path = ROOT / "deepspeed_qwen3_moe_pretrain.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }

    assert "Qwen3MoeEmbeddingPipe" in classes
    assert "MegatronMMapTokenSource" in classes
    assert "Qwen3MoeForCausalLM" in source
    assert "PipelineModule" in source
    assert "AutoEPMoELayer" in source
    assert "DeepSpeed PipelineModule is incompatible with ZeRO stage 2" in source
    assert "MOEGAMBIT_AUTOEP_GROUPED_MM" in source
    assert 'MOEGAMBIT_DEEPSPEED_HYBRID_RESTORE="${hot_swap}"' in (
        ROOT / "test_deepspeed_hotspare_replace.sh"
    ).read_text(encoding="utf-8")
    assert 'callable(getattr(torch, "_grouped_mm", None))' in source


def test_pipeline_backward_has_single_hook_owned_lifecycle():
    source = (
        ROOT
        / "DeepSpeed"
        / "deepspeed"
        / "runtime"
        / "pipe"
        / "engine.py"
    ).read_text(encoding="utf-8")
    start = source.index("    def _exec_backward_pass(self, buffer_id):")
    end = source.index("    def _exec_load_micro_batch", start)
    backward = source[start:end]

    assert "torch.autograd.backward(" in backward
    assert "out.backward(" not in backward
    assert "self._running_engine_backward = True" not in backward
    assert "self.timers(BACKWARD_MICRO_TIMER).start()" not in backward
    assert "self.optimizer.update_hp_grads" not in backward


def test_autoep_group_creation_receives_pipeline_topology_explicitly():
    engine = (
        ROOT / "DeepSpeed" / "deepspeed" / "runtime" / "engine.py"
    ).read_text(encoding="utf-8")
    groups = (
        ROOT / "DeepSpeed" / "deepspeed" / "utils" / "groups.py"
    ).read_text(encoding="utf-8")

    assert "pipeline_mpu=self.mpu" in engine
    assert "AutoEP group crosses pipeline stages" in engine
    assert "topology_mpu = pipeline_mpu" in groups
    assert "topology_mpu._topo.filter_match(pipe=stage)" in groups


def test_training_reports_barrier_and_first_iteration_boundaries():
    source = (
        ROOT / "deepspeed_qwen3_moe_pretrain.py"
    ).read_text(encoding="utf-8")

    phases = [
        "train_barrier_start",
        "train_barrier_done",
        "first_iteration_start",
        "first_iteration_done",
    ]
    offsets = [source.index(f'"{phase}"') for phase in phases]

    assert offsets == sorted(offsets)


def test_deepspeed_rank_log_directory_creation_is_idempotent():
    source = (
        ROOT
        / "DeepSpeed"
        / "deepspeed"
        / "launcher"
        / "launch.py"
    ).read_text(encoding="utf-8")

    assert "os.makedirs(args.enable_each_rank_log, exist_ok=True)" in source


def test_runtime_hooks_common_optimizer_boundary():
    source = (
        ROOT
        / "deepspeed_adapter"
        / "moegambit_deepspeed"
        / "integration.py"
    ).read_text(encoding="utf-8")

    assert "engine._take_model_step" in source
    assert "engine.step = wrapped_step" not in source
    assert "runtime._maybe_checkpoint(after)" in source


def test_hybrid_restore_runs_after_checkpoint_and_has_no_later_phase():
    integration = (
        ROOT
        / "deepspeed_adapter"
        / "moegambit_deepspeed"
        / "integration.py"
    ).read_text(encoding="utf-8")
    hybrid = (
        ROOT
        / "deepspeed_adapter"
        / "moegambit_deepspeed"
        / "hybrid_restore.py"
    ).read_text(encoding="utf-8")

    assert integration.index("self.engine.load_checkpoint(") < (
        integration.index("self._restore_non_expert_from_peer()")
    )
    assert '"expert_source": "checkpoint"' in hybrid
    assert '"non_expert_source": "live_dp_peer"' in hybrid
    assert '"peer_state_origin": "checkpoint_relaunch"' in hybrid
    assert '"optimizer_source": "checkpoint"' in hybrid
    assert '"two_phase": False' in hybrid


def test_runtime_checkpoint_hook_runs_for_common_model_step(tmp_path):
    from moegambit_deepspeed.integration import (
        DeepSpeedRecoveryRuntime,
        DeepSpeedRuntimeSettings,
    )

    class FakeEngine:
        def __init__(self):
            self.global_steps = 0
            self.saved = []

        def _take_model_step(self, *args, **kwargs):
            self.global_steps += 1
            return self.global_steps

        def save_checkpoint(self, path, tag, client_state):
            self.saved.append((path, tag, client_state))

    engine = FakeEngine()
    settings = DeepSpeedRuntimeSettings(
        hot_swap=True,
        zero2=False,
        application_checkpoint=False,
        checkpoint_dir=tmp_path,
        checkpoint_interval=2,
        restart_count=0,
        recovery_epoch=0,
        replica_timeout=1.0,
    )
    runtime = DeepSpeedRecoveryRuntime(engine, settings)
    runtime._install_optimizer_step_hook()

    engine._take_model_step(None)
    engine._take_model_step(None)

    assert engine.global_steps == 2
    assert engine.saved == [
        (
            str(tmp_path),
            "global_step2",
            {
                "moegambit_checkpoint_step": 2,
                "moegambit_recovery_epoch": 0,
            },
        )
    ]
    runtime.close()


def test_hybrid_restore_classifies_autoep_experts_without_name_heuristics():
    from moegambit_deepspeed.hybrid_restore import (
        is_expert_parameter,
        non_expert_model_tensors,
    )

    class Tensor:
        def __init__(self):
            self.data = self

    class Model:
        def __init__(self):
            self.dense = Tensor()
            self.unusually_named = Tensor()
            self.unusually_named.ds_zero_placement_family = "autoep_expert"
            self.running_value = Tensor()
            self.temporary = Tensor()

        def named_parameters(self):
            return [
                ("dense", self.dense),
                ("unusually_named", self.unusually_named),
            ]

        def named_buffers(self):
            return [
                ("running_value", self.running_value),
                ("temporary", self.temporary),
            ]

        def state_dict(self):
            return {
                "dense": self.dense,
                "unusually_named": self.unusually_named,
                "running_value": self.running_value,
            }

    model = Model()
    tensors = dict(non_expert_model_tensors(model))

    assert not is_expert_parameter(model.dense)
    assert is_expert_parameter(model.unusually_named)
    assert set(tensors) == {"dense", "running_value"}


class _FakeTopology:
    axes = ("pipe", "data")

    def __init__(self, mapping):
        self.mapping = mapping

    def get_coord(self, rank):
        pipe, data = self.mapping[rank]
        return type("Coord", (), {"pipe": pipe, "data": data})()

    def get_axis_names(self):
        return self.axes

    def filter_match(self, **filters):
        return [
            rank
            for rank, (pipe, data) in self.mapping.items()
            if all(
                {"pipe": pipe, "data": data}[axis] == value
                for axis, value in filters.items()
            )
        ]


def test_hybrid_restore_requires_a_live_dp_peer_outside_replacement_node():
    from moegambit_deepspeed.hybrid_restore import (
        DeepSpeedHybridRestoreError,
        build_peer_restore_plans,
    )

    # Default DeepSpeed PP-major layout puts every DP copy of stage 0 on
    # ranks 0..3. Replacing that physical node leaves no live stage-0 donor.
    pp_major = _FakeTopology(
        {
            rank: (rank // 4, rank % 4)
            for rank in range(8)
        }
    )
    with pytest.raises(DeepSpeedHybridRestoreError, match="no live DP peer"):
        build_peer_restore_plans(pp_major, range(4))


def test_hybrid_restore_data_major_layout_selects_same_stage_donor():
    from moegambit_deepspeed.hybrid_restore import (
        build_peer_restore_plans,
    )

    # Data-major layout gives each two-GPU physical node a complete pipeline.
    data_major = _FakeTopology(
        {
            rank: (rank % 2, rank // 2)
            for rank in range(8)
        }
    )
    plans = build_peer_restore_plans(data_major, range(2))

    assert [
        (plan.replacement_rank, plan.source_rank)
        for plan in plans
    ] == [(0, 2), (1, 3)]


def test_hybrid_restore_rejects_a_stale_peer_version():
    from moegambit_deepspeed.hybrid_restore import (
        DeepSpeedHybridRestoreError,
        PeerRestorePlan,
        _validate_peer_header,
    )

    plan = PeerRestorePlan(0, 2, (0, 2, 4, 6))
    manifest = [
        {
            "name": "dense",
            "shape": [2],
            "dtype": "torch.bfloat16",
            "numel": 2,
        }
    ]
    with pytest.raises(
        DeepSpeedHybridRestoreError, match="incompatible"
    ):
        _validate_peer_header(
            {
                "source_rank": 2,
                "replacement_rank": 0,
                "step": 10,
                "manifest_hash": "hash",
                "manifest": manifest,
            },
            plan=plan,
            expected_step=17,
            manifest=manifest,
            manifest_hash="hash",
        )


def test_hybrid_restore_uses_common_relaunch_checkpoint_version():
    from moegambit_deepspeed.hybrid_restore import (
        validate_relaunch_checkpoint_steps,
    )

    assert (
        validate_relaunch_checkpoint_steps([10] * 64, failure_step=17) == 10
    )


def test_hybrid_restore_rejects_mixed_relaunch_checkpoint_versions():
    from moegambit_deepspeed.hybrid_restore import (
        DeepSpeedHybridRestoreError,
        validate_relaunch_checkpoint_steps,
    )

    with pytest.raises(
        DeepSpeedHybridRestoreError,
        match="different checkpoint versions",
    ):
        validate_relaunch_checkpoint_steps(
            [10, 10, 15, 10], failure_step=17
        )


def test_hybrid_restore_rejects_checkpoint_newer_than_failure():
    from moegambit_deepspeed.hybrid_restore import (
        DeepSpeedHybridRestoreError,
        validate_relaunch_checkpoint_steps,
    )

    with pytest.raises(
        DeepSpeedHybridRestoreError,
        match="newer than the recorded failure",
    ):
        validate_relaunch_checkpoint_steps([20] * 4, failure_step=17)


def test_local_adapter_discovers_bsr_vendored_deepspeed():
    from moegambit.core.contracts import FeatureSwitches
    from moegambit.interfaces import LaunchRequest
    from moegambit_deepspeed import DeepSpeedAdapter

    prepared = DeepSpeedAdapter().prepare_launch(
        LaunchRequest(
            command=("python", "train.py"),
            environment={},
            features=FeatureSwitches(hot_swap=False, zero2=True),
        )
    )
    python_paths = prepared.environment["PYTHONPATH"].split(os.pathsep)

    assert str(ROOT / "deepspeed_adapter") in python_paths
    assert str(ROOT / "DeepSpeed") in python_paths
    assert prepared.environment["MOEGAMBIT_DEEPSPEED_ROOT"] == str(
        ROOT / "DeepSpeed"
    )


def test_deepspeed_adapter_does_not_inject_torchelastic_for_external_spare():
    from moegambit.core.contracts import FeatureSwitches
    from moegambit.interfaces import LaunchRequest
    from moegambit_deepspeed import DeepSpeedAdapter

    prepared = DeepSpeedAdapter().prepare_launch(
        LaunchRequest(
            command=(
                "python",
                "-m",
                "deepspeed.launcher.runner",
                "train.py",
            ),
            environment={
                "MOEGAMBIT_DEEPSPEED_EXTERNAL_ELASTIC": "1",
                "MOEGAMBIT_DEEPSPEED_CHECKPOINT_DIR": "/tmp/checkpoint",
                "MOEGAMBIT_DEEPSPEED_CHECKPOINT_INTERVAL": "10",
                "MOEGAMBIT_DEEPSPEED_RECOVERY_STRATEGY": (
                    "node_hot_spare_epoch_relaunch"
                ),
            },
            features=FeatureSwitches(hot_swap=True, zero2=False),
        )
    )

    assert "--elastic_training" not in prepared.command
    assert (
        prepared.metadata["recovery_strategy"]
        == "node_hot_spare_epoch_relaunch"
    )


def test_zero2_replica_slot_count_is_configurable():
    manager = (
        ROOT
        / "deepspeed_adapter"
        / "moegambit"
        / "runtime"
        / "zero2_replica.py"
    ).read_text(encoding="utf-8")
    adapter = (
        ROOT
        / "deepspeed_adapter"
        / "moegambit_deepspeed"
        / "zero2.py"
    ).read_text(encoding="utf-8")

    assert "buffer_slots: int = 2" in manager
    assert "range(self.buffer_slots)" in manager
    assert "MOEGAMBIT_ZERO2_BUFFER_SLOTS" in adapter


def test_multinode_script_dry_run_builds_real_commands(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "TEST_MODE": "all",
            "MASTER_ADDR": "10.0.0.1",
            "MASTER_PORT": "23991",
            "MOEGAMBIT_HOT_SPARE_COORDINATOR_ADDR": "10.0.0.9",
            "RUN_ROOT": str(tmp_path),
            "RESET_RUN": "0",
        }
    )
    result = subprocess.run(
        ["bash", str(ROOT / "test_deepspeed_hotspare_replace.sh")],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--pipeline-parallel-size 8" in result.stdout
    assert "--zero-stage 1" in result.stdout
    assert "--pipeline-parallel-size 1" in result.stdout
    assert "--zero-stage 2" in result.stdout
    assert "moegambit.runtime.hot_spare" in result.stdout
    assert "node_hot_spare_epoch_relaunch" not in result.stderr
    assert "--spare-node 8" in result.stdout
    assert "--node_rank \\{logical_node\\}" in result.stdout
    assert "--elastic_training" not in result.stdout
    assert "dry run complete" in result.stdout
    assert "hybrid_restore=1" in result.stdout
    assert "LOCAL_WORLD_SIZE" in (
        ROOT / "test_deepspeed_hotspare_replace.sh"
    ).read_text(encoding="utf-8")


def test_hot_spare_coordinator_replaces_failed_logical_node(tmp_path):
    from moegambit.runtime.hot_spare import HotSpareCoordinator
    from moegambit.runtime.protocol import WireMessage

    coordinator = HotSpareCoordinator(
        run_id="test-run",
        training_nodes=2,
        spare_physical_node=2,
        base_master_port=24000,
        heartbeat_timeout=30,
        state_path=tmp_path / "state.json",
    )

    def request(kind, physical_node, **payload):
        return coordinator.handle(
            WireMessage(
                kind,
                {
                    "run_id": "test-run",
                    "physical_node": physical_node,
                    **payload,
                },
            )
        ).payload

    assert request(
        "register", 0, role="active", advertise_addr="10.0.0.1"
    )["action"] == "wait"
    assert request(
        "register", 1, role="active", advertise_addr="10.0.0.2"
    )["action"] == "wait"
    spare = request(
        "register", 2, role="standby", advertise_addr="10.0.0.3"
    )
    assert spare["action"] == "standby"
    assert request("poll", 0)["logical_node"] == 0
    request("worker_ready", 0, epoch=0, logical_node=0)
    request("worker_ready", 1, epoch=0, logical_node=1)
    assert coordinator.ready_logical_nodes == {0, 1}

    replacement = request(
        "rank_failure",
        0,
        epoch=0,
        logical_node=0,
        rank=1,
        reason="injected_sigkill",
        global_step=17,
    )
    assert replacement["action"] == "retire"
    assert coordinator.epoch == 1
    assert coordinator.mapping == {0: 2, 1: 1}
    replacement_command = request("poll", 2)
    assert replacement_command["logical_node"] == 0
    assert replacement_command["master_addr"] == "10.0.0.3"
    assert replacement_command["master_port"] == 24001
    assert replacement_command["failed_logical_node"] == 0
    assert replacement_command["failure_step"] == 17
    assert request("poll", 1)["logical_node"] == 1
    assert (tmp_path / "state.json").is_file()


def test_hot_spare_coordinator_tracks_recovery_worker_phases(tmp_path):
    from moegambit.runtime.hot_spare import HotSpareCoordinator
    from moegambit.runtime.protocol import WireMessage

    coordinator = HotSpareCoordinator(
        run_id="test-run",
        training_nodes=2,
        spare_physical_node=2,
        base_master_port=24000,
        recovery_timeout=0.01,
        state_path=tmp_path / "state.json",
    )

    def request(kind, physical_node, **payload):
        return coordinator.handle(
            WireMessage(
                kind,
                {
                    "run_id": "test-run",
                    "physical_node": physical_node,
                    **payload,
                },
            )
        ).payload

    request("register", 0, role="active", advertise_addr="10.0.0.1")
    request("register", 1, role="active", advertise_addr="10.0.0.2")
    request("register", 2, role="standby", advertise_addr="10.0.0.3")
    request(
        "worker_phase",
        0,
        epoch=0,
        logical_node=0,
        phase="engine_init_done",
    )
    assert coordinator.worker_phases == {0: "engine_init_done"}

    request(
        "rank_failure",
        0,
        epoch=0,
        logical_node=0,
        rank=1,
    )
    assert coordinator.worker_phases == {}
    request(
        "worker_phase",
        2,
        epoch=1,
        logical_node=0,
        phase="checkpoint_restore_start",
    )
    request(
        "worker_phase",
        1,
        epoch=1,
        logical_node=1,
        phase="train_ready",
    )
    coordinator.recovery_started_at -= 1
    aborted = request("heartbeat", 1, epoch=1)

    assert aborted["action"] == "abort"
    assert "checkpoint_restore_start" in aborted["reason"]
    persisted = json.loads(
        (tmp_path / "state.json").read_text(encoding="utf-8")
    )
    assert persisted["worker_phases"] == {
        "0": "checkpoint_restore_start",
        "1": "train_ready",
    }


def test_hot_spare_coordinator_fails_closed_without_second_spare():
    from moegambit.runtime.hot_spare import HotSpareCoordinator
    from moegambit.runtime.protocol import WireMessage

    coordinator = HotSpareCoordinator(
        run_id="test-run",
        training_nodes=2,
        spare_physical_node=2,
        base_master_port=24000,
    )

    def request(kind, physical_node, **payload):
        return coordinator.handle(
            WireMessage(
                kind,
                {
                    "run_id": "test-run",
                    "physical_node": physical_node,
                    **payload,
                },
            )
        ).payload

    request("register", 0, role="active", advertise_addr="10.0.0.1")
    request("register", 1, role="active", advertise_addr="10.0.0.2")
    request("register", 2, role="standby", advertise_addr="10.0.0.3")
    request(
        "rank_failure",
        0,
        epoch=0,
        logical_node=0,
        rank=1,
    )
    aborted = request(
        "runner_failure",
        1,
        epoch=1,
        reason="second_failure",
    )

    assert aborted["action"] == "abort"
    assert coordinator.status == "aborted"


def test_hot_spare_does_not_mask_initial_program_failure():
    from moegambit.runtime.hot_spare import HotSpareCoordinator
    from moegambit.runtime.protocol import WireMessage

    coordinator = HotSpareCoordinator(
        run_id="test-run",
        training_nodes=2,
        spare_physical_node=2,
        base_master_port=24000,
    )

    def request(kind, physical_node, **payload):
        return coordinator.handle(
            WireMessage(
                kind,
                {
                    "run_id": "test-run",
                    "physical_node": physical_node,
                    **payload,
                },
            )
        ).payload

    request("register", 0, role="active", advertise_addr="10.0.0.1")
    request("register", 1, role="active", advertise_addr="10.0.0.2")
    request("register", 2, role="standby", advertise_addr="10.0.0.3")
    aborted = request(
        "runner_failure",
        1,
        epoch=0,
        return_code=2,
        reason="argument_error",
    )

    assert aborted["action"] == "abort"
    assert coordinator.epoch == 0
    assert coordinator.mapping == {0: 0, 1: 1}
    assert "refusing to consume the hot spare" in aborted["reason"]


def test_hot_spare_surfaces_rank_failure_artifact(tmp_path):
    from moegambit.runtime.hot_spare import (
        collect_worker_failure_diagnostic,
    )

    state_dir = tmp_path / "state"
    error_dir = state_dir / "errors"
    error_dir.mkdir(parents=True)
    artifact = error_dir / "epoch_0_rank_17.json"
    artifact.write_text(
        (
            '{"error":"RuntimeError: test failure",'
            '"traceback":"Traceback\\nRuntimeError: test failure"}'
        ),
        encoding="utf-8",
    )
    diagnostic = collect_worker_failure_diagnostic(
        (
            "python",
            "-m",
            "deepspeed.launcher.runner",
            "--num_gpus",
            "8",
            "train.py",
            "--state-dir",
            str(state_dir),
        ),
        epoch=0,
        logical_node=2,
        process_started_at=None,
    )

    assert "epoch_0_rank_17.json" in diagnostic
    assert "RuntimeError: test failure" in diagnostic


def test_hot_spare_prefetch_selects_replacement_checkpoint_shards(tmp_path):
    from moegambit.runtime.hot_spare import AgentSupervisor
    from moegambit.runtime.watcher_client import WatcherEndpoint

    checkpoint_dir = tmp_path / "checkpoint"
    tag_dir = checkpoint_dir / "global_step10"
    tag_dir.mkdir(parents=True)
    (checkpoint_dir / "latest").write_text(
        "global_step10\n", encoding="utf-8"
    )
    names = (
        "mp_rank_00_model_states.pt",
        "layer_0_expert_0_mp_rank_00_model_states.pt",
        "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt",
        "bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt",
        "other.txt",
    )
    for name in names:
        (tag_dir / name).write_bytes(b"x")

    supervisor = AgentSupervisor(
        endpoint=WatcherEndpoint("127.0.0.1", 1),
        run_id="test-run",
        physical_node=2,
        role="standby",
        advertise_addr="127.0.0.1",
        command=("train.py",),
        heartbeat_interval=1,
        startup_timeout=1,
    )
    tag, selected = supervisor._checkpoint_prefetch_files(
        checkpoint_dir, logical_node=0
    )

    assert tag == "global_step10"
    assert {path.name for path in selected} == {
        "mp_rank_00_model_states.pt",
        "layer_0_expert_0_mp_rank_00_model_states.pt",
        "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt",
    }


def test_hot_spare_relays_local_rank_zero_log_incrementally(
    tmp_path, capsys, monkeypatch
):
    from moegambit.runtime.hot_spare import AgentSupervisor
    from moegambit.runtime.watcher_client import WatcherEndpoint

    monkeypatch.setenv("MOEGAMBIT_RELAY_RANK_LOG", "1")
    log_dir = tmp_path / "rank_logs"
    log_dir.mkdir()
    rank_log = log_dir / "20260727120000_rank16.log"
    rank_log.write_text("first\n", encoding="utf-8")
    supervisor = AgentSupervisor(
        endpoint=WatcherEndpoint("127.0.0.1", 1),
        run_id="test-run",
        physical_node=2,
        role="active",
        advertise_addr="127.0.0.1",
        command=(
            "runner",
            "--num_gpus",
            "8",
            "--enable_each_rank_log",
            str(log_dir),
        ),
        heartbeat_interval=1,
        startup_timeout=1,
    )
    supervisor.process_logical_node = 2

    supervisor._relay_worker_log()
    with rank_log.open("a", encoding="utf-8") as stream:
        stream.write("second\n")
    supervisor._relay_worker_log()

    assert capsys.readouterr().out.splitlines() == [
        "[worker-rank16] first",
        "[worker-rank16] second",
    ]


def test_hot_spare_key_relay_keeps_progress_and_suppresses_noise(
    tmp_path, capsys, monkeypatch
):
    from moegambit.runtime.hot_spare import AgentSupervisor
    from moegambit.runtime.watcher_client import WatcherEndpoint

    monkeypatch.setenv("MOEGAMBIT_RELAY_RANK_LOG", "key")
    log_dir = tmp_path / "rank_logs" / "epoch_1" / "node_8"
    log_dir.mkdir(parents=True)
    rank_log = log_dir / "20260727120000_rank0.log"
    rank_log.write_text(
        "NCCL_IB_HCA=mlx5_bond\n"
        "[deepspeed-real] iteration 18/100 loss=1.0\n"
        "MoEGambit single-stage hybrid restore complete\n",
        encoding="utf-8",
    )
    supervisor = AgentSupervisor(
        endpoint=WatcherEndpoint("127.0.0.1", 1),
        run_id="test-run",
        physical_node=8,
        role="active",
        advertise_addr="127.0.0.1",
        command=("runner", "--num_gpus", "8"),
        heartbeat_interval=1,
        startup_timeout=1,
    )
    supervisor.process_command = (
        "runner",
        "--num_gpus",
        "8",
        "--enable_each_rank_log",
        str(log_dir),
    )
    supervisor.process_logical_node = 0

    supervisor._relay_worker_log()

    output = capsys.readouterr().out
    assert "NCCL_IB_HCA" not in output
    assert "iteration 18/100" in output
    assert "single-stage hybrid restore complete" in output


def test_hot_spare_formats_rank_logs_by_epoch_and_physical_node():
    from moegambit.runtime.hot_spare import AgentSupervisor
    from moegambit.runtime.watcher_client import WatcherEndpoint

    supervisor = AgentSupervisor(
        endpoint=WatcherEndpoint("127.0.0.1", 1),
        run_id="test-run",
        physical_node=8,
        role="active",
        advertise_addr="127.0.0.1",
        command=(
            "runner",
            "--enable_each_rank_log",
            "/tmp/logs/epoch_{recovery_epoch}/node_{physical_node}",
        ),
        heartbeat_interval=1,
        startup_timeout=1,
    )

    command = supervisor._formatted_command(
        logical_node=0,
        epoch=1,
        master_addr="10.0.0.1",
        master_port=24001,
    )

    assert command[-1] == "/tmp/logs/epoch_1/node_8"


def test_hot_spare_heartbeat_runs_independently_of_supervision():
    from moegambit.runtime.hot_spare import AgentSupervisor
    from moegambit.runtime.watcher_client import WatcherEndpoint

    supervisor = AgentSupervisor(
        endpoint=WatcherEndpoint("127.0.0.1", 1),
        run_id="test-run",
        physical_node=3,
        role="active",
        advertise_addr="127.0.0.1",
        command=("runner",),
        heartbeat_interval=0.01,
        startup_timeout=1,
    )
    received = threading.Event()
    calls = []

    def request(kind, **payload):
        calls.append((kind, payload))
        received.set()
        return {"action": "run"}

    supervisor._request = request
    supervisor._start_heartbeat()
    try:
        assert received.wait(timeout=1.0)
    finally:
        supervisor._stop_heartbeat()

    assert calls[0][0] == "heartbeat"
    assert calls[0][1]["state"] == "active"


def test_hot_spare_coordinator_aborts_stalled_recovery():
    from moegambit.runtime.hot_spare import HotSpareCoordinator
    from moegambit.runtime.protocol import WireMessage

    coordinator = HotSpareCoordinator(
        run_id="test-run",
        training_nodes=2,
        spare_physical_node=2,
        base_master_port=24000,
        recovery_timeout=0.01,
    )

    def request(kind, physical_node, **payload):
        return coordinator.handle(
            WireMessage(
                kind,
                {
                    "run_id": "test-run",
                    "physical_node": physical_node,
                    **payload,
                },
            )
        ).payload

    request("register", 0, role="active", advertise_addr="10.0.0.1")
    request("register", 1, role="active", advertise_addr="10.0.0.2")
    request("register", 2, role="standby", advertise_addr="10.0.0.3")
    request(
        "rank_failure",
        0,
        epoch=0,
        logical_node=0,
        rank=1,
    )
    coordinator.recovery_started_at -= 1
    aborted = request("heartbeat", 1, epoch=1)

    assert aborted["action"] == "abort"
    assert "did not reach TRAIN_READY" in aborted["reason"]


def test_hot_spare_supervisors_execute_recovery_epoch(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        """
import os
import time
from moegambit.runtime.protocol import WireMessage
from moegambit.runtime.watcher_client import WatcherClient, WatcherEndpoint

epoch = int(os.environ["MOEGAMBIT_RECOVERY_EPOCH"])
physical = int(os.environ["MOEGAMBIT_PHYSICAL_NODE_RANK"])
logical = int(os.environ["MOEGAMBIT_LOGICAL_NODE_RANK"])
WatcherClient(
    WatcherEndpoint(
        os.environ["MOEGAMBIT_HOT_SPARE_COORDINATOR_ADDR"],
        int(os.environ["MOEGAMBIT_HOT_SPARE_COORDINATOR_PORT"]),
    )
).request(
    WireMessage(
        "worker_ready",
        {
            "run_id": os.environ["MOEGAMBIT_HOT_SPARE_RUN_ID"],
            "physical_node": physical,
            "logical_node": logical,
            "epoch": epoch,
        },
    )
)
if epoch == 0 and physical == 0:
    time.sleep(0.3)
    raise SystemExit(23)
if epoch == 0:
    time.sleep(10)
time.sleep(0.2)
""".lstrip(),
        encoding="utf-8",
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    executable = [
        sys.executable,
        "-u",
        "-m",
        "moegambit.runtime.hot_spare",
    ]
    common_options = [
        "--coordinator-host",
        "127.0.0.1",
        "--coordinator-port",
        str(port),
        "--run-id",
        "supervisor-test",
        "--training-nodes",
        "2",
        "--spare-node",
        "2",
        "--base-master-port",
        "25000",
        "--heartbeat-interval",
        "0.05",
        "--heartbeat-timeout",
        "2",
        "--startup-timeout",
        "5",
    ]
    worker_command = ["--", sys.executable, str(worker)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT / "deepspeed_adapter"),
            env.get("PYTHONPATH", ""),
        ]
    )
    env["MOEGAMBIT_HOT_SPARE_COORDINATOR_ADDR"] = "127.0.0.1"
    env["MOEGAMBIT_HOT_SPARE_COORDINATOR_PORT"] = str(port)

    processes = [
        subprocess.Popen(
            [
                *executable,
                "--mode",
                "coordinator-agent",
                *common_options,
                "--physical-node",
                "2",
                *worker_command,
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    ]
    for physical in (0, 1):
        processes.append(
            subprocess.Popen(
                [
                    *executable,
                    "--mode",
                    "agent",
                    *common_options,
                    "--physical-node",
                    str(physical),
                    *worker_command,
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )

    results = [process.communicate(timeout=15) for process in processes]
    return_codes = [process.returncode for process in processes]
    assert return_codes == [0, 0, 0], results
    combined_logs = "\n".join(
        stdout + stderr for stdout, stderr in results
    )
    assert "starting recovery epoch 1" in combined_logs
    assert "physical_node=2 logical_node=0 epoch=1" in combined_logs


def test_vendored_deepspeed_version_is_pinned():
    metadata = (ROOT / "DeepSpeed" / "MOEGAMBIT_UPSTREAM.md").read_text(
        encoding="utf-8"
    )
    assert "v0.19.3" in metadata
    assert "0c36f6d3efb806c07a44e5c2c9b18a81d204b821" in metadata

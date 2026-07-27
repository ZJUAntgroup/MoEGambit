import ast
import os
import socket
import subprocess
import sys
from pathlib import Path


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

    replacement = request(
        "rank_failure",
        0,
        epoch=0,
        logical_node=0,
        rank=1,
        reason="injected_sigkill",
    )
    assert replacement["action"] == "retire"
    assert coordinator.epoch == 1
    assert coordinator.mapping == {0: 2, 1: 1}
    assert request("poll", 2)["logical_node"] == 0
    assert request("poll", 2)["master_addr"] == "10.0.0.3"
    assert request("poll", 2)["master_port"] == 24001
    assert request("poll", 1)["logical_node"] == 1
    assert (tmp_path / "state.json").is_file()


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

epoch = int(os.environ["MOEGAMBIT_RECOVERY_EPOCH"])
physical = int(os.environ["MOEGAMBIT_PHYSICAL_NODE_RANK"])
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

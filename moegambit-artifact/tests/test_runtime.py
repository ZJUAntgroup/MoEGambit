from itertools import product
from pathlib import Path

from elastic.elastic_watcher import ElasticWatcher
from moegambit.core.contracts import FeatureSwitches
from moegambit.interfaces import LaunchRequest
from moegambit.runtime.config import RuntimeConfig
from moegambit.runtime.discovery import discover_adapters, load_adapter
from moegambit.runtime.distributed import (
    GroupManifest,
    GroupSpec,
    TorchDistributedProtocol,
)
from moegambit.runtime.protocol import WireMessage
from moegambit_megatron import MegatronAdapter


def test_feature_switches_are_independent():
    for hot_swap, zero2 in product((False, True), repeat=2):
        config = RuntimeConfig(
            adapter="megatron",
            features=FeatureSwitches(hot_swap=hot_swap, zero2=zero2),
        )
        env = config.project_environment({})
        assert env["MOEGAMBIT_HOT_SWAP"] == ("1" if hot_swap else "0")
        assert env["MOEGAMBIT_ZERO2"] == ("1" if zero2 else "0")
        assert env["ELASTIC_HOT_SWAP_ENABLED"] == (
            "1" if hot_swap else "0"
        )
        assert env["ELASTIC_ZERO2_MEMORY_REPLICATION"] == (
            "1" if zero2 else "0"
        )


def test_disabling_hot_swap_removes_one_shot_recovery_state():
    config = RuntimeConfig(
        adapter="megatron",
        features=FeatureSwitches(hot_swap=False, zero2=True),
    )
    env = config.project_environment(
        {
            "ELASTIC_REBUILD_MODE": "1",
            "ELASTIC_RECOVERY_EPOCH": "9",
            "UNRELATED": "kept",
        }
    )
    assert "ELASTIC_REBUILD_MODE" not in env
    assert "ELASTIC_RECOVERY_EPOCH" not in env
    assert env["UNRELATED"] == "kept"


def test_megatron_adapter_applies_only_enabled_features():
    adapter = MegatronAdapter()
    request = LaunchRequest(
        command=(
            "python",
            "pretrain_gpt.py",
            "--moe-moegambit-hot-spare-pool",
            "--moe-moegambit-num-hot-spares",
            "8",
        ),
        environment={},
        features=FeatureSwitches(hot_swap=False, zero2=True),
    )

    prepared = adapter.prepare_launch(request)

    assert "--moe-moegambit-hot-spare-pool" not in prepared.command
    assert "--moe-moegambit-num-hot-spares" not in prepared.command
    assert "8" not in prepared.command
    assert prepared.command[-1] == "--use-distributed-optimizer"
    assert prepared.environment[
        "ELASTIC_ZERO2_USE_DISTRIBUTED_OPTIMIZER"
    ] == "1"


def test_megatron_watcher_backend_contains_spare_launcher():
    adapter = MegatronAdapter()
    command = adapter.watcher_command(
        FeatureSwitches(hot_swap=True, zero2=False), ()
    )
    assert command is not None
    assert Path(command[1]).name == "elastic_watcher.py"

    spare_command = ElasticWatcher._script_cmd(object())
    assert spare_command[0] == "bash"
    assert Path(spare_command[1]).is_file()


def test_discovery_finds_entry_point_adapters():
    adapters = discover_adapters()
    assert {"megatron", "deepspeed"} <= set(adapters)
    assert load_adapter(
        "auto", ("python", "Megatron-LM/pretrain_gpt.py")
    ).name == "megatron"
    assert load_adapter("auto", ("deepspeed", "train.py")).name == "deepspeed"


def test_wire_protocol_round_trip():
    message = WireMessage("heartbeat", {"rank": 3, "step": 17})
    assert WireMessage.decode(message.encode()) == message


def test_group_manifest_is_created_in_global_ordinal_order():
    class FakeDistributed:
        def __init__(self):
            self.calls = []

        def is_initialized(self):
            return True

        def new_group(self, **kwargs):
            self.calls.append(kwargs)
            return kwargs["ranks"]

    class RecordingBarrier:
        def __init__(self):
            self.calls = []

        def wait(self, phase, group, fingerprint):
            self.calls.append((phase, group.ordinal, fingerprint))

    manifest = GroupManifest.build(
        [
            GroupSpec(1, "ep", (1, 3)),
            GroupSpec(0, "dp", (0, 2)),
        ]
    )
    distributed = FakeDistributed()
    barrier = RecordingBarrier()
    created = TorchDistributedProtocol(distributed).create_groups(
        manifest, barrier=barrier, timeout_seconds=30
    )

    assert list(created) == ["dp", "ep"]
    assert [call["ranks"] for call in distributed.calls] == [[0, 2], [1, 3]]
    assert [(phase, ordinal) for phase, ordinal, _ in barrier.calls] == [
        ("start", 0),
        ("done", 0),
        ("start", 1),
        ("done", 1),
    ]
    assert len({fingerprint for _, _, fingerprint in barrier.calls}) == 1

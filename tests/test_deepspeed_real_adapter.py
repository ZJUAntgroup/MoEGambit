import ast
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import types
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
    assert "exit_code = main()" in source
    assert "raise SystemExit(main())" not in source


def test_resident_and_gpu_build_optimizations_are_recovery_scoped():
    workload = (
        ROOT / "deepspeed_qwen3_moe_pretrain.py"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT / "test_deepspeed_hotspare_replace.sh"
    ).read_text(encoding="utf-8")
    hot_spare = (
        ROOT
        / "deepspeed_adapter"
        / "moegambit"
        / "runtime"
        / "hot_spare.py"
    ).read_text(encoding="utf-8")

    assert "recovery_epoch > 0" in workload
    assert "MOEGAMBIT_RECOVERY_GPU_MODEL_BUILD" in workload
    assert "MOEGAMBIT_STANDBY_RESIDENT" in launcher
    assert "MOEGAMBIT_RECOVERY_GPU_MODEL_BUILD" in launcher
    assert "MOEGAMBIT_RECOVERY_FORCE_PREEMPT" in launcher
    assert 'self.role == "standby"' in hot_spare
    assert "standby_cache_for(args)" in workload


def test_packed_expert_checkpoint_and_pinned_cache_are_recovery_scoped():
    workload = (
        ROOT / "deepspeed_qwen3_moe_pretrain.py"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT / "test_deepspeed_hotspare_replace.sh"
    ).read_text(encoding="utf-8")
    engine = (
        ROOT / "DeepSpeed" / "deepspeed" / "runtime" / "engine.py"
    ).read_text(encoding="utf-8")

    assert "DEEPSPEED_MOEGAMBIT_PACKED_EXPERT_CHECKPOINT" in launcher
    assert "MOEGAMBIT_STANDBY_PACKED_EXPERT_CACHE" in launcher
    assert 'if [[ "${hot_swap}" == "1" ]]' in launcher
    assert "PackedExpertPrefetcher" in workload
    assert "activate_standby" in workload
    assert "build_packed_expert_state" in engine
    assert "take_cached_packed_expert" in engine
    assert "torch.stack(stacked[wname], dim=0)" in engine
    save_start = engine.index("    def _save_moe_checkpoint(")
    save_end = engine.index(
        "    def _create_checkpoint_file(", save_start
    )
    save_moe = engine[save_start:save_end]
    assert "packed_autoep_checkpoint = (" in save_moe
    assert "packed_mp_rank =" in save_moe


def test_standby_packed_expert_coordinates_cover_pp_and_ep_layouts():
    source = (
        ROOT / "deepspeed_qwen3_moe_pretrain.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "standby_packed_expert_coordinates"
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            "deepspeed_qwen3_moe_pretrain.py",
            "exec",
        ),
        namespace,
    )
    coordinates = namespace["standby_packed_expert_coordinates"]

    assert [
        coordinates(0, local_rank, 8, 8, 8)
        for local_rank in range(8)
    ] == [
        (local_rank, local_rank, 0, 0)
        for local_rank in range(8)
    ]
    assert [
        coordinates(7, local_rank, 8, 8, 8)
        for local_rank in range(8)
    ] == [
        (56 + local_rank, local_rank, 7, 7)
        for local_rank in range(8)
    ]
    assert [
        coordinates(0, local_rank, 8, 1, 8)
        for local_rank in range(8)
    ] == [
        (local_rank, 0, local_rank, local_rank)
        for local_rank in range(8)
    ]


def test_packed_expert_prefetch_hands_state_to_loader(tmp_path, monkeypatch):
    class FakeTensor:
        def __init__(self, shape):
            self.shape = tuple(shape)
            self.device = types.SimpleNamespace(type="cpu")
            self.pinned = False

        def dim(self):
            return len(self.shape)

        def numel(self):
            result = 1
            for value in self.shape:
                result *= value
            return result

        @staticmethod
        def element_size():
            return 2

        @staticmethod
        def is_contiguous():
            return True

        def is_pinned(self):
            return self.pinned

        def pin_memory(self):
            self.pinned = True
            return self

    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = FakeTensor
    states = {}
    fake_torch.load = lambda path, **_kwargs: states[Path(path).name]
    fake_constants = types.ModuleType("deepspeed.checkpoint.constants")
    fake_constants.FOLDING_METADATA_KEY = "folding"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "deepspeed", types.ModuleType("deepspeed"))
    monkeypatch.setitem(
        sys.modules,
        "deepspeed.checkpoint",
        types.ModuleType("deepspeed.checkpoint"),
    )
    monkeypatch.setitem(
        sys.modules,
        "deepspeed.checkpoint.constants",
        fake_constants,
    )

    path = (
        ROOT
        / "DeepSpeed"
        / "deepspeed"
        / "checkpoint"
        / "packed_moe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_packed_moe_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    checkpoint_dir = tmp_path / "checkpoint"
    tag = "global_step10"
    tag_dir = checkpoint_dir / tag
    tag_dir.mkdir(parents=True)
    packed_path = Path(
        module.packed_expert_checkpoint_name(
            checkpoint_dir,
            layer_id=0,
            ep_rank=2,
            mp_rank=3,
            tag=tag,
        )
    )
    packed_path.write_bytes(b"packed")
    states[packed_path.name] = module.build_packed_expert_state(
        layer_id=0,
        ep_rank=2,
        mp_rank=3,
        module_path="layers.0.mlp",
        global_expert_start=4,
        num_local_experts=2,
        tensors={
            "layers.0.mlp.experts.w1": FakeTensor((2, 4, 8)),
            "layers.0.mlp.experts.w2": FakeTensor((2, 8, 4)),
            "layers.0.mlp.experts.w3": FakeTensor((2, 4, 8)),
        },
    )
    (checkpoint_dir / "latest").write_text(
        f"{tag}\n", encoding="utf-8"
    )

    prefetcher = module.PackedExpertPrefetcher(
        checkpoint_dir,
        mp_rank=3,
        ep_rank=2,
        expected_layers=1,
        max_bytes=1024 * 1024,
        pin_memory=True,
        poll_interval=0.01,
    )
    prefetcher.start()
    deadline = time.monotonic() + 2
    while (
        not prefetcher.snapshot()["completed"]
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    snapshot = prefetcher.stop()

    assert snapshot["completed"] is True
    assert snapshot["files"] == 1
    assert snapshot["pinned_bytes"] == snapshot["bytes"]
    cached = module.take_cached_packed_expert(packed_path)
    assert cached is not None
    tensors = module.validate_packed_expert_state(
        cached,
        layer_id=0,
        ep_rank=2,
        mp_rank=3,
        module_path="layers.0.mlp",
        num_local_experts=2,
    )
    assert set(tensors) == {
        "layers.0.mlp.experts.w1",
        "layers.0.mlp.experts.w2",
        "layers.0.mlp.experts.w3",
    }


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


def test_mixed_restore_does_not_activate_on_plain_torchelastic(
    monkeypatch,
):
    from moegambit_deepspeed.integration import DeepSpeedRuntimeSettings

    monkeypatch.setenv("MOEGAMBIT_HOT_SWAP", "1")
    monkeypatch.setenv(
        "MOEGAMBIT_DEEPSPEED_RECOVERY_STRATEGY",
        "torch_elastic_checkpoint_relaunch",
    )
    monkeypatch.delenv(
        "MOEGAMBIT_DEEPSPEED_HYBRID_RESTORE", raising=False
    )

    settings = DeepSpeedRuntimeSettings.from_env()

    assert settings.hot_swap is True
    assert settings.hybrid_restore is False
    assert settings.survivor_handoff is False


def test_mixed_restore_requires_matching_recovery_strategy(monkeypatch):
    from moegambit_deepspeed.integration import DeepSpeedRuntimeSettings

    monkeypatch.setenv("MOEGAMBIT_HOT_SWAP", "1")
    monkeypatch.setenv(
        "MOEGAMBIT_DEEPSPEED_RECOVERY_STRATEGY",
        "torch_elastic_checkpoint_relaunch",
    )
    monkeypatch.setenv("MOEGAMBIT_DEEPSPEED_HYBRID_RESTORE", "1")

    with pytest.raises(ValueError, match="mixed_version_survivor_handoff"):
        DeepSpeedRuntimeSettings.from_env()


def test_hybrid_restore_uses_current_survivor_after_checkpoint_base_load():
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
        integration.index("self._restore_mixed_version_from_handoff()")
    )
    assert '"expert_source": "checkpoint"' in hybrid
    assert '"non_expert_source": "live_dp_peer"' in hybrid
    assert '"peer_state_origin": "survivor_handoff"' in hybrid
    assert '"optimizer_source": "survivor_peer_replica"' in hybrid
    assert '"rollback_steps": 0' in integration
    assert '"two_phase": False' in hybrid


def test_runtime_checkpoint_hook_runs_for_common_model_step(
    tmp_path, monkeypatch
):
    from moegambit_deepspeed import checkpoint_commit
    from moegambit_deepspeed.integration import (
        DeepSpeedRecoveryRuntime,
        DeepSpeedRuntimeSettings,
    )

    published = []
    monkeypatch.setattr(
        checkpoint_commit,
        "publish_checkpoint",
        lambda engine, path, tag: published.append((engine, path, tag)),
    )

    class FakeEngine:
        def __init__(self):
            self.global_steps = 0
            self.saved = []

        def _take_model_step(self, *args, **kwargs):
            self.global_steps += 1
            return self.global_steps

        def save_checkpoint(
            self, path, tag, client_state, save_latest=True
        ):
            self.saved.append((path, tag, client_state, save_latest))

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
            False,
        )
    ]
    assert published == [(engine, tmp_path, "global_step2")]
    runtime.close()


def test_checkpoint_manifest_requires_every_dense_pipeline_shard(tmp_path):
    from moegambit_deepspeed.checkpoint_commit import (
        build_checkpoint_manifest,
    )

    class FakeEngine:
        mp_world_size = 8
        dp_world_size = 8

        @staticmethod
        def zero_optimization_partition_weights():
            return False

    tag = "global_step10"
    tag_dir = tmp_path / tag
    tag_dir.mkdir()
    for rank in range(4):
        (tag_dir / f"mp_rank_{rank:02d}_model_states.pt").write_bytes(
            b"checkpoint"
        )

    with pytest.raises(RuntimeError, match="dense_shards=4/8"):
        build_checkpoint_manifest(FakeEngine(), tmp_path, tag)


def test_checkpoint_selection_falls_back_to_previous_complete_tag(tmp_path):
    from moegambit_deepspeed.checkpoint_commit import (
        MANIFEST_NAME,
        build_checkpoint_manifest,
        resolve_committed_checkpoint,
    )

    class FakeEngine:
        mp_world_size = 2
        dp_world_size = 1

        @staticmethod
        def zero_optimization_partition_weights():
            return False

    engine = FakeEngine()
    for step in (10, 20):
        tag = f"global_step{step}"
        tag_dir = tmp_path / tag
        tag_dir.mkdir()
        for rank in range(2):
            (tag_dir / f"mp_rank_{rank:02d}_model_states.pt").write_bytes(
                f"{step}-{rank}".encode()
            )
        manifest = build_checkpoint_manifest(engine, tmp_path, tag)
        (tag_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    (tmp_path / "latest").write_text("global_step20\n", encoding="utf-8")
    (tmp_path / "global_step20" / "mp_rank_01_model_states.pt").unlink()

    assert resolve_committed_checkpoint(tmp_path) == "global_step10"


def test_checkpoint_manifest_covers_packed_expert_shards(
    tmp_path, monkeypatch
):
    from moegambit_deepspeed import checkpoint_commit

    class FakeEngine:
        mp_world_size = 1
        dp_world_size = 1

        @staticmethod
        def zero_optimization_partition_weights():
            return False

    tag = "global_step10"
    tag_dir = tmp_path / tag
    tag_dir.mkdir()
    (tag_dir / "mp_rank_00_model_states.pt").write_bytes(b"dense")
    packed = [
        tag_dir
        / (
            f"layer_0_ep_rank_{ep_rank:04d}_mp_rank_00_"
            "packed_expert_states.pt"
        )
        for ep_rank in range(2)
    ]
    for path in packed:
        path.write_bytes(b"packed")

    monkeypatch.setenv(
        "DEEPSPEED_MOEGAMBIT_PACKED_EXPERT_CHECKPOINT", "1"
    )
    monkeypatch.setattr(
        checkpoint_commit,
        "_expected_packed_expert_shards",
        lambda _engine: 2,
    )
    manifest = checkpoint_commit.build_checkpoint_manifest(
        FakeEngine(), tmp_path, tag
    )
    (tag_dir / checkpoint_commit.MANIFEST_NAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    assert manifest["expected_packed_expert_shards"] == 2
    assert len(manifest["packed_expert_shards"]) == 2
    packed[1].unlink()
    with pytest.raises(RuntimeError, match="packed_shards=2/2"):
        checkpoint_commit.validate_checkpoint_manifest(tmp_path, tag)


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


def test_optimizer_replica_ring_places_node_failure_copy_off_node():
    from moegambit.runtime.replica_placement import (
        failure_domain_ring_order,
    )
    from moegambit_deepspeed.hybrid_restore import (
        build_optimizer_peer_restore_plans,
    )

    ring = failure_domain_ring_order(
        range(64), ranks_per_failure_domain=8
    )
    plans = build_optimizer_peer_restore_plans(
        {"ranks-0-63": ring}, range(8)
    )

    assert [
        (plan.replacement_rank, plan.source_rank)
        for plan in plans
    ] == [(rank, rank + 8) for rank in range(8)]
    assert all(
        plan.replacement_rank // 8 != plan.source_rank // 8
        for plan in plans
    )


def test_optimizer_replica_ring_rejects_same_node_only_group():
    from moegambit.runtime.replica_placement import (
        ReplicaPlacementError,
        failure_domain_ring_order,
    )

    with pytest.raises(
        ReplicaPlacementError,
        match="outside its owner's failure domain",
    ):
        failure_domain_ring_order(
            range(8), ranks_per_failure_domain=8
        )


def test_optimizer_restore_rejects_backup_on_failed_node():
    from moegambit_deepspeed.hybrid_restore import (
        DeepSpeedHybridRestoreError,
        build_optimizer_peer_restore_plans,
    )

    with pytest.raises(
        DeepSpeedHybridRestoreError,
        match="backup holder is also being replaced",
    ):
        build_optimizer_peer_restore_plans(
            {"flat": list(range(64))}, range(8)
        )


def test_optimizer_restore_rejects_tensor_manifest_drift():
    import hashlib

    from moegambit_deepspeed.hybrid_restore import (
        DeepSpeedHybridRestoreError,
        OptimizerPeerRestorePlan,
        _validate_optimizer_peer_entry,
    )

    manifest = [
        {
            "identity": "group/0/fp32_master",
            "shape": [16],
            "dtype": "torch.float32",
            "numel": 16,
            "offset": 0,
            "is_expert": False,
        }
    ]
    digest = hashlib.sha256(
        json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    plan = OptimizerPeerRestorePlan(
        namespace="dense",
        replacement_rank=0,
        source_rank=8,
        replica_ring_ranks=(0, 8, 16, 24),
    )
    entry = {
        "namespace": "dense",
        "owner_rank": 0,
        "holder_rank": 8,
        "step": 17,
        "replica_ring_ranks": [0, 8, 16, 24],
        "manifest_hash": digest,
        "manifest": [{**manifest[0], "shape": [8, 2]}],
    }

    with pytest.raises(
        DeepSpeedHybridRestoreError,
        match="optimizer state is incompatible",
    ):
        _validate_optimizer_peer_entry(
            entry,
            plan=plan,
            expected_step=17,
            local_manifest=manifest,
            local_manifest_hash=digest,
        )


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


def test_hybrid_restore_accepts_checkpoint_experts_and_current_survivors():
    from moegambit_deepspeed.hybrid_restore import (
        validate_mixed_version_steps,
    )

    assert (
        validate_mixed_version_steps(
            [10] * 64, [17] * 56, failure_step=17
        )
        == 10
    )


def test_hybrid_restore_rejects_checkpoint_aged_survivor_peer():
    from moegambit_deepspeed.hybrid_restore import (
        DeepSpeedHybridRestoreError,
        validate_mixed_version_steps,
    )

    with pytest.raises(
        DeepSpeedHybridRestoreError,
        match="not at the failure safe step",
    ):
        validate_mixed_version_steps(
            [10] * 64, [10] * 56, failure_step=17
        )


def test_fault_injection_holds_survivors_at_committed_step():
    source = (
        ROOT / "deepspeed_qwen3_moe_pretrain.py"
    ).read_text(encoding="utf-8")

    assert "FAULT_SAFE_STEP_WAITING_FOR_PREEMPTION" in source
    assert "runtime.wait_for_recovery_preemption(global_step)" in source


def test_hybrid_restore_rejects_mixed_checkpoint_base_versions():
    from moegambit_deepspeed.hybrid_restore import (
        DeepSpeedHybridRestoreError,
        validate_checkpoint_base_steps,
    )

    with pytest.raises(
        DeepSpeedHybridRestoreError,
        match="different checkpoint versions",
    ):
        validate_checkpoint_base_steps(
            [10, 10, 15, 10], failure_step=17
        )


def test_hybrid_restore_rejects_checkpoint_newer_than_failure():
    from moegambit_deepspeed.hybrid_restore import (
        DeepSpeedHybridRestoreError,
        validate_checkpoint_base_steps,
    )

    with pytest.raises(
        DeepSpeedHybridRestoreError,
        match="newer than the recorded failure",
    ):
        validate_checkpoint_base_steps([20] * 4, failure_step=17)


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
                    "mixed_version_survivor_handoff"
                ),
            },
            features=FeatureSwitches(hot_swap=True, zero2=False),
        )
    )

    assert "--elastic_training" not in prepared.command
    assert (
        prepared.metadata["recovery_strategy"]
        == "mixed_version_survivor_handoff"
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
    assert "mixed_version_survivor_handoff" not in result.stderr
    assert "--spare-node 8" in result.stdout
    assert "--node_rank \\{logical_node\\}" in result.stdout
    assert "--elastic_training" not in result.stdout
    assert "dry run complete" in result.stdout
    assert "hybrid_restore=1" in result.stdout
    assert "survivor_handoff=1" in result.stdout
    assert "MOEGAMBIT_DEEPSPEED_SURVIVOR_HANDOFF" in (
        ROOT / "test_deepspeed_hotspare_replace.sh"
    ).read_text(encoding="utf-8")
    assert "LOCAL_WORLD_SIZE" in (
        ROOT / "test_deepspeed_hotspare_replace.sh"
    ).read_text(encoding="utf-8")


def test_duplicate_spare_launcher_cannot_delete_active_case(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_flock = bin_dir / "flock"
    fake_flock.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake_flock.chmod(0o755)

    run_root = tmp_path / "run"
    case_root = run_root / "duplicate-run" / "hot_swap_pp8_ep8"
    case_root.mkdir(parents=True)
    sentinel = case_root / "active-checkpoint"
    sentinel.write_text("keep", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "DRY_RUN": "1",
            "TEST_MODE": "hot_swap",
            "NODE_RANK": "8",
            "MASTER_ADDR": "10.0.0.1",
            "MASTER_PORT": "23992",
            "MOEGAMBIT_HOT_SPARE_COORDINATOR_ADDR": "10.0.0.9",
            "RUN_ID": "duplicate-run",
            "RUN_ROOT": str(run_root),
            "RESET_RUN": "1",
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

    assert result.returncode == 64
    assert "another launcher already owns" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "keep"


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


def test_hot_spare_coordinator_does_not_persist_heartbeat_or_poll(
    tmp_path, monkeypatch
):
    from moegambit.runtime.hot_spare import HotSpareCoordinator
    from moegambit.runtime.protocol import WireMessage

    coordinator = HotSpareCoordinator(
        run_id="test-run",
        training_nodes=1,
        spare_physical_node=1,
        base_master_port=24000,
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
    request("register", 1, role="standby", advertise_addr="10.0.0.2")
    persisted = []
    monkeypatch.setattr(
        coordinator, "_persist", lambda: persisted.append(True)
    )

    request("heartbeat", 0, epoch=0)
    request("poll", 0, epoch=0)

    assert persisted == []


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
        "mp_rank_01_model_states.pt",
        "layer_0_expert_0_mp_rank_01_model_states.pt",
        "bf16_zero_pp_rank_0_mp_rank_01_optim_states.pt",
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
        command=("train.py", "--pipeline-parallel-size", "8"),
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
        "bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt",
        "mp_rank_01_model_states.pt",
        "layer_0_expert_0_mp_rank_01_model_states.pt",
        "bf16_zero_pp_rank_0_mp_rank_01_optim_states.pt",
    }
    assert [path.name for path in selected[:2]] == [
        "mp_rank_00_model_states.pt",
        "mp_rank_01_model_states.pt",
    ]

    _, stage_one = supervisor._checkpoint_prefetch_files(
        checkpoint_dir, logical_node=1
    )
    assert stage_one == selected

    supervisor.command = (
        "runner",
        "--num_gpus",
        "2",
        "train.py",
        "--pipeline-parallel-size",
        "2",
    )
    _, node_zero = supervisor._checkpoint_prefetch_files(
        checkpoint_dir, logical_node=0
    )
    assert {
        path.name
        for path in node_zero
        if path.name.endswith("_optim_states.pt")
    } == {
        "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt",
        "bf16_zero_pp_rank_0_mp_rank_01_optim_states.pt",
    }
    _, node_one = supervisor._checkpoint_prefetch_files(
        checkpoint_dir, logical_node=1
    )
    assert {
        path.name
        for path in node_one
        if path.name.endswith("_optim_states.pt")
    } == {
        "bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt",
    }

    supervisor.command = (
        "runner",
        "--num_gpus",
        "2",
        "train.py",
        "--pipeline-parallel-size",
        "2",
        "--zero-stage",
        "2",
    )
    _, zero_two = supervisor._checkpoint_prefetch_files(
        checkpoint_dir, logical_node=0
    )
    assert {
        path.name
        for path in zero_two
        if path.name.endswith("_optim_states.pt")
    } == {
        "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt",
        "bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt",
        "bf16_zero_pp_rank_0_mp_rank_01_optim_states.pt",
    }


def test_hot_spare_prefetch_rewarms_dataset_index(tmp_path, monkeypatch):
    from moegambit.runtime.hot_spare import AgentSupervisor
    from moegambit.runtime.watcher_client import WatcherEndpoint

    data_path = tmp_path / "dataset"
    index_path = Path(str(data_path) + ".idx")
    index_path.write_bytes(b"index")
    supervisor = AgentSupervisor(
        endpoint=WatcherEndpoint("127.0.0.1", 1),
        run_id="test-run",
        physical_node=2,
        role="standby",
        advertise_addr="127.0.0.1",
        command=("train.py", "--data-path", str(data_path)),
        heartbeat_interval=1,
        startup_timeout=1,
    )
    monkeypatch.setenv("MOEGAMBIT_STANDBY_PREFETCH_MAX_GIB", "1")
    supervisor._prefetch_stop.clear()

    assert supervisor._prefetch_dataset_index("recovery") == len(b"index")


def test_hot_spare_extracts_resident_deepspeed_workload(tmp_path):
    from moegambit.runtime.hot_spare import (
        _deepspeed_python_workload,
    )

    workload = tmp_path / "train.py"
    workload.write_text("", encoding="utf-8")
    extracted = _deepspeed_python_workload(
        (
            sys.executable,
            "-u",
            "-m",
            "deepspeed.launcher.runner",
            "--no_ssh",
            "--num_nodes",
            "8",
            "--num_gpus",
            "8",
            str(workload),
            "--state-dir",
            "/tmp/state",
        )
    )

    assert extracted == (
        workload,
        ("--state-dir", "/tmp/state"),
    )


def test_hot_spare_activates_only_fully_ready_resident_pool(
    tmp_path, monkeypatch
):
    from moegambit.runtime.hot_spare import AgentSupervisor
    from moegambit.runtime.watcher_client import WatcherEndpoint

    class RunningProcess:
        pid = 12345

        @staticmethod
        def poll():
            return None

    monkeypatch.setenv("MOEGAMBIT_STANDBY_RESIDENT", "1")
    control_dir = tmp_path / "control"
    control_dir.mkdir()
    session = "session-one"
    for local_rank in range(2):
        (control_dir / f"ready_{session}_{local_rank}.json").write_text(
            json.dumps(
                {
                    "session_id": session,
                    "local_rank": local_rank,
                    "allocated_gib": 8.5,
                }
            ),
            encoding="utf-8",
        )
    supervisor = AgentSupervisor(
        endpoint=WatcherEndpoint("127.0.0.1", 1),
        run_id="test-run",
        physical_node=2,
        role="standby",
        advertise_addr="127.0.0.1",
        command=(
            "runner",
            "--num_gpus",
            "2",
            "--enable_each_rank_log",
            str(tmp_path / "rank-logs"),
        ),
        heartbeat_interval=1,
        startup_timeout=1,
    )
    process = RunningProcess()
    supervisor._resident_process = process
    supervisor._resident_control_dir = control_dir
    supervisor._resident_session_id = session
    supervisor._resident_num_workers = 2

    activated = supervisor._activate_resident_standby(
        command=supervisor.command,
        environment={"MASTER_ADDR": "10.0.0.1", "MASTER_PORT": "25001"},
        logical_node=0,
        epoch=1,
    )

    assert activated is True
    assert supervisor.process is process
    assert supervisor._resident_process is None
    activation = json.loads(
        (control_dir / f"activate_{session}.json").read_text(
            encoding="utf-8"
        )
    )
    assert activation["logical_node"] == 0
    assert activation["epoch"] == 1
    assert activation["environment"]["MASTER_ADDR"] == "10.0.0.1"


def test_resident_standby_workers_activate_without_process_replacement(
    tmp_path,
):
    fake_modules = tmp_path / "fake-modules"
    fake_modules.mkdir()
    (fake_modules / "torch.py").write_text(
        """
class _Cuda:
    @staticmethod
    def set_device(_rank):
        pass

    @staticmethod
    def synchronize(_rank):
        pass

    @staticmethod
    def memory_allocated(_rank):
        return 1024**3

    @staticmethod
    def memory_reserved(_rank):
        return 2 * 1024**3


cuda = _Cuda()


def empty(*_args, **_kwargs):
    return object()
""".lstrip(),
        encoding="utf-8",
    )
    workload = tmp_path / "workload.py"
    workload.write_text(
        """
import json
import os
from pathlib import Path


activation = {}


def prepare_standby(argv, **context):
    return {
        "model_ready": True,
        "argv_count": len(argv),
        "prepared_local_rank": context["local_rank"],
    }


def activate_standby(_argv, **context):
    activation.update(context)
    return {"cache_handoff": True}


def main(_argv):
    output = Path(os.environ["OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    (output / f"rank_{rank}.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "rank": rank,
                "local_rank": int(os.environ["LOCAL_RANK"]),
                "world_size": int(os.environ["WORLD_SIZE"]),
                "activation_rank": activation["rank"],
                "activation_epoch": activation["epoch"],
            }
        ),
        encoding="utf-8",
    )
    return 0
""".lstrip(),
        encoding="utf-8",
    )
    control = tmp_path / "control"
    logs = tmp_path / "standby-logs"
    rank_logs = tmp_path / "rank-logs"
    output = tmp_path / "output"
    session = "resident-test"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(fake_modules),
            str(ROOT / "deepspeed_adapter"),
            environment.get("PYTHONPATH", ""),
        ]
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "moegambit.runtime.standby",
            "--mode",
            "launcher",
            "--control-dir",
            str(control),
            "--session-id",
            session,
            "--num-workers",
            "2",
            "--expected-logical-node",
            "1",
            "--world-size",
            "4",
            "--training-script",
            str(workload),
            "--log-dir",
            str(logs),
            "--",
            "--example",
            "value",
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if all(
                (
                    control / f"ready_{session}_{local_rank}.json"
                ).is_file()
                for local_rank in range(2)
            ):
                break
            assert process.poll() is None
            time.sleep(0.05)
        else:
            pytest.fail("resident workers did not become ready")

        (control / f"activate_{session}.json").write_text(
            json.dumps(
                {
                    "session_id": session,
                    "logical_node": 1,
                    "epoch": 1,
                    "rank_log_dir": str(rank_logs),
                    "environment": {
                        "MASTER_ADDR": "127.0.0.1",
                        "MASTER_PORT": "25001",
                        "MOEGAMBIT_RECOVERY_EPOCH": "1",
                        "OUTPUT_DIR": str(output),
                    },
                }
            ),
            encoding="utf-8",
        )
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)

    assert process.returncode == 0, (stdout, stderr)
    states = [
        json.loads(
            (output / f"rank_{rank}.json").read_text(
                encoding="utf-8"
            )
        )
        for rank in (2, 3)
    ]
    assert [state["local_rank"] for state in states] == [0, 1]
    assert {state["world_size"] for state in states} == {4}
    assert [state["activation_rank"] for state in states] == [2, 3]
    assert {state["activation_epoch"] for state in states} == {1}
    ready_pids = {
        json.loads(
            (
                control
                / f"ready_{session}_{local_rank}.json"
            ).read_text(encoding="utf-8")
        )["pid"]
        for local_rank in range(2)
    }
    assert {state["pid"] for state in states} == ready_pids


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


def test_hot_spare_heartbeat_preempts_old_recovery_epoch(monkeypatch):
    from moegambit.runtime.hot_spare import AgentSupervisor
    from moegambit.runtime.watcher_client import WatcherEndpoint

    class RunningProcess:
        pid = 43210

        @staticmethod
        def poll():
            return None

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
    supervisor.process = RunningProcess()
    supervisor.process_epoch = 0
    signal_sent = threading.Event()
    signals = []

    def killpg(pid, signum):
        signals.append((pid, signum))
        signal_sent.set()

    def request(kind, **payload):
        assert kind == "heartbeat"
        return {
            "action": "run",
            "epoch": 1,
            "logical_node": 3,
            "master_addr": "10.0.0.1",
            "master_port": 25001,
        }

    monkeypatch.setattr(os, "killpg", killpg)
    supervisor._request = request
    supervisor._start_heartbeat()
    try:
        assert signal_sent.wait(timeout=1.0)
    finally:
        supervisor._stop_heartbeat()

    assert signals == [(43210, signal.SIGKILL)]
    pending = supervisor._take_pending_command(
        {"action": "run", "epoch": 0}
    )
    assert pending is not None
    assert pending["epoch"] == 1


def test_hot_spare_same_epoch_command_never_preempts(monkeypatch):
    from moegambit.runtime.hot_spare import AgentSupervisor
    from moegambit.runtime.watcher_client import WatcherEndpoint

    class RunningProcess:
        pid = 43211

        @staticmethod
        def poll():
            return None

    supervisor = AgentSupervisor(
        endpoint=WatcherEndpoint("127.0.0.1", 1),
        run_id="test-run",
        physical_node=3,
        role="active",
        advertise_addr="127.0.0.1",
        command=("runner",),
        heartbeat_interval=1,
        startup_timeout=1,
    )
    supervisor.process = RunningProcess()
    supervisor.process_epoch = 0

    def unexpected_killpg(pid, signum):
        raise AssertionError(
            f"normal epoch attempted to kill pid={pid} signal={signum}"
        )

    monkeypatch.setattr(os, "killpg", unexpected_killpg)
    supervisor._observe_control_command(
        {
            "action": "run",
            "epoch": 0,
            "logical_node": 3,
            "master_addr": "10.0.0.1",
            "master_port": 25000,
        },
        source="test",
    )


def test_hot_spare_abort_preempts_same_epoch(monkeypatch):
    from moegambit.runtime.hot_spare import AgentSupervisor
    from moegambit.runtime.watcher_client import WatcherEndpoint

    class RunningProcess:
        pid = 43212

        @staticmethod
        def poll():
            return None

    supervisor = AgentSupervisor(
        endpoint=WatcherEndpoint("127.0.0.1", 1),
        run_id="test-run",
        physical_node=3,
        role="active",
        advertise_addr="127.0.0.1",
        command=("runner",),
        heartbeat_interval=1,
        startup_timeout=1,
    )
    supervisor.process = RunningProcess()
    supervisor.process_epoch = 1
    signals = []
    monkeypatch.setattr(
        os,
        "killpg",
        lambda pid, signum: signals.append((pid, signum)),
    )

    supervisor._observe_control_command(
        {"action": "abort", "epoch": 1},
        source="test",
    )

    assert signals == [(43212, signal.SIGKILL)]


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
    assert "RECOVERY_PREEMPT_SIGNAL" in combined_logs
    assert "physical_node=2 logical_node=0 epoch=1" in combined_logs


def test_vendored_deepspeed_version_is_pinned():
    metadata = (ROOT / "DeepSpeed" / "MOEGAMBIT_UPSTREAM.md").read_text(
        encoding="utf-8"
    )
    assert "v0.19.3" in metadata
    assert "0c36f6d3efb806c07a44e5c2c9b18a81d204b821" in metadata

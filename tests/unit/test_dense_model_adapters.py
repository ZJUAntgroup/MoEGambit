"""Dense recovery must not depend on expert state or its staleness policy."""

import sys
from types import ModuleType, SimpleNamespace

import pytest


def _fake_megatron_module(monkeypatch, leaf, **functions):
    name = f"moegambit.adapters.megatron.{leaf}"
    module = ModuleType(name)
    for key, value in functions.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_megatron_dense_startup_requires_matching_watcher_kind(monkeypatch):
    from moegambit.adapters.megatron.native_hooks import MegatronNativeHooks

    args = SimpleNamespace(moe_moegambit_enable=True, num_experts=None)
    with pytest.raises(ValueError, match="MOEGAMBIT_MODEL_KIND=dense"):
        MegatronNativeHooks().prepare_pretrain(args)

    from moegambit.adapters.megatron import integration

    monkeypatch.setenv("MOEGAMBIT_MODEL_KIND", "dense")
    _fake_megatron_module(
        monkeypatch,
        "elastic_client",
        elastic_sanitize_recovery_env_for_startup=lambda: None,
        elastic_expert_sidecar_available=lambda: False,
    )
    monkeypatch.setattr(integration, "bootstrap_control_plane", lambda: None)
    hooks = MegatronNativeHooks()
    monkeypatch.setattr(hooks, "is_rebuild_mode", lambda: False)
    assert hooks.prepare_pretrain(args).rebuild is False


def test_megatron_dense_replacement_skips_expert_sidecar(monkeypatch):
    from moegambit.adapters.megatron.native_hooks import MegatronNativeHooks
    from moegambit.adapters.megatron import integration

    args = SimpleNamespace(
        moe_moegambit_enable=True,
        num_experts=None,
        load="/tmp/checkpoint",
        no_load_optim=False,
        no_load_rng=False,
        enable_gloo_process_groups=True,
    )
    monkeypatch.setenv("MOEGAMBIT_MODEL_KIND", "dense")
    _fake_megatron_module(
        monkeypatch,
        "elastic_client",
        elastic_sanitize_recovery_env_for_startup=lambda: None,
        elastic_expert_sidecar_available=lambda: pytest.fail(
            "dense recovery must not inspect an expert sidecar"
        ),
    )
    monkeypatch.setattr(integration, "bootstrap_control_plane", lambda: None)
    hooks = MegatronNativeHooks()
    monkeypatch.setattr(hooks, "is_rebuild_mode", lambda: True)
    state = hooks.prepare_pretrain(args)
    assert state.rebuild and not state.use_expert_sidecar
    assert args.load == "/tmp/checkpoint"


def test_megatron_dense_checkpoint_is_recorded_without_expert_manifest(monkeypatch):
    from moegambit.adapters.megatron import integration
    from moegambit.adapters.megatron.native_hooks import MegatronNativeHooks

    recorded = []
    monkeypatch.setattr(integration, "get_runtime", lambda: SimpleNamespace(
        record_checkpoint=lambda path, step: recorded.append((path, step))
    ))
    _fake_megatron_module(
        monkeypatch,
        "moe_integration",
        moegambit_is_initialized=lambda: False,
        moegambit_pre_save_checkpoint=lambda step, state: state,
        moegambit_save_manifest=lambda *_: pytest.fail(
            "dense checkpoint must not write an expert manifest"
        ),
    )
    hooks = MegatronNativeHooks()
    hooks.checkpoint_pre_save(0, {})
    hooks.checkpoint_post_save("/tmp/checkpoints", 0)
    assert recorded == [("/tmp/checkpoints/iter_0000000", 0)]
    assert hooks._step_transaction.checkpoint_step == 0


def test_megatron_dense_guard_ignores_expert_gap(monkeypatch):
    from moegambit.adapters.megatron.compat_watcher import ElasticWatcher

    watcher = object.__new__(ElasticWatcher)
    watcher._latest_checkpoint_iteration = lambda: 200
    monkeypatch.setenv("MOEGAMBIT_MODEL_KIND", "dense")
    monkeypatch.setenv("MOEGAMBIT_MAX_SINGLE_GAP", "1")
    result = watcher._evaluate_moegambit_contract(1000, True)
    assert result["admitted"]
    assert result["name"] == "moegambit_dense_full_peer"
    assert result["expert_staleness_delta"] == 0
    assert not watcher._evaluate_moegambit_contract(1000, False)["admitted"]


def test_megatron_dense_descriptor_has_only_current_step_peer_sources(monkeypatch):
    from moegambit.adapters.megatron.compat_watcher import ElasticWatcher

    monkeypatch.setenv("MOEGAMBIT_MODEL_KIND", "dense")
    monkeypatch.setenv("EP_SIZE", "1")
    watcher = object.__new__(ElasticWatcher)
    watcher.training_nnodes = 2
    watcher.nproc_per_node = 1
    watcher.master_addr = "127.0.0.1"
    watcher.master_port = 29500
    watcher.fallback_exit_code = 75
    watcher.rank_quiescence_proof = {"status": "verified", "required": False}
    watcher._latest_checkpoint_iteration = lambda: 200
    descriptor = watcher._build_recovery_descriptor(0, 0, 1, resume_iteration=1000)

    assert descriptor["model_kind"] == "dense"
    assert descriptor["version_semantics"] == "full_current_step_peer"
    assert descriptor["recovery"]["hot_swap_feasible"]
    assert descriptor["state_sources"]["non_expert_optimizer"]["version_step"] == 1000
    assert "expert_parameters" not in descriptor["state_sources"]
    assert "expert_optimizer" not in descriptor["state_sources"]


def test_deepspeed_dense_classifier_catches_model_and_optimizer_experts():
    from moegambit.adapters.deepspeed.hybrid_restore import has_expert_state

    parameter = SimpleNamespace()
    module = SimpleNamespace(
        named_parameters=lambda: [("block.weight", parameter)],
        named_buffers=lambda: [],
    )
    optimizer = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"name": "dense"}]))
    engine = SimpleNamespace(module=module, optimizer=optimizer)
    assert not has_expert_state(engine)
    parameter.ds_zero_placement_family = "autoep_expert"
    assert has_expert_state(engine)
    del parameter.ds_zero_placement_family
    optimizer.optimizer.param_groups[0]["name"] = "expert"
    assert has_expert_state(engine)


def test_deepspeed_dense_claim_fails_closed_when_experts_exist(monkeypatch):
    from moegambit.adapters.deepspeed.integration import (
        DeepSpeedRecoveryRuntime,
        DeepSpeedRuntimeSettings,
    )

    parameter = SimpleNamespace(ds_zero_placement_family="autoep_expert")
    engine = SimpleNamespace(
        global_steps=0,
        _take_model_step=lambda: None,
        module=SimpleNamespace(
            named_parameters=lambda: [("weight", parameter)],
            named_buffers=lambda: [],
        ),
        optimizer=SimpleNamespace(optimizer=SimpleNamespace(param_groups=[])),
    )
    settings = DeepSpeedRuntimeSettings(False, False, None, 0, 0, 300.0)
    monkeypatch.setenv("MOEGAMBIT_MODEL_KIND", "dense")
    with pytest.raises(RuntimeError, match="contains expert state"):
        DeepSpeedRecoveryRuntime(engine, settings).start()


def test_deepspeed_dense_checkpoint_does_not_require_expert_shards(
    monkeypatch, tmp_path
):
    from moegambit.adapters.deepspeed.checkpoint_commit import build_checkpoint_manifest

    tag = "global_step10"
    tag_dir = tmp_path / tag
    tag_dir.mkdir()
    (tag_dir / "mp_rank_00_model_states.pt").write_bytes(b"dense")
    engine = SimpleNamespace(
        zero_optimization_partition_weights=lambda: False,
        mp_world_size=1,
    )
    monkeypatch.setenv("MOEGAMBIT_MODEL_KIND", "dense")
    monkeypatch.setenv("DEEPSPEED_MOEGAMBIT_PACKED_EXPERT_CHECKPOINT", "1")
    manifest = build_checkpoint_manifest(engine, tmp_path, tag)
    assert manifest["expected_dense_shards"] == 1
    assert "expected_packed_expert_shards" not in manifest


def test_deepspeed_dense_disables_packed_expert_checkpoint(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("deepspeed")
    from moegambit.adapters.deepspeed.packed_moe import (
        packed_expert_cache_enabled,
        packed_expert_checkpoint_enabled,
    )

    monkeypatch.setenv("MOEGAMBIT_MODEL_KIND", "dense")
    monkeypatch.setenv("DEEPSPEED_MOEGAMBIT_PACKED_EXPERT_CHECKPOINT", "1")
    monkeypatch.setenv("MOEGAMBIT_STANDBY_PACKED_EXPERT_CACHE", "1")
    assert not packed_expert_checkpoint_enabled()
    assert not packed_expert_cache_enabled()

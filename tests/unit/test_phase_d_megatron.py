import ast
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[2]
SRC = ROOT / "src"
MEGATRON = ROOT / "Megatron-LM"


def test_megatron_adapter_is_lazy_and_declares_preserved_capabilities():
    before = set(sys.modules)
    from moegambit.adapters.megatron import (
        MEGATRON_CAPABILITIES,
        build_megatron_adapter,
    )

    adapter = build_megatron_adapter([], object())
    assert adapter.name == "megatron"
    assert adapter.recovery_driver is not None
    assert MEGATRON_CAPABILITIES.static_world_replacement
    assert MEGATRON_CAPABILITIES.full_group_rebuild
    assert MEGATRON_CAPABILITIES.optimizer_memory_replication
    assert MEGATRON_CAPABILITIES.moe_state_classification
    assert MEGATRON_CAPABILITIES.two_phase_optimizer_restore
    assert MEGATRON_CAPABILITIES.supported_zero_stages == frozenset({0, 2})
    assert "torch" not in set(sys.modules) - before
    assert "megatron" not in set(sys.modules) - before


def test_framework_neutral_core_does_not_import_megatron():
    offenders = []
    adapter_root = SRC / "moegambit" / "adapters" / "megatron"
    for path in (SRC / "moegambit").rglob("*.py"):
        if adapter_root in path.parents:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name == "megatron" or name.startswith("megatron.") for name in names):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_latest_elastic_and_moe_sources_live_in_adapter_package():
    elastic = (SRC / "moegambit/adapters/megatron/elastic_client.py").read_text()
    integration = (SRC / "moegambit/adapters/megatron/moe_integration.py").read_text()
    assert "def elastic_do_rebuild(" in elastic
    assert "def elastic_zero2_quiesce_for_recovery(" in elastic
    assert "def moegambit_before_iteration(" in integration
    assert "def moegambit_pipeline_on_failure(" in integration


def test_adapter_does_not_import_deleted_megatron_recovery_modules():
    offenders = []
    for path in (SRC / "moegambit/adapters/megatron").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.module == "megatron.core.transformer.moe" or node.module.startswith(
                "megatron.core.transformer.moe."
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.module}")
            if node.module in {
                "megatron.training.elastic_client",
                "megatron.training.zero2_memory_checkpoint",
            }:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.module}")
    assert offenders == []


def test_megatron_recovery_cli_is_registered_by_adapter():
    from moegambit.adapters.megatron.native_hooks import MegatronNativeHooks

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("moe")
    MegatronNativeHooks().add_moe_arguments(group)
    args = parser.parse_args(
        [
            "--moe-moegambit-enable",
            "--moe-moegambit-group-rebuild",
            "--moe-moegambit-hot-spare-pool",
            "--moe-moegambit-num-hot-spares",
            "8",
            "--moe-moegambit-recovery-policy-type",
            "rank_exposure_guarded_hybrid",
        ]
    )
    assert args.moe_moegambit_enable
    assert args.moe_moegambit_group_rebuild
    assert args.moe_moegambit_hot_spare_pool
    assert args.moe_moegambit_num_hot_spares == 8
    assert args.moe_moegambit_recovery_policy_type == "rank_exposure_guarded_hybrid"


def test_megatron_tree_contains_hooks_but_no_recovery_modules():
    hooks = (SRC / "moegambit/adapters/megatron/hooks.py").read_text()
    training = (MEGATRON / "megatron/training/training.py").read_text()
    assert "from moegambit.adapters.megatron.hooks import" in training
    assert "Stable hook surface imported by patched Megatron source files" in hooks
    assert not (MEGATRON / "megatron/training/elastic_client.py").exists()
    assert not (
        MEGATRON / "megatron/core/transformer/moe/moegambit_integration.py"
    ).exists()
    assert not (
        MEGATRON / "megatron/core/transformer/moe/recovery_controller.py"
    ).exists()


def test_megatron_source_imports_only_the_public_hook_facade():
    offenders = []
    for path in (MEGATRON / "megatron").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.module.startswith("moegambit.") and (
                node.module != "moegambit.adapters.megatron.hooks"
            ):
                offenders.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}:{node.module}"
                )
    assert offenders == []


def test_megatron_patch_imports_only_the_single_hook_object():
    offenders = []
    for path in (MEGATRON / "megatron").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "moegambit.adapters.megatron.hooks":
                continue
            imported = [alias.name for alias in node.names]
            if imported != ["megatron_hooks"]:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}:{imported}")
    assert offenders == []

    transformer_config = (
        MEGATRON / "megatron/core/transformer/transformer_config.py"
    ).read_text()
    assert "moe_moegambit_" not in transformer_config


def test_training_loop_uses_stable_runtime_lifecycle_facade():
    source = (MEGATRON / "megatron/training/training.py").read_text()
    required_calls = (
        "megatron_hooks.prepare_pretrain(args)",
        "megatron_hooks.complete_model_setup(",
        "megatron_hooks.iteration_safe_point(",
        "megatron_hooks.before_optimizer_step(args.curr_iteration)",
        "megatron_hooks.after_optimizer_step(args.curr_iteration, committed=True)",
        "megatron_hooks.after_optimizer_step(args.curr_iteration, committed=False)",
        "megatron_hooks.handle_train_step_error(",
        "megatron_hooks.commit_iteration(args.curr_iteration)",
    )
    for call in required_calls:
        assert call in source

    imports = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
        and node.module == "moegambit.adapters.megatron.hooks"
    ]
    assert len(imports) == 1
    assert [alias.name for alias in imports[0].names] == ["megatron_hooks"]


def test_megatron_refuses_bookkeeping_rollback_after_optimizer_mutation(
    monkeypatch,
):
    from moegambit.adapters.megatron import integration
    from moegambit.adapters.megatron.native_hooks import MegatronNativeHooks
    from moegambit.core.step_transaction import StepPhase, StepTransaction

    fallback_calls = []

    class Runtime:
        def fail_closed(self, exc, *, reason, evidence):
            fallback_calls.append((exc, reason, evidence))
            return True

    monkeypatch.setattr(integration, "get_runtime", lambda: Runtime())
    hooks = MegatronNativeHooks()
    hooks._step_transaction = StepTransaction(4)
    hooks._step_transaction.mark_optimizer_snapshot(4)
    hooks._step_transaction.begin_step(4)
    hooks._step_transaction.enter(StepPhase.OPTIMIZER_DURING)

    disposition = hooks.handle_train_step_error(
        RuntimeError("NCCL peer failure during optimizer"),
        args=SimpleNamespace(moe_moegambit_enable=True),
        iteration=4,
    )

    assert not disposition.handled
    assert fallback_calls[0][1] == "unsafe_megatron_step_replay"
    assert fallback_calls[0][2]["optimizer_dirty"] is True


def test_migrated_moe_modules_do_not_leak_back_into_megatron():
    implementation_root = SRC / "moegambit/adapters/megatron/moe"
    megatron_moe_root = MEGATRON / "megatron/core/transformer/moe"
    leaves = {
        path.stem for path in implementation_root.glob("*.py") if path.name != "__init__.py"
    }
    assert leaves
    for leaf in leaves:
        assert not (megatron_moe_root / f"{leaf}.py").exists(), leaf

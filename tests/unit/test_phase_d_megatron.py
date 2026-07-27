import ast
import sys
from pathlib import Path


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


def test_legacy_paths_are_module_aliases_to_single_source_of_truth():
    elastic_shim = (MEGATRON / "megatron/training/elastic_client.py").read_text()
    integration_shim = (
        MEGATRON / "megatron/core/transformer/moe/moegambit_integration.py"
    ).read_text()
    recovery_shim = (
        MEGATRON / "megatron/core/transformer/moe/recovery_controller.py"
    ).read_text()
    assert "sys.modules[__name__] = _implementation" in elastic_shim
    assert "sys.modules[__name__] = _implementation" in integration_shim
    assert 'install_alias(__name__, "recovery_controller")' in recovery_shim


def test_training_loop_uses_stable_runtime_lifecycle_facade():
    source = (MEGATRON / "megatron/training/training.py").read_text()
    required_calls = (
        "moegambit_runtime_bootstrap()",
        "initialize_moegambit_runtime(",
        "moegambit_runtime_finalize_replacement(",
        "moegambit_runtime_iteration_boundary(iteration)",
        "moegambit_runtime_before_optimizer_step(args.curr_iteration)",
        "moegambit_runtime_after_optimizer_step(args.curr_iteration, committed=True)",
        "moegambit_runtime_after_optimizer_step(args.curr_iteration, committed=False)",
        "moegambit_runtime_on_distributed_error(_moegambit_exc)",
        "moegambit_runtime_commit_iteration(args.curr_iteration)",
    )
    for call in required_calls:
        assert call in source


def test_every_migrated_moe_module_has_a_legacy_alias():
    implementation_root = SRC / "moegambit/adapters/megatron/moe"
    legacy_root = MEGATRON / "megatron/core/transformer/moe"
    leaves = {
        path.stem for path in implementation_root.glob("*.py") if path.name != "__init__.py"
    }
    assert leaves
    for leaf in leaves:
        shim = legacy_root / f"{leaf}.py"
        assert shim.exists(), leaf
        assert f'install_alias(__name__, "{leaf}")' in shim.read_text()

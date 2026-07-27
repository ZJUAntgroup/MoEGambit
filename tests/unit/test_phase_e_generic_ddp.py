from __future__ import annotations

import inspect
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

import moegambit.adapters.generic_ddp as generic
from moegambit.adapters.base import (
    OptimizerAdapter,
    StateAdapter,
    TopologyAdapter,
    TrainingAdapter,
)
from moegambit.adapters.generic_ddp import (
    GENERIC_DDP_CAPABILITIES,
    GenericDDPOptimizerAdapter,
    GenericDDPStateAdapter,
    GenericDDPTopologyAdapter,
    GenericDDPTrainingAdapter,
    RebindableModel,
    build_generic_ddp_adapter,
)
from moegambit.adapters.registry import available_adapters, get_adapter
from moegambit.errors import AdapterUnsupportedError
from moegambit.runtime.recovery_plan import RecoveryMode, RecoveryPlan
from moegambit.state.catalog import StateSource, StateSourceKind
from moegambit.state.version import StateVersion


ROOT = Path(__file__).parents[2]


class _Module:
    def named_parameters(self):
        return ()


class _Optimizer:
    state = {}


def _signature_mismatches(implementation, protocol):
    mismatches = []
    for name, member in protocol.__dict__.items():
        if name.startswith("_") or not inspect.isfunction(member):
            continue
        actual = getattr(implementation, name, None)
        if actual is None or inspect.signature(actual) != inspect.signature(member):
            mismatches.append(name)
    return mismatches


@pytest.mark.parametrize(
    ("component", "implementation", "protocol"),
    [
        (GenericDDPTopologyAdapter(), GenericDDPTopologyAdapter, TopologyAdapter),
        (GenericDDPStateAdapter(_Module(), _Optimizer()), GenericDDPStateAdapter, StateAdapter),
        (GenericDDPOptimizerAdapter(_Module(), _Optimizer()), GenericDDPOptimizerAdapter, OptimizerAdapter),
        (GenericDDPTrainingAdapter(_Module()), GenericDDPTrainingAdapter, TrainingAdapter),
    ],
)
def test_components_conform_to_narrow_protocols(component, implementation, protocol):
    assert isinstance(component, protocol)
    assert _signature_mismatches(implementation, protocol) == []


def test_builtin_discovery_is_lazy_without_optional_frameworks():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    code = """
import sys
from moegambit.adapters.registry import available_adapters
assert {'generic_ddp', 'megatron'} <= set(available_adapters())
assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
assert not any(name == 'megatron' or name.startswith('megatron.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert {"generic_ddp", "megatron"} <= set(available_adapters())


def test_registry_builds_generic_adapter_without_importing_torch():
    before = set(sys.modules)
    adapter = get_adapter("generic_ddp", module=_Module(), optimizer=_Optimizer())
    assert adapter.name == "generic_ddp"
    assert adapter.support_level == "detection-only"
    assert not any(
        name == "torch" or name.startswith("torch.")
        for name in set(sys.modules) - before
    )


def test_observed_backend_is_used_for_rebuild(monkeypatch):
    dist = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_rank=lambda: 0,
        get_world_size=lambda: 2,
        get_backend=lambda: "nccl",
    )
    monkeypatch.setattr(generic, "_require_torch", lambda: SimpleNamespace(distributed=dist))
    adapter = GenericDDPTopologyAdapter(backend="gloo")
    topology = adapter.inspect()
    assert topology.groups[0].backend == "nccl"
    assert adapter._context.backend == "nccl"


def test_live_ddp_fails_before_teardown_without_replaceable_owner():
    class Wrapper(_Module):
        reducer = object()

    adapter = build_generic_ddp_adapter(Wrapper(), _Optimizer())
    assert not GENERIC_DDP_CAPABILITIES.full_group_rebuild
    with pytest.raises(AdapterUnsupportedError, match="before process-group teardown"):
        adapter.topology.prepare_rebuild(
            RecoveryPlan(1, 1, (0,), 0, RecoveryMode.PEER)
        )


def test_rebind_replaces_wrapper_and_reducer_through_stable_owner():
    class Wrapper(_Module):
        reducer = object()

    old, new = Wrapper(), Wrapper()
    owner = RebindableModel(old)
    calls = []
    adapter = build_generic_ddp_adapter(
        owner,
        _Optimizer(),
        replacement_loader=lambda plan: None,
        module_rebuilder=lambda wrapper, group: calls.append((wrapper, group)) or new,
    )
    adapter.topology._context.dp_group = "rebuilt-group"
    plan = RecoveryPlan(1, 1, (0,), 0, RecoveryMode.PEER)
    topology = generic.TopologySpec(
        1, 0, {"dp": 1}, {"dp": 0}, generation=1
    )
    adapter.topology.prepare_rebuild(plan)
    adapter.topology.rebind(topology)
    assert owner.current is new
    assert calls == [(old, "rebuilt-group")]
    assert adapter.capabilities.full_group_rebuild
    assert adapter.capabilities.static_world_replacement


def test_bare_replacement_is_wrapped_after_new_group_exists():
    bare = _Module()
    wrapped = SimpleNamespace(reducer=object(), named_parameters=lambda: ())
    owner = RebindableModel(bare)
    adapter = build_generic_ddp_adapter(
        owner,
        _Optimizer(),
        rank=1,
        world_size=2,
        replacement_loader=lambda plan: None,
        module_rebuilder=lambda module, group: wrapped,
    )
    adapter.topology._context.dp_group = "new-group"
    plan = RecoveryPlan(1, 1, (1,), 0, RecoveryMode.PEER)
    adapter.topology.prepare_rebuild(plan)
    adapter.topology.rebind(
        generic.TopologySpec(2, 1, {"dp": 2}, {"dp": 1}, generation=1)
    )
    assert owner.current is wrapped


def test_adamw_lazy_slots_are_materialized_from_frozen_metadata(monkeypatch):
    class Parameter:
        shape = (2, 3)
        dtype = "float32"
        device = "cpu"

    parameter = Parameter()

    class Module:
        def named_parameters(self):
            return (("weight", parameter),)

    optimizer = SimpleNamespace(state={})
    created = []

    class Torch:
        float32 = "float32"

        @staticmethod
        def zeros(shape, dtype=None, device=None):
            value = SimpleNamespace(shape=shape, dtype=dtype, device=device)
            created.append(value)
            return value

        @staticmethod
        def no_grad():
            return nullcontext()

    monkeypatch.setattr(generic, "_require_torch", lambda: Torch)
    adapter = GenericDDPStateAdapter(Module(), optimizer)
    source = StateSource(
        StateSourceKind.PEER,
        StateVersion(3, 3, 1),
        "rank://0",
        metadata={
            "parameter_name": "weight",
            "state_key": "exp_avg",
            "tensor": True,
            "shape": [2, 3],
            "dtype": "torch.float32",
        },
    )
    plan = RecoveryPlan(
        1, 1, (1,), 3, RecoveryMode.PEER,
        state_sources={"optim/weight/exp_avg": source},
    )
    adapter._materialize_optimizer_slots(plan)
    assert optimizer.state[parameter]["exp_avg"] is created[0]
    assert created[0].shape == (2, 3)


def test_cpu_and_gpu_conformance_programs_are_present_and_selectable():
    source = (ROOT / "examples/generic_ddp/fault_replacement.py").read_text()
    assert 'choices=("gloo", "nccl")' in source
    assert "os._exit(42)" in source
    assert "RebindableModel" in source
    assert "peer_state_sources" in source
    assert "post-recovery parameter mismatch" in source

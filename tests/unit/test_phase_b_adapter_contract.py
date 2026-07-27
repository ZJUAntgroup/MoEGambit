"""Phase B Protocol composition and lazy adapter-registry tests."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import pytest


SRC_ROOT = Path(__file__).parents[2] / "src"

from moegambit.adapters.base import (  # noqa: E402
    FrameworkAdapter,
    OptimizerAdapter,
    PauseRequest,
    ProgressToken,
    QuiescenceProof,
    RebuildHandle,
    StateAdapter,
    StateSources,
    StoreHandle,
    TopologyAdapter,
    TrainingAdapter,
)
from moegambit.adapters.registry import (  # noqa: E402
    available_adapters,
    get_adapter,
    register_adapter,
)
from moegambit.distributed.topology import TopologySpec, ValidationReport  # noqa: E402
from moegambit.errors import AdapterUnsupportedError  # noqa: E402
from moegambit.runtime.recovery_plan import RecoveryPlan  # noqa: E402
from moegambit.state.catalog import StateCatalog, StateRef  # noqa: E402


class FakeTopology:
    def inspect(self) -> TopologySpec:
        return TopologySpec(world_size=1, rank=0).sealed()

    def prepare_rebuild(self, plan: RecoveryPlan) -> RebuildHandle:
        return RebuildHandle(plan.recovery_epoch)

    def rebuild(self, plan: RecoveryPlan, store: StoreHandle) -> TopologySpec:
        del store
        return TopologySpec(
            world_size=1,
            rank=0,
            generation=plan.topology_generation,
        ).sealed()

    def rebind(self, topology: TopologySpec) -> None:
        del topology

    def validate(self, expected: TopologySpec) -> ValidationReport:
        return ValidationReport.success(manifest=expected.manifest_hash)


class FakeState:
    def catalog(self) -> StateCatalog:
        return StateCatalog()

    def load_replacement_base(self, plan: RecoveryPlan) -> None:
        del plan

    def restore(self, plan: RecoveryPlan, sources: StateSources) -> None:
        del plan, sources

    def validate_state(self, plan: RecoveryPlan) -> ValidationReport:
        return ValidationReport.success(step=plan.resume_step)


class FakeOptimizer:
    def local_state_refs(self) -> Sequence[StateRef]:
        return ()

    def before_step(self, step: int) -> None:
        del step

    def after_step(self, step: int, committed: bool) -> None:
        del step, committed

    def rebind(self, topology: TopologySpec) -> None:
        del topology


class FakeTraining:
    def current_progress(self) -> ProgressToken:
        return ProgressToken(0, committed=True)

    def quiesce(self, request: PauseRequest) -> QuiescenceProof:
        return QuiescenceProof(0, ProgressToken(0), True, False, {"reason": request.reason})

    def apply_resume(self, plan: RecoveryPlan) -> None:
        del plan

    def reset_transients(self, plan: RecoveryPlan) -> None:
        del plan

    def warmup_and_validate(self, plan: RecoveryPlan) -> ValidationReport:
        return ValidationReport.success(step=plan.resume_step)


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
    ("instance", "implementation", "protocol"),
    [
        (FakeTopology(), FakeTopology, TopologyAdapter),
        (FakeState(), FakeState, StateAdapter),
        (FakeOptimizer(), FakeOptimizer, OptimizerAdapter),
        (FakeTraining(), FakeTraining, TrainingAdapter),
    ],
)
def test_fake_components_satisfy_narrow_protocols(instance, implementation, protocol):
    assert isinstance(instance, protocol)
    assert _signature_mismatches(implementation, protocol) == []


def test_framework_adapter_composes_protocols_without_inheritance():
    adapter = FrameworkAdapter(
        name="fake",
        topology=FakeTopology(),
        state=FakeState(),
        optimizer=FakeOptimizer(),
        training=FakeTraining(),
        version="1",
    )

    assert adapter.describe()["adapter"] == "fake"
    assert isinstance(adapter.topology, TopologyAdapter)
    assert isinstance(adapter.state, StateAdapter)


def test_registry_starts_without_unimplemented_builtin_adapters():
    assert available_adapters() == ()


def test_process_local_adapter_registration_and_build():
    class Plugin:
        @classmethod
        def build(cls, **kwargs):
            return {"adapter": "phase_b_fake", **kwargs}

    register_adapter("phase_b_fake", Plugin)

    assert "phase_b_fake" in available_adapters()
    assert get_adapter("phase_b_fake", value=7) == {
        "adapter": "phase_b_fake",
        "value": 7,
    }


def test_unknown_adapter_lists_available_names():
    with pytest.raises(AdapterUnsupportedError) as raised:
        get_adapter("missing_adapter")

    assert "missing_adapter" in str(raised.value)
    assert "phase_b_fake" in str(raised.value)


def test_adapter_discovery_imports_no_optional_frameworks():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(SRC_ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    code = """
import sys
import moegambit.adapters
from moegambit.adapters import available_adapters
assert available_adapters() == ()
forbidden = ('torch', 'megatron', 'deepspeed')
assert not any(
    name == prefix or name.startswith(prefix + '.')
    for name in sys.modules
    for prefix in forbidden
)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

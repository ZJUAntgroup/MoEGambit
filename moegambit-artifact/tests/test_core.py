import ast
from pathlib import Path

from moegambit.core.contracts import (
    FailureEvent,
    FeatureSwitches,
    RecoveryContext,
    RecoveryMode,
)
from moegambit.core.events import EventBus
from moegambit.core.policy import StalenessDensityPolicy
from moegambit.core.controller import RecoveryController


CORE_ROOT = Path(__file__).parents[1] / "src" / "moegambit" / "core"


def test_core_has_no_training_engine_or_torch_imports():
    forbidden = ("torch", "megatron", "deepspeed")
    for path in CORE_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not [
            name
            for name in imports
            if name.startswith(forbidden)
        ], path


def test_policy_falls_back_when_hot_swap_is_disabled():
    context = RecoveryContext(
        failure=FailureEvent(rank=7, step=20, checkpoint_step=10),
        peer_available=True,
        spare_available=True,
        affected_experts=(1, 2),
        num_experts=128,
    )
    decision = StalenessDensityPolicy().decide(
        context, FeatureSwitches(hot_swap=False, zero2=True)
    )
    assert decision.mode == RecoveryMode.CHECKPOINT_RELAUNCH
    assert decision.reason == "hot_swap_disabled"


def test_controller_emits_auditable_decision_events():
    observed = []
    events = EventBus()
    events.subscribe("*", observed.append)
    controller = RecoveryController(
        policy=StalenessDensityPolicy(),
        features=FeatureSwitches(hot_swap=True, zero2=False),
        events=events,
    )
    context = RecoveryContext(
        failure=FailureEvent(rank=7, step=20, checkpoint_step=10),
        peer_available=True,
        spare_available=True,
        affected_experts=(1, 2),
        num_experts=128,
    )

    decision = controller.decide(context)
    controller.complete()

    assert decision.mode == RecoveryMode.HOT_SWAP
    assert [event.name for event in observed] == [
        "recovery.deciding",
        "recovery.decided",
        "recovery.completed",
    ]


def test_policy_falls_back_when_exposure_domain_is_unknown():
    context = RecoveryContext(
        failure=FailureEvent(rank=7, step=20, checkpoint_step=10),
        peer_available=True,
        spare_available=True,
        affected_experts=(1, 2),
        num_experts=0,
    )
    decision = StalenessDensityPolicy().decide(
        context, FeatureSwitches(hot_swap=True, zero2=False)
    )
    assert decision.mode == RecoveryMode.CHECKPOINT_RELAUNCH
    assert decision.reason == "invalid_exposure_domain"

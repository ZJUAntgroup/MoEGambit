"""Phase C end-to-end execution through framework-free fake adapters."""

from __future__ import annotations

from dataclasses import replace

import pytest

import moegambit
from moegambit.adapters.base import (
    FrameworkAdapter,
    ProgressToken,
    QuiescenceProof,
    RebuildHandle,
    StateSources,
    StoreHandle,
)
from moegambit.capabilities import AdapterCapabilities
from moegambit.config import FallbackMode, RuntimeConfig
from moegambit.control.coordinator import (
    RecoveryAssignment,
    StaticRecoveryCoordinator,
)
from moegambit.control.service import RecoveryCoordinatorService
from moegambit.control.watcher import ControlRequestProcessor
from moegambit.distributed.topology import GroupSpec, TopologySpec, ValidationReport
from moegambit.errors import ContractViolation, RecoverableDistributedError
from moegambit.observability.events import RecoveryOutcome
from moegambit.runtime.client import (
    ControlClient,
    ControlClientConfig,
    WatcherRecoveryCoordinator,
)
from moegambit.runtime.lifecycle import EpochState, RecoveryEpochTracker
from moegambit.runtime.recovery_plan import RecoveryMode, RecoveryPlan
from moegambit.runtime.runtime import RecoveryRuntime
from moegambit.state.catalog import (
    Placement,
    StateCatalog,
    StateKind,
    StateRef,
    StateSource,
    StateSourceKind,
)
from moegambit.state.version import StateVersion


def _topology(generation=0):
    return TopologySpec(
        world_size=2,
        rank=0,
        logical_axes={"dp": 2},
        coordinates={"dp": 0},
        groups=(GroupSpec("world", (0, 1), "gloo", "world", 0),),
        generation=generation,
    ).sealed()


class _Topology:
    def __init__(self, calls, *, wrong_generation=False):
        self.calls = calls
        self.current = _topology(0)
        self.wrong_generation = wrong_generation

    def inspect(self):
        self.calls.append("inspect")
        return self.current

    def prepare_rebuild(self, plan):
        self.calls.append("prepare_rebuild")
        return RebuildHandle(plan.recovery_epoch)

    def rebuild(self, plan, store):
        self.calls.append(("rebuild", store.prefix))
        generation = plan.topology_generation + (1 if self.wrong_generation else 0)
        self.current = _topology(generation)
        return self.current

    def rebind(self, topology):
        self.calls.append("topology_rebind")
        self.current = topology

    def validate(self, expected):
        self.calls.append("topology_validate")
        return self.current.agrees_with(expected)


class _State:
    def __init__(self, calls, *, valid=True):
        self.calls = calls
        self.valid = valid

    def catalog(self):
        return StateCatalog()

    def load_replacement_base(self, plan):
        self.calls.append("load_base")

    def restore(self, plan, sources):
        self.calls.append(("restore", tuple(sorted(sources.by_identity))))

    def validate_state(self, plan):
        self.calls.append("state_validate")
        if self.valid:
            return ValidationReport.success(state="consistent")
        return ValidationReport.failure("state mismatch")


class _Optimizer:
    def __init__(self, calls):
        self.calls = calls

    def local_state_refs(self):
        return ()

    def before_step(self, step):
        self.calls.append(("before_step", step))

    def after_step(self, step, committed):
        self.calls.append(("after_step", step, committed))

    def rebind(self, topology):
        self.calls.append("optimizer_rebind")


class _Training:
    def __init__(self, calls):
        self.calls = calls
        self.step = 9

    def current_progress(self):
        return ProgressToken(self.step, committed=True)

    def quiesce(self, request):
        self.calls.append(("quiesce", request.recovery_epoch))
        return QuiescenceProof(
            rank=0,
            progress=self.current_progress(),
            collectives_drained=True,
            process_groups_destroyed=False,
        )

    def apply_resume(self, plan):
        self.calls.append("apply_resume")
        self.step = plan.resume_step

    def reset_transients(self, plan):
        self.calls.append("reset_transients")

    def warmup_and_validate(self, plan):
        self.calls.append("warmup")
        return ValidationReport.success(first_collective=True)


def _adapter(calls, *, state_valid=True, wrong_generation=False):
    return FrameworkAdapter(
        name="fake",
        topology=_Topology(calls, wrong_generation=wrong_generation),
        state=_State(calls, valid=state_valid),
        optimizer=_Optimizer(calls),
        training=_Training(calls),
        capabilities=AdapterCapabilities(
            static_world_replacement=True,
            full_group_rebuild=True,
            peer_parameter_restore=True,
        ),
        version="test",
    )


def _assignment(epoch=1):
    source = StateSource(
        StateSourceKind.PEER,
        StateVersion(committed_step=9, optimizer_generation=9),
        "rank://0",
    )
    plan = RecoveryPlan(
        protocol_version=1,
        recovery_epoch=epoch,
        failed_ranks=(1,),
        resume_step=9,
        mode=RecoveryMode.PEER,
        topology_generation=1,
        group_manifest_hash=_topology(0).manifest_hash,
        state_sources={"parameter/a": source},
    )
    return RecoveryAssignment(
        plan,
        StoreHandle("127.0.0.1", 23456, prefix=f"epoch-{epoch}"),
        StateSources({"parameter/a": object()}),
    )


def _failure():
    return RecoverableDistributedError("rank 1 died", failed_ranks=(1,))


def test_runtime_executes_frozen_plan_and_commits_after_full_iteration():
    calls = []
    coordinator = StaticRecoveryCoordinator(_assignment())
    runtime = RecoveryRuntime(
        _adapter(calls),
        RuntimeConfig(enabled=True),
        coordinator=coordinator,
    )

    assert runtime.on_distributed_error(_failure())
    assert runtime.resume_step == 9
    assert runtime.epochs.state is EpochState.PROVISIONAL
    assert calls.index("prepare_rebuild") < calls.index(("quiesce", 1))
    assert not runtime.commit_iteration(9)
    assert runtime.commit_iteration(10)
    assert runtime.epochs.state is EpochState.COMMITTED
    assert coordinator.commits == [(1, 10)]
    assert calls.index("topology_rebind") < calls.index("optimizer_rebind")
    assert calls.index("optimizer_rebind") < calls.index("state_validate")
    description = runtime.describe()
    assert description["last_recovery"]["result"] == "committed"
    assert description["metrics"]["counters"][
        "recovery_result_committed_total"
    ] == 1.0
    assert description["metrics"]["latency_ms"]["quiesce"]["count"] == 1


def test_runtime_fails_closed_on_state_or_generation_validation_failure():
    state_runtime = RecoveryRuntime(
        _adapter([], state_valid=False),
        RuntimeConfig(enabled=True),
        coordinator=StaticRecoveryCoordinator(_assignment()),
    )
    assert not state_runtime.on_distributed_error(_failure())
    assert state_runtime.epochs.state is EpochState.FAILED

    generation_runtime = RecoveryRuntime(
        _adapter([], wrong_generation=True),
        RuntimeConfig(enabled=True),
        coordinator=StaticRecoveryCoordinator(_assignment()),
    )
    assert not generation_runtime.on_distributed_error(_failure())
    assert generation_runtime.epochs.state is EpochState.FAILED
    assert "generation" in generation_runtime.describe()["last_recovery"]["validation"][
        "error"
    ]


def test_replacement_rejects_stale_assignment_epoch():
    runtime = RecoveryRuntime(
        _adapter([]),
        RuntimeConfig(enabled=True),
        coordinator=StaticRecoveryCoordinator(_assignment()),
    )
    runtime.execute_assignment(_assignment(), at_step=9)
    with pytest.raises(ContractViolation, match="newer"):
        runtime.execute_assignment(_assignment(), at_step=9)


def test_checkpoint_fallback_is_requested_but_not_reported_as_in_place_recovery():
    class _Fallback:
        def __init__(self):
            self.requests = []

        def request_checkpoint_relaunch(self, request):
            self.requests.append(request)
            return True

    fallback = _Fallback()
    runtime = RecoveryRuntime(
        _adapter([]),
        RuntimeConfig(enabled=True, fallback=FallbackMode.CHECKPOINT_RELAUNCH),
        fallback_controller=fallback,
    )
    runtime.record_checkpoint("checkpoint://step-8", 8)

    assert not runtime.on_distributed_error(_failure())
    assert len(fallback.requests) == 1
    assert fallback.requests[0].checkpoint_locator == "checkpoint://step-8"
    assert fallback.requests[0].checkpoint_step == 8
    assert runtime.describe()["last_recovery"]["result"] == "fallback"


def test_unknown_application_error_is_not_swallowed():
    runtime = RecoveryRuntime(
        _adapter([]),
        RuntimeConfig(enabled=True),
        coordinator=StaticRecoveryCoordinator(_assignment()),
    )

    assert not runtime.on_distributed_error(RuntimeError("model bug"))
    assert runtime.recovery_epoch == 0


def test_epoch_tracker_rejects_nested_and_old_epochs():
    tracker = RecoveryEpochTracker()
    assert tracker.begin_recovery((1,), 9) == 1
    with pytest.raises(ContractViolation, match="cannot begin"):
        tracker.begin_recovery((1,), 9)
    tracker.mark_provisional(9)
    with pytest.raises(ContractViolation, match="newer"):
        tracker.adopt_recovery(1, (1,), 9)


def test_public_initialize_is_lazy_and_disabled_without_adapter():
    runtime = moegambit.initialize(config=RuntimeConfig(enabled=False))

    assert isinstance(runtime, moegambit.RecoveryRuntime)
    assert not runtime.enabled


def test_assignment_digest_is_rechecked_before_execution():
    assignment = _assignment()
    object.__setattr__(
        assignment,
        "plan",
        replace(assignment.plan, resume_step=8),
    )
    runtime = RecoveryRuntime(
        _adapter([]),
        RuntimeConfig(enabled=True),
        coordinator=StaticRecoveryCoordinator(_assignment()),
    )
    runtime.epochs.begin_recovery((1,), 9)

    with pytest.raises(ContractViolation, match="digest"):
        runtime.executor.execute(
            assignment,
            failure_class="fail_stop",
            at_step=9,
        )


def test_executor_rechecks_capabilities_and_resolved_source_identities():
    calls = []
    incapable = replace(
        _adapter(calls),
        capabilities=AdapterCapabilities(
            static_world_replacement=True,
            full_group_rebuild=True,
            peer_parameter_restore=False,
        ),
    )
    incapable_runtime = RecoveryRuntime(
        incapable,
        RuntimeConfig(enabled=True),
        coordinator=StaticRecoveryCoordinator(_assignment()),
    )
    assert not incapable_runtime.on_distributed_error(_failure())
    assert "cannot restore state from a peer" in incapable_runtime.describe()[
        "last_recovery"
    ]["validation"]["error"]

    unresolved = replace(_assignment(), sources=StateSources({}))
    source_runtime = RecoveryRuntime(
        _adapter([]),
        RuntimeConfig(enabled=True),
        coordinator=StaticRecoveryCoordinator(unresolved),
    )
    assert not source_runtime.on_distributed_error(_failure())
    assert "state-source identities" in source_runtime.describe()["last_recovery"][
        "validation"
    ]["error"]


class _LoopbackSocket:
    def __init__(self, processor):
        self.processor = processor
        self.response = bytearray()

    def settimeout(self, timeout):
        self.timeout = timeout

    def sendall(self, data):
        self.response.extend(self.processor.process(data))

    def recv(self, size):
        if not self.response:
            return b""
        chunk = bytes(self.response[:size])
        del self.response[:size]
        return chunk

    def close(self):
        pass


class _CatalogState(_State):
    def catalog(self):
        return StateCatalog.from_iterable(
            (
                StateRef(
                    identity="parameter/weight",
                    kind=StateKind.PARAMETER,
                    placement=Placement.REPLICATED,
                    owner=0,
                    version=StateVersion(
                        committed_step=9,
                        optimizer_generation=9,
                    ),
                    tensor=object(),
                ),
            )
        )


def _network_adapter(calls):
    return FrameworkAdapter(
        name="network-fake",
        topology=_Topology(calls),
        state=_CatalogState(calls),
        optimizer=_Optimizer(calls),
        training=_Training(calls),
        capabilities=AdapterCapabilities(
            static_world_replacement=True,
            full_group_rebuild=True,
            peer_parameter_restore=True,
        ),
        version="test",
    )


def _watcher_coordinator(processor, rank):
    client = ControlClient(
        ControlClientConfig(
            host="127.0.0.1",
            port=20200,
            job_id="job-e2e",
            attempt_id="attempt-e2e",
            sender={"global_rank": rank, "role": "worker"},
            job_token="shared-secret",
        ),
        socket_factory=lambda address, timeout: _LoopbackSocket(processor),
    )
    return WatcherRecoveryCoordinator(client)


def test_signed_control_plane_freezes_one_plan_for_survivor_and_replacement():
    service = RecoveryCoordinatorService(
        store_provider=lambda payload: {
            "host": "127.0.0.1",
            "port": 24000 + payload["recovery_epoch"],
            "prefix": f"epoch-{payload['recovery_epoch']}",
        }
    )
    processor = ControlRequestProcessor(
        service,
        job_token="shared-secret",
        require_token=True,
    )
    survivor_coordinator = _watcher_coordinator(processor, 0)
    replacement_coordinator = _watcher_coordinator(processor, 1)
    config = RuntimeConfig(
        enabled=True,
        job_id="job-e2e",
        attempt_id="attempt-e2e",
    )
    survivor = RecoveryRuntime(
        _network_adapter([]),
        config,
        coordinator=survivor_coordinator,
    )

    assert survivor.on_distributed_error(_failure())
    survivor_digest = survivor.describe()["last_recovery"]["validation"][
        "plan_digest"
    ]
    assignment = replacement_coordinator.fetch_assignment(
        1,
        plan_digest=survivor_digest,
    )
    replacement = RecoveryRuntime(
        _network_adapter([]),
        config,
        coordinator=replacement_coordinator,
    )

    assert assignment.plan_digest == survivor_digest
    assert replacement.execute_assignment(assignment, at_step=9) == 9
    assert survivor.commit_iteration(10)
    assert replacement.commit_iteration(10)
    snapshot = service.snapshot("job-e2e", "attempt-e2e")
    assert snapshot["latest_epoch"] == 1
    assert len(snapshot["assignments"]) == 1

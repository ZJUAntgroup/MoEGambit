"""Framework-neutral topology, state, version, and recovery-plan tests."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from moegambit.distributed.topology import GroupSpec, TopologySpec  # noqa: E402
from moegambit.runtime.recovery_plan import (  # noqa: E402
    RecoveryMode,
    RecoveryPlan,
    WorkerEndpoint,
)
from moegambit.state.catalog import (  # noqa: E402
    Placement,
    StateCatalog,
    StateKind,
    StateRef,
    StateSource,
    StateSourceKind,
)
from moegambit.state.version import StateVersion, newest_common_version  # noqa: E402


def _topology(**changes):
    base = TopologySpec(
        world_size=2,
        rank=0,
        logical_axes={"dp": 2},
        coordinates={"dp": 0},
        groups=(
            GroupSpec("world", (0, 1), "gloo", "world", 0),
            GroupSpec("dp", (0, 1), "gloo", "data_parallel", 1),
        ),
        generation=3,
    )
    return replace(base, **changes)


def test_manifest_hash_covers_group_creation_order():
    first = _topology()
    reversed_creation = _topology(
        groups=(
            GroupSpec("world", (0, 1), "gloo", "world", 1),
            GroupSpec("dp", (0, 1), "gloo", "data_parallel", 0),
        )
    )

    assert first.compute_manifest_hash() != reversed_creation.compute_manifest_hash()


def test_topology_agreement_allows_rank_specific_coordinates():
    first = _topology().sealed()
    other_rank = _topology(rank=1, coordinates={"dp": 1}).sealed()

    assert first.agrees_with(other_rank).ok


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        ({"world_size": 3}, "world_size mismatch"),
        ({"logical_axes": {"dp": 1, "tp": 2}}, "logical axes mismatch"),
        ({"generation": 4}, "topology generation mismatch"),
        (
            {
                "groups": (
                    GroupSpec("world", (0, 1), "gloo", "world", 0),
                    GroupSpec("other", (0, 1), "gloo", "data_parallel", 1),
                )
            },
            "group manifest mismatch",
        ),
    ],
)
def test_topology_agreement_reports_contract_mismatches(changed, expected):
    report = _topology().sealed().agrees_with(_topology(**changed).sealed())

    assert not report.ok
    assert any(expected in finding for finding in report.findings)


def test_state_version_ordering_consistency_and_intersection():
    old = StateVersion(8, optimizer_generation=2, recovery_epoch=9)
    new = StateVersion(9, optimizer_generation=0, recovery_epoch=0)
    same_state = StateVersion(8, optimizer_generation=2, recovery_epoch=10)

    assert old < new
    assert old.is_consistent_with(same_state)
    assert newest_common_version(((old, new), (old,))) == old
    assert newest_common_version(((old,), (new,))) is None
    assert newest_common_version(()) is None


def _state_ref(identity="parameter/layer.weight"):
    return StateRef(
        identity=identity,
        kind=StateKind.PARAMETER,
        placement=Placement.REPLICATED,
        owner=0,
        version=StateVersion(1),
        tensor=object(),
    )


def test_state_catalog_rejects_duplicates_and_builds_stable_digests():
    with pytest.raises(ValueError, match="duplicate StateRef identities"):
        StateCatalog.from_iterable((_state_ref(), _state_ref()))

    first = StateCatalog.from_iterable((_state_ref("b"), _state_ref("a")))
    second = StateCatalog.from_iterable((_state_ref("a"), _state_ref("b")))

    assert first.identity_digest() == second.identity_digest()
    assert first.manifest_digest() == second.manifest_digest()


def test_state_ref_requires_stable_identity_and_accessor():
    with pytest.raises(ValueError, match="non-empty string"):
        _state_ref("")
    with pytest.raises(ValueError, match="neither a tensor nor a scalar accessor"):
        StateRef(
            identity="scheduler/step",
            kind=StateKind.SCHEDULER,
            placement=Placement.UNIQUE,
            owner=0,
            version=StateVersion(1),
        )


@pytest.mark.parametrize(
    "kind",
    (
        StateSourceKind.PEER,
        StateSourceKind.CHECKPOINT,
        StateSourceKind.MEMORY_REPLICA,
    ),
)
def test_remote_state_sources_require_stable_locator(kind):
    with pytest.raises(ValueError, match="stable locator"):
        StateSource(kind, StateVersion(1))


def _plan(**changes):
    base = RecoveryPlan(
        protocol_version=1,
        recovery_epoch=4,
        failed_ranks=(1,),
        resume_step=17,
        mode=RecoveryMode.PEER,
        replacements={1: WorkerEndpoint("10.0.0.9", 24001, 8, 1)},
        topology_generation=5,
        group_manifest_hash="sha256:manifest",
        state_sources={"parameter/a": {"peer": 0}},
        policy_evidence={"reason": "peer available"},
    )
    return replace(base, **changes)


def test_recovery_plan_digest_covers_decisions_but_not_diagnostics():
    base = _plan()

    assert base.digest() == _plan(policy_evidence={"other": True}).digest()
    assert base.digest() != _plan(state_sources={"parameter/a": {"peer": 99}}).digest()
    assert base.digest() != _plan(mode=RecoveryMode.CHECKPOINT).digest()
    assert base.digest() != _plan(resume_step=16).digest()
    assert base.digest() != _plan(failed_ranks=(0, 1)).digest()


def test_recovery_plan_round_trips_typed_sources_and_replacements():
    source = StateSource(
        StateSourceKind.MEMORY_REPLICA,
        StateVersion(17, optimizer_generation=2, recovery_epoch=4),
        "rank://3/owner/1",
        metadata={"manifest": "sha256:state"},
    )
    original = _plan(state_sources={"optimizer/a": source})

    restored = RecoveryPlan.from_dict(original.as_dict())

    assert restored.as_dict() == original.as_dict()
    assert restored.digest() == original.digest()


def test_recovery_plan_rejects_opaque_only_state_source():
    with pytest.raises(TypeError, match="stable descriptor"):
        _plan(state_sources={"parameter/a": {"tensor": object()}}).digest()

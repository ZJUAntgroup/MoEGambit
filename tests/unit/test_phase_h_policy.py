"""Framework-neutral policy and state-source planning tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from moegambit.capabilities import AdapterCapabilities
from moegambit.control.service import RecoveryCoordinatorService
from moegambit.errors import ContractViolation, RecoveryRejected
from moegambit.policy import (
    DeterministicStateSourcePlanner,
    MoeHybridPolicy,
    PeerOrCheckpointPolicy,
    RecoveryFacts,
    parse_source_candidates,
    serialize_source_candidates,
)
from moegambit.runtime.recovery_plan import RecoveryMode
from moegambit.state.catalog import StateSource, StateSourceKind
from moegambit.state.version import StateVersion


def _version(step: int, generation: int | None = None) -> StateVersion:
    return StateVersion(
        committed_step=step,
        optimizer_generation=step if generation is None else generation,
        recovery_epoch=1,
    )


def _source(
    kind: StateSourceKind,
    step: int,
    locator: str,
    *,
    placement: str = "replicated",
    expert: bool = False,
) -> StateSource:
    return StateSource(
        kind=kind,
        version=_version(step),
        locator=locator,
        metadata={
            "kind": "parameter",
            "placement": placement,
            "owner": -1,
            "shape": [2],
            "dtype": "float32",
            "tags": ["expert"] if expert else [],
        },
    )


def _capabilities(**overrides: bool) -> AdapterCapabilities:
    values = {
        "static_world_replacement": True,
        "full_group_rebuild": True,
        "peer_parameter_restore": True,
    }
    values.update(overrides)
    return AdapterCapabilities(**values)


def _facts(sources, *, step=10, checkpoint=-1, capabilities=None):
    return RecoveryFacts(
        failed_ranks=(1,),
        resume_step=step,
        latest_checkpoint_step=checkpoint,
        available_state_sources=sources,
        exposure_history=(),
        capabilities=capabilities or _capabilities(),
    )


def test_peer_policy_and_planner_choose_one_deterministic_live_source():
    facts = _facts(
        {
            "param/weight": (
                _source(StateSourceKind.PEER, 10, "rank://3"),
                _source(StateSourceKind.PEER, 10, "rank://0"),
            )
        }
    )

    decision = PeerOrCheckpointPolicy().decide(facts)
    selected = DeterministicStateSourcePlanner().select(facts, decision)

    assert decision.mode is RecoveryMode.PEER
    assert selected["param/weight"].locator == "rank://0"


def test_memory_replica_can_cover_sharded_optimizer_state():
    facts = _facts(
        {
            "optim/shard-1": (
                _source(
                    StateSourceKind.MEMORY_REPLICA,
                    10,
                    "memory://rank/0/optim/shard-1",
                    placement="sharded",
                ),
            )
        },
        capabilities=_capabilities(optimizer_memory_replication=True),
    )

    decision = PeerOrCheckpointPolicy().decide(facts)
    selected = DeterministicStateSourcePlanner().select(facts, decision)

    assert decision.mode is RecoveryMode.PEER
    assert selected["optim/shard-1"].kind is StateSourceKind.MEMORY_REPLICA


def test_moe_policy_builds_hybrid_dense_peer_and_unique_expert_checkpoint():
    facts = _facts(
        {
            "param/dense": (
                _source(StateSourceKind.PEER, 10, "rank://0"),
            ),
            "expert/7": (
                _source(
                    StateSourceKind.CHECKPOINT,
                    8,
                    "checkpoint://step-8/expert/7",
                    placement="unique",
                    expert=True,
                ),
            ),
        },
        checkpoint=8,
        capabilities=_capabilities(moe_state_classification=True),
    )

    decision = MoeHybridPolicy().decide(facts)
    selected = DeterministicStateSourcePlanner().select(facts, decision)

    assert decision.mode is RecoveryMode.HYBRID
    assert selected["param/dense"].kind is StateSourceKind.PEER
    assert selected["expert/7"].kind is StateSourceKind.CHECKPOINT
    assert decision.evidence["checkpoint_gap"] == 2


def test_inconsistent_live_versions_fall_back_to_complete_checkpoint():
    sources = {
        "param/a": (
            _source(StateSourceKind.PEER, 10, "rank://0"),
            _source(StateSourceKind.CHECKPOINT, 8, "checkpoint://8/a"),
        ),
        "param/b": (
            _source(StateSourceKind.PEER, 9, "rank://0"),
            _source(StateSourceKind.CHECKPOINT, 8, "checkpoint://8/b"),
        ),
    }
    facts = _facts(sources, checkpoint=8)

    decision = PeerOrCheckpointPolicy().decide(facts)
    selected = DeterministicStateSourcePlanner().select(facts, decision)

    assert decision.mode is RecoveryMode.CHECKPOINT
    assert {source.kind for source in selected.values()} == {
        StateSourceKind.CHECKPOINT
    }


def test_missing_complete_source_fails_closed():
    facts = _facts(
        {
            "param/a": (_source(StateSourceKind.PEER, 10, "rank://0"),),
            "param/b": (),
        }
    )

    decision = PeerOrCheckpointPolicy().decide(facts)

    assert decision.mode is RecoveryMode.ABORT
    assert DeterministicStateSourcePlanner().select(facts, decision) == {}


def test_source_inventory_round_trip_is_canonical_and_locator_checked():
    candidates = {
        "param/b": (_source(StateSourceKind.PEER, 10, "rank://2"),),
        "param/a": (
            _source(StateSourceKind.PEER, 10, "rank://3"),
            _source(StateSourceKind.PEER, 10, "rank://0"),
        ),
    }

    encoded = serialize_source_candidates(candidates)
    decoded = parse_source_candidates(encoded)

    assert list(encoded) == ["param/a", "param/b"]
    assert [source.locator for source in decoded["param/a"]] == [
        "rank://0",
        "rank://3",
    ]
    invalid = deepcopy(encoded)
    invalid["param/a"][0]["locator"] = ""
    with pytest.raises(RecoveryRejected, match="stable locator"):
        parse_source_candidates(invalid)


def _service_payload(locator="memory://rank/0/shard", epoch=2):
    source = _source(
        StateSourceKind.MEMORY_REPLICA,
        10,
        locator,
        placement="sharded",
    )
    return {
        "at_step": 10,
        "recovery_epoch": epoch,
        "topology_generation": 2,
        "group_manifest_hash": "sha256:groups",
        "world_size": 2,
        "classification": {
            "recoverable": True,
            "failure_class": "fail_stop",
            "failed_ranks": [1],
        },
        "capabilities": _capabilities(
            optimizer_memory_replication=True
        ).as_dict(),
        "state_manifest": "sha256:state",
        "state_catalog": [
            {
                "identity": "optim/shard",
                "kind": "optimizer_tensor",
                "placement": "sharded",
                "owner": 1,
                "version": {
                    "committed_step": 10,
                    "optimizer_generation": 10,
                    "recovery_epoch": 1,
                },
                "shape": [2],
                "dtype": "float32",
                "tags": [],
                "metadata": {},
            }
        ],
        "available_state_sources": serialize_source_candidates(
            {"optim/shard": (source,)}
        ),
        "latest_checkpoint_step": -1,
        "exposure_history": [],
    }


def test_control_service_freezes_sharded_memory_source_and_digest_covers_locator():
    service = RecoveryCoordinatorService(
        store_provider=lambda _payload: {"host": "127.0.0.1", "port": 23456}
    )
    response = service.prepare(
        _service_payload(), job_id="job", attempt_id="attempt"
    )

    assert response["plan"]["mode"] == "peer"
    assert (
        response["plan"]["state_sources"]["optim/shard"]["locator"]
        == "memory://rank/0/shard"
    )
    with pytest.raises(ContractViolation, match="different recovery facts"):
        service.prepare(
            _service_payload("memory://rank/9/wrong"),
            job_id="job",
            attempt_id="attempt",
        )


def test_serialized_capability_booleans_are_not_truthy_strings():
    with pytest.raises(TypeError, match="must be a boolean"):
        AdapterCapabilities.from_dict({"full_group_rebuild": "false"})

"""Persistent control-store and watcher restart consistency tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from moegambit.control.service import RecoveryCoordinatorService
from moegambit.control.state_store import SQLiteControlStore
from moegambit.errors import ContractViolation


def _payload(*, epoch: int = 2, locator: str = "rank://0"):
    version = {
        "committed_step": 5,
        "optimizer_generation": 5,
        "recovery_epoch": 1,
    }
    metadata = {
        "kind": "parameter",
        "placement": "replicated",
        "owner": -1,
        "shape": [2],
        "dtype": "float32",
        "tags": [],
    }
    return {
        "at_step": 5,
        "recovery_epoch": epoch,
        "topology_generation": 3,
        "group_manifest_hash": "sha256:groups",
        "world_size": 2,
        "classification": {
            "recoverable": True,
            "failure_class": "fail_stop",
            "failed_ranks": [1],
        },
        "capabilities": {
            "static_world_replacement": True,
            "full_group_rebuild": True,
            "selective_group_rebuild": False,
            "peer_parameter_restore": True,
        },
        "state_manifest": "sha256:state",
        "state_catalog": [
            {
                "identity": "param/weight",
                "kind": "parameter",
                "placement": "replicated",
                "owner": -1,
                "version": dict(version),
                "shape": [2],
                "dtype": "float32",
                "tags": [],
                "metadata": {},
            }
        ],
        "available_state_sources": {
            "param/weight": [
                {
                    "kind": "peer",
                    "version": dict(version),
                    "locator": locator,
                    "metadata": metadata,
                }
            ]
        },
        "latest_checkpoint_step": -1,
        "exposure_history": [],
    }


def _service(store, port=23000):
    return RecoveryCoordinatorService(
        store_provider=lambda _payload: {"host": "127.0.0.1", "port": port},
        control_store=store,
    )


def test_sqlite_store_is_monotonic_idempotent_and_process_shareable(tmp_path):
    path = tmp_path / "control.db"
    first = SQLiteControlStore(str(path), poll_interval_s=0.01)
    second = SQLiteControlStore(str(path), poll_interval_s=0.01)

    assert first.put_if_epoch("plan", {"value": "epoch-2"}, 2)
    assert second.put_if_epoch("plan", {"value": "epoch-2"}, 2)
    assert not second.put_if_epoch("plan", {"value": "split-brain"}, 2)
    assert not second.put_if_epoch("plan", {"value": "stale"}, 1)
    assert second.compare_and_set(
        "plan", {"value": "epoch-2"}, {"value": "committed"}, 2
    )
    assert first.get("plan") == {"value": "committed"}
    assert first.get_epoch("plan") == 2


def test_service_recovers_frozen_plan_and_commits_after_watcher_restart(tmp_path):
    path = tmp_path / "control.db"
    first = _service(SQLiteControlStore(str(path)), port=23000)
    response = first.prepare(_payload(), job_id="job/a", attempt_id="attempt/1")
    assert first.committed(
        {"plan_digest": response["plan_digest"], "step": 6},
        job_id="job/a",
        attempt_id="attempt/1",
        rank=0,
        recovery_epoch=2,
    )["committed_count"] == 1

    restarted = _service(SQLiteControlStore(str(path)), port=29999)
    fetched = restarted.get_assignment(
        {"plan_digest": response["plan_digest"]},
        job_id="job/a",
        attempt_id="attempt/1",
        recovery_epoch=2,
    )
    second_commit = restarted.committed(
        {"plan_digest": response["plan_digest"], "step": 6},
        job_id="job/a",
        attempt_id="attempt/1",
        rank=1,
        recovery_epoch=2,
    )

    assert fetched == response
    assert fetched["store"]["port"] == 23000
    assert second_commit["committed_count"] == 2
    assert restarted.heartbeat(
        job_id="job/a", attempt_id="attempt/1"
    )["latest_recovery_epoch"] == 2


def test_two_watcher_writers_cannot_freeze_different_same_epoch_plans(tmp_path):
    path = tmp_path / "control.db"
    services = [
        _service(SQLiteControlStore(str(path)), port=23000),
        _service(SQLiteControlStore(str(path)), port=23001),
    ]
    payloads = [_payload(locator="rank://0"), _payload(locator="rank://9")]

    def prepare(index):
        return services[index].prepare(
            deepcopy(payloads[index]), job_id="job", attempt_id="attempt"
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(prepare, index) for index in range(2)]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(("ok", future.result()))
            except ContractViolation as exc:
                outcomes.append(("error", str(exc)))

    assert [kind for kind, _value in outcomes].count("ok") == 1
    assert [kind for kind, _value in outcomes].count("error") == 1
    assert "different recovery facts" in next(
        value for kind, value in outcomes if kind == "error"
    )


def test_sqlite_store_require_same_epoch_detects_incomplete_or_mixed_state(tmp_path):
    store = SQLiteControlStore(str(tmp_path / "control.db"))
    store.put_if_epoch("a", 1, 2)
    store.put_if_epoch("b", 2, 2)
    assert store.require_same_epoch("a", "b") == 2
    store.put_if_epoch("b", 3, 3)
    with pytest.raises(ContractViolation, match="complete epoch"):
        store.require_same_epoch("a", "b")


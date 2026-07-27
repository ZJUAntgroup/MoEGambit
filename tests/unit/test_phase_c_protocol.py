"""Phase C protocol, authentication, epoch and plan-freezing tests."""

from __future__ import annotations

import json

import pytest

from moegambit.control.protocol import Envelope, MessageDeduplicator
from moegambit.control.service import RecoveryCoordinatorService
from moegambit.control.state_store import InMemoryControlStore
from moegambit.control.watcher import ControlRequestProcessor, ControlServer
from moegambit.errors import ContractViolation, ProtocolVersionMismatch
from moegambit.runtime.client import ControlClient, ControlClientConfig


def _envelope(message_type="heartbeat", payload=None, epoch=2):
    return Envelope.new(
        message_type,
        payload or {"step": 7},
        job_id="job-1",
        attempt_id="attempt-1",
        recovery_epoch=epoch,
        sender={"node_rank": 0, "global_rank": 0},
    )


def _service_request(epoch=2):
    return {
        "at_step": 5,
        "recovery_epoch": epoch,
        "topology_generation": 3,
        "group_manifest_hash": "sha256:groups",
        "world_size": 2,
        "rank": 0,
        "classification": {
            "recoverable": True,
            "failure_class": "fail_stop",
            "failed_ranks": [1],
            "evidence": {"rank_local": "diagnostic"},
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
                "identity": "parameter/weight",
                "kind": "parameter",
                "placement": "replicated",
                "owner": 0,
                "version": {
                    "committed_step": 5,
                    "optimizer_generation": 5,
                    "recovery_epoch": 1,
                },
                "shape": [2, 2],
                "dtype": "float32",
                "tags": [],
                "metadata": {"parameter_name": "weight"},
            }
        ],
    }


def _service():
    return RecoveryCoordinatorService(
        store_provider=lambda payload: {
            "host": "127.0.0.1",
            "port": 23000 + int(payload["recovery_epoch"]),
            "prefix": f"epoch-{payload['recovery_epoch']}",
        }
    )


def test_envelope_round_trip_hmac_and_response_correlation():
    original = _envelope().sign("secret")
    restored = Envelope.from_json(original.to_json())

    restored.verify("secret", now_ns=restored.sent_at_ns)
    response = restored.response({"ok": True}).sign("secret")
    assert response.payload["request_id"] == restored.message_id
    assert response.job_id == restored.job_id
    assert response.recovery_epoch == restored.recovery_epoch


def test_envelope_rejects_tampering_staleness_and_protocol_mismatch():
    envelope = _envelope().sign("secret")
    envelope.payload["step"] = 999
    with pytest.raises(ValueError, match="invalid"):
        envelope.verify("secret", now_ns=envelope.sent_at_ns)

    stale = _envelope()
    stale.sent_at_ns = 1_000_000_000
    stale.sign("secret")
    with pytest.raises(ValueError, match="timestamp"):
        stale.verify("secret", max_clock_skew_s=1, now_ns=3_000_000_001)

    values = json.loads(_envelope().to_json())
    values["protocol_version"] = 99
    with pytest.raises(ProtocolVersionMismatch):
        Envelope.from_json(json.dumps(values))


def test_message_deduplicator_is_bounded():
    deduplicator = MessageDeduplicator(capacity=2)
    assert deduplicator.accept("a")
    assert not deduplicator.accept("a")
    assert deduplicator.accept("b")
    assert deduplicator.accept("c")
    assert deduplicator.accept("a")


def test_request_processor_requires_auth_and_replays_identical_response():
    processor = ControlRequestProcessor(
        _service(),
        job_token="secret",
        require_token=True,
    )
    unsigned = _envelope()
    with pytest.raises(ContractViolation, match="unsigned"):
        processor.process((unsigned.to_json() + "\n").encode())

    signed = _envelope().sign("secret")
    request = (signed.to_json() + "\n").encode()
    first = processor.process(request)
    repeated = processor.process(request)
    assert repeated == first
    response = Envelope.from_json(first.decode().strip())
    response.verify("secret", now_ns=response.sent_at_ns)
    assert response.payload["ok"] is True


def test_request_processor_rejects_message_id_reuse_and_oversize():
    processor = ControlRequestProcessor(
        _service(),
        job_token="secret",
        require_token=True,
        max_message_bytes=4096,
    )
    first = _envelope(payload={"value": 1}).sign("secret")
    processor.process((first.to_json() + "\n").encode())
    changed = _envelope(payload={"value": 2})
    changed.message_id = first.message_id
    changed.sign("secret")
    with pytest.raises(ContractViolation, match="reused"):
        processor.process((changed.to_json() + "\n").encode())
    with pytest.raises(ValueError, match="exceeds"):
        processor.process(b"x" * 4097)


def test_service_freezes_one_plan_and_rejects_split_brain_and_stale_epoch():
    service = _service()
    first = service.prepare(_service_request(), job_id="job", attempt_id="a")
    diagnostic_change = _service_request()
    diagnostic_change["classification"]["evidence"] = {"different": True}
    repeated = service.prepare(
        diagnostic_change,
        job_id="job",
        attempt_id="a",
    )
    assert repeated["plan_digest"] == first["plan_digest"]

    divergent = _service_request()
    divergent["state_catalog"][0]["metadata"] = {"parameter_name": "other"}
    with pytest.raises(ContractViolation, match="different recovery facts"):
        service.prepare(divergent, job_id="job", attempt_id="a")

    service.prepare(_service_request(epoch=3), job_id="job", attempt_id="a")
    with pytest.raises(ContractViolation, match="stale recovery epoch"):
        service.get_assignment(
            {},
            job_id="job",
            attempt_id="a",
            recovery_epoch=2,
        )


def test_service_rejects_non_replicated_state_and_invalid_commit():
    service = _service()
    sharded = _service_request()
    sharded["state_catalog"][0]["placement"] = "sharded"
    with pytest.raises(Exception, match="policy aborted"):
        service.prepare(sharded, job_id="job", attempt_id="a")

    response = service.prepare(_service_request(), job_id="job", attempt_id="b")
    with pytest.raises(ContractViolation, match="complete post-recovery"):
        service.committed(
            {"plan_digest": response["plan_digest"], "step": 5},
            job_id="job",
            attempt_id="b",
            rank=0,
            recovery_epoch=2,
        )


def test_in_memory_control_store_rejects_stale_epoch_and_supports_cas():
    store = InMemoryControlStore()
    assert store.put_if_epoch("plan", "epoch-2", 2)
    assert not store.put_if_epoch("plan", "stale", 1)
    assert store.get("plan") == "epoch-2"
    assert not store.compare_and_set("plan", "wrong", "new", 2)
    assert store.compare_and_set("plan", "epoch-2", "committed", 2)
    assert store.wait({"plan": lambda value: value == "committed"}, 0.1) == {
        "plan": "committed"
    }


def test_checkpoint_relaunch_request_is_auditable():
    service = _service()
    result = service.request_checkpoint_relaunch(
        {
            "at_step": 9,
            "reason": "peer restore failed",
            "error_type": "StateUnavailable",
            "evidence": {"source": "peer"},
            "checkpoint_locator": "checkpoint:///models/job/iter_0000007",
            "checkpoint_step": 7,
        },
        job_id="job",
        attempt_id="a",
        rank=0,
        recovery_epoch=2,
    )

    assert result["ok"] is True
    assert result["action"] == "checkpoint_relaunch"
    assert result["directive"]["checkpoint_step"] == 7
    snapshot = service.snapshot("job", "a")
    assert snapshot["fallback_requests"][0]["reason"] == "peer restore failed"


def test_node_agent_heartbeat_receives_and_acknowledges_relaunch_once():
    service = _service()
    result = service.request_checkpoint_relaunch(
        {
            "at_step": 9,
            "reason": "peer restore failed",
            "error_type": "StateUnavailable",
            "checkpoint_locator": "checkpoint://step-7",
            "checkpoint_step": 7,
            "command_digest": "sha256:command",
        },
        job_id="job",
        attempt_id="a",
        rank=0,
        recovery_epoch=2,
    )
    heartbeat = service.heartbeat(
        {
            "role": "node_agent",
            "node_rank": 0,
            "command_digest": "sha256:command",
        },
        job_id="job",
        attempt_id="a",
    )

    assert heartbeat["relaunch"] == result["directive"]
    acknowledgement = service.acknowledge_checkpoint_relaunch(
        {
            "directive_id": result["directive"]["directive_id"],
            "next_attempt_id": result["directive"]["next_attempt_id"],
            "command_digest": "sha256:command",
            "worker_count": 8,
        },
        job_id="job",
        attempt_id="a",
        node_rank=0,
        recovery_epoch=2,
    )
    repeated = service.heartbeat(
        {
            "role": "node_agent",
            "node_rank": 0,
            "command_digest": "sha256:command",
        },
        job_id="job",
        attempt_id="a",
    )

    assert acknowledgement["acknowledged_nodes"] == 1
    assert "relaunch" not in repeated


def test_same_epoch_cannot_choose_two_checkpoint_relaunch_targets():
    service = _service()
    base = {
        "at_step": 9,
        "reason": "test",
        "checkpoint_locator": "checkpoint://step-7",
        "checkpoint_step": 7,
    }
    service.request_checkpoint_relaunch(
        base,
        job_id="job",
        attempt_id="a",
        rank=0,
        recovery_epoch=2,
    )
    changed = dict(base, checkpoint_locator="checkpoint://other-step-7")

    with pytest.raises(ContractViolation, match="different checkpoint"):
        service.request_checkpoint_relaunch(
            changed,
            job_id="job",
            attempt_id="a",
            rank=1,
            recovery_epoch=2,
        )


def test_real_control_server_round_trip_on_loopback():
    processor = ControlRequestProcessor(
        _service(),
        job_token="secret",
        require_token=True,
    )
    try:
        server = ControlServer("127.0.0.1", 0, processor)
    except PermissionError:
        pytest.skip("the execution sandbox does not permit loopback listeners")
    with server:
        host, port = server.address
        client = ControlClient(
            ControlClientConfig(
                host=host,
                port=port,
                job_id="job",
                attempt_id="attempt",
                sender={"global_rank": 0},
                job_token="secret",
                request_timeout_s=2.0,
            )
        )
        response = client.request("heartbeat", {}, recovery_epoch=0)

    assert response["ok"] is True
    assert response["latest_recovery_epoch"] == 0

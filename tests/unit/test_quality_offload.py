"""Retention survives worker exit and never substitutes stale or partial inputs."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from moegambit.audit.io import digest
from moegambit.control.service import RecoveryCoordinatorService
from moegambit.control.watcher import ControlRequestProcessor, ControlServer
from moegambit.quality import AsyncQualityOffloader, CPUQualityFeatureStore
from moegambit.runtime.client import ControlClient, ControlClientConfig


IDENTITY = dict(model_id="test-moe", telemetry_version="v1", policy_version="v1")


def record(rank=0, step=1000, **updates):
    return {"schema_version": 1, **IDENTITY, "owner_rank": rank,
            "recovery_epoch": 0,
            "committed_step": step, "checkpoint_step": 800,
            "topology_generation": 0, "group_manifest_hash": "layout", "world_size": 2,
            "features": {"layer": [0], "expert": [rank], "routed_mass": [.125],
                         "sensitivity": [1.0], "parameter_drift": [.01],
                         "momentum_drift": [.02], "variance_drift": [.03]},
            "run_history": [], **updates}


def payload(**updates):
    return {"quality_context": {**IDENTITY, "features": {"forged": True}, "run_history": []},
            "recovery_epoch": 1, "quality_source_recovery_epoch": 0,
            "quality_source_topology_generation": 0, "topology_generation": 1,
            "group_manifest_hash": "layout", "world_size": 2,
            "at_step": 1000, "latest_checkpoint_step": 800, **updates}


def publish(store, value, **scope):
    return store.publish(value, job_id=scope.get("job_id", "job"),
                         attempt_id=scope.get("attempt_id", "a1"), rank=value["owner_rank"])


def context(store, value=None, **scope):
    return store.recovery_context(value or payload(), job_id=scope.get("job_id", "job"),
                                  attempt_id=scope.get("attempt_id", "a1"))


def complete_store(**kwargs):
    store = CPUQualityFeatureStore(**kwargs)
    publish(store, record(0)); publish(store, record(1))
    return store


def test_immutable_acknowledged_snapshot_replaces_worker_features_and_history():
    store = CPUQualityFeatureStore()
    first = record(0, run_history=[{"mode": "hybrid", "step": 900}])
    publish(store, first)
    first["features"]["routed_mass"][0] = 99
    publish(store, record(1, run_history=[{"mode": "hybrid", "step": 900}]))
    retained = context(store)
    assert retained["retained_features_complete"] is True
    assert retained["features"]["by_rank"]["0"]["routed_mass"] == [.125]
    assert retained["run_history"] == [{"mode": "hybrid", "step": 900}]
    retained["features"]["by_rank"]["0"]["routed_mass"][0] = 99
    assert context(store)["features"]["by_rank"]["0"]["routed_mass"] == [.125]


@pytest.mark.parametrize("change", [dict(at_step=1001), dict(latest_checkpoint_step=600),
    dict(group_manifest_hash="different"), dict(quality_source_topology_generation=1),
    dict(topology_generation=2), dict(world_size=3), dict(recovery_epoch=2),
    dict(quality_source_recovery_epoch=1)])
def test_scope_or_reference_mismatch_never_uses_an_older_snapshot(change):
    assert context(complete_store(), payload(**change))["retained_features_complete"] is False


def test_partial_snapshot_exposes_missing_rank_without_trusting_request():
    store = CPUQualityFeatureStore(); publish(store, record(0))
    result = context(store)
    assert result["missing_feature_ranks"] == [1]
    assert result["retained_features_complete"] is False
    assert "features" not in result
    assert context(store, attempt_id="other")["retained_features_complete"] is False


def test_duplicate_is_idempotent_conflict_is_rejected():
    store = CPUQualityFeatureStore(); ack = publish(store, record())
    assert publish(store, record()) == ack
    with pytest.raises(ValueError, match="conflicting"):
        publish(store, record(features={"changed": True}))
    publish(store, record(1))
    assert context(store)["feature_retention_error"] == "conflicting_feature_snapshot"


def test_owner_and_finite_values_are_checked():
    store = CPUQualityFeatureStore()
    with pytest.raises(ValueError, match="owner"):
        store.publish(record(), job_id="job", attempt_id="a1", rank=1)
    with pytest.raises(ValueError):
        publish(store, record(features={"x": float("nan")}))


def test_bounded_window_rejects_late_record_and_preserves_newest():
    store = complete_store(retain_steps=2)
    for step in (1001, 1002):
        for rank in (0, 1): publish(store, record(rank, step))
    with pytest.raises(ValueError, match="retained window"):
        publish(store, record(0, 999))
    assert context(store)["retained_features_complete"] is False
    assert context(store, payload(at_step=1002))["retained_features_complete"] is True


def test_byte_and_scope_limits_fail_without_partial_insertions():
    with pytest.raises(ValueError, match="record exceeds"):
        publish(CPUQualityFeatureStore(max_record_bytes=20), record())
    store = CPUQualityFeatureStore(max_total_bytes=10)
    with pytest.raises(ValueError, match="total byte"):
        publish(store, record())
    assert context(store)["missing_feature_ranks"] == [0, 1]
    store = complete_store(max_scopes=1)
    with pytest.raises(ValueError, match="scope limit"):
        publish(store, record(), attempt_id="a2")


def test_history_is_append_only_across_relaunch_and_checkpoint_rollback():
    store = CPUQualityFeatureStore()
    publish(store, record(run_history=[{"peak_violation_observed": True}]))
    with pytest.raises(ValueError, match="truncated"):
        publish(store, record(step=1001))
    with pytest.raises(ValueError, match="truncated"):
        publish(store, record(step=800), attempt_id="a2")
    with pytest.raises(ValueError, match="append-only"):
        publish(store, record(step=1001, run_history=[{"peak_violation_observed": False}]))


def test_ranks_with_different_history_cannot_form_complete_context():
    store = CPUQualityFeatureStore()
    publish(store, record(0)); publish(store, record(1, run_history=[{"mode": "restart"}]))
    assert context(store)["feature_retention_error"] == "rank_history_mismatch"


class BlockingClient:
    config = SimpleNamespace(sender={"global_rank": 0})
    def __init__(self):
        self.entered = threading.Event(); self.release = threading.Event(); self.records = []
    def request(self, kind, value, **kwargs):
        self.entered.set()
        assert self.release.wait(5), "test failed to release background upload"
        self.records.append(deepcopy(value))
        return {"ok": True, "record_digest": digest(value), "committed_step": value["committed_step"]}


def submit(offloader, features, **updates):
    return offloader.submit(features, committed_step=1000, checkpoint_step=800,
                           topology_generation=0, group_manifest_hash="layout", world_size=2,
                           run_history=[], **updates)


def test_training_can_continue_while_upload_pending_and_backpressure_is_bounded():
    client = BlockingClient()
    with AsyncQualityOffloader(client, **IDENTITY, max_pending=1) as offloader:
        features = {"x": [1]}
        future = submit(offloader, features)
        assert client.entered.wait(2)
        features["x"][0] = 999  # simulates next training step writing its buffer
        assert not future.done()
        assert submit(offloader, {"x": [2]}) is None
        client.release.set(); future.result(timeout=5)
        assert client.records[0]["features"] == {"x": [1]}
        assert offloader.stats()["dropped"] == 1
        assert offloader.stats()["acknowledged"] == 1


def test_upload_errors_are_recorded_and_slot_is_released(caplog):
    client = BlockingClient(); client.release.set()
    client.request = lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("watcher down"))
    with AsyncQualityOffloader(client, **IDENTITY, max_pending=1) as offloader:
        for _ in range(2):
            with pytest.raises(ConnectionError): submit(offloader, {}).result(timeout=5)
        assert offloader.stats()["failed"] == 2
        assert "watcher down" in offloader.stats()["last_error"]
        assert "quality CPU publication failed" in caplog.text


def test_element_limit_includes_empty_containers():
    with AsyncQualityOffloader(BlockingClient(), **IDENTITY, max_elements=4) as offloader:
        with pytest.raises(ValueError, match="element limit"):
            submit(offloader, {"empty": [{} for _ in range(20)]})


def test_cpu_tensor_is_frozen_before_background_upload():
    torch = pytest.importorskip("torch")
    client = BlockingClient()
    with AsyncQualityOffloader(client, **IDENTITY) as offloader:
        value = torch.tensor([1., 2.])
        future = submit(offloader, {"x": value})
        assert client.entered.wait(2)
        value.add_(100)
        client.release.set(); future.result(timeout=5)
        assert client.records[0]["features"] == {"x": [1., 2.]}


def test_cuda_copy_is_ordered_and_independent_of_next_step_mutation():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available(): pytest.skip("CUDA is not available")
    client = BlockingClient(); client.release.set()
    with AsyncQualityOffloader(client, **IDENTITY) as offloader:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            value = torch.zeros(1024, device="cuda")
            value.fill_(7)
            future = submit(offloader, {"x": value})
            value.fill_(99)
        future.result(timeout=10)
        assert client.records[0]["features"]["x"] == [7.] * 1024


def test_independent_watcher_retains_features_after_worker_abrupt_exit():
    store = CPUQualityFeatureStore()
    service = RecoveryCoordinatorService(store_provider=lambda payload: {}, quality_feature_store=store)
    processor = ControlRequestProcessor(service, job_token="test-token", require_token=True)
    with ControlServer("127.0.0.1", 0, processor) as server:
        host, port = server.address
        config = ControlClientConfig(host, port, "job", "a1", {"global_rank": 0}, job_token="test-token")
        with AsyncQualityOffloader(ControlClient(config), **IDENTITY) as offloader:
            submit(offloader, {"survivor": 1}).result(timeout=5)
        source = Path(__file__).resolve().parents[2] / "src"
        code = '''
import os, sys
from moegambit.runtime.client import ControlClient, ControlClientConfig
from moegambit.quality import AsyncQualityOffloader
client = ControlClient(ControlClientConfig(sys.argv[1], int(sys.argv[2]), "job", "a1",
                        {"global_rank": 1}, job_token="test-token"))
offload = AsyncQualityOffloader(client, model_id="test-moe", telemetry_version="v1", policy_version="v1")
offload.submit({"lost_rank_drift": [.01, .02, .03]}, committed_step=1000, checkpoint_step=800,
               topology_generation=0, group_manifest_hash="layout", world_size=2,
               run_history=[]).result(timeout=5)
os._exit(23)
'''
        child = subprocess.run([sys.executable, "-c", code, host, str(port)],
                               env={**os.environ, "PYTHONPATH": str(source)}, capture_output=True, timeout=15)
        assert child.returncode == 23, child.stderr.decode()
        retained = context(store)
        assert retained["retained_features_complete"] is True
        assert retained["features"]["by_rank"]["1"] == {"lost_rank_drift": [.01, .02, .03]}
        assert context(CPUQualityFeatureStore())["retained_features_complete"] is False

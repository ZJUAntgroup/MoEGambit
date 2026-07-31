"""Framework-neutral optimizer replication contracts."""

from __future__ import annotations

import inspect
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from moegambit.replication import (
    FrameLimits,
    MemoryReplicaLocator,
    OptimizerMemorySnapshot,
    TransportIntegrityError,
    memory_replica_sources,
    ring_neighbors,
)
from moegambit.replication.transport import (
    payload_digest,
    receive_json,
    send_json,
)
from moegambit.state.catalog import StateSourceKind


def test_memory_replica_locator_round_trip_keeps_identity_and_generation():
    locator = MemoryReplicaLocator(
        generation=4,
        owner_rank=3,
        holder_rank=7,
        identity="optim/layer 1/exp_avg",
    )

    restored = MemoryReplicaLocator.parse(str(locator))

    assert restored == locator


def test_ring_mapping_is_deterministic_and_rejects_duplicates():
    assert ring_neighbors([1, 9, 17, 25], 9) == (1, 17)
    with pytest.raises(RuntimeError, match="duplicate"):
        ring_neighbors([0, 1, 1], 0)


def test_bounded_json_frame_round_trip_over_socket_pair():
    left, right = socket.socketpair()
    received = {}

    def receive():
        received.update(
            receive_json(right, limits=FrameLimits(max_header_bytes=1024))
        )

    thread = threading.Thread(target=receive)
    thread.start()
    try:
        send_json(
            left,
            {"generation": 2, "step": 8, "committed": True},
            limits=FrameLimits(max_header_bytes=1024),
        )
        thread.join(timeout=2.0)
    finally:
        left.close()
        right.close()

    assert not thread.is_alive()
    assert received == {"generation": 2, "step": 8, "committed": True}


def test_frame_limit_and_payload_digest_detect_invalid_or_changed_data():
    left, right = socket.socketpair()
    try:
        with pytest.raises(TransportIntegrityError, match="exceeds"):
            send_json(
                left,
                {"payload": "x" * 100},
                limits=FrameLimits(max_header_bytes=16),
            )
    finally:
        left.close()
        right.close()

    assert payload_digest(b"abc") != payload_digest(b"abd")


def test_snapshot_projects_to_stable_policy_sources_without_framework_objects():
    snapshot = OptimizerMemorySnapshot(
        owner_rank=2,
        holder_rank=3,
        step=11,
        generation=5,
        manifest_hash="sha256:manifest",
        manifest=[
            {
                "identity": "optim/layer/exp_avg",
                "shape": [4],
                "dtype": "torch.float32",
                "numel": 4,
                "offset": 0,
                "is_expert": False,
            }
        ],
        scalars={"optim/layer/step": 11},
        segments=[],
        buffers={},
    )

    sources = memory_replica_sources(snapshot, recovery_epoch=1)

    assert set(sources) == {"optim/layer/exp_avg", "optim/layer/step"}
    tensor = sources["optim/layer/exp_avg"]
    assert tensor.kind is StateSourceKind.MEMORY_REPLICA
    assert tensor.version.committed_step == 11
    assert MemoryReplicaLocator.parse(tensor.locator).holder_rank == 3


def test_replication_package_import_does_not_require_or_import_torch():
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(Path(__file__).parents[2] / 'src')!r}); "
        "import moegambit.replication; "
        "raise SystemExit(1 if 'torch' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_legacy_runtime_replica_path_exports_the_canonical_implementation():
    from moegambit.replication import (
        Zero2MemoryReplicaManager,
        capture_optimizer_snapshot as canonical_capture,
    )
    from moegambit.runtime.zero2_replica import (
        Zero2MemoryReplicaManager as CompatibilityManager,
        capture_optimizer_snapshot as compatibility_capture,
    )

    assert CompatibilityManager is Zero2MemoryReplicaManager
    assert compatibility_capture is canonical_capture
    parameters = inspect.signature(Zero2MemoryReplicaManager).parameters
    assert parameters["buffer_slots"].default == 2
    assert hasattr(Zero2MemoryReplicaManager, "wait_until_peer_committed")


def test_canonical_replica_manager_preserves_deepspeed_slot_contract(
    monkeypatch,
):
    from moegambit.replication import optimizer_memory

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setattr(optimizer_memory, "_require_torch", lambda: fake_torch)
    manager = optimizer_memory.Zero2MemoryReplicaManager(
        rank=0,
        tensor_refs_fn=lambda: [],
        scalar_refs_fn=lambda: [],
        publish_endpoint_fn=lambda *_args: True,
        wait_endpoint_fn=lambda *_args: None,
        buffer_slots=1,
    )

    assert manager.buffer_slots == 1
    assert len(manager._slots) == 1
    assert len(manager._peer_slots) == 1
    manager._peer_committed_step = 4
    manager.wait_until_peer_committed(4, timeout=0.01)

    with pytest.raises(ValueError, match="buffer_slots"):
        optimizer_memory.Zero2MemoryReplicaManager(
            rank=0,
            tensor_refs_fn=lambda: [],
            scalar_refs_fn=lambda: [],
            publish_endpoint_fn=lambda *_args: True,
            wait_endpoint_fn=lambda *_args: None,
            buffer_slots=3,
        )

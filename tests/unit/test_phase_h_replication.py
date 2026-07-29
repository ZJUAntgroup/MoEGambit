"""Framework-neutral optimizer replication contracts."""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
from pathlib import Path

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

import importlib.util
from pathlib import Path
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

torch = pytest.importorskip("torch")


MODULE_PATH = (
    Path(__file__).parents[1]
    / "Megatron-LM"
    / "megatron"
    / "training"
    / "zero2_memory_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location("zero2_memory_checkpoint", MODULE_PATH)
zero2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = zero2
SPEC.loader.exec_module(zero2)


class _EndpointRegistry:
    def __init__(self):
        self.cv = threading.Condition()
        self.endpoints = {}

    def publish(self, peer_id, port, src_rank, dst_rank):
        with self.cv:
            self.endpoints[peer_id] = {
                "host": "127.0.0.1",
                "port": port,
                "src_rank": src_rank,
                "dst_rank": dst_rank,
            }
            self.cv.notify_all()
        return True

    def wait(self, peer_id, timeout):
        with self.cv:
            return self.cv.wait_for(
                lambda: self.endpoints.get(peer_id), timeout=timeout
            )


def _refs(dense, expert, dense_state, expert_state):
    return [
        zero2.OptimizerTensorRef("dense:main", dense, False),
        zero2.OptimizerTensorRef("dense:exp_avg", dense_state, False),
        zero2.OptimizerTensorRef("expert:main", expert, True),
        zero2.OptimizerTensorRef("expert:exp_avg", expert_state, True),
    ]


def test_ring_neighbors_use_the_next_dp_rank_as_backup_holder():
    assert zero2.ring_neighbors([1, 9, 17, 25], 9) == (1, 17)
    assert zero2.backup_holder_for_owner([1, 9, 17, 25], 25) == 1
    with pytest.raises(RuntimeError, match="DP size"):
        zero2.ring_neighbors([1], 1)


def test_optimizer_snapshot_h2h_commit_and_nonexpert_restore(monkeypatch):
    registry = _EndpointRegistry()
    real_create_connection = zero2.socket.create_connection
    connection_attempts = 0

    def create_connection_with_one_transient_failure(*args, **kwargs):
        nonlocal connection_attempts
        connection_attempts += 1
        if connection_attempts == 1:
            raise ConnectionRefusedError("listener endpoint is not ready yet")
        return real_create_connection(*args, **kwargs)

    monkeypatch.setattr(
        zero2.socket, "create_connection", create_connection_with_one_transient_failure
    )
    rank0_tensors = (
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0]),
        torch.tensor([4.0, 5.0]),
        torch.tensor([6.0]),
    )
    rank1_tensors = tuple(torch.zeros_like(tensor) for tensor in rank0_tensors)

    manager0 = zero2.Zero2MemoryReplicaManager(
        rank=0,
        tensor_refs_fn=lambda: _refs(*rank0_tensors),
        scalar_refs_fn=lambda: [],
        publish_endpoint_fn=registry.publish,
        wait_endpoint_fn=registry.wait,
        timeout=5.0,
    )
    manager1 = zero2.Zero2MemoryReplicaManager(
        rank=1,
        tensor_refs_fn=lambda: _refs(*rank1_tensors),
        scalar_refs_fn=lambda: [],
        publish_endpoint_fn=registry.publish,
        wait_endpoint_fn=registry.wait,
        timeout=5.0,
    )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(manager0.start_transport, [0, 1], 0),
                pool.submit(manager1.start_transport, [0, 1], 0),
            ]
            for future in futures:
                future.result(timeout=10.0)

        manager0.schedule_snapshot(7)
        manager1.schedule_snapshot(7)
        manager0.wait_until_replicated(7, timeout=5.0)
        manager1.wait_until_replicated(7, timeout=5.0)

        snapshot = manager1.get_peer_snapshot(owner_rank=0, step=7)
        destination = (
            torch.tensor([-1.0, -1.0]),
            torch.tensor([-1.0]),
            torch.tensor([-2.0, -2.0]),
            torch.tensor([-2.0]),
        )
        summary = zero2.apply_optimizer_snapshot(
            snapshot,
            _refs(*destination),
            [],
            restore_expert=False,
        )

        assert torch.equal(destination[0], rank0_tensors[0])
        assert torch.equal(destination[2], rank0_tensors[2])
        assert torch.equal(destination[1], torch.tensor([-1.0]))
        assert torch.equal(destination[3], torch.tensor([-2.0]))
        assert summary["step"] == 7
        assert summary["restore_scope"] == "non_expert"

        rank0_tensors[0].add_(10.0)
        rank0_tensors[2].add_(20.0)
        manager0.schedule_snapshot(8)
        manager1.schedule_snapshot(8)
        manager0.wait_until_replicated(8, timeout=5.0)
        manager1.wait_until_replicated(8, timeout=5.0)
        next_snapshot = manager1.get_peer_snapshot(owner_rank=0, step=8)
        assert next_snapshot.step == 8
        assert next_snapshot.manifest_hash == snapshot.manifest_hash
        assert connection_attempts >= 3

        with pytest.raises(RuntimeError, match="unavailable"):
            manager1.get_peer_snapshot(owner_rank=0, step=9)
    finally:
        manager0.stop_transport()
        manager1.stop_transport()


def test_optimizer_snapshot_rejects_manifest_mismatch():
    source = torch.tensor([1.0, 2.0])
    manifest, _, digest = zero2._manifest_for_refs(
        [zero2.OptimizerTensorRef("dense", source, False)]
    )
    snapshot = zero2.OptimizerMemorySnapshot(
        owner_rank=0,
        holder_rank=1,
        step=3,
        manifest_hash=digest,
        manifest=manifest,
        scalars={},
        segments=[{"dtype": "torch.float32", "numel": 2, "byte_count": 8}],
        buffers={"torch.float32": bytearray(source.view(torch.uint8).numpy())},
    )
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        zero2.apply_optimizer_snapshot(
            snapshot,
            [zero2.OptimizerTensorRef("different", torch.zeros(2), False)],
            [],
            restore_expert=False,
        )

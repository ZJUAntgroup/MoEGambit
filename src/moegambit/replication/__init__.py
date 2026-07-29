"""Framework-independent optimizer replication public API."""

from .locator import MemoryReplicaLocator
from .optimizer_memory import (
    OptimizerMemoryReplicaManager,
    OptimizerMemorySnapshot,
    OptimizerScalarRef,
    OptimizerTensorRef,
    Zero2MemoryReplicaManager,
    apply_optimizer_snapshot,
    backup_holder_for_owner,
    memory_replica_sources,
    ring_neighbors,
)
from .transport import FrameLimits, TransportIntegrityError

__all__ = [
    "FrameLimits",
    "MemoryReplicaLocator",
    "OptimizerMemoryReplicaManager",
    "OptimizerMemorySnapshot",
    "OptimizerScalarRef",
    "OptimizerTensorRef",
    "TransportIntegrityError",
    "Zero2MemoryReplicaManager",
    "apply_optimizer_snapshot",
    "backup_holder_for_owner",
    "memory_replica_sources",
    "ring_neighbors",
]

"""Compatibility exports for the canonical optimizer replication package.

Framework-neutral optimizer replication is implemented only in
``moegambit.replication``.  This module keeps the historical import path for
external callers while Megatron and DeepSpeed share the same implementation.
"""

from moegambit.replication import (
    OptimizerMemoryReplicaManager,
    OptimizerMemorySnapshot,
    OptimizerScalarRef,
    OptimizerTensorRef,
    Zero2MemoryReplicaManager,
    apply_optimizer_snapshot,
    backup_holder_for_owner,
    capture_optimizer_snapshot,
    memory_replica_sources,
    ring_neighbors,
)

__all__ = [
    "OptimizerMemoryReplicaManager",
    "OptimizerMemorySnapshot",
    "OptimizerScalarRef",
    "OptimizerTensorRef",
    "Zero2MemoryReplicaManager",
    "apply_optimizer_snapshot",
    "backup_holder_for_owner",
    "capture_optimizer_snapshot",
    "memory_replica_sources",
    "ring_neighbors",
]

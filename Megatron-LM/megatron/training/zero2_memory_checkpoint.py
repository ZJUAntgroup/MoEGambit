"""Backward-compatible import path for optimizer memory replication.

The implementation moved to ``moegambit.replication`` so it can be shared by
framework adapters.  Existing Megatron patches and experiment scripts may keep
using this module while migrating.
"""

from moegambit.replication import (  # noqa: F401
    FrameLimits,
    MemoryReplicaLocator,
    OptimizerMemoryReplicaManager,
    OptimizerMemorySnapshot,
    OptimizerScalarRef,
    OptimizerTensorRef,
    TransportIntegrityError,
    Zero2MemoryReplicaManager,
    apply_optimizer_snapshot,
    backup_holder_for_owner,
    memory_replica_sources,
    ring_neighbors,
)
from moegambit.replication.optimizer_memory import (  # noqa: F401
    _manifest_for_refs,
)

# Some focused legacy tests monkeypatch ``socket.create_connection`` through
# this historical module.  Re-exporting the module preserves that behavior;
# the new transport module uses the same stdlib module object.
from moegambit.replication.transport import socket  # noqa: E402,F401

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

"""DeepSpeed ZeRO-2 state mapping for the generic host replica runtime."""

from __future__ import annotations

import hashlib
import json
import os
import socket
from datetime import timedelta
from typing import Any

from moegambit.runtime.zero2_replica import (
    OptimizerScalarRef,
    OptimizerTensorRef,
    Zero2MemoryReplicaManager,
    apply_optimizer_snapshot,
)


class DeepSpeedZero2Error(RuntimeError):
    pass


def _advertise_host() -> str:
    configured = os.environ.get("MOEGAMBIT_REPLICA_ADVERTISE_ADDR")
    if configured:
        return configured
    peer = os.environ.get("MASTER_ADDR", "127.0.0.1")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((peer, 9))
            host = probe.getsockname()[0]
        if host and host != "0.0.0.0":
            return host
    except OSError:
        pass
    return socket.gethostbyname(socket.gethostname())


def _key(peer_id: str) -> str:
    digest = hashlib.sha256(peer_id.encode("utf-8")).hexdigest()
    return f"moegambit/deepspeed/zero2/{digest}"


def _is_expert_group(group: dict[str, Any]) -> bool:
    name = str(group.get("name", "")).lower()
    return bool(group.get("moe")) or "expert" in name


class DeepSpeedZero2Replica:
    """Expose DeepSpeed's rank-local ZeRO-2 shard as stable replica refs."""

    def __init__(self, engine: Any, timeout: float = 300.0) -> None:
        self.engine = engine
        self.optimizer = engine.optimizer
        self.timeout = float(timeout)
        self.buffer_slots = int(
            os.environ.get("MOEGAMBIT_ZERO2_BUFFER_SLOTS", "2")
        )
        if self.buffer_slots not in (1, 2):
            raise DeepSpeedZero2Error(
                "MOEGAMBIT_ZERO2_BUFFER_SLOTS must be 1 or 2"
            )
        self.manager: Zero2MemoryReplicaManager | None = None
        self.managers: dict[str, Zero2MemoryReplicaManager] = {}
        self.group_ranks: dict[str, list[int]] = {}
        self._group_indices: dict[str, tuple[int, ...]] = {}
        self._store = None

    def _validate_optimizer(self) -> None:
        optimizer = self.optimizer
        required = (
            "single_partition_of_fp32_groups",
            "optimizer",
            "dp_process_group",
        )
        missing = [name for name in required if not hasattr(optimizer, name)]
        if missing or not getattr(optimizer, "partition_gradients", False):
            raise DeepSpeedZero2Error(
                "MoEGambit requires DeepSpeed ZeRO stage 2; missing "
                + ", ".join(missing or ["partition_gradients"])
            )

    def _tensor_refs(
        self, group_indices: tuple[int, ...] | None = None
    ) -> list[OptimizerTensorRef]:
        refs: list[OptimizerTensorRef] = []
        base_optimizer = self.optimizer.optimizer
        groups = self.optimizer.single_partition_of_fp32_groups
        param_groups = base_optimizer.param_groups
        indices = group_indices or tuple(range(len(groups)))
        for index in indices:
            master = groups[index]
            group = param_groups[index]
            expert = _is_expert_group(group)
            refs.append(
                OptimizerTensorRef(
                    f"group/{index}/fp32_master", master, expert
                )
            )
            for key, value in sorted(
                base_optimizer.state.get(master, {}).items(),
                key=lambda item: str(item[0]),
            ):
                if hasattr(value, "numel"):
                    refs.append(
                        OptimizerTensorRef(
                            f"group/{index}/optimizer/{key}",
                            value,
                            expert,
                        )
                    )
        return refs

    def _scalar_refs(
        self, group_indices: tuple[int, ...] | None = None
    ) -> list[OptimizerScalarRef]:
        refs: list[OptimizerScalarRef] = []
        base_optimizer = self.optimizer.optimizer
        groups = self.optimizer.single_partition_of_fp32_groups
        indices = group_indices or tuple(range(len(groups)))
        for index in indices:
            master = groups[index]
            state = base_optimizer.state.get(master, {})
            expert = _is_expert_group(base_optimizer.param_groups[index])
            for key, value in sorted(
                state.items(), key=lambda item: str(item[0])
            ):
                if not hasattr(value, "numel"):
                    refs.append(
                        OptimizerScalarRef(
                            f"group/{index}/optimizer/{key}",
                            state,
                            key,
                            expert,
                        )
                    )
        return refs

    def _publish_endpoint(
        self,
        namespace: str,
        peer_id: str,
        port: int,
        src_rank: int,
        dst_rank: int,
    ) -> bool:
        payload = {
            "host": _advertise_host(),
            "port": int(port),
            "src_rank": int(src_rank),
            "dst_rank": int(dst_rank),
        }
        self._store.set(_key(f"{namespace}/{peer_id}"), json.dumps(payload))
        return True

    def _wait_endpoint(
        self, namespace: str, peer_id: str, timeout: float
    ):
        key = _key(f"{namespace}/{peer_id}")
        try:
            self._store.wait([key], timedelta(seconds=float(timeout)))
            value = self._store.get(key)
        except Exception:
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(value)

    def start(self, initial_step: int = 0) -> dict[str, Any]:
        self._validate_optimizer()
        import torch.distributed as dist

        if not dist.is_initialized():
            raise DeepSpeedZero2Error(
                "torch.distributed must be initialized before ZeRO-2 replication"
            )
        self._store = dist.distributed_c10d._get_default_store()
        rank = dist.get_rank()
        generation = int(
            os.environ.get(
                "MOEGAMBIT_RECOVERY_EPOCH",
                os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"),
            )
        )
        process_groups = getattr(
            self.optimizer,
            "real_dp_process_group",
            [self.optimizer.dp_process_group]
            * len(self.optimizer.single_partition_of_fp32_groups),
        )
        buckets: dict[tuple[int, ...], list[int]] = {}
        for index, process_group in enumerate(process_groups):
            ranks = tuple(dist.get_process_group_ranks(process_group))
            buckets.setdefault(ranks, []).append(index)

        summaries = {}
        ordered = sorted(
            buckets.items(), key=lambda item: (-len(item[0]), item[0])
        )
        for ranks, indices_value in ordered:
            if rank not in ranks:
                continue
            if len(ranks) < 2:
                raise DeepSpeedZero2Error(
                    "ZeRO-2 replication requires every optimizer DP group "
                    "to contain at least two ranks"
                )
            namespace = "ranks-" + "-".join(str(item) for item in ranks)
            indices = tuple(indices_value)
            manager = Zero2MemoryReplicaManager(
                rank=rank,
                tensor_refs_fn=lambda selected=indices: self._tensor_refs(
                    selected
                ),
                scalar_refs_fn=lambda selected=indices: self._scalar_refs(
                    selected
                ),
                publish_endpoint_fn=(
                    lambda peer_id, port, src_rank, dst_rank, ns=namespace: (
                        self._publish_endpoint(
                            ns, peer_id, port, src_rank, dst_rank
                        )
                    )
                ),
                wait_endpoint_fn=(
                    lambda peer_id, timeout, ns=namespace: self._wait_endpoint(
                        ns, peer_id, timeout
                    )
                ),
                timeout=self.timeout,
                buffer_slots=self.buffer_slots,
            )
            manager.start_transport(list(ranks), generation=generation)
            self.managers[namespace] = manager
            self.group_ranks[namespace] = list(ranks)
            self._group_indices[namespace] = indices
            summaries[namespace] = manager.schedule_snapshot(
                int(initial_step)
            )
        if not self.managers:
            raise DeepSpeedZero2Error(
                f"rank {rank} has no ZeRO-2 optimizer replication group"
            )
        self.manager = next(iter(self.managers.values()))
        # Ranks that do not belong to a smaller expert-DP group would otherwise
        # return to the application while expert ranks are still creating TCP
        # replica rings. This barrier is only on the healthy initialization
        # path and uses DeepSpeed's already-established global process group.
        dist.barrier()
        return summaries

    def before_step(self, step: int) -> None:
        if step < 0:
            return
        for manager in self.managers.values():
            manager.wait_until_replicated(int(step))

    def after_step(self, step: int) -> dict[str, Any]:
        return {
            namespace: manager.schedule_snapshot(int(step))
            for namespace, manager in self.managers.items()
        }

    def restore_local_snapshot(self, owner_rank: int, step: int) -> None:
        if not self.managers:
            raise DeepSpeedZero2Error("replica manager is not started")
        failures = []
        restored = []
        for namespace, manager in self.managers.items():
            try:
                snapshot = manager.get_peer_snapshot(owner_rank, step)
            except RuntimeError as exc:
                failures.append(str(exc))
                continue
            indices = self._group_indices[namespace]
            apply_optimizer_snapshot(
                snapshot,
                self._tensor_refs(indices),
                self._scalar_refs(indices),
            )
            restored.append(namespace)
        if not restored:
            raise DeepSpeedZero2Error(
                f"no local replica for owner={owner_rank} step={step}: "
                f"{failures}"
            )

    def close(self) -> None:
        for manager in self.managers.values():
            manager.stop_transport()
        self.managers.clear()
        self.manager = None

"""Framework-neutral static-world node launcher and replacement agent."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple

from ..errors import ContractViolation
from ..runtime.recovery_plan import RecoveryPlan
from .worker_supervisor import WorkerHandle, WorkerSpec, WorkerSupervisor

__all__ = ["NodeLaunchSpec", "NodeAgent"]

_TORCHELASTIC_ENV_VARS = (
    "TORCHELASTIC_USE_AGENT_STORE",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_ERROR_FILE",
    "TORCHELASTIC_ROLE",
    "TORCHELASTIC_ROLE_RANK",
    "TORCHELASTIC_ROLE_WORLD_SIZE",
)


def _clean_worker_env(env: Mapping[str, str]) -> Mapping[str, str]:
    cleaned = dict(env)
    for key in _TORCHELASTIC_ENV_VARS:
        cleaned.pop(key, None)
    return cleaned


@dataclass(frozen=True)
class NodeLaunchSpec:
    nnodes: int
    nproc_per_node: int
    node_rank: int
    master_addr: str
    master_port: int
    argv: Sequence[str]
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None

    def __post_init__(self) -> None:
        if self.nnodes <= 0 or self.nproc_per_node <= 0:
            raise ValueError("nnodes and nproc_per_node must be positive")
        if not 0 <= self.node_rank < self.nnodes:
            raise ValueError("node_rank must be within nnodes")
        if not self.master_addr:
            raise ValueError("master_addr is required")
        if not 0 < int(self.master_port) <= 65535:
            raise ValueError("master_port is out of range")
        if isinstance(self.argv, str) or not self.argv:
            raise ValueError("training argv must be a non-empty sequence")
        if any(not isinstance(item, str) or not item for item in self.argv):
            raise ValueError("training argv must contain non-empty strings")

    @property
    def world_size(self) -> int:
        return self.nnodes * self.nproc_per_node

    @property
    def command_digest(self) -> str:
        blob = "\0".join(self.argv).encode("utf-8")
        return "sha256:" + hashlib.sha256(blob).hexdigest()

    def owns_rank(self, logical_rank: int) -> bool:
        first = self.node_rank * self.nproc_per_node
        return first <= int(logical_rank) < first + self.nproc_per_node

    def worker_spec(
        self,
        local_rank: int,
        *,
        extra_env: Optional[Mapping[str, str]] = None,
    ) -> WorkerSpec:
        if not 0 <= int(local_rank) < self.nproc_per_node:
            raise ValueError("local_rank is outside this node")
        logical_rank = self.node_rank * self.nproc_per_node + int(local_rank)
        env = dict(_clean_worker_env(self.env))
        env.update(
            {
                "RANK": str(logical_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(self.world_size),
                "LOCAL_WORLD_SIZE": str(self.nproc_per_node),
                "MASTER_ADDR": self.master_addr,
                "MASTER_PORT": str(self.master_port),
                "NODE_RANK": str(self.node_rank),
                "GROUP_RANK": str(self.node_rank),
                "MOEGAMBIT_COMMAND_DIGEST": self.command_digest,
            }
        )
        env.update(dict(extra_env or {}))
        return WorkerSpec(
            logical_rank=logical_rank,
            local_rank=int(local_rank),
            argv=tuple(self.argv),
            env=env,
            cwd=self.cwd,
        )


class NodeAgent:
    """Own all local workers while preserving logical rank identity."""

    def __init__(
        self,
        spec: NodeLaunchSpec,
        *,
        supervisor: Optional[WorkerSupervisor] = None,
    ) -> None:
        self.spec = spec
        self.supervisor = supervisor or WorkerSupervisor()

    def start_all(self) -> Tuple[WorkerHandle, ...]:
        handles = []
        try:
            for local_rank in range(self.spec.nproc_per_node):
                handles.append(
                    self.supervisor.start(self.spec.worker_spec(local_rank))
                )
        except Exception:
            for handle in handles:
                self.supervisor.terminate(handle.spec.logical_rank)
            raise
        return tuple(handles)

    def replace(
        self,
        logical_rank: int,
        *,
        recovery_epoch: int,
        plan_digest: str,
        timeout_s: float = 10.0,
    ) -> WorkerHandle:
        if not self.spec.owns_rank(logical_rank):
            raise ContractViolation(
                f"logical rank {logical_rank} is not owned by node {self.spec.node_rank}"
            )
        if recovery_epoch <= 0 or not plan_digest:
            raise ContractViolation(
                "replacement requires a positive epoch and frozen plan digest"
            )
        local_rank = int(logical_rank) % self.spec.nproc_per_node
        worker = self.spec.worker_spec(
            local_rank,
            extra_env={
                "MOEGAMBIT_RECOVERY_EPOCH": str(recovery_epoch),
                "MOEGAMBIT_RECOVERY_PLAN_DIGEST": str(plan_digest),
            },
        )
        return self.supervisor.replace(
            int(logical_rank),
            worker,
            timeout_s=timeout_s,
        )

    def apply_plan(
        self,
        plan: RecoveryPlan,
        *,
        timeout_s: float = 10.0,
    ) -> Tuple[WorkerHandle, ...]:
        plan_digest = plan.digest()
        replaced = []
        for logical_rank, endpoint in sorted(plan.replacements.items()):
            if logical_rank not in plan.failed_ranks:
                raise ContractViolation(
                    "replacement plan attempts to replace a rank not marked failed"
                )
            if endpoint.node_rank != self.spec.node_rank:
                continue
            expected_local_rank = int(logical_rank) % self.spec.nproc_per_node
            if endpoint.local_rank != expected_local_rank:
                raise ContractViolation(
                    "replacement endpoint local_rank disagrees with logical rank"
                )
            replaced.append(
                self.replace(
                    logical_rank,
                    recovery_epoch=plan.recovery_epoch,
                    plan_digest=plan_digest,
                    timeout_s=timeout_s,
                )
            )
        return tuple(replaced)

    def terminate_all(self, *, timeout_s: float = 10.0) -> None:
        self.supervisor.terminate_all(timeout_s=timeout_s)

    def wait(self, *, poll_interval_s: float = 0.1) -> Mapping[int, int]:
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        while True:
            snapshot = self.supervisor.snapshot()
            if not snapshot:
                raise ContractViolation("node agent has no workers to wait for")
            if snapshot and not any(item["running"] for item in snapshot.values()):
                return {
                    rank: int(item["returncode"])
                    for rank, item in snapshot.items()
                }
            time.sleep(poll_interval_s)

    def heartbeat_payload(self) -> Mapping[str, object]:
        return {
            "node_rank": self.spec.node_rank,
            "world_size": self.spec.world_size,
            "command_digest": self.spec.command_digest,
            "workers": self.supervisor.snapshot(),
        }

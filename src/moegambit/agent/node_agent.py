"""Framework-neutral static-world node launcher and replacement agent."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple

from ..errors import ContractViolation
from ..control.relaunch import RelaunchDirective
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
    job_id: str = "default"
    attempt_id: str = "0"
    checkpoint_argv_template: Sequence[str] = field(default_factory=tuple)

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
        if not self.job_id or not self.attempt_id:
            raise ValueError("job_id and attempt_id must be non-empty")
        if isinstance(self.checkpoint_argv_template, str) or any(
            not isinstance(item, str) or not item
            for item in self.checkpoint_argv_template
        ):
            raise ValueError(
                "checkpoint_argv_template must contain non-empty arguments"
            )

    @property
    def world_size(self) -> int:
        return self.nnodes * self.nproc_per_node

    @property
    def command_digest(self) -> str:
        blob = json.dumps(
            {
                "argv": list(self.argv),
                "checkpoint_argv_template": list(self.checkpoint_argv_template),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(blob).hexdigest()

    def owns_rank(self, logical_rank: int) -> bool:
        first = self.node_rank * self.nproc_per_node
        return first <= int(logical_rank) < first + self.nproc_per_node

    def worker_spec(
        self,
        local_rank: int,
        *,
        extra_env: Optional[Mapping[str, str]] = None,
        extra_argv: Sequence[str] = (),
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
                "MOEGAMBIT_JOB_ID": self.job_id,
                "MOEGAMBIT_ATTEMPT_ID": self.attempt_id,
            }
        )
        env.update(dict(extra_env or {}))
        return WorkerSpec(
            logical_rank=logical_rank,
            local_rank=int(local_rank),
            argv=tuple(self.argv) + tuple(extra_argv),
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
        self.current_attempt_id = spec.attempt_id
        self.relaunch_count = 0

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
                "MOEGAMBIT_ATTEMPT_ID": self.current_attempt_id,
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

    def _checkpoint_argv(
        self, directive: RelaunchDirective
    ) -> Tuple[str, ...]:
        values = {
            "checkpoint_locator": directive.checkpoint_locator,
            "checkpoint_step": str(directive.checkpoint_step),
            "next_attempt_id": directive.next_attempt_id,
            "recovery_epoch": str(directive.recovery_epoch),
        }
        rendered = []
        for argument in self.spec.checkpoint_argv_template:
            try:
                rendered.append(argument.format_map(values))
            except KeyError as exc:
                raise ContractViolation(
                    f"unknown checkpoint argv placeholder: {exc.args[0]}"
                ) from exc
        return tuple(rendered)

    def relaunch_from_checkpoint(
        self,
        directive: RelaunchDirective,
        *,
        timeout_s: float = 10.0,
    ) -> Tuple[WorkerHandle, ...]:
        """Replace the whole local worker set with a new cold attempt."""

        if directive.next_attempt_id == self.current_attempt_id:
            handles = tuple(
                handle
                for rank in sorted(self.supervisor.snapshot())
                for handle in (self.supervisor.get(rank),)
                if handle is not None and handle.running
            )
            if len(handles) != self.spec.nproc_per_node:
                raise ContractViolation(
                    "already relaunched attempt does not own all live workers"
                )
            return handles
        if directive.parent_attempt_id != self.current_attempt_id:
            raise ContractViolation(
                "checkpoint relaunch parent attempt does not match node agent"
            )
        if (
            directive.command_digest
            and directive.command_digest != self.spec.command_digest
        ):
            raise ContractViolation(
                "checkpoint relaunch command digest does not match node agent"
            )
        extra_argv = self._checkpoint_argv(directive)
        extra_env = {
            "MOEGAMBIT_CHECKPOINT_RELAUNCH": "1",
            "MOEGAMBIT_CHECKPOINT_LOCATOR": directive.checkpoint_locator,
            "MOEGAMBIT_CHECKPOINT_STEP": str(directive.checkpoint_step),
            "MOEGAMBIT_PARENT_ATTEMPT_ID": directive.parent_attempt_id,
            "MOEGAMBIT_ATTEMPT_ID": directive.next_attempt_id,
            "MOEGAMBIT_RECOVERY_EPOCH": str(directive.recovery_epoch),
            # A cold checkpoint restart must not execute the in-process
            # replacement assignment path.
            "MOEGAMBIT_REPLACEMENT": "0",
        }
        self.terminate_all(timeout_s=timeout_s)
        handles = []
        try:
            for local_rank in range(self.spec.nproc_per_node):
                worker = self.spec.worker_spec(
                    local_rank,
                    extra_env=extra_env,
                    extra_argv=extra_argv,
                )
                handles.append(self.supervisor.start(worker))
        except Exception:
            for handle in handles:
                self.supervisor.terminate(
                    handle.spec.logical_rank, timeout_s=timeout_s
                )
            raise
        self.current_attempt_id = directive.next_attempt_id
        self.relaunch_count += 1
        return tuple(handles)

    def apply_relaunch_directive(
        self,
        client: object,
        values: Mapping[str, object],
        *,
        timeout_s: float = 10.0,
    ) -> Tuple[WorkerHandle, ...]:
        directive = RelaunchDirective.from_dict(values)
        handles = self.relaunch_from_checkpoint(
            directive, timeout_s=timeout_s
        )
        notify = getattr(client, "notify", None)
        if not callable(notify):
            raise ContractViolation(
                "node-agent control client cannot acknowledge relaunch"
            )
        notify(
            "checkpoint_relaunch_ack",
            {
                "directive_id": directive.directive_id,
                "next_attempt_id": directive.next_attempt_id,
                "command_digest": self.spec.command_digest,
                "worker_count": len(handles),
            },
            recovery_epoch=directive.recovery_epoch,
        )
        return handles

    def run_with_control(
        self,
        client: object,
        *,
        poll_interval_s: float = 0.5,
        relaunch_grace_s: float = 30.0,
        terminate_timeout_s: float = 10.0,
        max_relaunches: int = 3,
    ) -> Mapping[int, int]:
        """Supervise workers, polling the watcher for cold-relaunch work."""

        if poll_interval_s <= 0 or relaunch_grace_s <= 0:
            raise ValueError("node-agent poll and grace intervals must be positive")
        if max_relaunches < 0:
            raise ValueError("max_relaunches must be non-negative")
        active_client = client
        all_stopped_at: Optional[float] = None
        while True:
            snapshot = self.supervisor.snapshot()
            if not snapshot:
                raise ContractViolation("node agent has no workers to supervise")
            heartbeat = getattr(active_client, "request", None)
            if not callable(heartbeat):
                raise ContractViolation("node-agent control client cannot heartbeat")
            try:
                response = heartbeat(
                    "heartbeat",
                    self.heartbeat_payload(),
                    recovery_epoch=0,
                )
            except Exception:
                response = {}
            raw_directive = response.get("relaunch") if isinstance(response, Mapping) else None
            if isinstance(raw_directive, Mapping):
                next_attempt = str(raw_directive["next_attempt_id"])
                if (
                    next_attempt != self.current_attempt_id
                    and self.relaunch_count >= max_relaunches
                ):
                    raise ContractViolation(
                        "checkpoint relaunch count exceeds configured maximum"
                    )
                try:
                    self.apply_relaunch_directive(
                        active_client,
                        raw_directive,
                        timeout_s=terminate_timeout_s,
                    )
                except Exception:
                    # If the local relaunch completed but its ACK was lost,
                    # keep polling the parent scope and retry the idempotent
                    # acknowledgement instead of launching twice.
                    if self.current_attempt_id != next_attempt:
                        raise
                    time.sleep(poll_interval_s)
                    continue
                client_for_attempt = getattr(active_client, "with_attempt", None)
                if callable(client_for_attempt):
                    active_client = client_for_attempt(next_attempt)
                all_stopped_at = None
                time.sleep(poll_interval_s)
                continue

            running = any(bool(item["running"]) for item in snapshot.values())
            if running:
                all_stopped_at = None
            else:
                exit_codes = {
                    rank: int(item["returncode"])
                    for rank, item in snapshot.items()
                }
                if all(code == 0 for code in exit_codes.values()):
                    return exit_codes
                if all_stopped_at is None:
                    all_stopped_at = time.monotonic()
                if time.monotonic() - all_stopped_at >= relaunch_grace_s:
                    return exit_codes
            time.sleep(poll_interval_s)

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
            "role": "node_agent",
            "node_rank": self.spec.node_rank,
            "nnodes": self.spec.nnodes,
            "world_size": self.spec.world_size,
            "command_digest": self.spec.command_digest,
            "attempt_id": self.current_attempt_id,
            "workers": self.supervisor.snapshot(),
        }

"""Node-level hot-spare supervision for static distributed training worlds.

The spare node never joins the healthy training world.  A coordinator assigns
physical agents to logical node ranks.  After a failure, every surviving agent
enters a new recovery epoch while the spare takes over the failed logical node.
The training world size and all logical ranks therefore remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from moegambit.runtime.protocol import WireMessage
from moegambit.runtime.watcher import WatcherRuntime
from moegambit.runtime.watcher_client import WatcherClient, WatcherEndpoint


logger = logging.getLogger(__name__)
_ZERO_OPTIM_SHARD = re.compile(
    r"^(?:bf16_|fp16_)?zero_pp_rank_(?P<dp>\d+)_"
    r"mp_rank_(?P<mp>\d+)_optim_states\.pt$"
)


def request_worker_command(
    kind: str, rank: int, **payload: Any
) -> Mapping[str, Any] | None:
    """Send a rank-scoped event and return the coordinator command."""
    coordinator_host = os.environ.get(
        "MOEGAMBIT_HOT_SPARE_COORDINATOR_ADDR"
    )
    coordinator_port = os.environ.get(
        "MOEGAMBIT_HOT_SPARE_COORDINATOR_PORT"
    )
    run_id = os.environ.get("MOEGAMBIT_HOT_SPARE_RUN_ID")
    if not coordinator_host or not coordinator_port or not run_id:
        return None

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    logical_node = rank // local_world_size
    physical_node = int(
        os.environ.get(
            "MOEGAMBIT_PHYSICAL_NODE_RANK", str(logical_node)
        )
    )
    recovery_epoch = int(
        os.environ.get(
            "MOEGAMBIT_RECOVERY_EPOCH",
            os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"),
        )
    )
    response = WatcherClient(
        WatcherEndpoint(
            coordinator_host,
            int(coordinator_port),
            timeout=10.0,
        )
    ).request(
        WireMessage(
            kind,
            {
                "run_id": run_id,
                "physical_node": physical_node,
                "logical_node": logical_node,
                "rank": rank,
                "epoch": recovery_epoch,
                **payload,
            },
        )
    )
    if response.kind == "error":
        raise RuntimeError(
            f"hot-spare coordinator rejected {kind}: "
            f"{response.payload.get('error', 'unknown error')}"
        )
    if response.kind != "command":
        raise RuntimeError(
            f"unexpected hot-spare response {response.kind!r}"
        )
    return dict(response.payload)


def send_worker_event(kind: str, rank: int, **payload: Any) -> bool:
    """Send a rank-scoped lifecycle event when hot-spare control is active."""
    return request_worker_command(kind, rank, **payload) is not None


def report_worker_phase(phase: str, rank: int) -> bool:
    """Best-effort startup phase reporting from one rank per logical node."""
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if rank % local_world_size:
        return False
    try:
        return send_worker_event("worker_phase", rank, phase=phase)
    except (ConnectionError, OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "could not report worker phase=%s rank=%d: %s",
            phase,
            rank,
            exc,
        )
        return False


def _command_option(
    command: Sequence[str], option: str
) -> str | None:
    prefix = f"{option}="
    for index, item in enumerate(command):
        if item == option and index + 1 < len(command):
            return command[index + 1]
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


def _deepspeed_python_workload(
    command: Sequence[str],
) -> tuple[Path, tuple[str, ...]] | None:
    """Extract a Python training script from a DeepSpeed runner command."""
    try:
        module_index = command.index("deepspeed.launcher.runner")
    except ValueError:
        return None
    for index in range(module_index + 1, len(command)):
        candidate = Path(command[index])
        if candidate.suffix != ".py":
            continue
        return candidate, tuple(command[index + 1 :])
    return None


def _tail_text(path: Path, line_count: int = 100) -> str:
    try:
        lines = path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-line_count:])


def collect_worker_failure_diagnostic(
    command: Sequence[str],
    *,
    epoch: int,
    logical_node: int,
    process_started_at: float | None,
) -> str:
    """Recover the first local-rank traceback hidden by DeepSpeed logging."""
    local_world_size_text = _command_option(command, "--num_gpus")
    try:
        local_world_size = int(local_world_size_text or "1")
    except ValueError:
        local_world_size = 1
    first_rank = logical_node * local_world_size
    ranks = range(first_rank, first_rank + local_world_size)

    state_dir_text = _command_option(command, "--state-dir")
    if state_dir_text:
        error_dir = Path(state_dir_text) / "errors"
        for rank in ranks:
            artifact = error_dir / f"epoch_{epoch}_rank_{rank}.json"
            if not artifact.is_file():
                continue
            try:
                payload = json.loads(
                    artifact.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            error = str(payload.get("error", "unknown worker error"))
            traceback_text = str(payload.get("traceback", "")).strip()
            result = (
                f"fatal artifact={artifact} rank={rank}: {error}"
            )
            if traceback_text:
                result = f"{result}\n{traceback_text}"
            return result[-16000:]

    log_dir_text = _command_option(
        command, "--enable_each_rank_log"
    )
    if not log_dir_text:
        return ""
    log_dir = Path(log_dir_text)
    candidates: list[Path] = []
    for rank in ranks:
        candidates.extend(log_dir.glob(f"*_rank{rank}.log"))
    candidates = [
        path
        for path in candidates
        if process_started_at is None
        or path.stat().st_mtime >= process_started_at - 5.0
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    markers = (
        "Traceback (most recent call last)",
        "FATAL ",
        "RuntimeError:",
        "OutOfMemoryError:",
        "Error:",
        "Exception:",
    )
    fallback = ""
    fallback_path: Path | None = None
    for path in candidates:
        tail = _tail_text(path)
        if not tail:
            continue
        if not fallback:
            fallback = tail
            fallback_path = path
        if any(marker in tail for marker in markers):
            return f"rank log={path}\n{tail}"[-16000:]
    if fallback_path is not None:
        return (
            f"no traceback marker found; newest rank log="
            f"{fallback_path}\n{fallback}"
        )[-16000:]
    return ""


@dataclass
class AgentRecord:
    physical_node: int
    role: str
    advertise_addr: str
    last_seen: float
    state: str = "registered"


@dataclass
class HotSpareCoordinator:
    run_id: str
    training_nodes: int
    spare_physical_node: int
    base_master_port: int
    port_stride: int = 1
    heartbeat_timeout: float = 30.0
    recovery_timeout: float = 300.0
    state_path: Path | None = None
    local_world_size: int = 1
    rank_hot_swap: bool = False
    epoch: int = 0
    status: str = "forming"
    mapping: dict[int, int] = field(default_factory=dict)
    agents: dict[int, AgentRecord] = field(default_factory=dict)
    retired: set[int] = field(default_factory=set)
    completed: set[int] = field(default_factory=set)
    completion_acks: set[int] = field(default_factory=set)
    abort_acks: set[int] = field(default_factory=set)
    failure: dict[str, Any] | None = None
    abort_reason: str | None = None
    recovery_started_at: float | None = None
    ready_logical_nodes: set[int] = field(default_factory=set)
    worker_phases: dict[int, str] = field(default_factory=dict)
    rank_mapping: dict[int, int] = field(default_factory=dict)
    recovery_ready_ranks: set[int] = field(default_factory=set)
    recovery_rank_phases: dict[int, str] = field(default_factory=dict)
    dispatched_epochs: dict[int, int] = field(default_factory=dict)
    group_ordinal_barriers: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    _group_barrier_cv: threading.Condition = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.training_nodes <= 0:
            raise ValueError("training_nodes must be positive")
        if self.spare_physical_node < self.training_nodes:
            raise ValueError(
                "spare_physical_node must be outside the initial training set"
            )
        if self.port_stride <= 0:
            raise ValueError("port_stride must be positive")
        if self.heartbeat_timeout <= 0:
            raise ValueError("heartbeat_timeout must be positive")
        if self.recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be positive")
        if self.local_world_size <= 0:
            raise ValueError("local_world_size must be positive")
        self.mapping = {
            logical_node: logical_node
            for logical_node in range(self.training_nodes)
        }
        self.rank_mapping = {
            rank: rank // self.local_world_size
            for rank in range(self.world_size)
        }
        self._group_barrier_cv = threading.Condition(self._lock)

    @property
    def world_size(self) -> int:
        return self.training_nodes * self.local_world_size

    @property
    def master_port(self) -> int:
        return self.base_master_port + self.epoch * self.port_stride

    @property
    def active_physical_nodes(self) -> set[int]:
        if self.rank_hot_swap and self.epoch > 0:
            return set(self.rank_mapping.values())
        return set(self.mapping.values())

    def handle(self, message: WireMessage) -> WireMessage:
        payload = dict(message.payload)
        if payload.get("run_id") != self.run_id:
            return WireMessage(
                "error",
                {
                    "error": "run_id_mismatch",
                    "expected": self.run_id,
                },
            )
        try:
            physical_node = int(payload["physical_node"])
        except (KeyError, TypeError, ValueError):
            return WireMessage("error", {"error": "invalid_physical_node"})

        with self._lock:
            now = time.monotonic()
            durable_state_before = (
                self.status,
                self.epoch,
                self.abort_reason,
            )
            if message.kind == "register":
                self._register(
                    physical_node,
                    str(payload.get("role", "active")),
                    str(payload.get("advertise_addr", "")),
                    now,
                )
            elif physical_node not in self.agents:
                return WireMessage("error", {"error": "agent_not_registered"})
            else:
                record = self.agents[physical_node]
                record.last_seen = now
                record.state = str(payload.get("state", record.state))

            if message.kind == "group_ordinal_barrier":
                response = self._handle_group_ordinal_barrier(
                    physical_node, payload
                )
                if self.status == "aborted":
                    self._persist()
                return response

            if message.kind == "rank_failure":
                self._handle_rank_failure(physical_node, payload)
            elif message.kind == "runner_failure":
                self._handle_runner_failure(physical_node, payload)
            elif message.kind == "runner_complete":
                self._handle_runner_complete(physical_node, payload)
            elif message.kind == "worker_ready":
                self._handle_worker_ready(physical_node, payload)
            elif message.kind == "worker_phase":
                self._handle_worker_phase(physical_node, payload)
            elif message.kind == "rank_recovery_phase":
                self._handle_rank_recovery_phase(
                    physical_node, payload
                )
            elif message.kind == "rank_recovery_ready":
                self._handle_rank_recovery_ready(
                    physical_node, payload
                )
            elif message.kind == "rank_recovery_failed":
                self._handle_rank_recovery_failed(
                    physical_node, payload
                )
            elif message.kind == "ack_complete":
                self.completion_acks.add(physical_node)
            elif message.kind == "ack_abort":
                self.abort_acks.add(physical_node)
            elif message.kind not in {
                "register",
                "heartbeat",
                "poll",
            }:
                return WireMessage(
                    "error", {"error": f"unsupported_message:{message.kind}"}
                )

            self._activate_when_formed()
            self._detect_timeouts(now)
            durable_state_changed = durable_state_before != (
                self.status,
                self.epoch,
                self.abort_reason,
            )
            transient_message = message.kind in {
                "heartbeat",
                "poll",
                "rank_recovery_phase",
                "rank_recovery_ready",
            }
            recovery_ready = (
                message.kind == "rank_recovery_ready"
                and len(self.recovery_ready_ranks) == self.world_size
            )
            if (
                durable_state_changed
                or not transient_message
                or recovery_ready
            ):
                self._persist()
            command = self._command_for(physical_node)
            self._record_command_dispatch(physical_node, command)
            return WireMessage("command", command)

    def _handle_group_ordinal_barrier(
        self,
        physical_node: int,
        payload: Mapping[str, Any],
    ) -> WireMessage:
        """Globally order recovery-only process-group creation."""
        epoch = self._payload_epoch(payload)
        try:
            rank = int(payload["rank"])
            ordinal = int(payload["ordinal"])
            world_size = int(payload["world_size"])
            ranks = tuple(int(item) for item in payload["ranks"])
            timeout = float(payload.get("timeout_seconds", 300.0))
        except (KeyError, TypeError, ValueError):
            return WireMessage(
                "error", {"error": "invalid_group_ordinal_barrier"}
            )
        if (
            not self.rank_hot_swap
            or epoch <= 0
            or epoch != self.epoch
            or self.status != "running"
        ):
            return WireMessage(
                "error",
                {
                    "error": "inactive_recovery_group_barrier",
                    "epoch": self.epoch,
                    "status": self.status,
                },
            )
        if world_size != self.world_size:
            return WireMessage(
                "error",
                {
                    "error": "group_barrier_world_size_mismatch",
                    "expected": self.world_size,
                    "actual": world_size,
                },
            )
        if self.rank_mapping.get(rank) != physical_node:
            return WireMessage(
                "error",
                {
                    "error": "group_barrier_rank_mapping_mismatch",
                    "rank": rank,
                    "physical_node": physical_node,
                    "expected_physical_node": self.rank_mapping.get(rank),
                },
            )
        if (
            not ranks
            or len(set(ranks)) != len(ranks)
            or any(item < 0 or item >= self.world_size for item in ranks)
        ):
            return WireMessage(
                "error", {"error": "invalid_group_barrier_ranks"}
            )

        phase = str(payload.get("phase", "")).strip()
        name = str(payload.get("group", "")).strip()
        backend = str(payload.get("backend", "")).strip()
        manifest = str(payload.get("manifest", "")).strip()
        if phase not in {"ready", "start", "done"} or not name or not manifest:
            return WireMessage(
                "error", {"error": "incomplete_group_barrier_manifest"}
            )

        barrier_id = f"{epoch}:{ordinal}:{phase}"
        descriptor = {
            "name": name,
            "ranks": ranks,
            "backend": backend,
            "manifest": manifest,
            "world_size": world_size,
        }
        state = self.group_ordinal_barriers.setdefault(
            barrier_id,
            {
                "descriptor": descriptor,
                "arrived": set(),
                "released": False,
                "created": time.monotonic(),
            },
        )
        if state["descriptor"] != descriptor:
            self.status = "aborted"
            self.abort_reason = (
                "recovery group manifest mismatch at "
                f"epoch={epoch} ordinal={ordinal} phase={phase}: "
                f"expected={state['descriptor']} actual={descriptor} "
                f"rank={rank}"
            )
            logger.error(self.abort_reason)
            self._group_barrier_cv.notify_all()
            return WireMessage(
                "error",
                {
                    "error": "group_barrier_manifest_mismatch",
                    "reason": self.abort_reason,
                },
            )

        arrived = state["arrived"]
        arrived.add(rank)
        self.recovery_rank_phases[rank] = (
            f"group_{ordinal:04d}_{phase}"
        )
        self._group_barrier_cv.notify_all()
        deadline = time.monotonic() + max(1.0, timeout)
        while len(arrived) < self.world_size:
            if self.status != "running" or self.epoch != epoch:
                return WireMessage(
                    "error",
                    {
                        "error": "group_barrier_recovery_aborted",
                        "reason": self.abort_reason,
                    },
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = sorted(set(range(self.world_size)) - arrived)
                self.status = "aborted"
                self.abort_reason = (
                    "recovery group ordinal barrier timed out: "
                    f"epoch={epoch} ordinal={ordinal} phase={phase} "
                    f"group={name} arrived={sorted(arrived)} "
                    f"missing={missing}"
                )
                logger.error(self.abort_reason)
                self._group_barrier_cv.notify_all()
                return WireMessage(
                    "error",
                    {
                        "error": "group_barrier_timeout",
                        "reason": self.abort_reason,
                        "missing": missing,
                    },
                )
            self._group_barrier_cv.wait(timeout=min(1.0, remaining))

        if not state["released"]:
            state["released"] = True
            logger.info(
                "GROUP_ORDINAL_READY epoch=%d ordinal=%d phase=%s "
                "group=%s ranks=%s",
                epoch,
                ordinal,
                phase,
                name,
                list(ranks),
            )
        return WireMessage(
            "group_ordinal_ready",
            {
                "epoch": epoch,
                "ordinal": ordinal,
                "phase": phase,
                "count": len(arrived),
            },
        )

    def _record_command_dispatch(
        self, physical_node: int, command: Mapping[str, Any]
    ) -> None:
        try:
            command_epoch = int(command.get("epoch", -1))
        except (TypeError, ValueError):
            return
        previous_epoch = self.dispatched_epochs.get(physical_node, -1)
        if command_epoch <= previous_epoch:
            return
        self.dispatched_epochs[physical_node] = command_epoch
        if command_epoch <= 0 or self.recovery_started_at is None:
            return
        elapsed = time.monotonic() - self.recovery_started_at
        logger.warning(
            "RECOVERY_COMMAND_DISPATCH physical_node=%d action=%s "
            "epoch=%d recovery_elapsed_s=%.2f",
            physical_node,
            command.get("action", "unknown"),
            command_epoch,
            elapsed,
        )
        if self.failure is not None:
            agent_phases = self.failure.setdefault(
                "agent_phase_seconds", {}
            )
            physical_phases = agent_phases.setdefault(
                str(physical_node), {}
            )
            physical_phases["command_dispatched"] = elapsed

    def _register(
        self,
        physical_node: int,
        role: str,
        advertise_addr: str,
        now: float,
    ) -> None:
        if role not in {"active", "standby"}:
            raise ValueError(f"unsupported agent role {role!r}")
        if role == "standby" and physical_node != self.spare_physical_node:
            raise ValueError(
                f"standby must run on physical node {self.spare_physical_node}"
            )
        if role == "active" and not 0 <= physical_node < self.training_nodes:
            raise ValueError(
                f"active physical node must be in [0, {self.training_nodes})"
            )
        if not advertise_addr:
            raise ValueError("agent advertise_addr cannot be empty")
        self.agents[physical_node] = AgentRecord(
            physical_node=physical_node,
            role=role,
            advertise_addr=advertise_addr,
            last_seen=now,
        )
        logger.info(
            "registered %s physical_node=%d", role, physical_node
        )

    def _activate_when_formed(self) -> None:
        expected = set(range(self.training_nodes))
        expected.add(self.spare_physical_node)
        if self.status == "forming" and expected.issubset(self.agents):
            self.status = "running"
            logger.info(
                "recovery epoch 0 released mapping=%s", self.mapping
            )

    def _payload_epoch(self, payload: Mapping[str, Any]) -> int:
        try:
            return int(payload.get("epoch", -1))
        except (TypeError, ValueError):
            return -1

    def _handle_rank_failure(
        self, reporting_physical_node: int, payload: Mapping[str, Any]
    ) -> None:
        if self.status not in {"running", "recovering"}:
            return
        if self._payload_epoch(payload) != self.epoch:
            return
        try:
            logical_node = int(payload["logical_node"])
        except (KeyError, TypeError, ValueError):
            return
        failed_physical_node = self.mapping.get(logical_node)
        if failed_physical_node is None:
            return
        if self.rank_hot_swap:
            try:
                failed_rank = int(payload["rank"])
            except (KeyError, TypeError, ValueError):
                self.status = "aborted"
                self.abort_reason = (
                    "rank-granular recovery requires the failed global rank"
                )
                return
            if not 0 <= failed_rank < self.world_size:
                self.status = "aborted"
                self.abort_reason = (
                    f"failed rank {failed_rank} is outside world size "
                    f"{self.world_size}"
                )
                return
            if failed_rank // self.local_world_size != logical_node:
                self.status = "aborted"
                self.abort_reason = (
                    "rank failure topology mismatch: "
                    f"rank={failed_rank} logical_node={logical_node}"
                )
                return
            expected_physical_node = self.rank_mapping.get(failed_rank)
            if reporting_physical_node != expected_physical_node:
                self.status = "aborted"
                self.abort_reason = (
                    "rank failure reporter does not own the failed rank: "
                    f"rank={failed_rank} reporter={reporting_physical_node} "
                    f"expected={expected_physical_node}"
                )
                return
            self._start_rank_failover(
                failed_rank,
                failed_physical_node,
                str(payload.get("reason", "rank_failure")),
                reporting_physical_node=reporting_physical_node,
                global_step=payload.get("global_step"),
            )
            return
        self._start_failover(
            logical_node,
            failed_physical_node,
            str(payload.get("reason", "rank_failure")),
            reporting_physical_node=reporting_physical_node,
            failed_rank=payload.get("rank"),
            global_step=payload.get("global_step"),
        )

    def _handle_rank_recovery_phase(
        self, physical_node: int, payload: Mapping[str, Any]
    ) -> None:
        if not self.rank_hot_swap or self._payload_epoch(payload) != self.epoch:
            return
        try:
            rank = int(payload["rank"])
        except (KeyError, TypeError, ValueError):
            return
        if self.rank_mapping.get(rank) != physical_node:
            logger.error(
                "rejecting rank recovery phase with inconsistent topology: "
                "rank=%d physical_node=%d expected_physical=%s",
                rank,
                physical_node,
                self.rank_mapping.get(rank),
            )
            return
        phase = str(payload.get("phase", "")).strip()
        if not phase:
            return
        previous = self.recovery_rank_phases.get(rank)
        self.recovery_rank_phases[rank] = phase
        if phase != previous:
            logger.info(
                "RANK_RECOVERY_PHASE rank=%d physical_node=%d phase=%s",
                rank,
                physical_node,
                phase,
            )

    def _handle_rank_recovery_ready(
        self, physical_node: int, payload: Mapping[str, Any]
    ) -> None:
        if not self.rank_hot_swap or self._payload_epoch(payload) != self.epoch:
            return
        try:
            rank = int(payload["rank"])
        except (KeyError, TypeError, ValueError):
            return
        if self.rank_mapping.get(rank) != physical_node:
            logger.error(
                "rejecting rank recovery ready with inconsistent topology: "
                "rank=%d physical_node=%d expected_physical=%s",
                rank,
                physical_node,
                self.rank_mapping.get(rank),
            )
            return
        self.recovery_ready_ranks.add(rank)
        self.recovery_rank_phases[rank] = "train_ready"
        logger.info(
            "RANK_RECOVERY_READY rank=%d physical_node=%d (%d/%d)",
            rank,
            physical_node,
            len(self.recovery_ready_ranks),
            self.world_size,
        )
        if (
            self.recovery_started_at is not None
            and len(self.recovery_ready_ranks) == self.world_size
        ):
            elapsed = time.monotonic() - self.recovery_started_at
            if self.failure is not None:
                self.failure["train_ready_seconds"] = elapsed
            logger.warning(
                "rank recovery epoch %d reached TRAIN_READY on all ranks "
                "in %.2fs",
                self.epoch,
                elapsed,
            )
            self.recovery_started_at = None

    def _handle_rank_recovery_failed(
        self, physical_node: int, payload: Mapping[str, Any]
    ) -> None:
        if not self.rank_hot_swap or self._payload_epoch(payload) != self.epoch:
            return
        self.status = "aborted"
        self.abort_reason = (
            "rank-granular recovery failed closed: "
            f"physical_node={physical_node} rank={payload.get('rank')} "
            f"error={payload.get('error', 'unknown error')}"
        )
        logger.error(self.abort_reason)

    def _handle_runner_failure(
        self, physical_node: int, payload: Mapping[str, Any]
    ) -> None:
        if self.status not in {"running", "recovering"}:
            return
        if self._payload_epoch(payload) != self.epoch:
            return
        if self.rank_hot_swap and self.epoch > 0:
            self.status = "aborted"
            self.abort_reason = (
                "a launcher exited during rank-granular recovery; "
                "refusing to continue with a partial world: "
                f"physical_node={physical_node} "
                f"return_code={payload.get('return_code', 'unknown')}"
            )
            logger.error(self.abort_reason)
            return
        logical_node = next(
            (
                logical
                for logical, physical in self.mapping.items()
                if physical == physical_node
            ),
            None,
        )
        if logical_node is None:
            return
        diagnostic = str(payload.get("diagnostic", "")).strip()
        if diagnostic:
            logger.error(
                "worker failure diagnostic from physical_node=%d:\n%s",
                physical_node,
                diagnostic,
            )
        if (
            self.epoch == 0
            and len(self.ready_logical_nodes) < self.training_nodes
        ):
            self.status = "aborted"
            self.abort_reason = (
                "initial training runner exited before every logical node "
                "reached TRAIN_READY; refusing to consume the hot spare: "
                f"physical_node={physical_node} "
                f"return_code={payload.get('return_code', 'unknown')}"
            )
            logger.error(self.abort_reason)
            return
        if self.rank_hot_swap:
            self.status = "aborted"
            self.abort_reason = (
                "an active launcher exited without a committed rank-scoped "
                "failure; refusing to guess which rank can be replaced: "
                f"physical_node={physical_node} "
                f"return_code={payload.get('return_code', 'unknown')}"
            )
            logger.error(self.abort_reason)
            return
        self._start_failover(
            logical_node,
            physical_node,
            str(payload.get("reason", "runner_failure")),
        )

    def _handle_runner_complete(
        self, physical_node: int, payload: Mapping[str, Any]
    ) -> None:
        if self._payload_epoch(payload) != self.epoch:
            return
        if physical_node not in self.active_physical_nodes:
            return
        self.completed.add(physical_node)
        if self.completed == self.active_physical_nodes:
            self.status = "complete"
            logger.info("training completed in recovery epoch %d", self.epoch)

    def _handle_worker_ready(
        self, physical_node: int, payload: Mapping[str, Any]
    ) -> None:
        if self._payload_epoch(payload) != self.epoch:
            return
        try:
            logical_node = int(payload["logical_node"])
        except (KeyError, TypeError, ValueError):
            return
        if self.mapping.get(logical_node) != physical_node:
            logger.error(
                "rejecting TRAIN_READY with inconsistent topology: "
                "physical_node=%d logical_node=%d expected_physical=%s",
                physical_node,
                logical_node,
                self.mapping.get(logical_node),
            )
            return
        self.ready_logical_nodes.add(logical_node)
        self.worker_phases[logical_node] = "train_ready"
        ready_elapsed: float | None = None
        if self.recovery_started_at is not None:
            ready_elapsed = time.monotonic() - self.recovery_started_at
            if self.failure is not None:
                phase_seconds = self.failure.setdefault(
                    "phase_seconds", {}
                )
                logical_phases = phase_seconds.setdefault(
                    str(logical_node), {}
                )
                logical_phases["train_ready"] = ready_elapsed
        logger.info(
            "TRAIN_READY physical_node=%d logical_node=%d (%d/%d)",
            physical_node,
            logical_node,
            len(self.ready_logical_nodes),
            self.training_nodes,
        )
        if (
            self.recovery_started_at is not None
            and len(self.ready_logical_nodes) == self.training_nodes
        ):
            elapsed = ready_elapsed
            assert elapsed is not None
            logger.warning(
                "recovery epoch %d reached TRAIN_READY on all logical "
                "nodes in %.2fs",
                self.epoch,
                elapsed,
            )
            if self.failure is not None:
                self.failure["train_ready_seconds"] = elapsed
            self.recovery_started_at = None

    def _handle_worker_phase(
        self, physical_node: int, payload: Mapping[str, Any]
    ) -> None:
        if self._payload_epoch(payload) != self.epoch:
            return
        try:
            logical_node = int(payload["logical_node"])
        except (KeyError, TypeError, ValueError):
            return
        if self.mapping.get(logical_node) != physical_node:
            logger.error(
                "rejecting worker phase with inconsistent topology: "
                "physical_node=%d logical_node=%d expected_physical=%s "
                "phase=%s",
                physical_node,
                logical_node,
                self.mapping.get(logical_node),
                payload.get("phase", "unknown"),
            )
            return
        phase = str(payload.get("phase", "")).strip()
        if not phase:
            return
        previous = self.worker_phases.get(logical_node)
        self.worker_phases[logical_node] = phase
        elapsed: float | None = None
        if self.recovery_started_at is not None:
            elapsed = time.monotonic() - self.recovery_started_at
            if self.failure is not None:
                phase_seconds = self.failure.setdefault(
                    "phase_seconds", {}
                )
                logical_phases = phase_seconds.setdefault(
                    str(logical_node), {}
                )
                logical_phases[phase] = elapsed
        if phase != previous:
            logger.info(
                "WORKER_PHASE physical_node=%d logical_node=%d phase=%s%s",
                physical_node,
                logical_node,
                phase,
                (
                    f" recovery_elapsed_s={elapsed:.2f}"
                    if elapsed is not None
                    else ""
                ),
            )

    def _start_failover(
        self,
        logical_node: int,
        failed_physical_node: int,
        reason: str,
        **metadata: Any,
    ) -> None:
        if failed_physical_node in self.retired:
            return
        spare = self.spare_physical_node
        spare_record = self.agents.get(spare)
        spare_available = (
            spare not in self.active_physical_nodes
            and spare not in self.retired
            and spare_record is not None
            and time.monotonic() - spare_record.last_seen
            <= self.heartbeat_timeout
        )
        if not spare_available:
            self.status = "aborted"
            self.abort_reason = (
                f"no hot spare available for logical node {logical_node}: "
                f"{reason}"
            )
            logger.error(self.abort_reason)
            return

        previous_epoch = self.epoch
        self.retired.add(failed_physical_node)
        self.mapping[logical_node] = spare
        self.epoch += 1
        self.status = "recovering"
        self.completed.clear()
        self.completion_acks.clear()
        self.ready_logical_nodes.clear()
        self.worker_phases.clear()
        self.recovery_started_at = time.monotonic()
        self.failure = {
            "previous_epoch": previous_epoch,
            "epoch": self.epoch,
            "logical_node": logical_node,
            "failed_physical_node": failed_physical_node,
            "replacement_physical_node": spare,
            "reason": reason,
            **metadata,
        }
        logger.warning(
            "starting recovery epoch %d: logical node %d moves physical %d -> %d",
            self.epoch,
            logical_node,
            failed_physical_node,
            spare,
        )
        # Agents enter the new epoch as soon as they observe this mapping.
        self.status = "running"

    def _start_rank_failover(
        self,
        failed_rank: int,
        failed_physical_node: int,
        reason: str,
        **metadata: Any,
    ) -> None:
        if self.failure is not None or self.epoch > 0:
            self.status = "aborted"
            self.abort_reason = (
                "a second failure occurred after the hot spare was consumed: "
                f"rank={failed_rank} reason={reason}"
            )
            logger.error(self.abort_reason)
            return
        try:
            failure_step = int(metadata["global_step"])
        except (KeyError, TypeError, ValueError):
            self.status = "aborted"
            self.abort_reason = (
                "rank-granular recovery requires a committed non-negative "
                f"failure step: rank={failed_rank}"
            )
            logger.error(self.abort_reason)
            return
        if failure_step < 0:
            self.status = "aborted"
            self.abort_reason = (
                "rank-granular recovery requires a committed non-negative "
                f"failure step: rank={failed_rank} step={failure_step}"
            )
            logger.error(self.abort_reason)
            return
        metadata["global_step"] = failure_step
        spare = self.spare_physical_node
        spare_record = self.agents.get(spare)
        spare_available = (
            spare not in self.rank_mapping.values()
            and spare_record is not None
            and time.monotonic() - spare_record.last_seen
            <= self.heartbeat_timeout
        )
        if not spare_available:
            self.status = "aborted"
            self.abort_reason = (
                f"no hot spare available for rank {failed_rank}: {reason}"
            )
            logger.error(self.abort_reason)
            return
        if len(self.ready_logical_nodes) < self.training_nodes:
            self.status = "aborted"
            self.abort_reason = (
                "rank failure occurred before every logical node reached "
                "TRAIN_READY; refusing in-process recovery"
            )
            logger.error(self.abort_reason)
            return

        previous_epoch = self.epoch
        logical_node = failed_rank // self.local_world_size
        self.rank_mapping[failed_rank] = spare
        self.epoch += 1
        self.status = "running"
        self.completed.clear()
        self.completion_acks.clear()
        self.recovery_ready_ranks.clear()
        self.recovery_rank_phases.clear()
        self.group_ordinal_barriers.clear()
        self.recovery_started_at = time.monotonic()
        self.failure = {
            "previous_epoch": previous_epoch,
            "epoch": self.epoch,
            "logical_node": logical_node,
            "failed_rank": failed_rank,
            "failed_local_rank": failed_rank % self.local_world_size,
            "failed_physical_node": failed_physical_node,
            "replacement_physical_node": spare,
            "reason": reason,
            **metadata,
        }
        logger.warning(
            "starting rank recovery epoch %d: global rank %d moves "
            "physical %d -> %d; %d survivor processes stay resident",
            self.epoch,
            failed_rank,
            failed_physical_node,
            spare,
            self.world_size - 1,
        )

    def _detect_timeouts(self, now: float) -> None:
        if self.status != "running":
            return
        if (
            self.recovery_started_at is not None
            and now - self.recovery_started_at > self.recovery_timeout
        ):
            if self.rank_hot_swap:
                missing = sorted(
                    set(range(self.world_size))
                    - self.recovery_ready_ranks
                )
                detail = (
                    f"missing ranks={missing}; last rank phases="
                    f"{dict(sorted(self.recovery_rank_phases.items()))}"
                )
            else:
                missing = sorted(
                    set(range(self.training_nodes))
                    - self.ready_logical_nodes
                )
                detail = (
                    f"missing logical nodes={missing}; last phases="
                    f"{dict(sorted(self.worker_phases.items()))}"
                )
            self.status = "aborted"
            self.abort_reason = (
                f"recovery epoch {self.epoch} did not reach TRAIN_READY "
                f"within {self.recovery_timeout:.1f}s; "
                f"{detail}"
            )
            logger.error(self.abort_reason)
            return
        if self.rank_hot_swap and self.epoch > 0:
            for physical_node in sorted(self.active_physical_nodes):
                record = self.agents.get(physical_node)
                if (
                    record is not None
                    and now - record.last_seen <= self.heartbeat_timeout
                ):
                    continue
                self.status = "aborted"
                self.abort_reason = (
                    "an active physical agent disappeared after rank "
                    f"replacement: physical_node={physical_node}"
                )
                logger.error(self.abort_reason)
                return
            return
        for logical_node, physical_node in tuple(self.mapping.items()):
            record = self.agents.get(physical_node)
            if (
                record is not None
                and now - record.last_seen <= self.heartbeat_timeout
            ):
                continue
            if (
                self.epoch == 0
                and len(self.ready_logical_nodes) < self.training_nodes
            ):
                self.status = "aborted"
                self.abort_reason = (
                    "initial training agent disappeared before every logical "
                    "node reached TRAIN_READY; refusing to consume the hot "
                    f"spare: physical_node={physical_node}"
                )
                logger.error(self.abort_reason)
                return
            if self.rank_hot_swap:
                self.status = "aborted"
                self.abort_reason = (
                    "an active physical agent disappeared without a "
                    "committed rank-scoped failure; refusing node-level "
                    f"fallback: physical_node={physical_node}"
                )
                logger.error(self.abort_reason)
                return
            self._start_failover(
                logical_node,
                physical_node,
                "agent_heartbeat_timeout",
            )
            break

    def _command_for(self, physical_node: int) -> dict[str, Any]:
        rank_zero_physical = (
            self.rank_mapping[0]
            if self.rank_hot_swap
            else self.mapping[0]
        )
        rank_zero_agent = self.agents.get(rank_zero_physical)
        common = {
            "status": self.status,
            "epoch": self.epoch,
            "master_port": self.master_port,
            "master_addr": (
                rank_zero_agent.advertise_addr
                if rank_zero_agent is not None
                else ""
            ),
            "mapping": {
                str(logical): physical
                for logical, physical in sorted(self.mapping.items())
            },
            "rank_mapping": {
                str(rank): physical
                for rank, physical in sorted(self.rank_mapping.items())
            },
            "recovery_mode": (
                "rank_in_process"
                if self.rank_hot_swap
                else "node_relaunch"
            ),
            "failed_logical_node": (
                self.failure.get("logical_node")
                if self.failure is not None
                else None
            ),
            "failure_step": (
                self.failure.get("global_step")
                if self.failure is not None
                else None
            ),
            "failed_rank": (
                self.failure.get("failed_rank")
                if self.failure is not None
                else None
            ),
        }
        if self.status == "forming":
            return {**common, "action": "wait"}
        if self.status == "aborted":
            return {
                **common,
                "action": "abort",
                "reason": self.abort_reason or "recovery aborted",
            }
        if self.status == "complete":
            return {**common, "action": "complete"}
        if self.rank_hot_swap and self.failure is not None:
            replacement = int(
                self.failure["replacement_physical_node"]
            )
            if physical_node == replacement:
                return {
                    **common,
                    "action": "replace_rank",
                    "logical_node": int(self.failure["logical_node"]),
                    "replacement_rank": int(
                        self.failure["failed_rank"]
                    ),
                    "replacement_local_rank": int(
                        self.failure["failed_local_rank"]
                    ),
                    "checkpoint_required": True,
                }
            if 0 <= physical_node < self.training_nodes:
                return {
                    **common,
                    "action": "run",
                    "logical_node": physical_node,
                    "preserve_process": True,
                    "checkpoint_required": False,
                }
            return {**common, "action": "standby"}
        if physical_node in self.retired:
            return {**common, "action": "retire"}
        logical_node = next(
            (
                logical
                for logical, physical in self.mapping.items()
                if physical == physical_node
            ),
            None,
        )
        if logical_node is None:
            return {**common, "action": "standby"}
        return {
            **common,
            "action": "run",
            "logical_node": logical_node,
            "checkpoint_required": self.epoch > 0,
        }

    def _persist(self) -> None:
        if self.state_path is None:
            return
        value = {
            "run_id": self.run_id,
            "status": self.status,
            "epoch": self.epoch,
            "master_port": self.master_port,
            "mapping": {
                str(logical): physical
                for logical, physical in sorted(self.mapping.items())
            },
            "rank_mapping": {
                str(rank): physical
                for rank, physical in sorted(self.rank_mapping.items())
            },
            "rank_hot_swap": self.rank_hot_swap,
            "retired": sorted(self.retired),
            "completed": sorted(self.completed),
            "failure": self.failure,
            "abort_reason": self.abort_reason,
            "ready_logical_nodes": sorted(self.ready_logical_nodes),
            "worker_phases": {
                str(logical): phase
                for logical, phase in sorted(self.worker_phases.items())
            },
            "recovery_ready_ranks": sorted(self.recovery_ready_ranks),
            "recovery_rank_phases": {
                str(rank): phase
                for rank, phase in sorted(
                    self.recovery_rank_phases.items()
                )
            },
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(
            self.state_path.suffix + ".tmp"
        )
        temporary.write_text(
            json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self.state_path)


class AgentSupervisor:
    def __init__(
        self,
        *,
        endpoint: WatcherEndpoint,
        run_id: str,
        physical_node: int,
        role: str,
        advertise_addr: str,
        command: Sequence[str],
        heartbeat_interval: float,
        startup_timeout: float,
    ) -> None:
        if not command:
            raise ValueError("worker command cannot be empty")
        self.client = WatcherClient(endpoint)
        self.run_id = run_id
        self.physical_node = physical_node
        self.role = role
        self.advertise_addr = advertise_addr
        self.command = tuple(command)
        self.heartbeat_interval = heartbeat_interval
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen | None = None
        self.process_epoch: int | None = None
        self.completed_epoch: int | None = None
        self.retired_logged = False
        self.process_started_at: float | None = None
        self.process_logical_node: int | None = None
        self.process_rank: int | None = None
        self.process_command: tuple[str, ...] | None = None
        self._relay_log_path: Path | None = None
        self._relay_log_offset = 0
        self._relay_log_buffer = ""
        self._prefetch_stop = threading.Event()
        self._prefetch_thread: threading.Thread | None = None
        self._prefetched_checkpoint_tag: str | None = None
        self._resident_process: subprocess.Popen | None = None
        self._resident_session_id: str | None = None
        self._resident_control_dir: Path | None = None
        self._resident_num_workers = 0
        self._resident_failed = False
        self._resident_ready_logged = False
        self._resident_last_status_log = 0.0
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._last_control_warning = 0.0
        self._control_lock = threading.Lock()
        self._pending_command: dict[str, Any] | None = None
        self._control_signature: tuple[Any, ...] | None = None
        self._preempted_processes: set[tuple[int, int]] = set()
        self._recovery_command_received_at: dict[int, float] = {}

    def _request(
        self, kind: str, **payload: Any
    ) -> Mapping[str, Any]:
        response = self.client.request(
            WireMessage(
                kind,
                {
                    "run_id": self.run_id,
                    "physical_node": self.physical_node,
                    "role": self.role,
                    "advertise_addr": self.advertise_addr,
                    **payload,
                },
            )
        )
        if response.kind == "error":
            raise RuntimeError(str(response.payload.get("error", "watcher error")))
        if response.kind != "command":
            raise RuntimeError(f"unexpected coordinator response {response.kind!r}")
        return response.payload

    def _register(self) -> Mapping[str, Any]:
        deadline = time.monotonic() + self.startup_timeout
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                return self._request("register", state="waiting")
            except (ConnectionError, OSError) as exc:
                last_error = exc
                time.sleep(min(self.heartbeat_interval, 1.0))
        raise TimeoutError(
            f"could not register with hot-spare coordinator: {last_error}"
        )

    def _heartbeat_loop(self) -> None:
        last_warning = 0.0
        while not self._heartbeat_stop.wait(self.heartbeat_interval):
            process = self.process
            if process is not None and process.poll() is None:
                state = "running"
            elif (
                self._resident_process is not None
                and self._resident_process.poll() is None
            ):
                ready, _ = self._resident_ready_snapshot()
                state = (
                    "standby_ready"
                    if len(ready) == self._resident_num_workers
                    else "standby_warming"
                )
            else:
                state = self.role
            try:
                command = self._request(
                    "heartbeat",
                    epoch=int(self.process_epoch or 0),
                    state=state,
                )
                self._observe_control_command(
                    command, source="heartbeat"
                )
            except (
                ConnectionError,
                OSError,
                RuntimeError,
                TimeoutError,
            ) as exc:
                now = time.monotonic()
                if now - last_warning >= 30.0:
                    logger.warning(
                        "heartbeat request failed physical_node=%d: %s",
                        self.physical_node,
                        exc,
                    )
                    last_warning = now

    @staticmethod
    def _command_signature(
        command: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        mapping = command.get("mapping", {})
        if isinstance(mapping, Mapping):
            mapping_signature = tuple(
                sorted(
                    (str(key), str(value))
                    for key, value in mapping.items()
                )
            )
        else:
            mapping_signature = ()
        rank_mapping = command.get("rank_mapping", {})
        if isinstance(rank_mapping, Mapping):
            rank_mapping_signature = tuple(
                sorted(
                    (str(key), str(value))
                    for key, value in rank_mapping.items()
                )
            )
        else:
            rank_mapping_signature = ()
        return (
            command.get("epoch"),
            command.get("action"),
            command.get("logical_node"),
            command.get("replacement_rank"),
            command.get("preserve_process"),
            command.get("master_addr"),
            command.get("master_port"),
            mapping_signature,
            rank_mapping_signature,
        )

    @staticmethod
    def _command_supersedes(
        candidate: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> bool:
        try:
            candidate_epoch = int(candidate.get("epoch", -1))
        except (TypeError, ValueError):
            candidate_epoch = -1
        try:
            current_epoch = int(current.get("epoch", -1))
        except (TypeError, ValueError):
            current_epoch = -1
        if candidate_epoch != current_epoch:
            return candidate_epoch > current_epoch
        action_priority = {
            "wait": 0,
            "standby": 0,
            "run": 1,
            "replace_rank": 1,
            "retire": 2,
            "complete": 3,
            "abort": 3,
        }
        return action_priority.get(
            str(candidate.get("action", "")), -1
        ) >= action_priority.get(str(current.get("action", "")), -1)

    def _recovery_force_preempt_enabled(self) -> bool:
        return os.environ.get(
            "MOEGAMBIT_RECOVERY_FORCE_PREEMPT", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}

    def _observe_control_command(
        self,
        command: Mapping[str, Any],
        *,
        source: str,
    ) -> None:
        command_copy = dict(command)
        signature = self._command_signature(command_copy)
        try:
            command_epoch = int(command_copy.get("epoch", -1))
        except (TypeError, ValueError):
            command_epoch = -1
        command_action = str(command_copy.get("action", ""))

        with self._control_lock:
            if signature != self._control_signature:
                self._control_signature = signature
                if (
                    self._pending_command is None
                    or self._command_supersedes(
                        command_copy, self._pending_command
                    )
                ):
                    self._pending_command = command_copy

            process = self.process
            process_epoch = self.process_epoch
            if command_epoch > 0 and (
                process_epoch is None or command_epoch > process_epoch
            ):
                self._recovery_command_received_at.setdefault(
                    command_epoch, time.monotonic()
                )
            terminal_action = command_action in {"abort", "retire"}
            should_preempt = (
                process is not None
                and process.poll() is None
                and process_epoch is not None
                and (
                    terminal_action
                    or (
                        self._recovery_force_preempt_enabled()
                        and command_epoch > process_epoch
                        and not bool(command_copy.get("preserve_process"))
                        and command_copy.get("recovery_mode")
                        != "rank_in_process"
                    )
                )
            )
            preempt_key = (
                (process.pid, command_epoch)
                if should_preempt and process is not None
                else None
            )
            if (
                preempt_key is not None
                and preempt_key in self._preempted_processes
            ):
                should_preempt = False
            elif preempt_key is not None:
                self._preempted_processes.add(preempt_key)

        if not should_preempt or process is None:
            return
        with self._control_lock:
            if (
                self.process is not process
                or self.process_epoch != process_epoch
            ):
                return
            logger.warning(
                "RECOVERY_COMMAND_RECEIVED physical_node=%d action=%s "
                "old_epoch=%d new_epoch=%d source=%s",
                self.physical_node,
                command_action or "unknown",
                process_epoch,
                command_epoch,
                source,
            )
            try:
                os.killpg(process.pid, signal.SIGKILL)
                logger.warning(
                    "RECOVERY_PREEMPT_SIGNAL physical_node=%d pid=%d "
                    "old_epoch=%d new_epoch=%d signal=SIGKILL",
                    self.physical_node,
                    process.pid,
                    process_epoch,
                    command_epoch,
                )
            except PermissionError as exc:
                logger.warning(
                    "could not signal old worker process group; falling "
                    "back to launcher process physical_node=%d pid=%d "
                    "error=%s",
                    self.physical_node,
                    process.pid,
                    exc,
                )
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                logger.info(
                    "old worker already exited before recovery preemption "
                    "physical_node=%d pid=%d old_epoch=%d new_epoch=%d",
                    self.physical_node,
                    process.pid,
                    process_epoch,
                    command_epoch,
                )

    def _take_pending_command(
        self, current: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        with self._control_lock:
            pending = self._pending_command
            self._pending_command = None
        if (
            pending is not None
            and self._command_supersedes(pending, current)
        ):
            return pending
        return None

    def _recovery_command_delay(self, epoch: int) -> float | None:
        received_at = self._recovery_command_received_at.get(epoch)
        if received_at is None:
            return None
        return time.monotonic() - received_at

    def _start_heartbeat(self) -> None:
        if (
            self._heartbeat_thread is not None
            and self._heartbeat_thread.is_alive()
        ):
            return
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"moegambit-heartbeat-{self.physical_node}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        thread = self._heartbeat_thread
        if thread is None:
            return
        self._heartbeat_stop.set()
        thread.join(timeout=max(2.0, self.heartbeat_interval * 2))
        self._heartbeat_thread = None

    def _formatted_command(
        self,
        logical_node: int,
        epoch: int,
        master_addr: str,
        master_port: int,
    ) -> tuple[str, ...]:
        values = {
            "logical_node": logical_node,
            "recovery_epoch": epoch,
            "master_port": master_port,
            "master_addr": master_addr,
            "physical_node": self.physical_node,
        }
        return tuple(item.format_map(values) for item in self.command)

    def _resident_enabled(self) -> bool:
        return (
            self.role == "standby"
            and os.environ.get(
                "MOEGAMBIT_STANDBY_RESIDENT", "0"
            ).strip().lower()
            in {"1", "true", "yes", "on"}
        )

    def _resident_ready_snapshot(
        self,
    ) -> tuple[list[Mapping[str, Any]], list[int]]:
        if (
            self._resident_control_dir is None
            or self._resident_session_id is None
            or self._resident_num_workers <= 0
        ):
            return [], []
        ready: list[Mapping[str, Any]] = []
        missing: list[int] = []
        for local_rank in range(self._resident_num_workers):
            path = (
                self._resident_control_dir
                / (
                    f"ready_{self._resident_session_id}_"
                    f"{local_rank}.json"
                )
            )
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                missing.append(local_rank)
                continue
            if (
                not isinstance(value, dict)
                or value.get("session_id")
                != self._resident_session_id
                or int(value.get("local_rank", -1)) != local_rank
            ):
                missing.append(local_rank)
                continue
            ready.append(value)
        return ready, missing

    def _log_resident_readiness(self) -> None:
        if self._resident_ready_logged:
            return
        ready, missing = self._resident_ready_snapshot()
        if missing or not ready:
            now = time.monotonic()
            if now - self._resident_last_status_log >= 30.0:
                phases: dict[int, str] = {}
                if (
                    self._resident_control_dir is not None
                    and self._resident_session_id is not None
                ):
                    for local_rank in missing:
                        status_path = (
                            self._resident_control_dir
                            / (
                                f"status_{self._resident_session_id}_"
                                f"{local_rank}.json"
                            )
                        )
                        try:
                            status = json.loads(
                                status_path.read_text(
                                    encoding="utf-8"
                                )
                            )
                        except (OSError, ValueError):
                            phases[local_rank] = "not_started"
                        else:
                            phases[local_rank] = str(
                                status.get("phase", "unknown")
                            )
                logger.info(
                    "resident standby warming ready=%d/%d "
                    "missing_local_ranks=%s phases=%s",
                    len(ready),
                    self._resident_num_workers,
                    missing,
                    phases,
                )
                self._resident_last_status_log = now
            return
        allocated = sum(
            float(value.get("allocated_gib", 0.0))
            for value in ready
        )
        reserved = sum(
            float(value.get("reserved_gib", 0.0))
            for value in ready
        )
        warmup = max(
            float(value.get("warmup_seconds", 0.0))
            for value in ready
        )
        per_gpu = ",".join(
            (
                f"{int(value['local_rank'])}:"
                f"{float(value.get('allocated_gib', 0.0)):.2f}G"
            )
            for value in ready
        )
        logger.warning(
            "resident standby ready workers=%d/%d "
            "allocated_gib=%.2f reserved_gib=%.2f "
            "warmup_seconds=%.2f per_gpu=[%s]",
            len(ready),
            self._resident_num_workers,
            allocated,
            reserved,
            warmup,
            per_gpu,
        )
        self._resident_ready_logged = True

    def _resident_failure_tail(self) -> str:
        state_dir_text = _command_option(
            self.command, "--state-dir"
        )
        if not state_dir_text or self._resident_session_id is None:
            return ""
        log_dir = (
            Path(state_dir_text)
            / "standby_logs"
            / f"node_{self.physical_node}"
        )
        values: list[str] = []
        for local_rank in range(self._resident_num_workers):
            path = (
                log_dir
                / (
                    f"{self._resident_session_id}_"
                    f"local_rank{local_rank}.log"
                )
            )
            tail = _tail_text(path, line_count=20)
            if tail:
                values.append(
                    f"--- standby local_rank={local_rank} ---\n{tail}"
                )
        return "\n".join(values)[-16000:]

    def _ensure_resident_standby(self) -> None:
        if not self._resident_enabled() or self._resident_failed:
            return
        process = self._resident_process
        if process is not None:
            return_code = process.poll()
            if return_code is None:
                self._log_resident_readiness()
                return
            self._resident_process = None
            self._resident_failed = True
            logger.error(
                "resident standby exited before activation code=%d\n%s",
                return_code,
                self._resident_failure_tail(),
            )
            return

        workload = _deepspeed_python_workload(self.command)
        num_workers_text = _command_option(
            self.command, "--num_gpus"
        )
        num_nodes_text = _command_option(
            self.command, "--num_nodes"
        )
        state_dir_text = _command_option(
            self.command, "--state-dir"
        )
        if (
            workload is None
            or num_workers_text is None
            or num_nodes_text is None
            or state_dir_text is None
        ):
            logger.error(
                "resident standby requires a DeepSpeed Python workload "
                "with --num_nodes, --num_gpus, and --state-dir; "
                "falling back to cold launch"
            )
            self._resident_failed = True
            return
        try:
            num_workers = int(num_workers_text)
            num_nodes = int(num_nodes_text)
        except ValueError:
            logger.error(
                "invalid resident standby topology nodes=%s workers=%s",
                num_nodes_text,
                num_workers_text,
            )
            self._resident_failed = True
            return
        if num_workers <= 0 or num_nodes <= 0:
            self._resident_failed = True
            return

        training_script, training_args = workload
        expected_logical = int(
            os.environ.get(
                "MOEGAMBIT_STANDBY_PREFETCH_LOGICAL_NODE", "0"
            )
        )
        session_id = (
            f"{int(time.time())}-{os.getpid()}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        state_dir = Path(state_dir_text)
        control_dir = (
            state_dir
            / "standby_control"
            / f"node_{self.physical_node}"
        )
        log_dir = (
            state_dir
            / "standby_logs"
            / f"node_{self.physical_node}"
        )
        command = (
            sys.executable,
            "-u",
            "-m",
            "moegambit.runtime.standby",
            "--mode",
            "launcher",
            "--control-dir",
            str(control_dir),
            "--session-id",
            session_id,
            "--num-workers",
            str(num_workers),
            "--expected-logical-node",
            str(expected_logical),
            "--world-size",
            str(num_nodes * num_workers),
            "--training-script",
            str(training_script),
            "--log-dir",
            str(log_dir),
            "--",
            *training_args,
        )
        control_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "starting resident standby workers=%d expected_logical=%d "
            "world_size=%d control_dir=%s",
            num_workers,
            expected_logical,
            num_nodes * num_workers,
            control_dir,
        )
        self._resident_session_id = session_id
        self._resident_control_dir = control_dir
        self._resident_num_workers = num_workers
        self._resident_ready_logged = False
        self._resident_last_status_log = 0.0
        self._resident_process = subprocess.Popen(
            command,
            env=os.environ.copy(),
            start_new_session=True,
        )

    def _stop_resident_standby(self) -> None:
        process = self._resident_process
        self._resident_process = None
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)

    def _activate_resident_standby(
        self,
        *,
        command: Sequence[str],
        environment: Mapping[str, str],
        logical_node: int,
        epoch: int,
        target_local_rank: int | None = None,
    ) -> bool:
        process = self._resident_process
        if (
            not self._resident_enabled()
            or process is None
            or self._resident_control_dir is None
            or self._resident_session_id is None
        ):
            return False

        wait_seconds = float(
            os.environ.get(
                "MOEGAMBIT_STANDBY_READY_TIMEOUT",
                "30",
            )
        )
        deadline = time.monotonic() + max(0.0, wait_seconds)
        ready: list[Mapping[str, Any]] = []
        missing: list[int] = list(
            range(self._resident_num_workers)
        )
        while time.monotonic() <= deadline:
            if process.poll() is not None:
                logger.error(
                    "resident standby failed before activation code=%d\n%s",
                    process.returncode,
                    self._resident_failure_tail(),
                )
                self._resident_process = None
                self._resident_failed = True
                return False
            ready, missing = self._resident_ready_snapshot()
            if len(ready) == self._resident_num_workers:
                break
            time.sleep(0.2)
        if len(ready) != self._resident_num_workers:
            logger.error(
                "resident standby is not fully ready; ready=%d/%d "
                "missing_local_ranks=%s after %.1fs; using cold launch",
                len(ready),
                self._resident_num_workers,
                missing,
                wait_seconds,
            )
            self._stop_resident_standby()
            self._resident_failed = True
            return False

        rank_log_dir = _command_option(
            command, "--enable_each_rank_log"
        )
        if not rank_log_dir:
            rank_log_dir = str(
                self._resident_control_dir / "rank_logs"
            )
        activation_path = (
            self._resident_control_dir
            / f"activate_{self._resident_session_id}.json"
        )
        temporary = activation_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "session_id": self._resident_session_id,
                    "logical_node": logical_node,
                    "epoch": epoch,
                    "target_local_rank": target_local_rank,
                    "rank_log_dir": rank_log_dir,
                    "environment": dict(environment),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, activation_path)

        self._resident_process = None
        self.process_started_at = time.time()
        self.process_logical_node = logical_node
        self.process_command = tuple(command)
        self.process_epoch = epoch
        self.completed_epoch = None
        self._relay_log_path = None
        self._relay_log_offset = 0
        self._relay_log_buffer = ""
        with self._control_lock:
            self.process = process
        command_delay = self._recovery_command_delay(epoch)
        logger.warning(
            "activated resident standby workers=%d/%d "
            "physical_node=%d logical_node=%d epoch=%d "
            "target_local_rank=%s%s",
            len(ready),
            self._resident_num_workers,
            self.physical_node,
            logical_node,
            epoch,
            (
                str(target_local_rank)
                if target_local_rank is not None
                else "all"
            ),
            (
                f" recovery_command_to_activate_s={command_delay:.2f}"
                if command_delay is not None
                else ""
            ),
        )
        return True

    def _start_worker(
        self,
        logical_node: int,
        epoch: int,
        master_addr: str,
        master_port: int,
        failed_logical_node: int | None = None,
        failure_step: int | None = None,
        replacement_rank: int | None = None,
    ) -> None:
        self._stop_standby_prefetch()
        previous_epoch = self.process_epoch
        force_recovery_stop = (
            self._recovery_force_preempt_enabled()
            and self.process is not None
            and previous_epoch is not None
            and epoch > previous_epoch
        )
        self._stop_worker(
            force=force_recovery_stop,
            reason=(
                f"recovery_epoch_{previous_epoch}_to_{epoch}"
                if force_recovery_stop
                else "worker_restart"
            ),
        )
        command = self._formatted_command(
            logical_node, epoch, master_addr, master_port
        )
        environment = os.environ.copy()
        local_world_size = _command_option(command, "--num_gpus")
        environment.update(
            {
                "NODE_RANK": str(logical_node),
                "GROUP_RANK": str(logical_node),
                "MOEGAMBIT_LOGICAL_NODE_RANK": str(logical_node),
                "MOEGAMBIT_PHYSICAL_NODE_RANK": str(self.physical_node),
                "MOEGAMBIT_RECOVERY_EPOCH": str(epoch),
                "TORCHELASTIC_RESTART_COUNT": str(epoch),
                "MASTER_PORT": str(master_port),
                "MASTER_ADDR": master_addr,
                "MOEGAMBIT_HOT_SPARE_RUN_ID": self.run_id,
            }
        )
        target_local_rank = None
        if replacement_rank is not None:
            if local_world_size is None:
                raise RuntimeError(
                    "rank replacement requires --num_gpus"
                )
            local_world_size_value = int(local_world_size)
            target_local_rank = (
                int(replacement_rank) % local_world_size_value
            )
            environment.update(
                {
                    "RANK": str(replacement_rank),
                    "LOCAL_RANK": str(target_local_rank),
                    "LOCAL_WORLD_SIZE": str(local_world_size_value),
                    "LOCAL_SIZE": str(local_world_size_value),
                    "WORLD_SIZE": str(
                        int(
                            _command_option(command, "--num_nodes")
                            or "1"
                        )
                        * local_world_size_value
                    ),
                    "MOEGAMBIT_DEEPSPEED_INPROCESS_REPLACEMENT": "1",
                    "MOEGAMBIT_RECOVERY_FAILED_RANK": str(
                        replacement_rank
                    ),
                }
            )
        self.process_rank = (
            int(replacement_rank)
            if replacement_rank is not None
            else int(logical_node) * int(local_world_size or "1")
        )
        if failed_logical_node is not None:
            environment[
                "MOEGAMBIT_RECOVERY_FAILED_LOGICAL_NODE"
            ] = str(failed_logical_node)
        else:
            environment.pop(
                "MOEGAMBIT_RECOVERY_FAILED_LOGICAL_NODE", None
            )
        if failure_step is not None:
            environment[
                "MOEGAMBIT_RECOVERY_FAILURE_STEP"
            ] = str(failure_step)
        else:
            environment.pop("MOEGAMBIT_RECOVERY_FAILURE_STEP", None)
        if local_world_size is not None:
            environment["LOCAL_WORLD_SIZE"] = local_world_size
        rank_log_dir = _command_option(
            command, "--enable_each_rank_log"
        )
        if (
            self.role == "standby"
            and epoch > 0
            and self._activate_resident_standby(
                command=command,
                environment=environment,
                logical_node=logical_node,
                epoch=epoch,
                target_local_rank=target_local_rank,
            )
        ):
            return
        if self.role == "standby" and epoch > 0:
            self._stop_resident_standby()
            if replacement_rank is not None:
                workload = _deepspeed_python_workload(command)
                if workload is None or target_local_rank is None:
                    raise RuntimeError(
                        "rank replacement has no direct Python workload "
                        "fallback"
                    )
                training_script, training_args = workload
                direct_command = [
                    sys.executable,
                    "-u",
                    str(training_script),
                    *training_args,
                ]
                if not any(
                    item == "--local_rank"
                    or item.startswith("--local_rank=")
                    for item in direct_command
                ):
                    direct_command.append(
                        f"--local_rank={target_local_rank}"
                    )
                command = tuple(direct_command)
            # Keep the old page-cache path as a fail-safe when resident
            # preparation was disabled or did not reach all local ranks.
            self._prefetch_stop.clear()
            self._prefetch_dataset_index("recovery-cold-fallback")
        command_delay = self._recovery_command_delay(epoch)
        logger.warning(
            "starting worker physical_node=%d logical_node=%d epoch=%d "
            "local_world_size=%s master=%s:%d rank_logs=%s%s",
            self.physical_node,
            logical_node,
            epoch,
            environment.get("LOCAL_WORLD_SIZE", "unknown"),
            master_addr,
            master_port,
            rank_log_dir or "disabled",
            (
                f" recovery_command_to_spawn_s={command_delay:.2f}"
                if command_delay is not None
                else ""
            ),
        )
        logger.debug("worker command=%s", command)
        launched_process = subprocess.Popen(
            command,
            env=environment,
            start_new_session=True,
        )
        self.process_started_at = time.time()
        self.process_logical_node = logical_node
        self.process_command = command
        self.process_epoch = epoch
        self.completed_epoch = None
        self._relay_log_path = None
        self._relay_log_offset = 0
        self._relay_log_buffer = ""
        # Publish the process only after its epoch and metadata are coherent.
        with self._control_lock:
            self.process = launched_process

    def _rank_log_path(self) -> Path | None:
        if self.process_logical_node is None:
            return None
        log_dir_text = _command_option(
            self.process_command or self.command,
            "--enable_each_rank_log",
        )
        if not log_dir_text:
            return None
        local_world_size_text = _command_option(
            self.command, "--num_gpus"
        )
        try:
            local_world_size = int(local_world_size_text or "1")
        except ValueError:
            local_world_size = 1
        rank = (
            self.process_rank
            if self.process_rank is not None
            else self.process_logical_node * local_world_size
        )
        try:
            candidates = list(
                Path(log_dir_text).glob(f"*_rank{rank}.log")
            )
        except OSError:
            return None
        eligible: list[tuple[float, Path]] = []
        for path in candidates:
            try:
                modified = path.stat().st_mtime
            except OSError:
                continue
            if (
                self.process_started_at is None
                or modified >= self.process_started_at - 5.0
            ):
                eligible.append((modified, path))
        if not eligible:
            return None
        return max(eligible, key=lambda item: item[0])[1]

    def _relay_worker_log(self) -> None:
        relay_mode = os.environ.get(
            "MOEGAMBIT_RELAY_RANK_LOG", "key"
        ).strip().lower()
        if relay_mode in {"0", "false", "no", "off", "none"}:
            return
        relay_all = relay_mode in {"1", "true", "yes", "on", "full"}
        if (
            not relay_all
            and self.process_logical_node != 0
            and self.role != "standby"
        ):
            return
        path = self._rank_log_path()
        if path is None:
            return
        if path != self._relay_log_path:
            self._relay_log_path = path
            self._relay_log_offset = 0
            self._relay_log_buffer = ""
        try:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(self._relay_log_offset)
                content = stream.read()
                self._relay_log_offset = stream.tell()
        except OSError:
            return
        if not content:
            return
        rank_text = path.stem.rsplit("_rank", 1)[-1]
        lines = (self._relay_log_buffer + content).splitlines(keepends=True)
        self._relay_log_buffer = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._relay_log_buffer = lines.pop()
        for line in lines:
            if not relay_all and not any(
                marker in line
                for marker in (
                    "iteration ",
                    "FAULT_",
                    "single-stage hybrid restore",
                    "in-process hybrid restore",
                    "STANDBY_ACTIVATED",
                    "STANDBY_CACHE_HIT",
                    "STANDBY_PACKED_CACHE",
                    "MODEL_BUILD_START",
                    "MODEL_BUILD_DONE",
                    "DEEPSPEED_ENGINE_INIT_START",
                    "DEEPSPEED_ENGINE_INIT_DONE",
                    "RANK_INPROCESS_INITIAL_BARRIER",
                    "MoEGambit packed expert load",
                    "Traceback (most recent call last)",
                    "FATAL ",
                    "RuntimeError:",
                    "OutOfMemoryError:",
                    "Error:",
                    "Exception:",
                )
            ):
                continue
            print(
                f"[worker-rank{rank_text}] {line}",
                end="",
                flush=True,
            )

    def _prefetch_file(self, path: Path, budget: int) -> int:
        try:
            size = path.stat().st_size
        except OSError:
            return 0
        if size > budget:
            return 0
        read_bytes = 0
        try:
            with path.open("rb", buffering=0) as stream:
                while not self._prefetch_stop.is_set():
                    chunk = stream.read(min(16 * 1024 * 1024, budget))
                    if not chunk:
                        break
                    read_bytes += len(chunk)
                    budget -= len(chunk)
                    if budget <= 0:
                        break
        except OSError as exc:
            logger.warning(
                "standby prefetch could not read %s: %s", path, exc
            )
        return read_bytes

    def _checkpoint_prefetch_files(
        self, checkpoint_dir: Path, logical_node: int
    ) -> tuple[str | None, list[Path]]:
        latest = checkpoint_dir / "latest"
        try:
            tag = latest.read_text(encoding="utf-8").strip()
        except OSError:
            return None, []
        if not tag:
            return None, []
        tag_dir = checkpoint_dir / tag
        try:
            candidates = [
                path for path in tag_dir.iterdir() if path.is_file()
            ]
        except OSError:
            return tag, []

        needed_optimizer_shards: set[tuple[int, int]] | None = None
        pipeline_text = _command_option(
            self.command, "--pipeline-parallel-size"
        )
        workers_text = _command_option(self.command, "--num_gpus")
        zero_stage_text = _command_option(
            self.command, "--zero-stage"
        )
        try:
            pipeline_size = int(pipeline_text or "")
            local_workers = int(workers_text or "")
            zero_stage = int(zero_stage_text or "1")
        except ValueError:
            pipeline_size = 0
            local_workers = 0
            zero_stage = 1
        if pipeline_size > 0 and local_workers > 0:
            needed_optimizer_shards = set()
            first_rank = logical_node * local_workers
            for local_rank in range(local_workers):
                global_rank = first_rank + local_rank
                needed_optimizer_shards.add(
                    (
                        global_rank // pipeline_size,
                        global_rank % pipeline_size,
                    )
                )

        def selected_for_replacement(path: Path) -> bool:
            if path.name.endswith("_model_states.pt"):
                return True
            match = _ZERO_OPTIM_SHARD.match(path.name)
            if match is None:
                return False
            if needed_optimizer_shards is None:
                return True
            if zero_stage == 2:
                needed_mp_ranks = {
                    mp_rank
                    for _, mp_rank in needed_optimizer_shards
                }
                return int(match.group("mp")) in needed_mp_ranks
            return (
                int(match.group("dp")),
                int(match.group("mp")),
            ) in needed_optimizer_shards

        # In the supported hot-swap topology, replacement local workers span
        # PP stages. ZeRO-1 loads one local DP shard per worker, while the
        # elastic ZeRO-2 path loads every DP shard for that worker's MP rank.
        # Warm exactly that union, then cap total coverage by the byte budget.
        selected = [
            path
            for path in candidates
            if selected_for_replacement(path)
        ]
        selected.sort(
            key=lambda path: (
                0
                if (
                    path.name.startswith("mp_rank_")
                    and path.name.endswith("_model_states.pt")
                )
                else (
                    1
                    if path.name.endswith("_model_states.pt")
                    else 2
                ),
                path.name,
            )
        )
        return tag, selected

    def _prefetch_dataset_index(self, reason: str) -> int:
        data_path = _command_option(self.command, "--data-path")
        if not data_path:
            return 0
        max_gib = float(
            os.environ.get("MOEGAMBIT_STANDBY_PREFETCH_MAX_GIB", "128")
        )
        max_bytes = max(0, int(max_gib * 1024**3))
        if not max_bytes:
            return 0
        index_path = Path(data_path + ".idx")
        started = time.monotonic()
        warmed = self._prefetch_file(index_path, max_bytes)
        if warmed:
            logger.info(
                "standby prefetched dataset index reason=%s path=%s "
                "bytes=%d seconds=%.2f",
                reason,
                index_path,
                warmed,
                time.monotonic() - started,
            )
        return warmed

    def _standby_prefetch_loop(self) -> None:
        max_gib = float(
            os.environ.get("MOEGAMBIT_STANDBY_PREFETCH_MAX_GIB", "128")
        )
        max_bytes = max(0, int(max_gib * 1024**3))
        checkpoint_text = _command_option(
            self.command, "--checkpoint-dir"
        )
        logical_node = int(
            os.environ.get(
                "MOEGAMBIT_STANDBY_PREFETCH_LOGICAL_NODE", "0"
            )
        )
        self._prefetch_dataset_index("standby")

        while not self._prefetch_stop.wait(2.0):
            if not checkpoint_text or not max_bytes:
                continue
            tag, paths = self._checkpoint_prefetch_files(
                Path(checkpoint_text), logical_node
            )
            if not tag or tag == self._prefetched_checkpoint_tag or not paths:
                continue
            started = time.monotonic()
            remaining = max_bytes
            warmed = 0
            for path in paths:
                if self._prefetch_stop.is_set() or remaining <= 0:
                    break
                amount = self._prefetch_file(path, remaining)
                warmed += amount
                remaining -= amount
            completed = not self._prefetch_stop.is_set()
            if warmed:
                logger.info(
                    "standby checkpoint prefetch progress tag=%s "
                    "logical_node=%d files=%d bytes=%d seconds=%.2f "
                    "completed=%s",
                    tag,
                    logical_node,
                    len(paths),
                    warmed,
                    time.monotonic() - started,
                    completed,
                )
            if completed:
                self._prefetched_checkpoint_tag = tag
                self._prefetch_dataset_index("post-checkpoint")

    def _ensure_standby_prefetch(self) -> None:
        enabled = os.environ.get(
            "MOEGAMBIT_STANDBY_PREFETCH", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if (
            self.role != "standby"
            or not enabled
            or (
                self._prefetch_thread is not None
                and self._prefetch_thread.is_alive()
            )
        ):
            return
        self._prefetch_stop.clear()
        self._prefetch_thread = threading.Thread(
            target=self._standby_prefetch_loop,
            name="moegambit-standby-prefetch",
            daemon=True,
        )
        self._prefetch_thread.start()

    def _stop_standby_prefetch(self) -> None:
        thread = self._prefetch_thread
        if thread is None:
            return
        self._prefetch_stop.set()
        thread.join(timeout=5.0)
        self._prefetch_thread = None

    def _stop_worker(
        self,
        *,
        force: bool = False,
        reason: str = "supervisor_stop",
    ) -> None:
        with self._control_lock:
            process = self.process
            self.process = None
        if not force:
            self._relay_worker_log()
        if process is None:
            return
        if process.poll() is not None:
            if force:
                self._relay_worker_log()
            return
        started = time.monotonic()
        stop_signal = signal.SIGKILL if force else signal.SIGTERM
        logger.warning(
            "stopping worker physical_node=%d pid=%d epoch=%s "
            "signal=%s reason=%s",
            self.physical_node,
            process.pid,
            self.process_epoch,
            signal.Signals(stop_signal).name,
            reason,
        )
        try:
            os.killpg(process.pid, stop_signal)
        except PermissionError as exc:
            logger.warning(
                "could not signal worker process group; falling back to "
                "launcher process physical_node=%d pid=%d error=%s",
                self.physical_node,
                process.pid,
                exc,
            )
            try:
                process.send_signal(stop_signal)
            except ProcessLookupError:
                return
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5 if force else 20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except PermissionError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
        self._relay_worker_log()
        logger.warning(
            "worker stopped physical_node=%d pid=%d epoch=%s "
            "return_code=%s elapsed_s=%.2f reason=%s",
            self.physical_node,
            process.pid,
            self.process_epoch,
            process.returncode,
            time.monotonic() - started,
            reason,
        )

    def run(self) -> int:
        self._ensure_resident_standby()
        command = self._register()
        self._observe_control_command(command, source="register")
        pending = self._take_pending_command(command)
        if pending is not None:
            command = pending
        self._start_heartbeat()
        try:
            while True:
                pending = self._take_pending_command(command)
                if pending is not None:
                    command = pending
                action = str(command.get("action", "wait"))
                epoch = int(command.get("epoch", 0))

                if action == "run":
                    logical_node = int(command["logical_node"])
                    master_addr = str(command["master_addr"])
                    master_port = int(command["master_port"])
                    failed_logical_node = command.get(
                        "failed_logical_node"
                    )
                    failure_step = command.get("failure_step")
                    failed_logical_node = (
                        int(failed_logical_node)
                        if failed_logical_node is not None
                        else None
                    )
                    failure_step = (
                        int(failure_step)
                        if failure_step is not None
                        else None
                    )
                    preserve_process = bool(
                        command.get("preserve_process")
                    )
                    if (
                        preserve_process
                        and self.process is not None
                        and self.process.poll() is None
                    ):
                        self.process_epoch = epoch
                    elif self.process_epoch != epoch:
                        self._start_worker(
                            logical_node,
                            epoch,
                            master_addr,
                            master_port,
                            failed_logical_node,
                            failure_step,
                        )
                    elif (
                        self.process is None
                        and self.completed_epoch != epoch
                    ):
                        self._start_worker(
                            logical_node,
                            epoch,
                            master_addr,
                            master_port,
                            failed_logical_node,
                            failure_step,
                        )

                    if self.process is not None:
                        return_code = self.process.poll()
                        self._relay_worker_log()
                        if return_code is not None:
                            diagnostic = ""
                            if return_code != 0:
                                diagnostic = (
                                    collect_worker_failure_diagnostic(
                                        self.process_command
                                        or self.command,
                                        epoch=epoch,
                                        logical_node=logical_node,
                                        process_started_at=(
                                            self.process_started_at
                                        ),
                                    )
                                )
                                if diagnostic:
                                    logger.error(
                                        "worker failure diagnostic "
                                        "physical_node=%d logical_node=%d "
                                        "epoch=%d:\n%s",
                                        self.physical_node,
                                        logical_node,
                                        epoch,
                                        diagnostic,
                                    )
                                else:
                                    logger.error(
                                        "worker exited with code %d but no "
                                        "fatal artifact or rank traceback "
                                        "was found",
                                        return_code,
                                    )
                            with self._control_lock:
                                self.process = None
                            if return_code == 0:
                                self.completed_epoch = epoch
                                command = self._request(
                                    "runner_complete",
                                    epoch=epoch,
                                    state="completed",
                                )
                                continue
                            latest = self._request(
                                "poll",
                                epoch=epoch,
                                state="runner_exited",
                                return_code=return_code,
                            )
                            if (
                                int(latest.get("epoch", epoch)) != epoch
                                or latest.get("action") != "run"
                            ):
                                command = latest
                                continue
                            command = self._request(
                                "runner_failure",
                                epoch=epoch,
                                state="failed",
                                return_code=return_code,
                                reason=f"runner_exit_{return_code}",
                                diagnostic=diagnostic[-8000:],
                            )
                            continue

                elif action == "replace_rank":
                    logical_node = int(command["logical_node"])
                    replacement_rank = int(
                        command["replacement_rank"]
                    )
                    master_addr = str(command["master_addr"])
                    master_port = int(command["master_port"])
                    failed_logical_node = command.get(
                        "failed_logical_node"
                    )
                    failure_step = command.get("failure_step")
                    if self.process_epoch != epoch:
                        self._start_worker(
                            logical_node,
                            epoch,
                            master_addr,
                            master_port,
                            (
                                int(failed_logical_node)
                                if failed_logical_node is not None
                                else None
                            ),
                            (
                                int(failure_step)
                                if failure_step is not None
                                else None
                            ),
                            replacement_rank,
                        )

                    if self.process is not None:
                        return_code = self.process.poll()
                        self._relay_worker_log()
                        if return_code is not None:
                            with self._control_lock:
                                self.process = None
                            if return_code == 0:
                                self.completed_epoch = epoch
                                command = self._request(
                                    "runner_complete",
                                    epoch=epoch,
                                    state="completed",
                                )
                                continue
                            command = self._request(
                                "runner_failure",
                                epoch=epoch,
                                state="failed",
                                return_code=return_code,
                                reason=(
                                    "replacement_rank_exit_"
                                    f"{return_code}"
                                ),
                            )
                            continue

                elif action in {"wait", "standby"}:
                    self._stop_worker()
                    self._ensure_resident_standby()
                    self._ensure_standby_prefetch()
                elif action == "retire":
                    self._stop_worker()
                    self._stop_resident_standby()
                    if not self.retired_logged:
                        logger.error(
                            "physical node %d retired after failover; "
                            "waiting for cluster completion",
                            self.physical_node,
                        )
                        self.retired_logged = True
                elif action == "complete":
                    self._stop_worker()
                    self._stop_resident_standby()
                    self._request(
                        "ack_complete", epoch=epoch, state="complete"
                    )
                    return 0
                elif action == "abort":
                    logical_node = next(
                        (
                            logical
                            for logical, physical in {
                                int(key): int(value)
                                for key, value in dict(
                                    command.get("mapping", {})
                                ).items()
                            }.items()
                            if physical == self.physical_node
                        ),
                        None,
                    )
                    if (
                        self.process is not None
                        and self.process.poll() is None
                        and logical_node is not None
                    ):
                        diagnostic = collect_worker_failure_diagnostic(
                            self.process_command or self.command,
                            epoch=epoch,
                            logical_node=logical_node,
                            process_started_at=self.process_started_at,
                        )
                        if diagnostic:
                            logger.error(
                                "live worker diagnostic before abort "
                                "physical_node=%d logical_node=%d epoch=%d:\n%s",
                                self.physical_node,
                                logical_node,
                                epoch,
                                diagnostic,
                            )
                    self._stop_worker()
                    self._stop_resident_standby()
                    logger.error(
                        "hot-spare recovery aborted: %s",
                        command.get("reason", "unknown reason"),
                    )
                    try:
                        self._request(
                            "ack_abort", epoch=epoch, state="aborted"
                        )
                    except (ConnectionError, OSError):
                        pass
                    return 70
                else:
                    raise RuntimeError(
                        f"unsupported coordinator action {action!r}"
                    )

                time.sleep(self.heartbeat_interval)
                try:
                    latest = self._request(
                        "poll",
                        epoch=epoch,
                        state=(
                            "running"
                            if self.process is not None
                            and self.process.poll() is None
                            else action
                        ),
                    )
                    self._observe_control_command(
                        latest, source="poll"
                    )
                    if self._command_supersedes(latest, command):
                        command = dict(latest)
                except (
                    ConnectionError,
                    OSError,
                    TimeoutError,
                ) as exc:
                    now = time.monotonic()
                    if now - self._last_control_warning >= 30.0:
                        logger.warning(
                            "control poll failed physical_node=%d; "
                            "keeping current command: %s",
                            self.physical_node,
                            exc,
                        )
                        self._last_control_warning = now
        finally:
            self._stop_heartbeat()
            self._stop_standby_prefetch()
            self._stop_resident_standby()
            self._stop_worker()


def _infer_advertise_addr(remote_host: str) -> str:
    """Return the local address selected by the route to the coordinator."""
    last_error: OSError | None = None
    for family, socktype, protocol, _, address in socket.getaddrinfo(
        remote_host, 9, type=socket.SOCK_DGRAM
    ):
        try:
            with socket.socket(family, socktype, protocol) as probe:
                probe.connect(address)
                return str(probe.getsockname()[0])
        except OSError as exc:
            last_error = exc
    raise OSError(
        f"cannot infer a routable local address toward {remote_host}: "
        f"{last_error}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MoEGambit node-level hot-spare supervisor"
    )
    parser.add_argument(
        "--mode",
        choices=("agent", "coordinator-agent"),
        required=True,
    )
    parser.add_argument("--coordinator-host", required=True)
    parser.add_argument("--coordinator-port", type=int, required=True)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--training-nodes", type=int, required=True)
    parser.add_argument("--spare-node", type=int, required=True)
    parser.add_argument("--physical-node", type=int, required=True)
    parser.add_argument("--local-world-size", type=int, default=1)
    parser.add_argument("--rank-hot-swap", action="store_true")
    parser.add_argument("--advertise-addr")
    parser.add_argument("--base-master-port", type=int, required=True)
    parser.add_argument("--master-port-stride", type=int, default=1)
    parser.add_argument("--heartbeat-interval", type=float, default=1.0)
    parser.add_argument("--heartbeat-timeout", type=float, default=30.0)
    parser.add_argument("--recovery-timeout", type=float, default=300.0)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--state-path")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("MOEGAMBIT_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    endpoint = WatcherEndpoint(
        args.coordinator_host,
        args.coordinator_port,
        timeout=max(10.0, args.heartbeat_interval * 4),
    )
    role = (
        "standby"
        if args.physical_node == args.spare_node
        else "active"
    )
    advertise_addr = args.advertise_addr or _infer_advertise_addr(
        args.coordinator_host
    )
    supervisor = AgentSupervisor(
        endpoint=endpoint,
        run_id=args.run_id,
        physical_node=args.physical_node,
        role=role,
        advertise_addr=advertise_addr,
        command=command,
        heartbeat_interval=args.heartbeat_interval,
        startup_timeout=args.startup_timeout,
    )
    if args.mode == "agent":
        if role != "active":
            raise ValueError("standby node must use coordinator-agent mode")
        return supervisor.run()

    if args.physical_node != args.spare_node:
        raise ValueError("coordinator-agent must run on the spare node")
    coordinator = HotSpareCoordinator(
        run_id=args.run_id,
        training_nodes=args.training_nodes,
        spare_physical_node=args.spare_node,
        base_master_port=args.base_master_port,
        port_stride=args.master_port_stride,
        heartbeat_timeout=args.heartbeat_timeout,
        recovery_timeout=args.recovery_timeout,
        state_path=Path(args.state_path) if args.state_path else None,
        local_world_size=args.local_world_size,
        rank_hot_swap=args.rank_hot_swap,
    )
    runtime = WatcherRuntime(
        args.listen_host, args.coordinator_port, coordinator.handle
    )
    server_thread = threading.Thread(
        target=runtime.serve_forever,
        name="moegambit-hot-spare-coordinator",
        daemon=True,
    )
    server_thread.start()
    try:
        return_code = supervisor.run()
        deadline = time.monotonic() + max(
            10.0, args.heartbeat_timeout
        )
        while (
            coordinator.status in {"complete", "aborted"}
            and time.monotonic() < deadline
        ):
            now = time.monotonic()
            expected_acks = {
                physical
                for physical, record in coordinator.agents.items()
                if now - record.last_seen <= args.heartbeat_timeout
            }
            observed_acks = (
                coordinator.completion_acks
                if coordinator.status == "complete"
                else coordinator.abort_acks
            )
            if expected_acks.issubset(observed_acks):
                break
            time.sleep(0.2)
        return return_code
    finally:
        runtime.shutdown()
        server_thread.join(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())

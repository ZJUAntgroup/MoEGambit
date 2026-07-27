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
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from moegambit.runtime.protocol import WireMessage
from moegambit.runtime.watcher import WatcherRuntime
from moegambit.runtime.watcher_client import WatcherClient, WatcherEndpoint


logger = logging.getLogger(__name__)


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
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
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
        self.mapping = {
            logical_node: logical_node
            for logical_node in range(self.training_nodes)
        }

    @property
    def master_port(self) -> int:
        return self.base_master_port + self.epoch * self.port_stride

    @property
    def active_physical_nodes(self) -> set[int]:
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

            if message.kind == "rank_failure":
                self._handle_rank_failure(physical_node, payload)
            elif message.kind == "runner_failure":
                self._handle_runner_failure(physical_node, payload)
            elif message.kind == "runner_complete":
                self._handle_runner_complete(physical_node, payload)
            elif message.kind == "worker_ready":
                self._handle_worker_ready(physical_node, payload)
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
            self._persist()
            return WireMessage("command", self._command_for(physical_node))

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
        self._start_failover(
            logical_node,
            failed_physical_node,
            str(payload.get("reason", "rank_failure")),
            reporting_physical_node=reporting_physical_node,
            failed_rank=payload.get("rank"),
        )

    def _handle_runner_failure(
        self, physical_node: int, payload: Mapping[str, Any]
    ) -> None:
        if self.status not in {"running", "recovering"}:
            return
        if self._payload_epoch(payload) != self.epoch:
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
            return
        self.ready_logical_nodes.add(logical_node)
        if (
            self.recovery_started_at is not None
            and len(self.ready_logical_nodes) == self.training_nodes
        ):
            elapsed = time.monotonic() - self.recovery_started_at
            logger.warning(
                "recovery epoch %d reached TRAIN_READY on all logical "
                "nodes in %.2fs",
                self.epoch,
                elapsed,
            )
            if self.failure is not None:
                self.failure["train_ready_seconds"] = elapsed
            self.recovery_started_at = None

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

    def _detect_timeouts(self, now: float) -> None:
        if self.status != "running":
            return
        if (
            self.recovery_started_at is not None
            and now - self.recovery_started_at > self.recovery_timeout
        ):
            missing = sorted(
                set(range(self.training_nodes)) - self.ready_logical_nodes
            )
            self.status = "aborted"
            self.abort_reason = (
                f"recovery epoch {self.epoch} did not reach TRAIN_READY "
                f"within {self.recovery_timeout:.1f}s; "
                f"missing logical nodes={missing}"
            )
            logger.error(self.abort_reason)
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
            self._start_failover(
                logical_node,
                physical_node,
                "agent_heartbeat_timeout",
            )
            break

    def _command_for(self, physical_node: int) -> dict[str, Any]:
        rank_zero_physical = self.mapping[0]
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
            "retired": sorted(self.retired),
            "completed": sorted(self.completed),
            "failure": self.failure,
            "abort_reason": self.abort_reason,
            "ready_logical_nodes": sorted(self.ready_logical_nodes),
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

    def _start_worker(
        self,
        logical_node: int,
        epoch: int,
        master_addr: str,
        master_port: int,
    ) -> None:
        self._stop_worker()
        command = self._formatted_command(
            logical_node, epoch, master_addr, master_port
        )
        environment = os.environ.copy()
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
        logger.warning(
            "starting worker physical_node=%d logical_node=%d epoch=%d "
            "master=%s:%d command=%s",
            self.physical_node,
            logical_node,
            epoch,
            master_addr,
            master_port,
            command,
        )
        self.process = subprocess.Popen(
            command,
            env=environment,
            start_new_session=True,
        )
        self.process_epoch = epoch
        self.completed_epoch = None

    def _stop_worker(self) -> None:
        process = self.process
        self.process = None
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

    def run(self) -> int:
        command = self._register()
        try:
            while True:
                action = str(command.get("action", "wait"))
                epoch = int(command.get("epoch", 0))

                if action == "run":
                    logical_node = int(command["logical_node"])
                    master_addr = str(command["master_addr"])
                    master_port = int(command["master_port"])
                    if self.process_epoch != epoch:
                        self._start_worker(
                            logical_node, epoch, master_addr, master_port
                        )
                    elif (
                        self.process is None
                        and self.completed_epoch != epoch
                    ):
                        self._start_worker(
                            logical_node, epoch, master_addr, master_port
                        )

                    if self.process is not None:
                        return_code = self.process.poll()
                        if return_code is not None:
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
                            )
                            continue

                elif action in {"wait", "standby"}:
                    self._stop_worker()
                elif action == "retire":
                    self._stop_worker()
                    if not self.retired_logged:
                        logger.error(
                            "physical node %d retired after failover; "
                            "waiting for cluster completion",
                            self.physical_node,
                        )
                        self.retired_logged = True
                elif action == "complete":
                    self._stop_worker()
                    self._request(
                        "ack_complete", epoch=epoch, state="complete"
                    )
                    return 0
                elif action == "abort":
                    self._stop_worker()
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
                command = self._request(
                    "heartbeat",
                    epoch=epoch,
                    state=(
                        "running"
                        if self.process is not None
                        and self.process.poll() is None
                        else action
                    ),
                )
        finally:
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

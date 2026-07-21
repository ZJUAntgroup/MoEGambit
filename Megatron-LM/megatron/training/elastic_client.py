"""
elastic_client.py — Training-side client for hot-spare elastic recovery.

In launcher-controlled mode, every rank publishes local state tags over a Unix
socket and the node launcher owns the watcher connection.  The legacy mode,
where local_rank 0 owns that connection, remains available for compatibility.

Architecture (single-rank replacement):
  - Watcher sends PAUSE to ALL nodes (including target)
  - ALL nodes: detect pause → destroy_process_group → send ready_to_rebuild
  - Watcher: receives all ready → sends kill_rank to target node
  - Target node: kills only the specified local_rank worker
  - Watcher: launches 1 spare process on spare node
  - Watcher: sends rebuild signal to all surviving ranks
  - All surviving ranks: re-init_process_group → sync params → resume
  - Spare process: init_process_group → receive params → join training

CRITICAL: elastic_check_pause() must NEVER use dist.all_reduce or any global
collective.  If a node is dead, a global collective hangs forever.  We use
file-based signaling between local ranks on the same node.

Usage in training code:
    from megatron.training.elastic_client import (
        elastic_client_start,
        elastic_check_pause,
        elastic_do_rebuild,
        elastic_on_nccl_error,
    )
"""

import json
import hashlib
import fcntl
import logging
import os
import io
import signal as signal_module
import socket
import struct
import subprocess
import threading
import time
from datetime import timedelta
from inspect import signature
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Module state
_CLIENT: Optional["ElasticClient"] = None
_PAUSE_REQUESTED = False
_REBUILD_INFO: Optional[dict] = None
_FALLBACK_RELAUNCH_INFO: Optional[dict] = None
_REBUILD_STORE = None
_LOCK = threading.Lock()
_CURRENT_STEP = -1
_CURRENT_STEP_TAG = -1
_CURRENT_TRAIN_PHASE = "startup"
_LAUNCHER_STATUS_SOCKET = None

_POST_REBUILD_STATE_ENV = (
    "ELASTIC_RECOVERY_STATE",
    "ELASTIC_POST_REBUILD_PENDING",
    "ELASTIC_POST_REBUILD_TRACE_ACTIVE",
    "ELASTIC_POST_REBUILD_TRACE_TOKEN",
    "ELASTIC_POST_REBUILD_TRACE_ITERATION",
    "ELASTIC_POST_REBUILD_COMM_WARMUP_DONE",
    "ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER_TOKEN",
)

_POST_REBUILD_STABILIZATION_PHASES = {
    "rerun_state_contract_start",
    "rerun_state_contract_ready",
    "rerun_state_contract_error",
    "iteration_prologue_start",
    "iteration_prologue_done",
    "iteration_safe_point_start",
    "iteration_safe_point_done",
    "moegambit_before_iteration_start",
    "moegambit_before_iteration_done",
    "forward_backward_start",
    "pipeline_p2p_start",
    "pipeline_p2p_returned",
    "moe_first_collective_prepared",
    "moe_first_collective_store_ready",
    "moe_first_collective_store_error",
    "moe_first_collective_start",
    "moe_first_collective_done",
    "moe_first_collective_error",
    "moe_first_collective_timeout",
    "optimizer_pg_contract_ready",
    "optimizer_pg_contract_error",
    "optimizer_step_start",
    "optimizer_step_done",
    "optimizer_skipped",
    "train_step_finalize_done",
    "training_log_start",
    "training_log_done",
    "post_step_callbacks_start",
    "post_step_callbacks_done",
    "checkpoint_exit_start",
    "checkpoint_exit_done",
}


def _elastic_ipv4_interface_for_peer(peer_host):
    """Return the interface carrying IPv4 traffic to a recovery peer."""
    if not peer_host:
        return None, None
    try:
        peer_ip = socket.gethostbyname(peer_host)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((peer_ip, 9))
            local_ip = probe.getsockname()[0]
    except OSError:
        return None, None

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control:
            for _, interface in socket.if_nameindex():
                request = struct.pack("256s", interface[:15].encode())
                try:
                    response = fcntl.ioctl(control.fileno(), 0x8915, request)
                except OSError:
                    continue
                if socket.inet_ntoa(response[20:24]) == local_ip:
                    return interface, local_ip
    except OSError:
        pass
    return None, local_ip


def elastic_configure_recovery_nccl_transport():
    """Pin recovery NCCL to the control-plane-reachable IPv4 socket path."""
    enabled = os.environ.get("ELASTIC_RECOVERY_NCCL_SOCKET_ONLY", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    recovery_active = is_rebuild_mode() or os.environ.get("ELASTIC_PG_GENERATION", "0") not in (
        "",
        "0",
    )
    if not enabled or not recovery_active:
        return None

    peer_host = os.environ.get("ELASTIC_WATCHER_ADDR")
    if peer_host in (None, "", "127.0.0.1", "localhost", "::1"):
        peer_host = os.environ.get("MASTER_ADDR")
    interface, local_ip = _elastic_ipv4_interface_for_peer(peer_host)

    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_SOCKET_FAMILY"] = "AF_INET"
    recovery_ifname = os.environ.get("ELASTIC_RECOVERY_NCCL_SOCKET_IFNAME")
    if recovery_ifname:
        os.environ["NCCL_SOCKET_IFNAME"] = recovery_ifname
    elif interface:
        os.environ["NCCL_SOCKET_IFNAME"] = f"={interface}"

    debug_level = os.environ.get("ELASTIC_RECOVERY_NCCL_DEBUG", "")
    if debug_level:
        os.environ["NCCL_DEBUG"] = debug_level
        os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,NET,ENV")

    config = {
        "peer": peer_host,
        "local_ip": local_ip,
        "socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME", "auto"),
        "socket_family": os.environ["NCCL_SOCKET_FAMILY"],
        "ib_disabled": os.environ["NCCL_IB_DISABLE"],
        "debug": os.environ.get("NCCL_DEBUG", "WARN"),
    }
    logger.warning("[elastic] Recovery NCCL transport configured: %s", config)
    return config


def elastic_create_rebuild_store(host, port, world_size, rank, timeout):
    """Create the same unprefixed TCPStore on survivor and replacement paths."""
    global _REBUILD_STORE

    kwargs = {
        "host_name": str(host),
        "port": int(port),
        "world_size": int(world_size),
        "is_master": int(rank) == 0,
        "timeout": timeout,
        "wait_for_workers": True,
        "multi_tenant": False,
        "use_libuv": True,
    }
    logger.warning(
        "[elastic] Rank %d: opening rebuild TCPStore endpoint=%s:%s role=%s",
        int(rank),
        host,
        port,
        "server" if int(rank) == 0 else "client",
    )
    try:
        store = dist.TCPStore(**kwargs)
    except TypeError:
        # Older supported PyTorch builds expose only the original constructor.
        # All removed options match that constructor's defaults.
        kwargs.pop("wait_for_workers")
        kwargs.pop("multi_tenant")
        kwargs.pop("use_libuv")
        store = dist.TCPStore(**kwargs)

    _REBUILD_STORE = store
    logger.warning(
        "[elastic] Rank %d: rebuild TCPStore ready endpoint=%s:%s "
        "role=%s type=%s torchelastic_agent_store=%s",
        int(rank),
        host,
        port,
        "server" if int(rank) == 0 else "client",
        type(store).__name__,
        os.environ.get("TORCHELASTIC_USE_AGENT_STORE", "unset"),
    )
    return store


def elastic_sanitize_recovery_env_for_startup():
    """Clear one-shot recovery state for a fresh, non-rebuild training process."""
    if is_rebuild_mode():
        return
    removed = []
    for key in _POST_REBUILD_STATE_ENV + (
        "ELASTIC_REPLACEMENT_RANK",
        "ELASTIC_RESUME_ITERATION",
        "ELASTIC_RECOVERY_EPOCH",
        "ELASTIC_RECOVERY_DESCRIPTOR",
        "ELASTIC_RECOVERY_DESCRIPTOR_SHA256",
        "ELASTIC_PG_GENERATION",
        "ELASTIC_CHECKPOINT_STEP",
        "ELASTIC_EXPERT_STALENESS_DELTA",
        "ELASTIC_MOEGAMBIT_RECOVERY_MODE",
        "ELASTIC_TWO_PHASE_RECOVERY",
    ):
        if key in os.environ:
            removed.append(key)
            os.environ.pop(key, None)
    if removed:
        logger.warning(
            "[elastic] Cleared stale recovery env for fresh startup: %s",
            ",".join(sorted(removed)),
        )


def _elastic_post_rebuild_token(iteration: Optional[int] = None) -> str:
    if iteration is None:
        iteration = os.environ.get("ELASTIC_RESUME_ITERATION", "-1")
    replacement_rank = os.environ.get("ELASTIC_REPLACEMENT_RANK", "-1")
    return f"iter{iteration}:replacement{replacement_rank}"


def _elastic_recovery_epoch_payload() -> dict:
    epoch = os.environ.get("ELASTIC_RECOVERY_EPOCH")
    if epoch is None:
        return {}
    payload = {"recovery_epoch": epoch}
    descriptor = os.environ.get("ELASTIC_RECOVERY_DESCRIPTOR")
    if descriptor:
        payload["descriptor"] = descriptor
    return payload


def _launcher_control_enabled() -> bool:
    return (
        os.environ.get("ELASTIC_LAUNCHER_CONTROL_PLANE", "0") == "1"
        and bool(os.environ.get("ELASTIC_LAUNCHER_CONTROL_SOCKET"))
    )


def _launcher_control_send(event: str, **extra) -> bool:
    """Publish rank state to the fault-isolated node launcher."""
    global _LAUNCHER_STATUS_SOCKET

    if not _launcher_control_enabled():
        return False
    socket_path = os.environ.get("ELASTIC_LAUNCHER_CONTROL_SOCKET")
    try:
        recovery_epoch = int(os.environ.get("ELASTIC_RECOVERY_EPOCH", "0") or 0)
    except ValueError:
        recovery_epoch = 0
    payload = {
        "event": event,
        "rank": int(os.environ.get("RANK", "-1")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "-1")),
        "pid": os.getpid(),
        "step": _CURRENT_STEP,
        "step_tag": _CURRENT_STEP_TAG,
        "train_phase": _CURRENT_TRAIN_PHASE,
        "recovery_epoch": recovery_epoch,
        **extra,
    }
    try:
        if _LAUNCHER_STATUS_SOCKET is None:
            _LAUNCHER_STATUS_SOCKET = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            _LAUNCHER_STATUS_SOCKET.setblocking(False)
        _LAUNCHER_STATUS_SOCKET.sendto(
            json.dumps(payload, sort_keys=True).encode(), socket_path
        )
        return True
    except (BlockingIOError, FileNotFoundError, OSError, ValueError) as exc:
        logger.debug("[elastic] launcher control status send failed: %s", exc)
        return False


def elastic_is_post_rebuild_trace_active(iteration: Optional[int] = None) -> bool:
    """Return True only inside an explicit post-rebuild validation step."""
    if os.environ.get("ELASTIC_RECOVERY_STATE") not in (
        "post_rebuild_trace",
        "post_rebuild_stabilization_trace",
    ):
        return False
    if os.environ.get("ELASTIC_POST_REBUILD_PENDING") != "0":
        return False
    if os.environ.get("ELASTIC_POST_REBUILD_TRACE_ACTIVE") != "1":
        return False
    if not os.environ.get("ELASTIC_POST_REBUILD_TRACE_TOKEN"):
        return False
    trace_iteration = os.environ.get("ELASTIC_POST_REBUILD_TRACE_ITERATION")
    if iteration is not None and trace_iteration != str(iteration):
        return False
    return True


class ElasticClient:
    """Background heartbeat client that communicates with the watcher."""

    def __init__(self, watcher_addr: str, watcher_port: int, node_rank: int):
        self.watcher_addr = watcher_addr
        self.watcher_port = watcher_port
        self.node_rank = node_rank
        self.sock: Optional[socket.socket] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.step = -1
        self.step_tag = -1
        self.train_phase = "startup"

    def start(self):
        """Connect to watcher and start heartbeat thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass

    def update_step(self, step: int, phase: Optional[str] = None, step_tag: Optional[int] = None):
        self.step = step
        self.step_tag = step if step_tag is None else step_tag
        if phase is not None:
            self.train_phase = phase

    def _connect(self):
        """Establish TCP connection to watcher."""
        while self.running:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(10.0)
                self.sock.connect((self.watcher_addr, self.watcher_port))
                logger.info(f"[elastic] Connected to watcher at "
                            f"{self.watcher_addr}:{self.watcher_port}")
                return True
            except (ConnectionRefusedError, OSError, socket.timeout) as e:
                logger.warning(f"[elastic] Cannot connect to watcher: {e}, retrying in 5s...")
                time.sleep(5.0)
        return False

    def _send(self, msg: dict):
        """Send a JSON message to watcher."""
        if self.sock is None:
            return
        try:
            data = (json.dumps(msg) + "\n").encode()
            self.sock.sendall(data)
        except (BrokenPipeError, OSError) as e:
            logger.warning(f"[elastic] Send failed: {e}")
            self.sock = None

    def _recv_nonblocking(self) -> Optional[dict]:
        """Try to receive a message from watcher (non-blocking)."""
        if self.sock is None:
            return None
        try:
            self.sock.setblocking(False)
            data = self.sock.recv(4096)
            self.sock.setblocking(True)
            if not data:
                self.sock = None
                return None
            lines = data.decode().strip().split("\n")
            for line in reversed(lines):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        except BlockingIOError:
            try:
                self.sock.setblocking(True)
            except OSError:
                pass
            return None
        except (OSError, ConnectionResetError):
            self.sock = None
            return None

    def _run(self):
        """Main heartbeat loop."""
        global _PAUSE_REQUESTED, _REBUILD_INFO, _FALLBACK_RELAUNCH_INFO

        if not self._connect():
            return

        while self.running:
            # Send heartbeat
            self._send({
                "type": "heartbeat",
                "node_rank": self.node_rank,
                "step": self.step,
                "step_tag": self.step_tag,
                "train_phase": self.train_phase,
                **_elastic_recovery_epoch_payload(),
            })

            # Check for incoming messages
            msg = self._recv_nonblocking()
            if msg is not None:
                msg_type = msg.get("type")
                if msg_type == "pause":
                    with _LOCK:
                        _PAUSE_REQUESTED = True
                    # Write pause signal file for other local ranks
                    _write_pause_signal(msg)
                    logger.warning(f"[elastic] PAUSE signal received! "
                                   f"Failed node: {msg.get('failed_node')}")
                elif msg_type == "rebuild":
                    with _LOCK:
                        _REBUILD_INFO = msg
                    logger.info(f"[elastic] REBUILD signal received: {msg}")
                elif msg_type == "fallback_relaunch":
                    with _LOCK:
                        _FALLBACK_RELAUNCH_INFO = msg
                    logger.error("[elastic] FALLBACK_RELAUNCH received: %s", msg)
                    _request_fallback_relaunch(msg)
                    return
                elif msg_type == "kill_rank":
                    # Watcher tells us to kill a specific local_rank worker.
                    # This happens AFTER all nodes have paused and destroyed
                    # their process groups, so it's safe.
                    target_node = msg.get("target_node", -1)
                    target_local_rank = msg.get("local_rank", 0)
                    if target_node == self.node_rank:
                        logger.warning(
                            "[elastic] KILL_RANK received: killing local_rank %d "
                            "on node %d", target_local_rank, self.node_rank)
                        _kill_single_worker(target_local_rank)
                elif msg_type == "kill_node":
                    # Legacy: kill entire node (backward compat)
                    target_node = msg.get("target_node", -1)
                    if target_node == self.node_rank:
                        logger.error(
                            "[elastic] KILL_NODE received! Killing all "
                            "workers on node %d", self.node_rank)
                        _kill_all_local_workers()
                        return

            # Reconnect if disconnected
            if self.sock is None:
                self._connect()

            time.sleep(5.0)  # heartbeat interval

    def send_ready_to_rebuild(self):
        """Notify watcher that this node is paused and ready to rebuild."""
        self._send({
            "type": "ready_to_rebuild",
            "node_rank": self.node_rank,
            "step": self.step,
            "step_tag": self.step_tag,
            "train_phase": self.train_phase,
            **_elastic_recovery_epoch_payload(),
        })


def _write_pause_signal(msg=None):
    """Write the pause signal file so all local ranks can see it."""
    fault_dir = os.environ.get("ELASTIC_FAULT_DIR", "/tmp/elastic_faults")
    pause_file = os.path.join(fault_dir, "pause_signal")
    try:
        os.makedirs(fault_dir, exist_ok=True)
        content = json.dumps(msg if msg is not None else {})
        with open(pause_file, "w") as f:
            f.write(content)
        logger.info("[elastic] Wrote pause signal file: %s", pause_file)
    except OSError as e:
        logger.warning("[elastic] Failed to write pause file: %s", e)


def _fallback_relaunch_signal_path() -> str:
    fault_dir = os.environ.get("ELASTIC_FAULT_DIR", "/tmp/elastic_faults")
    return os.path.join(fault_dir, "fallback_relaunch_signal.json")


def _write_fallback_relaunch_signal(msg=None):
    fault_dir = os.environ.get("ELASTIC_FAULT_DIR", "/tmp/elastic_faults")
    signal_file = _fallback_relaunch_signal_path()
    try:
        os.makedirs(fault_dir, exist_ok=True)
        tmp_file = signal_file + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(msg if isinstance(msg, dict) else {}, f)
        os.replace(tmp_file, signal_file)
        logger.error("[elastic] Wrote fallback relaunch signal: %s", signal_file)
    except OSError as e:
        logger.warning("[elastic] Failed to write fallback relaunch signal: %s", e)


def _read_fallback_relaunch_signal() -> Optional[dict]:
    signal_file = _fallback_relaunch_signal_path()
    if not os.path.exists(signal_file):
        return None
    try:
        with open(signal_file, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _fallback_exit_code(info=None) -> int:
    if isinstance(info, dict):
        try:
            return int(info.get("exit_code", os.environ.get("ELASTIC_FALLBACK_EXIT_CODE", "75")))
        except (TypeError, ValueError):
            pass
    try:
        return int(os.environ.get("ELASTIC_FALLBACK_EXIT_CODE", "75"))
    except ValueError:
        return 75


def _exit_if_fallback_relaunch_requested():
    info = _read_fallback_relaunch_signal()
    if info is None:
        return
    exit_code = _fallback_exit_code(info)
    logger.error("[elastic] Exiting for fallback relaunch: %s", info)
    os._exit(exit_code)


def _request_fallback_relaunch(msg):
    _write_fallback_relaunch_signal(msg)
    exit_code = _fallback_exit_code(msg)
    ppid = os.getppid()
    try:
        pgid = os.getpgid(ppid)
        logger.error(
            "[elastic] Requesting checkpoint relaunch: sending SIGTERM to launcher "
            "process group pgid=%s exit_code=%s",
            pgid,
            exit_code,
        )
        os.killpg(pgid, signal_module.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError) as exc:
        logger.error("[elastic] Failed to signal launcher process group: %s", exc)
    time.sleep(1.0)
    os._exit(exit_code)


def _read_pause_signal_info(pause_file: str) -> dict:
    """Read pause metadata while tolerating legacy flag-only files."""
    try:
        with open(pause_file, "r") as f:
            pause_info = json.loads(f.read() or "{}")
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return pause_info if isinstance(pause_info, dict) else {}


def _validate_recovery_descriptor(rebuild_info: dict, world_size: int) -> dict:
    """Fail closed if a recovery epoch is not backed by one state contract."""
    descriptor = rebuild_info.get("descriptor_data")
    if not isinstance(descriptor, dict):
        descriptor_path = rebuild_info.get("descriptor")
        try:
            with open(descriptor_path, "r") as f:
                descriptor = json.load(f)
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"[elastic] recovery descriptor is unavailable: {descriptor_path}: {exc}"
            ) from exc

    errors = []
    expected_epoch = int(rebuild_info.get("recovery_epoch", -1))
    expected_rank = int(rebuild_info.get("killed_global_rank", -1))
    expected_iteration = int(rebuild_info.get("resume_iteration", -1))
    if int(descriptor.get("recovery_epoch", -1)) != expected_epoch:
        errors.append("recovery_epoch")
    if int(descriptor.get("killed_global_rank", -1)) != expected_rank:
        errors.append("killed_global_rank")
    if int(descriptor.get("resume_iteration", -1)) != expected_iteration:
        errors.append("resume_iteration")

    topology = descriptor.get("topology", {})
    if int(topology.get("world_size", -1)) != int(world_size):
        errors.append("world_size")
    if str(topology.get("rebuild_master_port")) != str(rebuild_info.get("new_master_port")):
        errors.append("rebuild_master_port")

    rank_table = descriptor.get("rank_table", [])
    try:
        logical_ranks = [
            int(entry.get("logical_rank"))
            for entry in rank_table
            if isinstance(entry, dict)
        ]
    except (TypeError, ValueError):
        logical_ranks = []
    replacements = [
        entry for entry in rank_table
        if isinstance(entry, dict) and entry.get("role") == "replacement"
    ]
    if len(rank_table) != world_size or sorted(logical_ranks) != list(range(world_size)):
        errors.append("rank_table")
    try:
        replacement_rank = (
            int(replacements[0].get("logical_rank", -1))
            if len(replacements) == 1
            else -1
        )
    except (TypeError, ValueError):
        replacement_rank = -1
    if len(replacements) != 1 or replacement_rank != expected_rank:
        errors.append("replacement_ownership")

    quiescence = descriptor.get("safe_point", {}).get("rank_quiescence", {})
    if quiescence.get("required") and quiescence.get("status") != "verified":
        errors.append("rank_quiescence")
    if quiescence.get("status") == "verified":
        if int(quiescence.get("resume_iteration", -1)) != expected_iteration:
            errors.append("quiescence_iteration")
        if int(quiescence.get("observed_survivors", -1)) != world_size - 1:
            errors.append("survivor_quorum")

    state_sources = descriptor.get("state_sources", {})
    for state_name in ("non_expert_parameters", "non_expert_optimizer"):
        if int(state_sources.get(state_name, {}).get("version_step", -1)) != expected_iteration:
            errors.append(f"{state_name}_version")
    checkpoint_step = int(descriptor.get("checkpoint_step", -1))
    for state_name in ("expert_parameters", "expert_optimizer"):
        if int(state_sources.get(state_name, {}).get("version_step", -1)) != checkpoint_step:
            errors.append(f"{state_name}_version")

    expected_digest = rebuild_info.get("descriptor_sha256")
    actual_digest = hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected_digest and actual_digest != expected_digest:
        errors.append("descriptor_sha256")

    if errors:
        raise RuntimeError(
            "[elastic] recovery descriptor validation failed: "
            + ",".join(sorted(set(errors)))
        )
    os.environ["ELASTIC_RECOVERY_DESCRIPTOR_SHA256"] = actual_digest
    return descriptor


def _kill_single_worker(target_local_rank: int):
    """Kill a single worker process by its local_rank.

    This is called AFTER all process groups have been destroyed, so it's
    safe — no NCCL operations are in flight.

    The elastic_launcher tracks worker PIDs.  We find the target worker's
    PID from the launcher's shared state and send SIGKILL to it.
    """
    import sys

    # Read the PID file written by elastic_launcher for this local_rank
    fault_dir = os.environ.get("ELASTIC_FAULT_DIR", "/tmp/elastic_faults")
    pid_file = os.path.join(fault_dir, f"worker_pid_{target_local_rank}")

    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                target_pid = int(f.read().strip())
            logger.warning("[elastic] Killing worker local_rank=%d pid=%d",
                           target_local_rank, target_pid)
            os.kill(target_pid, signal_module.SIGKILL)
            return
        except (ValueError, ProcessLookupError, PermissionError, OSError) as e:
            logger.warning("[elastic] Failed to kill via PID file: %s", e)

    # Fallback: if we ARE the target local_rank, kill ourselves
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == target_local_rank:
        logger.warning("[elastic] I am the target rank, killing self (pid=%d)", os.getpid())
        os.kill(os.getpid(), signal_module.SIGKILL)
    else:
        logger.error("[elastic] Cannot find PID for local_rank=%d, no PID file at %s",
                     target_local_rank, pid_file)


def _kill_all_local_workers():
    """Kill all worker processes on this node (for fault injection).

    We kill by sending SIGKILL to the current process group, which kills
    all workers forked by elastic_launcher.
    """
    import sys
    pid = os.getpid()
    ppid = os.getppid()
    logger.error("[elastic] Sending SIGKILL to process group (pid=%d, ppid=%d)", pid, ppid)
    # Kill the parent (elastic_launcher) process group, which kills all workers
    try:
        os.killpg(os.getpgid(ppid), signal_module.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    # Also kill self
    os.kill(pid, signal_module.SIGKILL)


def elastic_client_start():
    """Initialize and start the elastic client (call once after init_process_group).

    Only rank 0 on each node (local_rank=0) runs the heartbeat client.
    All ranks on the node share the pause/rebuild state via the global flags.
    """
    global _CLIENT

    if _launcher_control_enabled() and not is_rebuild_mode():
        _launcher_control_send("rank_state")
        logger.info(
            "[elastic] Watcher control is owned by the fault-isolated launcher "
            "(local_rank=%s)",
            os.environ.get("LOCAL_RANK", "?"),
        )
        return

    if is_rebuild_mode():
        logger.info(
            "[elastic] Replacement worker: watcher heartbeat disabled to avoid "
            "duplicate node_rank ownership"
        )
        return

    watcher_addr = os.environ.get("ELASTIC_WATCHER_ADDR")
    watcher_port = os.environ.get("ELASTIC_WATCHER_PORT")

    if not watcher_addr or not watcher_port:
        logger.info("[elastic] No ELASTIC_WATCHER_ADDR/PORT set, elastic client disabled")
        return

    # Only local_rank 0 connects to watcher
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank != 0:
        return

    node_rank = int(os.environ.get("NODE_RANK", "0"))
    _CLIENT = ElasticClient(watcher_addr, int(watcher_port), node_rank)
    _CLIENT.start()
    logger.info(f"[elastic] Client started on node {node_rank}")


def elastic_client_update_step(
    step: int, phase: str = "forward_backward", step_tag: Optional[int] = None
):
    """Update heartbeat step metadata.

    FlashRecovery-style recovery needs to distinguish failures during
    forward/backward from failures during optimizer step.  ``step_tag`` follows
    that convention: i during forward/backward, -1 while optimizer step is in
    flight, and i+1 after a committed optimizer step.
    """
    global _CURRENT_STEP, _CURRENT_STEP_TAG, _CURRENT_TRAIN_PHASE

    _CURRENT_STEP = int(step)
    _CURRENT_STEP_TAG = int(step if step_tag is None else step_tag)
    _CURRENT_TRAIN_PHASE = phase
    if _CLIENT is not None:
        _CLIENT.update_step(step, phase=phase, step_tag=step_tag)
    _launcher_control_send("rank_state")


def elastic_mark_post_rebuild_pending(iteration: Optional[int] = None):
    """Mark that the next training iteration must align all rebuilt ranks.

    The first post-rebuild forward is where Megatron lazily creates several
    NCCL communicators.  Keep the alignment on the watcher/TCP control plane so
    replacement-only setup cannot race survivor ranks into MoE/P2P collectives.
    """
    for key in _POST_REBUILD_STATE_ENV:
        os.environ.pop(key, None)
    os.environ["ELASTIC_RECOVERY_STATE"] = "post_rebuild_pending"
    os.environ["ELASTIC_POST_REBUILD_PENDING"] = "1"
    if iteration is not None and iteration >= 0:
        os.environ["ELASTIC_RESUME_ITERATION"] = str(iteration)


def elastic_post_rebuild_iteration_barrier(iteration: int) -> bool:
    """Align the first recovered step or its stabilization successor."""
    if os.environ.get("ELASTIC_POST_REBUILD_PENDING") != "1":
        return False
    recovery_state = os.environ.get("ELASTIC_RECOVERY_STATE")
    if recovery_state not in (
        "post_rebuild_pending",
        "post_rebuild_stabilization_pending",
    ):
        logger.warning(
            "[elastic] Ignoring stale post-rebuild pending flag without recovery state "
            "(state=%s)",
            os.environ.get("ELASTIC_RECOVERY_STATE"),
        )
        elastic_clear_post_rebuild_trace()
        return False
    if not dist.is_available() or not dist.is_initialized():
        return False

    os.environ["ELASTIC_RESUME_ITERATION"] = str(iteration)
    world_size = dist.get_world_size()
    timeout = _elastic_phase_timeout_seconds()
    stabilization = recovery_state == "post_rebuild_stabilization_pending"
    phase = (
        "post_rebuild_stabilization_ready"
        if stabilization
        else "post_rebuild_iteration_ready"
    )
    elastic_report_recovery_phase(phase, step=iteration)
    if not elastic_wait_for_recovery_phase_count(phase, world_size, timeout):
        raise RuntimeError(
            f"[elastic] Not all {world_size} ranks reached {phase} "
            f"before first post-rebuild train step within {timeout}s"
        )
    _elastic_validate_rerun_state_machine(iteration, stabilization=stabilization)

    os.environ["ELASTIC_POST_REBUILD_PENDING"] = "0"
    os.environ["ELASTIC_POST_REBUILD_TRACE_ACTIVE"] = "1"
    os.environ["ELASTIC_RECOVERY_STATE"] = (
        "post_rebuild_stabilization_trace" if stabilization else "post_rebuild_trace"
    )
    os.environ["ELASTIC_POST_REBUILD_TRACE_ITERATION"] = str(iteration)
    trace_token = _elastic_post_rebuild_token(iteration)
    if stabilization:
        trace_token = f"{trace_token}:stabilization"
    os.environ["ELASTIC_POST_REBUILD_TRACE_TOKEN"] = trace_token
    os.environ["ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER_TOKEN"] = trace_token
    logger.warning(
        "[elastic] Rank %d: all ranks aligned before %s post-rebuild train step "
        "(iteration=%d token=%s)",
        dist.get_rank(),
        "stabilization" if stabilization else "first",
        iteration,
        trace_token,
    )
    return True


def _elastic_replacement_rank_from_env() -> int:
    try:
        return int(os.environ.get("ELASTIC_REPLACEMENT_RANK", "-1"))
    except ValueError:
        return -1


def _elastic_group_ranks(group):
    try:
        return tuple(dist.get_process_group_ranks(group))
    except Exception:
        return None


def _elastic_replacement_warmup_groups(replacement_rank: int):
    """Return rebuilt first-step communicators that include the replacement rank."""
    if replacement_rank < 0:
        return []

    from megatron.core import parallel_state as mpu

    candidates = (
        ("tp_ep", "collective", lambda: mpu.get_expert_tensor_and_model_parallel_group(
            check_initialized=False
        )),
        ("ep", "collective", lambda: mpu.get_expert_model_parallel_group(check_initialized=False)),
        ("pipeline", "p2p", lambda: mpu.get_pipeline_model_parallel_group(
            check_initialized=False
        )),
        ("dp", "collective", lambda: mpu.get_data_parallel_group(with_context_parallel=False)),
        ("dp_cp", "collective", lambda: mpu.get_data_parallel_group(with_context_parallel=True)),
        (
            "tensor_data",
            "collective",
            lambda: mpu.get_tensor_and_data_parallel_group(check_initialized=False),
        ),
        (
            "tensor_data_cp",
            "collective",
            lambda: mpu.get_tensor_and_data_parallel_group(
                check_initialized=False, with_context_parallel=True
            ),
        ),
        ("embedding", "collective", lambda: mpu.get_embedding_group(check_initialized=False)),
        (
            "position_embedding",
            "collective",
            lambda: mpu.get_position_embedding_group(check_initialized=False),
        ),
    )
    selected_env = os.environ.get("ELASTIC_POST_REBUILD_WARMUP_GROUPS")
    selected = None
    if selected_env:
        selected = {name.strip() for name in selected_env.split(",") if name.strip()}
    rank = dist.get_rank()
    groups = []
    seen = set()
    for name, kind, getter in candidates:
        if selected is not None and name not in selected:
            continue
        try:
            group = getter()
        except Exception as exc:
            logger.debug("[elastic] post-rebuild communicator warmup: %s unavailable: %s", name, exc)
            continue
        if group is None:
            continue
        ranks = _elastic_group_ranks(group)
        if ranks is None or len(ranks) <= 1:
            continue
        if rank not in ranks or replacement_rank not in ranks:
            continue
        key = id(group)
        if key in seen:
            continue
        seen.add(key)
        groups.append((name, kind, ranks, group))
    return groups


def elastic_warmup_post_rebuild_communicators(iteration: int) -> bool:
    """Optionally warm replacement-facing communicators before the first real forward.

    Keep this opt-in.  Megatron normally creates communicators lazily in the
    exact order used by forward/backward.  Probing replacement-facing groups at
    recovery time can introduce a new order that does not match all ranks
    (for example pipeline P2P vs embedding/model collectives), so the default
    is to rely on the aligned first train step to initialize them naturally.
    """
    if not elastic_is_post_rebuild_trace_active(iteration):
        return False
    if os.environ.get("ELASTIC_POST_REBUILD_COMM_WARMUP_DONE") == "1":
        return False
    if os.environ.get("ELASTIC_POST_REBUILD_COMM_WARMUP", "0") == "0":
        os.environ["ELASTIC_POST_REBUILD_COMM_WARMUP_DONE"] = "1"
        if dist.is_available() and dist.is_initialized():
            logger.warning(
                "[elastic] Rank %d: skipping post-rebuild communicator warmup; "
                "using Megatron's lazy first-step communicator order",
                dist.get_rank(),
            )
        return False
    if not dist.is_available() or not dist.is_initialized() or not torch.cuda.is_available():
        return False

    replacement_rank = _elastic_replacement_rank_from_env()
    world_size = dist.get_world_size()
    timeout = _elastic_phase_timeout_seconds()
    group_timeout = float(
        os.environ.get(
            "ELASTIC_POST_REBUILD_COMM_WARMUP_TIMEOUT",
            os.environ.get("ELASTIC_REBUILD_WARMUP_GROUP_TIMEOUT", str(min(timeout, 60.0))),
        )
    )
    start_phase = "post_rebuild_comm_warmup_start"
    done_phase = "post_rebuild_comm_warmup_done"

    elastic_report_recovery_phase(start_phase, step=iteration, replacement_rank=replacement_rank)
    if not elastic_wait_for_recovery_phase_count(start_phase, world_size, timeout):
        raise RuntimeError(
            f"[elastic] Not all {world_size} ranks reached {start_phase} "
            f"within {timeout}s"
        )

    groups = _elastic_replacement_warmup_groups(replacement_rank)
    rank = dist.get_rank()
    if groups:
        warmup = torch.ones(1, device=torch.cuda.current_device())
        for name, kind, ranks, group in groups:
            logger.warning(
                "[elastic] Rank %d: warming post-rebuild communicator %s ranks=%s",
                rank,
                name,
                list(ranks),
            )
            if kind == "p2p":
                _elastic_warmup_pipeline_p2p(group, ranks, group_timeout)
            else:
                work = dist.all_reduce(warmup, group=group, async_op=True)
                _elastic_wait_distributed_works(
                    [work],
                    group_timeout,
                    f"post-rebuild-comm-warmup rank={rank} group={name} ranks={list(ranks)}",
                )
            logger.warning(
                "[elastic] Rank %d: warmed post-rebuild communicator %s",
                rank,
                name,
            )
        torch.cuda.synchronize()
    else:
        logger.info(
            "[elastic] Rank %d: no replacement-facing communicator to warm "
            "(replacement=%d)",
            rank,
            replacement_rank,
        )

    elastic_report_recovery_phase(done_phase, step=iteration, replacement_rank=replacement_rank)
    if not elastic_wait_for_recovery_phase_count(done_phase, world_size, timeout):
        raise RuntimeError(
            f"[elastic] Not all {world_size} ranks reached {done_phase} within {timeout}s"
        )
    os.environ["ELASTIC_POST_REBUILD_COMM_WARMUP_DONE"] = "1"
    return bool(groups)


def elastic_trace_post_rebuild_phase(
    phase: str, iteration: Optional[int] = None, optimizer=None
):
    """Report a diagnostic phase for the first post-rebuild train step."""
    if not elastic_is_post_rebuild_trace_active(iteration):
        return
    if phase == "optimizer_step_start" and optimizer is not None:
        elastic_validate_optimizer_process_groups(optimizer, iteration)
    extra = {}
    if iteration is not None:
        extra["step"] = iteration
    elastic_report_recovery_phase(phase, **extra)


def elastic_clear_post_rebuild_trace():
    for key in _POST_REBUILD_STATE_ENV:
        os.environ.pop(key, None)


def _send_one_shot_to_watcher(msg: dict) -> bool:
    watcher_addr = os.environ.get("ELASTIC_WATCHER_ADDR")
    watcher_port = os.environ.get("ELASTIC_WATCHER_PORT")
    if not watcher_addr or not watcher_port:
        return False

    last_error = None
    for attempt in range(10):
        try:
            with socket.create_connection((watcher_addr, int(watcher_port)), timeout=5.0) as sock:
                sock.sendall((json.dumps(msg) + "\n").encode())
            return True
        except (OSError, ValueError) as e:
            last_error = e
            if attempt < 9:
                time.sleep(0.2)
    logger.warning("[elastic] Failed to send one-shot watcher event: %s", last_error)
    return False


def _elastic_effective_recovery_phase(phase: str) -> str:
    if (
        os.environ.get("ELASTIC_RECOVERY_STATE") == "post_rebuild_stabilization_trace"
        and phase in _POST_REBUILD_STABILIZATION_PHASES
    ):
        return f"stabilization_{phase}"
    return phase


def elastic_report_recovery_phase(phase: str, **extra):
    """Report a rebuild/replacement milestone to the watcher for diagnostics."""
    if not os.environ.get("ELASTIC_WATCHER_ADDR"):
        return

    phase = _elastic_effective_recovery_phase(phase)

    rank = int(os.environ.get("RANK", "-1"))
    node_rank = int(os.environ.get("NODE_RANK", "-1"))
    step = int(os.environ.get("ELASTIC_RESUME_ITERATION", "-1"))
    role = "replacement" if is_rebuild_mode() else "survivor"
    msg = {
        "type": "recovery_phase",
        "node_rank": node_rank,
        "rank": rank,
        "role": role,
        "phase": phase,
        "step": step,
    }
    msg.update(_elastic_recovery_epoch_payload())
    msg.update(extra)

    if _CLIENT is not None and _CLIENT.sock is not None:
        _CLIENT._send(msg)
        if _CLIENT.sock is None:
            _send_one_shot_to_watcher(msg)
    else:
        _send_one_shot_to_watcher(msg)


def elastic_wait_for_recovery_phase(role: str, rank: int, phase: str, timeout: float = 300.0) -> bool:
    watcher_addr = os.environ.get("ELASTIC_WATCHER_ADDR")
    watcher_port = os.environ.get("ELASTIC_WATCHER_PORT")
    if not watcher_addr or not watcher_port:
        return False

    msg = {
        "type": "wait_phase",
        "node_rank": int(os.environ.get("NODE_RANK", "-1")),
        "role": role,
        "rank": rank,
        "phase": phase,
        "timeout": timeout,
    }
    msg.update(_elastic_recovery_epoch_payload())
    deadline = time.time() + timeout
    last_error = None
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        remaining = max(1.0, deadline - time.time())
        msg["timeout"] = remaining
        try:
            with socket.create_connection(
                (watcher_addr, int(watcher_port)),
                timeout=min(10.0, remaining),
            ) as sock:
                sock.settimeout(remaining + 5.0)
                sock.sendall((json.dumps(msg) + "\n").encode())
                data = b""
                while b"\n" not in data:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                if not data:
                    last_error = RuntimeError("empty watcher response")
                    time.sleep(min(1.0, max(0.0, deadline - time.time())))
                    continue
                response = json.loads(data.split(b"\n", 1)[0].decode())
                return bool(response.get("ok"))
        except (OSError, ValueError, json.JSONDecodeError) as e:
            last_error = e
            if time.time() < deadline:
                logger.warning(
                    "[elastic] wait_phase retry %d failed role=%s rank=%s phase=%s: %s",
                    attempt,
                    role,
                    rank,
                    phase,
                    e,
                )
                time.sleep(min(1.0, max(0.0, deadline - time.time())))
    logger.warning(
        "[elastic] Failed waiting for recovery phase role=%s rank=%s phase=%s: %s",
        role,
        rank,
        phase,
        last_error,
    )
    return False


def elastic_wait_for_recovery_phase_count(
    phase: str, min_count: int, timeout: float = 300.0
) -> bool:
    watcher_addr = os.environ.get("ELASTIC_WATCHER_ADDR")
    watcher_port = os.environ.get("ELASTIC_WATCHER_PORT")
    if not watcher_addr or not watcher_port:
        return False

    msg = {
        "type": "wait_phase_count",
        "node_rank": int(os.environ.get("NODE_RANK", "-1")),
        "phase": phase,
        "min_count": min_count,
        "timeout": timeout,
    }
    msg.update(_elastic_recovery_epoch_payload())
    deadline = time.time() + timeout
    last_error = None
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        remaining = max(1.0, deadline - time.time())
        msg["timeout"] = remaining
        try:
            with socket.create_connection(
                (watcher_addr, int(watcher_port)),
                timeout=min(10.0, remaining),
            ) as sock:
                sock.settimeout(remaining + 5.0)
                sock.sendall((json.dumps(msg) + "\n").encode())
                data = b""
                while b"\n" not in data:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                if not data:
                    last_error = RuntimeError("empty watcher response")
                    time.sleep(min(1.0, max(0.0, deadline - time.time())))
                    continue
                response = json.loads(data.split(b"\n", 1)[0].decode())
                ok = bool(response.get("ok"))
                if not ok:
                    logger.warning(
                        "[elastic] phase-count wait failed: phase=%s count=%s min_count=%s "
                        "missing=%s pending=%s unreported=%s",
                        phase,
                        response.get("count"),
                        response.get("min_count"),
                        response.get("missing"),
                        response.get("pending"),
                        response.get("unreported"),
                    )
                return ok
        except (OSError, ValueError, json.JSONDecodeError) as e:
            last_error = e
            if time.time() < deadline:
                logger.warning(
                    "[elastic] wait_phase_count retry %d failed phase=%s min_count=%s: %s",
                    attempt,
                    phase,
                    min_count,
                    e,
                )
                time.sleep(min(1.0, max(0.0, deadline - time.time())))
    logger.warning(
        "[elastic] Failed waiting for recovery phase count phase=%s min_count=%s: %s",
        phase,
        min_count,
        last_error,
    )
    return False


def elastic_commit_post_rebuild_iteration(iteration: int) -> bool:
    """Advance or commit recovery at a true train-loop boundary.

    A phase report is asynchronous, so reporting ``step_complete`` and then
    immediately clearing the recovery state lets fast ranks enter the next
    iteration before slow ranks have left Megatron's post-step callbacks.  Use
    recovery-epoch barriers to make the commit atomic from the training ranks'
    point of view.  This function is a no-op outside the
    explicit post-rebuild trace window.
    """
    if not elastic_is_post_rebuild_trace_active(iteration):
        return False
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "[elastic] Cannot commit post-rebuild iteration without an initialized "
            "process group"
        )

    recovery_state = os.environ.get("ELASTIC_RECOVERY_STATE")
    world_size = dist.get_world_size()
    timeout = _elastic_phase_timeout_seconds()
    if recovery_state == "post_rebuild_trace":
        completed_phase = "post_rebuild_step_complete"
        stabilization_phase = "post_rebuild_stabilization_pending"

        elastic_report_recovery_phase(completed_phase, step=iteration)
        if not elastic_wait_for_recovery_phase_count(completed_phase, world_size, timeout):
            raise RuntimeError(
                f"[elastic] Not all {world_size} ranks completed the first post-rebuild "
                f"iteration within {timeout}s"
            )

        elastic_report_recovery_phase(stabilization_phase, step=iteration)
        if not elastic_wait_for_recovery_phase_count(
            stabilization_phase, world_size, timeout
        ):
            raise RuntimeError(
                f"[elastic] Not all {world_size} ranks entered post-rebuild "
                f"stabilization within {timeout}s"
            )

        next_iteration = iteration + 1
        os.environ["ELASTIC_RECOVERY_STATE"] = "post_rebuild_stabilization_pending"
        os.environ["ELASTIC_POST_REBUILD_PENDING"] = "1"
        os.environ["ELASTIC_POST_REBUILD_TRACE_ACTIVE"] = "0"
        os.environ["ELASTIC_POST_REBUILD_TRACE_ITERATION"] = str(next_iteration)
        os.environ["ELASTIC_RESUME_ITERATION"] = str(next_iteration)
        os.environ.pop("ELASTIC_POST_REBUILD_TRACE_TOKEN", None)
        os.environ.pop("ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER_TOKEN", None)
        logger.warning(
            "[elastic] Rank %d: first recovered iteration %d completed; "
            "stabilization iteration %d required before recovery commit",
            dist.get_rank(),
            iteration,
            next_iteration,
        )
        return True

    if recovery_state != "post_rebuild_stabilization_trace":
        raise RuntimeError(
            f"[elastic] Cannot commit unexpected recovery state {recovery_state!r}"
        )

    completed_phase = "post_rebuild_stabilization_complete"
    commit_phase = "post_rebuild_commit_ready"

    elastic_report_recovery_phase(completed_phase, step=iteration)
    if not elastic_wait_for_recovery_phase_count(completed_phase, world_size, timeout):
        raise RuntimeError(
            f"[elastic] Not all {world_size} ranks completed the post-rebuild "
            f"stabilization iteration within {timeout}s"
        )

    elastic_report_recovery_phase(commit_phase, step=iteration)
    if not elastic_wait_for_recovery_phase_count(commit_phase, world_size, timeout):
        raise RuntimeError(
            f"[elastic] Not all {world_size} ranks acknowledged the post-rebuild "
            f"commit within {timeout}s"
        )

    logger.warning(
        "[elastic] Rank %d: recovery epoch committed after stabilization iteration %d",
        dist.get_rank(),
        iteration,
    )
    elastic_clear_post_rebuild_trace()
    return True


def elastic_wait_for_ordinal_barrier(
    barrier_id: str,
    rank: int,
    min_count: int,
    timeout: float = 300.0,
    **extra,
) -> bool:
    watcher_addr = os.environ.get("ELASTIC_WATCHER_ADDR")
    watcher_port = os.environ.get("ELASTIC_WATCHER_PORT")
    if not watcher_addr or not watcher_port:
        return False

    msg = {
        "type": "ordinal_barrier",
        "node_rank": int(os.environ.get("NODE_RANK", "-1")),
        "barrier_id": barrier_id,
        "rank": int(rank),
        "min_count": int(min_count),
        "timeout": float(timeout),
    }
    msg.update(_elastic_recovery_epoch_payload())
    msg.update(extra)
    deadline = time.time() + timeout
    last_error = None
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        remaining = max(1.0, deadline - time.time())
        msg["timeout"] = remaining
        try:
            with socket.create_connection(
                (watcher_addr, int(watcher_port)),
                timeout=min(10.0, remaining),
            ) as sock:
                sock.settimeout(remaining + 5.0)
                sock.sendall((json.dumps(msg) + "\n").encode())
                data = b""
                while b"\n" not in data:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                if not data:
                    last_error = RuntimeError("empty watcher response")
                    time.sleep(min(1.0, max(0.0, deadline - time.time())))
                    continue
                response = json.loads(data.split(b"\n", 1)[0].decode())
                ok = bool(response.get("ok"))
                if not ok:
                    logger.warning(
                        "[elastic] ordinal barrier failed: id=%s count=%s "
                        "min_count=%s missing=%s arrived=%s",
                        barrier_id,
                        response.get("count"),
                        response.get("min_count"),
                        response.get("missing"),
                        response.get("arrived"),
                    )
                return ok
        except (OSError, ValueError, json.JSONDecodeError) as e:
            last_error = e
            if time.time() < deadline:
                logger.warning(
                    "[elastic] ordinal barrier retry %d failed id=%s: %s",
                    attempt,
                    barrier_id,
                    e,
                )
                time.sleep(min(1.0, max(0.0, deadline - time.time())))
    logger.warning("[elastic] Failed waiting for ordinal barrier id=%s: %s", barrier_id, last_error)
    return False


def _elastic_wait_for_peer_sync_endpoint(peer_id: str, timeout: float = 300.0) -> Optional[dict]:
    watcher_addr = os.environ.get("ELASTIC_WATCHER_ADDR")
    watcher_port = os.environ.get("ELASTIC_WATCHER_PORT")
    if not watcher_addr or not watcher_port:
        return None

    msg = {
        "type": "wait_peer_sync_endpoint",
        "node_rank": int(os.environ.get("NODE_RANK", "-1")),
        "peer_id": peer_id,
        "timeout": timeout,
    }
    msg.update(_elastic_recovery_epoch_payload())
    try:
        with socket.create_connection((watcher_addr, int(watcher_port)), timeout=5.0) as sock:
            sock.settimeout(timeout + 5.0)
            sock.sendall((json.dumps(msg) + "\n").encode())
            data = b""
            while b"\n" not in data:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
            if not data:
                return None
            response = json.loads(data.split(b"\n", 1)[0].decode())
            if not response.get("ok"):
                return None
            return response.get("endpoint")
    except (OSError, ValueError, json.JSONDecodeError) as e:
        logger.warning("[elastic] Failed waiting for peer sync endpoint %s: %s", peer_id, e)
        return None


def _wait_for_replacement_phase_before_global_barrier(
    replacement_rank: int, phase: str, timeout: float
):
    if is_rebuild_mode() or replacement_rank < 0:
        return

    logger.warning(
        "[elastic] Rank %d: waiting for replacement rank %d %s before global barrier",
        dist.get_rank() if dist.is_initialized() else -1,
        replacement_rank,
        phase,
    )
    if not elastic_wait_for_recovery_phase("replacement", replacement_rank, phase, timeout):
        raise RuntimeError(
            f"[elastic] Replacement rank {replacement_rank} did not reach "
            f"{phase} within {timeout}s before global barrier"
        )


def _elastic_barrier(label: str):
    """Run a default-group barrier with explicit CUDA device and useful logs."""
    rank = dist.get_rank() if dist.is_initialized() else -1
    device_ids = None
    if torch.cuda.is_available():
        device_ids = [torch.cuda.current_device()]
    logger.warning("[elastic] Rank %d: entering %s barrier", rank, label)
    if device_ids is None:
        dist.barrier()
    else:
        dist.barrier(device_ids=device_ids)
    logger.warning("[elastic] Rank %d: exited %s barrier", rank, label)


def _elastic_rebuild_final_barrier():
    """Avoid a default NCCL barrier on the hot path after replacement sync.

    The replacement and source ranks have already completed the required data
    transfer, and survivor ranks gate on the replacement reaching
    ``param_sync_done`` through the watcher.  Running a fresh full-world NCCL
    barrier immediately after communicator rebuild has repeatedly been the next
    hang point, so keep it opt-in for debugging only.
    """
    rank = dist.get_rank() if dist.is_initialized() else -1
    if os.environ.get("ELASTIC_REBUILD_FINAL_NCCL_BARRIER", "0") == "1":
        _elastic_barrier("rebuild-final")
        return
    logger.warning(
        "[elastic] Rank %d: skipping rebuild-final NCCL barrier "
        "(phase-gated by watcher param_sync_done)",
        rank,
    )


def _elastic_iter_groups(groups):
    if isinstance(groups, (list, tuple)):
        for group in groups:
            yield group
    else:
        yield groups


def _elastic_wait_distributed_works(works, timeout: float, label: str):
    """Wait for asynchronous distributed works without blocking past timeout."""
    works = list(works)
    if not works:
        return

    pending = works
    deadline = time.monotonic() + max(float(timeout), 0.0)
    while pending:
        next_pending = []
        for work in pending:
            try:
                if work.is_completed():
                    continue
            except Exception:
                wait_timeout = max(deadline - time.monotonic(), 0.0)
                try:
                    wait_result = work.wait(timeout=timedelta(seconds=wait_timeout))
                except TypeError:
                    wait_result = work.wait()
                if wait_result is False:
                    next_pending.append(work)
                continue
            next_pending.append(work)

        if not next_pending:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"[elastic] distributed work timed out: {label}")
        time.sleep(0.05)
        pending = next_pending

    for work in works:
        work.wait()


def _elastic_warmup_pipeline_p2p(group, ranks, timeout: float):
    """Warm the rebuilt pipeline communicator with Megatron-style P2P ops."""
    rank = dist.get_rank()
    group_size = dist.get_world_size(group=group)
    if group_size <= 1:
        return

    try:
        group_rank = dist.get_rank(group=group)
    except Exception:
        group_rank = list(ranks).index(rank)

    device = torch.cuda.current_device()
    dtype = torch.float32

    def global_rank(group_index: int) -> int:
        try:
            return dist.get_global_rank(group, group_index)
        except Exception:
            return list(ranks)[group_index]

    def run_direction(direction: str, ops, peers):
        if not ops:
            logger.info(
                "[elastic] Rank %d: no-op pipeline P2P warmup direction=%s",
                rank,
                direction,
            )
            return
        logger.info(
            "[elastic] Rank %d: warming pipeline P2P direction=%s group_rank=%d peers=%s ranks=%s",
            rank,
            direction,
            group_rank,
            peers,
            list(ranks),
        )
        reqs = dist.batch_isend_irecv(ops)
        _elastic_wait_distributed_works(
            reqs,
            timeout,
            f"pipeline-p2p-{direction} rank={rank} group_rank={group_rank} ranks={list(ranks)}",
        )
        logger.info(
            "[elastic] Rank %d: warmed pipeline P2P direction=%s group_rank=%d",
            rank,
            direction,
            group_rank,
        )

    # Match the real training directions instead of using a full pipeline
    # collective.  The first train step will receive/send along the PP chain,
    # not all-reduce the model group.  Keeping this warmup semantically close
    # to the real P2P path avoids creating another recovery-only NCCL ordering.
    forward_ops = []
    forward_peers = []
    if group_rank > 0:
        prev_rank = global_rank(group_rank - 1)
        recv_prev = torch.empty(1, device=device, dtype=dtype)
        forward_ops.append(dist.P2POp(dist.irecv, recv_prev, prev_rank, group))
        forward_peers.append(("recv_prev", prev_rank))
    if group_rank < group_size - 1:
        next_rank = global_rank(group_rank + 1)
        send_next = torch.ones(1, device=device, dtype=dtype)
        forward_ops.append(dist.P2POp(dist.isend, send_next, next_rank, group))
        forward_peers.append(("send_next", next_rank))
    run_direction("forward", forward_ops, forward_peers)

    backward_ops = []
    backward_peers = []
    if group_rank > 0:
        prev_rank = global_rank(group_rank - 1)
        send_prev = torch.ones(1, device=device, dtype=dtype)
        backward_ops.append(dist.P2POp(dist.isend, send_prev, prev_rank, group))
        backward_peers.append(("send_prev", prev_rank))
    if group_rank < group_size - 1:
        next_rank = global_rank(group_rank + 1)
        recv_next = torch.empty(1, device=device, dtype=dtype)
        backward_ops.append(dist.P2POp(dist.irecv, recv_next, next_rank, group))
        backward_peers.append(("recv_next", next_rank))
    run_direction("backward", backward_ops, backward_peers)


def _elastic_warmup_rebuild_communicators(replacement_rank: int = -1, timeout: Optional[float] = None):
    """Eagerly initialize replacement-facing NCCL communicators before resume.

    Survivor-only communicators were already used before the rebuild.  Warming
    them again can interleave unrelated overlapping NCCL groups in different
    rank-local orders.  Restrict warmup to groups containing the replacement
    rank, and order those groups with a global rank-list key so every overlap
    observes the same collective order.
    """
    if not dist.is_initialized() or not torch.cuda.is_available():
        return

    from megatron.core import parallel_state as mpu

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if replacement_rank < 0:
        replacement_rank = int(
            os.environ.get("ELASTIC_REPLACEMENT_RANK", os.environ.get("RANK", "-1"))
        )
    if timeout is None:
        timeout = _elastic_phase_timeout_seconds()
    warm_full_groups = os.environ.get("ELASTIC_REBUILD_WARMUP_FULL_GROUPS", "0") == "1"
    warm_data_groups = os.environ.get("ELASTIC_REBUILD_WARMUP_DATA_GROUPS", "0") == "1"
    allow_collective_warmup = (
        os.environ.get("ELASTIC_REBUILD_ALLOW_COLLECTIVE_WARMUP", "0") == "1"
    )
    selected_group_names_env = os.environ.get("ELASTIC_REBUILD_WARMUP_GROUPS")
    if selected_group_names_env:
        if selected_group_names_env.strip().lower() in ("0", "none", "off", "false"):
            selected_group_names = set()
        else:
            selected_group_names = {
                name.strip() for name in selected_group_names_env.split(",") if name.strip()
            }
    else:
        # Keep recovery-stage NCCL warmup opt-in.  Even narrow expert-group
        # collectives can deadlock here because rebuilt ranks have not yet
        # re-entered Megatron's normal forward-order communicator creation path.
        selected_group_names = set()
    group_timeout = float(
        os.environ.get(
            "ELASTIC_REBUILD_WARMUP_GROUP_TIMEOUT",
            str(min(timeout, 60.0)),
        )
    )
    skipped_groups = []
    candidates = []

    def add_group(name, getter):
        try:
            groups = getter()
        except Exception:
            return
        for group in _elastic_iter_groups(groups):
            if group is None:
                continue
            try:
                ranks = tuple(dist.get_process_group_ranks(group))
            except Exception:
                continue
            if rank not in ranks or len(ranks) <= 1:
                continue
            if replacement_rank >= 0 and replacement_rank not in ranks:
                continue
            if name not in selected_group_names:
                skipped_groups.append((ranks, name, "not-selected"))
                continue
            if name != "pipeline" and not allow_collective_warmup:
                skipped_groups.append((ranks, name, "collective-disabled"))
                continue
            if not warm_data_groups and "data" in name:
                skipped_groups.append((ranks, name, "data"))
                continue
            if not warm_full_groups and (len(ranks) >= world_size or name == "expert_tensor_model_pipeline"):
                skipped_groups.append((ranks, name, "full-or-metadata"))
                continue
            candidates.append((ranks, name, group))

    add_group("model", lambda: mpu.get_model_parallel_group(check_initialized=False))
    add_group("tensor", lambda: mpu.get_tensor_model_parallel_group(check_initialized=False))
    add_group("pipeline", lambda: mpu.get_pipeline_model_parallel_group(check_initialized=False))
    add_group("data", lambda: mpu.get_data_parallel_group())
    add_group("data_cp", lambda: mpu.get_data_parallel_group(with_context_parallel=True))
    add_group(
        "tensor_data",
        lambda: mpu.get_tensor_and_data_parallel_group(check_initialized=False),
    )
    add_group(
        "tensor_data_cp",
        lambda: mpu.get_tensor_and_data_parallel_group(
            check_initialized=False, with_context_parallel=True
        ),
    )
    add_group(
        "tensor_context",
        lambda: mpu.get_tensor_and_context_parallel_group(check_initialized=False),
    )
    add_group("embedding", lambda: mpu.get_embedding_group(check_initialized=False))
    add_group("position_embedding", lambda: mpu.get_position_embedding_group(check_initialized=False))
    add_group("expert", lambda: mpu.get_expert_model_parallel_group(check_initialized=False))
    add_group("expert_tensor", lambda: mpu.get_expert_tensor_parallel_group(check_initialized=False))
    add_group(
        "expert_tensor_model",
        lambda: mpu.get_expert_tensor_and_model_parallel_group(check_initialized=False),
    )
    add_group(
        "expert_tensor_model_pipeline",
        lambda: mpu.get_expert_tensor_model_pipeline_parallel_group(check_initialized=False),
    )
    add_group("expert_data", lambda: mpu.get_expert_data_parallel_group(check_initialized=False))

    seen = set()
    ordered = []
    for ranks, name, group in sorted(candidates, key=lambda item: (-len(item[0]), item[0], item[1])):
        key = id(group)
        if key in seen:
            continue
        seen.add(key)
        ordered.append((ranks, name, group))

    device = torch.cuda.current_device()
    warmup = torch.ones(1, device=device)
    logger.warning(
        "[elastic] Rank %d: warming %d replacement-facing rebuild communicators "
        "before train_ready (replacement=%d)",
        rank,
        len(ordered),
        replacement_rank,
    )
    if skipped_groups:
        logger.info(
            "[elastic] Rank %d: skipped %d rebuild warmup groups: %s",
            rank,
            len(skipped_groups),
            [(name, reason, list(ranks)) for ranks, name, reason in skipped_groups],
        )
    elastic_report_recovery_phase("comm_warmup_start")
    if not elastic_wait_for_recovery_phase_count("comm_warmup_start", world_size, timeout):
        raise RuntimeError(
            "[elastic] not all ranks reached comm_warmup_start before communicator warmup"
        )
    for ranks, name, group in ordered:
        logger.info(
            "[elastic] Rank %d: warming communicator %s ranks=%s",
            rank,
            name,
            list(ranks),
        )
        if name == "pipeline" and os.environ.get("ELASTIC_REBUILD_PIPELINE_P2P_WARMUP", "1") != "0":
            _elastic_warmup_pipeline_p2p(group, ranks, group_timeout)
        elif allow_collective_warmup:
            work = dist.all_reduce(warmup, group=group, async_op=True)
            _elastic_wait_distributed_works(
                [work],
                group_timeout,
                f"communicator warmup rank={rank} group={name} ranks={list(ranks)}",
            )
        else:
            logger.info(
                "[elastic] Rank %d: skipping communicator %s collective warmup ranks=%s",
                rank,
                name,
                list(ranks),
            )
            continue
        logger.info("[elastic] Rank %d: warmed communicator %s", rank, name)
    torch.cuda.synchronize()
    elastic_report_recovery_phase("comm_warmup_done")
    logger.warning("[elastic] Rank %d: rebuild communicator warmup complete", rank)


def _elastic_report_and_wait_train_ready(timeout: float):
    """Use watcher/TCP as the post-rebuild full-rank readiness barrier."""
    rank = dist.get_rank() if dist.is_initialized() else -1
    world_size = dist.get_world_size() if dist.is_initialized() else int(
        os.environ.get("WORLD_SIZE", "1")
    )
    elastic_report_recovery_phase("train_ready")
    logger.warning(
        "[elastic] Rank %d: waiting for %d ranks to reach train_ready via watcher",
        rank,
        world_size,
    )
    if not elastic_wait_for_recovery_phase_count("train_ready", world_size, timeout):
        raise RuntimeError(
            f"[elastic] Not all {world_size} ranks reached train_ready within {timeout}s"
        )
    logger.warning("[elastic] Rank %d: all ranks reached train_ready", rank)


def _elastic_rebuild_timeout(args):
    timeout_minutes = int(
        os.environ.get(
            "ELASTIC_REBUILD_TIMEOUT_MINUTES",
            str(max(30, int(getattr(args, "distributed_timeout_minutes", 10)))),
        )
    )
    return timedelta(minutes=timeout_minutes)


def _elastic_rebuild_timeout_minutes(args):
    return int(_elastic_rebuild_timeout(args).total_seconds() // 60)


def _elastic_rebuild_subgroup_timeout_minutes(args):
    timeout_minutes = getattr(args, "distributed_timeout_minutes", None)
    if timeout_minutes is None:
        timeout_minutes = _elastic_rebuild_timeout_minutes(args)
    return int(timeout_minutes)


def _elastic_phase_timeout_seconds(args=None, default_seconds: float = 300.0) -> float:
    """Return a watcher/control-plane timeout compatible with NCCL group setup."""
    env_value = os.environ.get(
        "ELASTIC_PHASE_TIMEOUT_SECONDS",
        os.environ.get("ELASTIC_REBUILD_PHASE_TIMEOUT"),
    )
    if env_value:
        return float(env_value)

    timeout_minutes = None
    if args is not None:
        timeout_minutes = getattr(args, "distributed_timeout_minutes", None)
    if timeout_minutes is None:
        timeout_minutes = os.environ.get("DISTRIBUTED_TIMEOUT_MINUTES")

    try:
        group_timeout = float(timeout_minutes) * 60.0
    except (TypeError, ValueError):
        group_timeout = 0.0

    margin = float(os.environ.get("ELASTIC_PHASE_TIMEOUT_MARGIN_SECONDS", "120"))
    return max(float(default_seconds), group_timeout + margin)


def _elastic_c10d_generation_state():
    """Return the process-local c10d registry state used by PG naming."""
    state = {
        "initialized": bool(dist.is_available() and dist.is_initialized()),
        "pg_map_count": -1,
        "pg_name_count": -1,
        "group_count": -1,
    }
    try:
        world = torch.distributed.distributed_c10d._world
        state.update(
            pg_map_count=len(world.pg_map),
            pg_name_count=len(world.pg_names),
            group_count=int(world.group_count),
        )
    except Exception:
        pass
    return state


def _elastic_destroy_process_group_generation(mpu, rank):
    """Atomically retire one c10d generation before rebuilding another.

    PyTorch's WORLD teardown sorts and shuts down every registered subgroup,
    clears the Python/C++ registries, and resets the implicit group-name
    counter. Destroying each rank's Megatron subgroup list first bypasses that
    recovery path and is asymmetric because pipeline stages own different PGs.
    """
    global _REBUILD_STORE

    before = _elastic_c10d_generation_state()
    if not before["initialized"]:
        raise RuntimeError(
            f"rank {rank} cannot retire c10d generation: default PG is not initialized"
        )
    unknown_before = [
        key for key in ("pg_map_count", "pg_name_count", "group_count")
        if before[key] < 0
    ]
    if unknown_before:
        raise RuntimeError(
            "cannot validate c10d generation state: " + ",".join(unknown_before)
        )

    dist.destroy_process_group()
    after = _elastic_c10d_generation_state()
    errors = []
    unknown_after = [
        key for key in ("pg_map_count", "pg_name_count", "group_count")
        if after[key] < 0
    ]
    if unknown_after:
        errors.append("unavailable=" + ",".join(unknown_after))
    if after["initialized"]:
        errors.append("default PG remains initialized")
    if after["pg_map_count"] not in (-1, 0):
        errors.append(f"pg_map_count={after['pg_map_count']}")
    if after["pg_name_count"] not in (-1, 0):
        errors.append(f"pg_name_count={after['pg_name_count']}")
    if after["group_count"] not in (-1, 0):
        errors.append(f"group_count={after['group_count']}")
    if errors:
        raise RuntimeError(
            "incomplete c10d generation teardown: " + "; ".join(errors)
        )

    # The c10d generation is already shut down; this only drops Megatron's
    # Python references and cached topology fields.
    mpu.destroy_model_parallel()
    _REBUILD_STORE = None
    logger.warning(
        "[elastic] Rank %d: retired c10d generation atomically "
        "(before=%s, after=%s)",
        rank,
        before,
        after,
    )


def _initialize_model_parallel_for_rebuild(mpu, args):
    subgroup_timeout_minutes = _elastic_rebuild_subgroup_timeout_minutes(args)
    logger.warning(
        "[elastic] Rank %d: rebuilding Megatron subgroups with timeout=%d minutes",
        dist.get_rank() if dist.is_initialized() else -1,
        subgroup_timeout_minutes,
    )
    old_trace_mpu_groups = os.environ.get("ELASTIC_TRACE_MPU_GROUPS")
    if old_trace_mpu_groups is None:
        os.environ["ELASTIC_TRACE_MPU_GROUPS"] = "1"
    try:
        if hasattr(mpu, "reset_elastic_mpu_group_ordinal"):
            mpu.reset_elastic_mpu_group_ordinal()
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=getattr(
                args, "virtual_pipeline_model_parallel_size", None
            ),
            pipeline_model_parallel_comm_backend=getattr(
                args, "pipeline_model_parallel_comm_backend", None
            ),
            use_sharp=getattr(args, "use_sharp", False),
            context_parallel_size=getattr(args, "context_parallel_size", 1),
            hierarchical_context_parallel_sizes=getattr(
                args, "hierarchical_context_parallel_sizes", None
            ),
            expert_model_parallel_size=getattr(args, "expert_model_parallel_size", 1),
            num_distributed_optimizer_instances=getattr(
                args, "num_distributed_optimizer_instances", 1
            ),
            expert_tensor_parallel_size=getattr(args, "expert_tensor_parallel_size", None),
            distributed_timeout_minutes=subgroup_timeout_minutes,
            nccl_communicator_config_path=getattr(args, "nccl_communicator_config_path", None),
            order="tp-cp-ep-dp-pp"
            if not getattr(args, "use_tp_pp_dp_mapping", False)
            else "tp-cp-ep-pp-dp",
            create_gloo_process_groups=False,
            high_priority_stream_groups=getattr(args, "high_priority_stream_groups", None),
            sharp_enabled_group=getattr(args, "sharp_enabled_group", None),
        )
    finally:
        if old_trace_mpu_groups is None:
            os.environ.pop("ELASTIC_TRACE_MPU_GROUPS", None)


def _elastic_safe_get_group(name, getter):
    try:
        return getter()
    except Exception as exc:
        logger.debug("[elastic] current process group %s unavailable: %s", name, exc)
        return None


def _elastic_current_pg_dict():
    from megatron.core import parallel_state as mpu

    return {
        "tp": _elastic_safe_get_group(
            "tp", lambda: mpu.get_tensor_model_parallel_group(check_initialized=False)
        ),
        "pp": _elastic_safe_get_group(
            "pp", lambda: mpu.get_pipeline_model_parallel_group(check_initialized=False)
        ),
        "mp": _elastic_safe_get_group(
            "mp", lambda: mpu.get_model_parallel_group(check_initialized=False)
        ),
        "embd": _elastic_safe_get_group(
            "embd", lambda: mpu.get_embedding_group(check_initialized=False)
        ),
        "pos_embd": _elastic_safe_get_group(
            "pos_embd", lambda: mpu.get_position_embedding_group(check_initialized=False)
        ),
        "cp": _elastic_safe_get_group(
            "cp", lambda: mpu.get_context_parallel_group(check_initialized=False)
        ),
        "tp_cp": _elastic_safe_get_group(
            "tp_cp", lambda: mpu.get_tensor_and_context_parallel_group(check_initialized=False)
        ),
        "hcp": _elastic_safe_get_group(
            "hcp", lambda: mpu.get_hierarchical_context_parallel_groups(check_initialized=False)
        ),
        "ep": _elastic_safe_get_group(
            "ep", lambda: mpu.get_expert_model_parallel_group(check_initialized=False)
        ),
        "expt_tp": _elastic_safe_get_group(
            "expt_tp", lambda: mpu.get_expert_tensor_parallel_group(check_initialized=False)
        ),
        "tp_ep": _elastic_safe_get_group(
            "tp_ep", lambda: mpu.get_expert_tensor_and_model_parallel_group(
                check_initialized=False
            )
        ),
        "tp_ep_pp": _elastic_safe_get_group(
            "tp_ep_pp", lambda: mpu.get_expert_tensor_model_pipeline_parallel_group(
                check_initialized=False
            )
        ),
        "tp_dp_cp": _elastic_safe_get_group(
            "tp_dp_cp", lambda: mpu.get_tensor_and_data_parallel_group(
                check_initialized=False, with_context_parallel=True
            )
        ),
        "dp": _elastic_safe_get_group(
            "dp", lambda: mpu.get_data_parallel_group(with_context_parallel=False)
        ),
        "dp_cp": _elastic_safe_get_group(
            "dp_cp", lambda: mpu.get_data_parallel_group(with_context_parallel=True)
        ),
        "intra_dp_cp": _elastic_safe_get_group(
            "intra_dp_cp", lambda: mpu.get_data_parallel_group(
                with_context_parallel=True, partial_data_parallel=True
            )
        ),
        "expt_dp": _elastic_safe_get_group(
            "expt_dp", lambda: mpu.get_expert_data_parallel_group(check_initialized=False)
        ),
        "intra_expt_dp": _elastic_safe_get_group(
            "intra_expt_dp", lambda: mpu.get_expert_data_parallel_group(
                check_initialized=False, partial_expert_data_parallel=True
            )
        ),
        "inter_dist_opt": _elastic_safe_get_group(
            "inter_dist_opt",
            lambda: mpu.get_inter_distributed_optimizer_instance_group(check_initialized=False),
        ),
        "intra_dist_opt": _elastic_safe_get_group(
            "intra_dist_opt",
            lambda: mpu.get_intra_distributed_optimizer_instance_group(check_initialized=False),
        ),
        "intra_dp_cp_gloo": _elastic_safe_get_group(
            "intra_dp_cp_gloo", lambda: mpu.get_data_parallel_group_gloo(
                with_context_parallel=True, partial_data_parallel=True
            )
        ),
        "intra_expt_dp_gloo": _elastic_safe_get_group(
            "intra_expt_dp_gloo", lambda: mpu.get_expert_data_parallel_group_gloo(
                partial_expert_data_parallel=True
            )
        ),
    }


def _elastic_set_attr(obj, attr_name, new_value):
    if new_value is None or not hasattr(obj, attr_name):
        return 0
    if getattr(obj, attr_name, None) is new_value:
        return 0
    setattr(obj, attr_name, new_value)
    return 1


def _elastic_rebind_pg_collection(module, pg_dict):
    pg_collection = getattr(module, "pg_collection", None)
    if pg_collection is None:
        return 0

    count = 0
    for field_name, group in pg_dict.items():
        if field_name.endswith("_gloo") or group is None:
            continue
        if hasattr(pg_collection, field_name):
            count += _elastic_set_attr(pg_collection, field_name, group)
    return count


def _elastic_refresh_group_derived_attrs(module):
    count = 0
    for group_attr, prefix in (
        ("tp_group", "tp"),
        ("cp_group", "cp"),
        ("ep_group", "ep"),
        ("dp_group", "dp"),
    ):
        group = getattr(module, group_attr, None)
        if group is None:
            continue
        try:
            if hasattr(module, f"{prefix}_size"):
                new_size = group.size()
                if getattr(module, f"{prefix}_size", None) != new_size:
                    setattr(module, f"{prefix}_size", new_size)
                    count += 1
            if hasattr(module, f"{prefix}_rank"):
                new_rank = group.rank()
                if getattr(module, f"{prefix}_rank", None) != new_rank:
                    setattr(module, f"{prefix}_rank", new_rank)
                    count += 1
        except Exception:
            pass
    return count


def _elastic_try_call_group_setter(module, setter_name, group, *extra_args):
    if group is None:
        return 0
    setter = getattr(module, setter_name, None)
    if not callable(setter):
        return 0
    try:
        setter(group, *extra_args)
        return 1
    except TypeError:
        if extra_args:
            return 0
        try:
            ranks = dist.get_process_group_ranks(group)
            setter(group, ranks)
            return 1
        except Exception as exc:
            logger.debug(
                "[elastic] failed calling %s on %s: %s",
                setter_name,
                type(module).__name__,
                exc,
            )
            return 0
    except Exception as exc:
        logger.debug(
            "[elastic] failed calling %s on %s: %s",
            setter_name,
            type(module).__name__,
            exc,
        )
        return 0


def _elastic_rebind_extension_group_setters(module, pg_dict, handled_specific=False):
    count = 0
    tp_key = "expt_tp" if getattr(module, "is_expert", False) else "tp"
    if not handled_specific:
        count += _elastic_try_call_group_setter(
            module, "set_tensor_parallel_group", pg_dict.get(tp_key)
        )
    count += _elastic_try_call_group_setter(
        module, "set_context_parallel_group", pg_dict.get("cp")
    )
    return count


def _elastic_rebind_extension_stashed_groups(module, pg_dict):
    if not hasattr(module, "stashed_tp_group"):
        return 0
    if getattr(module, "stashed_tp_group", None) is None:
        return 0
    tp_key = "expt_tp" if getattr(module, "stashed_is_expert", False) else "tp"
    return _elastic_set_attr(module, "stashed_tp_group", pg_dict.get(tp_key))


def _elastic_rebind_direct_module_groups(module, pg_dict, handled_specific=False):
    """Refresh direct group attrs that Megatron modules cache outside pg_collection."""
    count = 0

    direct_map = {
        "cp_group": "cp",
        "pp_group": "pp",
        "embd_group": "embd",
        "attn_tp_group": "tp",
        "ep_group": "ep",
        "dp_cp_group": "dp_cp",
        "tp_ep_group": "tp_ep",
        "expt_dp_group": "expt_dp",
        "intra_expt_dp_group": "intra_expt_dp",
    }
    for attr_name, pg_key in direct_map.items():
        count += _elastic_set_attr(module, attr_name, pg_dict.get(pg_key))

    if not handled_specific and hasattr(module, "tp_group"):
        tp_key = "expt_tp" if getattr(module, "is_expert", False) else "tp"
        count += _elastic_set_attr(module, "tp_group", pg_dict.get(tp_key))

    count += _elastic_rebind_extension_group_setters(module, pg_dict, handled_specific)
    count += _elastic_rebind_extension_stashed_groups(module, pg_dict)
    count += _elastic_refresh_group_derived_attrs(module)
    return count


def _elastic_group_size(group):
    try:
        return group.size()
    except Exception:
        return 1


def _elastic_group_rank(group):
    try:
        return group.rank()
    except Exception:
        return 0


def _elastic_rebind_param_buffers(buffers, data_parallel_group, tp_group, dp_cp_group):
    count = 0
    for buffer in buffers or []:
        count += _elastic_set_attr(buffer, "data_parallel_group", data_parallel_group)
        if data_parallel_group is not None and hasattr(buffer, "data_parallel_world_size"):
            new_world_size = _elastic_group_size(data_parallel_group)
            if getattr(buffer, "data_parallel_world_size", None) != new_world_size:
                buffer.data_parallel_world_size = new_world_size
                count += 1
        count += _elastic_set_attr(buffer, "tp_group", tp_group)
        count += _elastic_set_attr(buffer, "dp_cp_group", dp_cp_group)
    return count


def _elastic_rebind_bucket_groups(bucket_groups, collective_group, inter_group=None):
    count = 0
    for bucket_group in bucket_groups or []:
        if collective_group is not None and getattr(
            bucket_group.ddp_config, "use_distributed_optimizer", False
        ):
            count += _elastic_set_attr(
                bucket_group, "intra_distributed_optimizer_instance_group", collective_group
            )
            new_size = _elastic_group_size(collective_group)
            if (
                getattr(bucket_group, "intra_distributed_optimizer_instance_size", None)
                != new_size
            ):
                bucket_group.intra_distributed_optimizer_instance_size = new_size
                count += 1
            new_rank = _elastic_group_rank(collective_group)
            if (
                getattr(bucket_group, "intra_distributed_optimizer_instance_rank", None)
                != new_rank
            ):
                bucket_group.intra_distributed_optimizer_instance_rank = new_rank
                count += 1
            count += _elastic_set_attr(
                bucket_group, "inter_distributed_optimizer_instance_group", inter_group
            )
        elif collective_group is not None:
            count += _elastic_set_attr(bucket_group, "data_parallel_group", collective_group)
        if hasattr(bucket_group, "param_gather_handle"):
            bucket_group.param_gather_handle = None
        if hasattr(bucket_group, "param_gather_dispatched"):
            bucket_group.param_gather_dispatched = False
        if hasattr(bucket_group, "grad_reduce_handle"):
            bucket_group.grad_reduce_handle = None
        if hasattr(bucket_group, "cached_param_buffer_shard_list"):
            bucket_group.cached_param_buffer_shard_list = [None] * len(bucket_group.buckets)
        if hasattr(bucket_group, "cached_grad_buffer_shard_list"):
            bucket_group.cached_grad_buffer_shard_list = [None] * len(bucket_group.buckets)
    return count


def _elastic_is_ddp_wrapper(module):
    return (
        type(module).__name__ == "DistributedDataParallel"
        or (
            hasattr(module, "ddp_config")
            and hasattr(module, "bucket_groups")
            and hasattr(module, "expert_parallel_bucket_groups")
        )
    )


def _elastic_rebind_ddp_wrapper(module, pg_dict):
    count = 0
    attr_map = {
        "dp_group": "dp",
        "dp_cp_group": "dp_cp",
        "intra_dp_cp_group": "intra_dp_cp",
        "expt_dp_group": "expt_dp",
        "intra_expt_dp_group": "intra_expt_dp",
        "tp_group": "tp",
        "pp_group": "pp",
        "ep_group": "ep",
        "inter_dist_opt_group": "inter_dist_opt",
    }
    for attr_name, pg_key in attr_map.items():
        count += _elastic_set_attr(module, attr_name, pg_dict.get(pg_key))

    count += _elastic_rebind_param_buffers(
        getattr(module, "buffers", None),
        pg_dict.get("intra_dp_cp"),
        pg_dict.get("tp"),
        pg_dict.get("dp_cp"),
    )
    count += _elastic_rebind_param_buffers(
        getattr(module, "expert_parallel_buffers", None),
        pg_dict.get("intra_expt_dp"),
        pg_dict.get("tp"),
        pg_dict.get("dp_cp"),
    )
    count += _elastic_rebind_bucket_groups(
        getattr(module, "bucket_groups", None),
        pg_dict.get("intra_dp_cp"),
        pg_dict.get("inter_dist_opt"),
    )
    count += _elastic_rebind_bucket_groups(
        getattr(module, "expert_parallel_bucket_groups", None),
        pg_dict.get("intra_expt_dp"),
        pg_dict.get("inter_dist_opt"),
    )
    return count


def _elastic_iter_model_chunks(model):
    if model is None:
        return []
    return list(model) if isinstance(model, (list, tuple)) else [model]


def _elastic_model_buffer_kind_by_id(model):
    buffer_kind = {}
    for model_chunk in _elastic_iter_model_chunks(model):
        for buffer in getattr(model_chunk, "buffers", []) or []:
            buffer_kind[id(buffer)] = "dense"
        for buffer in getattr(model_chunk, "expert_parallel_buffers", []) or []:
            buffer_kind[id(buffer)] = "expert"
    return buffer_kind


def _elastic_classify_optimizer_buffers(megatron_optimizer, buffer_kind):
    kinds = set()
    for buffer in getattr(megatron_optimizer, "buffers", []) or []:
        kind = buffer_kind.get(id(buffer))
        if kind is not None:
            kinds.add(kind)
    if kinds == {"expert"}:
        return "expert"
    if kinds == {"dense"}:
        return "dense"
    return None


def _elastic_classify_optimizer_param_groups(megatron_optimizer):
    """Classify an inner optimizer using Megatron's persistent group metadata."""
    base_optimizer = getattr(megatron_optimizer, "optimizer", None)
    param_groups = getattr(base_optimizer, "param_groups", None) or []
    expert_flags = {
        bool(group["is_expert_parallel"])
        for group in param_groups
        if "is_expert_parallel" in group
    }
    if expert_flags == {True}:
        return "expert"
    if expert_flags == {False}:
        return "dense"
    if len(expert_flags) > 1:
        raise RuntimeError(
            "one inner Megatron optimizer contains both dense and expert parameter groups"
        )
    return None


def _elastic_classify_optimizer_kind(megatron_optimizer, buffer_kind):
    # Non-distributed Float16Optimizer does not retain `.buffers`, but its
    # underlying optimizer param groups keep `is_expert_parallel`.  Prefer that
    # canonical metadata so dense/expert grad-stat groups remain distinct.
    kind = _elastic_classify_optimizer_param_groups(megatron_optimizer)
    if kind is not None:
        return kind, "param_groups"

    kind = _elastic_classify_optimizer_buffers(megatron_optimizer, buffer_kind)
    if kind is not None:
        return kind, "buffers"

    # A parameterless/stub optimizer may not have either source.  Preserve its
    # original semantic identity from the retired ProcessGroup when available.
    old_group = getattr(megatron_optimizer, "grad_stats_parallel_group", None)
    old_desc = str(getattr(old_group, "group_desc", "")).upper()
    if "EXPERT_TENSOR" in old_desc and "PIPELINE" in old_desc:
        return "expert", "retired_group_desc"
    if "MODEL_PARALLEL" in old_desc:
        return "dense", "retired_group_desc"
    return None, "unresolved"


def _elastic_optimizer_uses_distributed_optimizer(megatron_optimizer):
    config = getattr(megatron_optimizer, "config", None)
    return bool(getattr(config, "use_distributed_optimizer", False))


def _elastic_expected_optimizer_grad_group(megatron_optimizer, kind, pg_dict):
    if _elastic_optimizer_uses_distributed_optimizer(megatron_optimizer):
        return "intra_dist_opt", pg_dict.get("intra_dist_opt")
    if kind == "expert":
        return "tp_ep_pp", pg_dict.get("tp_ep_pp")
    return "mp", pg_dict.get("mp")


def _elastic_process_group_contract(group):
    if group is None:
        return {"group_name": "missing", "group_desc": "missing", "ranks": []}
    contract = {
        "group_name": str(getattr(group, "group_name", "unavailable")),
        "group_desc": str(getattr(group, "group_desc", "unavailable")),
        "ranks": [],
    }
    try:
        contract["ranks"] = list(dist.get_process_group_ranks(group))
    except Exception as exc:
        contract["ranks_error"] = str(exc)
    return contract


def _elastic_rebind_optimizer_process_groups(optimizer, model, pg_dict):
    if optimizer is None:
        return 0

    count = 0
    buffer_kind = _elastic_model_buffer_kind_by_id(model)
    contracts = []
    for optimizer_idx, megatron_optimizer in enumerate(_iter_megatron_optimizers(optimizer)):
        kind, kind_source = _elastic_classify_optimizer_kind(
            megatron_optimizer, buffer_kind
        )
        if kind is None:
            if getattr(megatron_optimizer, "is_stub_optimizer", False):
                continue
            raise RuntimeError(
                f"cannot classify optimizer[{optimizer_idx}] as dense or expert; "
                "missing is_expert_parallel metadata"
            )
        if kind == "expert":
            data_group = pg_dict.get("intra_expt_dp")
            data_group_gloo = pg_dict.get("intra_expt_dp_gloo")
        elif kind == "dense":
            data_group = pg_dict.get("intra_dp_cp")
            data_group_gloo = pg_dict.get("intra_dp_cp_gloo")
        else:
            data_group = None
            data_group_gloo = None

        count += _elastic_rebind_param_buffers(
            getattr(megatron_optimizer, "buffers", None),
            data_group,
            pg_dict.get("tp"),
            pg_dict.get("dp_cp"),
        )
        per_model_bucket_groups = (
            getattr(megatron_optimizer, "per_model_bucket_groups", {}) or {}
        )
        for bucket_groups in per_model_bucket_groups.values():
            count += _elastic_rebind_bucket_groups(
                bucket_groups, data_group, pg_dict.get("inter_dist_opt")
            )

        if data_group is not None:
            count += _elastic_set_attr(megatron_optimizer, "data_parallel_group", data_group)
            if hasattr(megatron_optimizer, "data_parallel_group_gloo"):
                if (
                    getattr(megatron_optimizer, "data_parallel_group_gloo", None)
                    is not data_group_gloo
                ):
                    megatron_optimizer.data_parallel_group_gloo = data_group_gloo
                    count += 1

        expected_key, grad_stats_group = _elastic_expected_optimizer_grad_group(
            megatron_optimizer, kind, pg_dict
        )
        if grad_stats_group is None:
            raise RuntimeError(
                f"optimizer[{optimizer_idx}] kind={kind} expected ProcessGroup "
                f"{expected_key}, but it is unavailable after rebuild"
            )
        count += _elastic_set_attr(
            megatron_optimizer, "grad_stats_parallel_group", grad_stats_group
        )
        contracts.append(
            {
                "index": optimizer_idx,
                "kind": kind,
                "kind_source": kind_source,
                "expected_group": expected_key,
                "use_distributed_optimizer": _elastic_optimizer_uses_distributed_optimizer(
                    megatron_optimizer
                ),
                **_elastic_process_group_contract(grad_stats_group),
            }
        )

    logger.warning(
        "[elastic] Rank %d: rebound optimizer ProcessGroup contract=%s",
        dist.get_rank(),
        contracts,
    )

    return count


def elastic_validate_optimizer_process_groups(optimizer, iteration=None):
    """Validate first-step optimizer collective groups without running a collective."""
    if not elastic_is_post_rebuild_trace_active(iteration):
        return False

    pg_dict = _elastic_current_pg_dict()
    buffer_kind = {}
    contracts = []
    try:
        for optimizer_idx, megatron_optimizer in enumerate(
            _iter_megatron_optimizers(optimizer)
        ):
            kind, kind_source = _elastic_classify_optimizer_kind(
                megatron_optimizer, buffer_kind
            )
            if kind is None:
                if getattr(megatron_optimizer, "is_stub_optimizer", False):
                    continue
                raise RuntimeError(
                    f"cannot classify optimizer[{optimizer_idx}] as dense or expert"
                )
            expected_key, expected_group = _elastic_expected_optimizer_grad_group(
                megatron_optimizer, kind, pg_dict
            )
            actual_group = megatron_optimizer.get_grad_stats_parallel_group()
            contract = {
                "index": optimizer_idx,
                "kind": kind,
                "kind_source": kind_source,
                "expected_group": expected_key,
                "use_distributed_optimizer": _elastic_optimizer_uses_distributed_optimizer(
                    megatron_optimizer
                ),
                **_elastic_process_group_contract(actual_group),
            }
            contracts.append(contract)
            if expected_group is None or actual_group is not expected_group:
                raise RuntimeError(
                    f"optimizer[{optimizer_idx}] kind={kind} grad-stat ProcessGroup "
                    f"does not match current {expected_key}: contract={contract}"
                )

        non_distributed = [
            item
            for item in contracts
            if not item["use_distributed_optimizer"]
        ]
        dense_groups = {
            item["group_name"] for item in non_distributed if item["kind"] == "dense"
        }
        expert_groups = {
            item["group_name"] for item in non_distributed if item["kind"] == "expert"
        }
        if dense_groups and expert_groups and dense_groups == expert_groups:
            raise RuntimeError(
                "dense and expert optimizers share one grad-stat ProcessGroup after rebuild: "
                f"dense={dense_groups} expert={expert_groups}"
            )
    except Exception as exc:
        elastic_report_recovery_phase(
            "optimizer_pg_contract_error",
            optimizer_pg_contract=contracts,
            error=str(exc),
        )
        raise

    ready_phase = _elastic_effective_recovery_phase("optimizer_pg_contract_ready")
    elastic_report_recovery_phase(
        "optimizer_pg_contract_ready", optimizer_pg_contract=contracts
    )
    world_size = dist.get_world_size()
    timeout = _elastic_phase_timeout_seconds()
    if not elastic_wait_for_recovery_phase_count(
        ready_phase, world_size, timeout
    ):
        error = (
            "not all ranks validated optimizer ProcessGroups before the first "
            f"post-rebuild optimizer step within {timeout}s"
        )
        elastic_report_recovery_phase(
            "optimizer_pg_contract_error",
            optimizer_pg_contract=contracts,
            error=error,
        )
        raise RuntimeError(error)
    logger.warning(
        "[elastic] Rank %d: optimizer ProcessGroup contract verified: %s",
        dist.get_rank(),
        contracts,
    )
    return True


def _elastic_rebind_model_process_groups(model, optimizer=None):
    """Refresh cached Megatron process-group handles after hot-spare rebuild.

    Reinitializing parallel_state is not enough: pre-existing module instances,
    DDP wrappers, grad buffers, and distributed optimizers store ProcessGroup
    objects created before the failed rank was replaced.
    """
    if not dist.is_available() or not dist.is_initialized():
        return

    from megatron.core.transformer.moe.group_rebuild import (
        MOE_DISPATCHER_REBIND_MAP,
        MOE_EXPERTS_REBIND_MAP,
        MOE_LAYER_REBIND_MAP,
        MOE_ROUTER_REBIND_MAP,
        rebind_dispatcher_derived_values,
        rebind_moe_module_groups,
    )

    pg_dict = _elastic_current_pg_dict()
    rank = dist.get_rank()
    total_count = 0
    module_count = 0
    ddp_count = 0
    dispatcher_count = 0
    optimizer_count = 0

    for chunk_idx, model_chunk in enumerate(_elastic_iter_model_chunks(model)):
        if not hasattr(model_chunk, "named_modules"):
            continue
        for name, module in model_chunk.named_modules():
            module_count += 1
            module_name = f"model[{chunk_idx}].{name}" if name else f"model[{chunk_idx}]"
            module_type = type(module).__name__
            handled_specific = False

            if _elastic_is_ddp_wrapper(module):
                changed = _elastic_rebind_ddp_wrapper(module, pg_dict)
                if changed:
                    ddp_count += 1
                total_count += changed
                handled_specific = True

            if "Dispatcher" in module_type or "dispatcher" in name:
                changed = rebind_moe_module_groups(
                    module, pg_dict, MOE_DISPATCHER_REBIND_MAP, module_name=module_name
                )
                if changed:
                    rebind_dispatcher_derived_values(module)
                    dispatcher_count += 1
                total_count += changed
                handled_specific = True
            elif "Router" in module_type or "router" in name:
                total_count += rebind_moe_module_groups(
                    module, pg_dict, MOE_ROUTER_REBIND_MAP, module_name=module_name
                )
                handled_specific = True
            elif "MoELayer" in module_type or "moe_layer" in name:
                total_count += rebind_moe_module_groups(
                    module, pg_dict, MOE_LAYER_REBIND_MAP, module_name=module_name
                )
                handled_specific = True
            elif "Expert" in module_type or "expert" in name:
                total_count += rebind_moe_module_groups(
                    module, pg_dict, MOE_EXPERTS_REBIND_MAP, module_name=module_name
                )
                handled_specific = True

            token_dispatcher = getattr(module, "token_dispatcher", None)
            if token_dispatcher is not None:
                changed = rebind_moe_module_groups(
                    token_dispatcher,
                    pg_dict,
                    MOE_DISPATCHER_REBIND_MAP,
                    module_name=f"{module_name}.token_dispatcher",
                )
                if changed:
                    rebind_dispatcher_derived_values(token_dispatcher)
                    dispatcher_count += 1
                total_count += changed

            total_count += _elastic_rebind_pg_collection(module, pg_dict)
            total_count += _elastic_rebind_direct_module_groups(
                module, pg_dict, handled_specific=handled_specific
            )

    optimizer_count = _elastic_rebind_optimizer_process_groups(optimizer, model, pg_dict)
    total_count += optimizer_count
    logger.warning(
        "[elastic] Rank %d: rebound cached process groups after rebuild "
        "(attrs=%d, modules=%d, ddp_wrappers=%d, dispatchers=%d, optimizer_attrs=%d)",
        rank,
        total_count,
        module_count,
        ddp_count,
        dispatcher_count,
        optimizer_count,
    )


def elastic_check_pause() -> bool:
    """Check if a pause has been requested by the watcher.

    This should be called at every iteration boundary (safe point).
    Returns True if training should pause for group rebuild.

    IMPORTANT: This must NOT use dist.all_reduce or any global collective,
    because the whole point is that some node may have died.  A global
    collective would hang waiting for the dead node.

    Instead we use a **local file signal**:
      - local_rank 0 has the TCP connection to the watcher and receives
        the pause signal.  When it does, it writes a flag file.
      - All local ranks check the flag file (no distributed communication).
    """
    global _PAUSE_REQUESTED

    # If no watcher configured, never pause
    if not os.environ.get("ELASTIC_WATCHER_ADDR"):
        return False

    _exit_if_fallback_relaunch_requested()

    fault_dir = os.environ.get("ELASTIC_FAULT_DIR", "/tmp/elastic_faults")
    pause_file = os.path.join(fault_dir, "pause_signal")

    # All ranks: check for the flag file (non-blocking, no dist calls)
    if os.path.exists(pause_file):
        with _LOCK:
            _PAUSE_REQUESTED = True
        return True

    return False


def elastic_on_nccl_error(exception: Exception):
    """Called when an NCCL communication error is caught in train_step.

    This sets the pause flag so that at the next safe point (or immediately
    if we can reach one), the training enters the rebuild path.

    This function must be safe to call from any rank, and must NOT use any
    distributed calls (the process group may be corrupted).
    """
    global _PAUSE_REQUESTED

    logger.error("[elastic] NCCL error detected: %s", exception)
    with _LOCK:
        _PAUSE_REQUESTED = True

    # Write pause signal file for all local ranks
    _write_pause_signal()
    _launcher_control_send("nccl_error", error=str(exception)[:200])

    # Notify watcher (best-effort — the watcher may already know via
    # heartbeat timeout, but sending explicit notification is faster)
    if _CLIENT is not None:
        _CLIENT._send({
            "type": "nccl_error",
            "node_rank": _CLIENT.node_rank,
            "step": _CLIENT.step,
            "step_tag": _CLIENT.step_tag,
            "train_phase": _CLIENT.train_phase,
            "error": str(exception)[:200],
        })


def elastic_wait_for_rebuild_signal() -> dict:
    """Block until the watcher sends the rebuild signal.

    Called after all surviving nodes have paused and notified the watcher.
    Returns the rebuild info dict with new_master_addr, new_master_port, etc.

    NOTE: This is called AFTER destroy_process_group, so we cannot use
    dist.broadcast. Launcher-controlled ranks receive the rebuild contract via
    an atomic local file; the legacy path still proxies through local_rank 0.
    """
    global _REBUILD_INFO

    launcher_owned = _launcher_control_enabled()
    _launcher_control_send("rebuild_ready")
    next_ready_report = time.time() + 1.0

    # Legacy path: local_rank 0 owns the watcher connection.
    if _CLIENT is not None and not launcher_owned:
        _CLIENT.send_ready_to_rebuild()

    # Wait for rebuild signal (only local_rank 0 gets it via TCP)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    fault_dir = os.environ.get("ELASTIC_FAULT_DIR", "/tmp/elastic_faults")
    rebuild_file = os.path.join(fault_dir, "rebuild_signal.json")

    if local_rank == 0 and not launcher_owned:
        while True:
            _exit_if_fallback_relaunch_requested()
            with _LOCK:
                if _FALLBACK_RELAUNCH_INFO is not None:
                    _write_fallback_relaunch_signal(_FALLBACK_RELAUNCH_INFO)
                    _exit_if_fallback_relaunch_requested()
                if _REBUILD_INFO is not None:
                    info = _REBUILD_INFO
                    _REBUILD_INFO = None
                    break
            time.sleep(0.5)

        # Write to shared file so other local ranks can read it
        with open(rebuild_file, "w") as f:
            json.dump(info, f)
    else:
        # Launcher-controlled ranks and legacy nonzero local ranks wait for the file.
        while True:
            _exit_if_fallback_relaunch_requested()
            if launcher_owned and time.time() >= next_ready_report:
                _launcher_control_send("rebuild_ready")
                next_ready_report = time.time() + 1.0
            if os.path.exists(rebuild_file):
                try:
                    with open(rebuild_file, "r") as f:
                        info = json.load(f)
                    break
                except (json.JSONDecodeError, OSError):
                    pass
            time.sleep(0.5)

    for _ in range(3):
        _launcher_control_send("rebuild_consumed")
    return info


def elastic_align_resume_state(args, opt_param_scheduler, resume_iteration):
    if resume_iteration is None:
        return None
    try:
        resume_iteration = int(resume_iteration)
    except (TypeError, ValueError):
        logger.warning("[elastic] Invalid resume_iteration from watcher: %s", resume_iteration)
        return None
    if resume_iteration < 0:
        return resume_iteration

    args.iteration = resume_iteration
    args.curr_iteration = resume_iteration
    consumed_train_samples = resume_iteration * args.global_batch_size
    args.consumed_train_samples = consumed_train_samples
    if getattr(args, "eval_interval", None):
        eval_iters = args.eval_iters
        if isinstance(eval_iters, list):
            eval_iters = sum(eval_iters)
        args.consumed_valid_samples = (
            (resume_iteration // args.eval_interval) * eval_iters * args.global_batch_size
        )

    if opt_param_scheduler is not None:
        try:
            current_steps = getattr(opt_param_scheduler, "num_steps", None)
            if current_steps != consumed_train_samples:
                opt_param_scheduler.num_steps = 0
                opt_param_scheduler.step(increment=consumed_train_samples)
                logger.warning(
                    "[elastic] Aligned optimizer scheduler to resume_iteration=%d "
                    "(num_steps=%d, previous=%s)",
                    resume_iteration,
                    consumed_train_samples,
                    current_steps,
                )
        except Exception as exc:
            logger.warning("[elastic] Failed to align optimizer scheduler: %s", exc)

    return resume_iteration


def _elastic_reset_rerun_state_machine(resume_iteration: int):
    """Make Megatron's local rerun FSM part of the recovery state contract."""
    from megatron.core.rerun_state_machine import get_rerun_state_machine

    try:
        resume_iteration = int(resume_iteration)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"[elastic] invalid rerun-state recovery iteration: {resume_iteration!r}"
        ) from exc
    if resume_iteration < 0:
        raise RuntimeError(
            f"[elastic] invalid rerun-state recovery iteration: {resume_iteration}"
        )

    machine = get_rerun_state_machine()
    summary = machine.reset_after_external_recovery(resume_iteration)
    elastic_report_recovery_phase("rerun_state_reset", rerun_state=summary)
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        logger.warning(
            "[elastic] Rank %d: reset rerun state after external recovery: %s",
            rank,
            summary,
        )
        epoch = os.environ.get("ELASTIC_RECOVERY_EPOCH", "0")
        canonical = {
            key: summary[key]
            for key in (
                "mode",
                "state",
                "current_iteration",
                "first_iteration_complete",
            )
        }
        if not elastic_wait_for_ordinal_barrier(
            f"rerun_state_reset:{epoch}:{resume_iteration}",
            rank,
            world_size,
            _elastic_phase_timeout_seconds(),
            group_desc="RERUN_STATE_MACHINE",
            group_backend="control_plane",
            group_size=world_size,
            group_ranks=list(range(world_size)),
            barrier_stage="post_rebuild_reset",
            state_contract=canonical,
        ):
            raise RuntimeError(
                "[elastic] rerun-state recovery contract did not match across "
                f"{world_size} ranks: local={canonical}"
            )
    return summary


def _elastic_validate_rerun_state_machine(iteration: int, *, stabilization: bool):
    """Verify the local rerun FSM contract before a recovered train step."""
    from megatron.core.rerun_state_machine import RerunState, get_rerun_state_machine

    machine = get_rerun_state_machine()
    local = {
        "mode": machine.mode.value,
        "state": machine.state.value,
        "state_name": machine.state.name,
        "current_iteration": machine.current_iteration,
        "first_iteration_complete": machine.first_iteration_complete,
        "rerun_requested": machine.rerun_requested,
        "checkpoint_requested": machine.checkpoint_requested,
        "restart_again_requested": machine.restart_again_requested,
        "continue_requested": machine.continue_requested,
    }
    expected = (
        machine.state == RerunState.NOT_RUNNING_YET
        and machine.current_iteration == int(iteration)
        and not machine.rerun_requested
        and not machine.checkpoint_requested
        and not machine.restart_again_requested
        and not machine.continue_requested
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    epoch = os.environ.get("ELASTIC_RECOVERY_EPOCH", "0")
    stage = "stabilization" if stabilization else "first"
    elastic_report_recovery_phase(
        "rerun_state_contract_start", rerun_state=local, contract_ok=expected
    )
    matched = elastic_wait_for_ordinal_barrier(
        f"rerun_state_pre_step:{epoch}:{iteration}:{stage}",
        rank,
        world_size,
        _elastic_phase_timeout_seconds(),
        group_desc="RERUN_STATE_MACHINE",
        group_backend="control_plane",
        group_size=world_size,
        group_ranks=list(range(world_size)),
        barrier_stage=f"pre_{stage}_recovered_step",
        state_contract=local,
    )
    if not matched or not expected:
        elastic_report_recovery_phase(
            "rerun_state_contract_error",
            rerun_state=local,
            manifest_matched=matched,
            contract_ok=expected,
        )
        raise RuntimeError(
            "[elastic] rerun-state contract invalid before recovered train step: "
            f"stage={stage} manifest_matched={matched} local={local}"
        )
    elastic_report_recovery_phase(
        "rerun_state_contract_ready", rerun_state=local, contract_ok=True
    )


def elastic_do_rebuild(model, optimizer, opt_param_scheduler):
    """Execute the full group rebuild sequence.

    Called when elastic_check_pause() returns True.  ALL nodes (including
    the target node) go through this path.

    Flow:
      1. Destroy current process groups (stops NCCL watchdog)
      2. Notify watcher: ready_to_rebuild
      3. Wait for rebuild signal (watcher kills target rank + launches spare)
      4. Re-initialize process groups (with spare replacing dead rank)
      5. Re-initialize model parallel groups
      6. Broadcast params to new rank (from DP peer)
      7. Return to training loop

    The REPLACEMENT process goes through pretrain() normally with
    ELASTIC_REBUILD_MODE=1.
    """
    from megatron.core import parallel_state as mpu
    from megatron.training.global_vars import get_args

    args = get_args()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logger.warning(f"[elastic] Rank {rank}: entering rebuild sequence")

    # CRITICAL: Destroy the complete c10d generation before waiting for the
    # rebuild signal. WORLD teardown owns subgroup shutdown order and registry
    # reset; per-subgroup teardown is not a valid generation boundary.
    # The NCCL watchdog runs in a C++ background thread and will SIGABRT
    # the process if it detects a timeout on any process group — even while
    # Python is blocked waiting for the rebuild signal.  Destroying the
    # groups stops the watchdog immediately.
    logger.info(f"[elastic] Rank {rank}: destroying process groups (stop watchdog)")
    try:
        _elastic_destroy_process_group_generation(mpu, rank)
    except Exception as exc:
        logger.exception(
            "[elastic] Rank %d: failed to retire c10d generation", rank
        )
        elastic_report_recovery_phase(
            "state_contract_error",
            contract="c10d_generation_teardown",
            error=str(exc),
        )
        raise

    # Brief sleep to let NCCL resources release
    time.sleep(2.0)

    # If this node is the target node, kill the specified local_rank worker
    # BEFORE sending ready_to_rebuild.  This way, when the watcher receives
    # all ready_to_rebuild messages, the target rank is already dead and the
    # watcher can immediately launch the spare + send rebuild.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    fault_dir = os.environ.get("ELASTIC_FAULT_DIR", "/tmp/elastic_faults")
    pause_file = os.path.join(fault_dir, "pause_signal")
    pause_info = _read_pause_signal_info(pause_file)
    recovery_epoch = pause_info.get("recovery_epoch")
    if recovery_epoch is not None:
        os.environ["ELASTIC_RECOVERY_EPOCH"] = str(recovery_epoch)
        os.environ["ELASTIC_PG_GENERATION"] = str(recovery_epoch)
    descriptor = pause_info.get("descriptor")
    if descriptor:
        os.environ["ELASTIC_RECOVERY_DESCRIPTOR"] = str(descriptor)

    if local_rank == 0 and not _launcher_control_enabled():
        # Read pause signal to get target info
        killed_local_rank = pause_info.get("killed_local_rank", -1)
        failed_node_from_pause = pause_info.get("failed_node", -1)

        if failed_node_from_pause == node_rank and killed_local_rank >= 0:
            logger.warning(f"[elastic] This is the target node! "
                           f"Killing local_rank={killed_local_rank} before ready_to_rebuild")
            _kill_single_worker(killed_local_rank)
            time.sleep(1.0)  # Let the kill take effect

    # Step 1: Wait for rebuild signal from watcher.
    # The watcher will:
    #   - Receive ready_to_rebuild from all nodes
    #   - Launch spare process (target rank is already dead)
    #   - Send rebuild signal
    rebuild_info = elastic_wait_for_rebuild_signal()
    failed_node = rebuild_info.get("failed_node", -1)
    killed_global_rank = rebuild_info.get("killed_global_rank", -1)
    new_master_addr = rebuild_info.get("new_master_addr", os.environ.get("MASTER_ADDR"))
    new_master_port = rebuild_info.get("new_master_port", os.environ.get("MASTER_PORT"))
    recovery_epoch = rebuild_info.get("recovery_epoch", recovery_epoch)
    if recovery_epoch is not None:
        os.environ["ELASTIC_RECOVERY_EPOCH"] = str(recovery_epoch)
        os.environ["ELASTIC_PG_GENERATION"] = str(recovery_epoch)
    descriptor = rebuild_info.get("descriptor", descriptor)
    if descriptor:
        os.environ["ELASTIC_RECOVERY_DESCRIPTOR"] = str(descriptor)
    os.environ["ELASTIC_CHECKPOINT_STEP"] = str(
        rebuild_info.get("checkpoint_step", -1)
    )
    os.environ["ELASTIC_EXPERT_STALENESS_DELTA"] = str(
        rebuild_info.get("expert_staleness_delta", -1)
    )
    os.environ["ELASTIC_MOEGAMBIT_RECOVERY_MODE"] = str(
        rebuild_info.get("recovery_mode", "intentional_mixed_version")
    )
    os.environ["ELASTIC_TWO_PHASE_RECOVERY"] = (
        "1" if rebuild_info.get("two_phase_enabled", False) else "0"
    )
    resume_iteration = rebuild_info.get("resume_iteration")
    try:
        _validate_recovery_descriptor(rebuild_info, world_size)
    except RuntimeError as exc:
        elastic_report_recovery_phase("state_contract_error", error=str(exc))
        raise
    resume_iteration = elastic_align_resume_state(args, opt_param_scheduler, resume_iteration)
    if resume_iteration is not None:
        os.environ["ELASTIC_RESUME_ITERATION"] = str(resume_iteration)
    if killed_global_rank >= 0:
        os.environ["ELASTIC_REPLACEMENT_RANK"] = str(killed_global_rank)
        os.environ.setdefault("ELASTIC_TRACE_REPLACEMENT_GROUP_MEMBERS", "1")

    logger.warning(f"[elastic] Rank {rank}: rebuild signal received. "
                   f"Failed node={failed_node}, killed_rank={killed_global_rank}, "
                   f"resume_iteration={resume_iteration}, "
                   f"new master={new_master_addr}:{new_master_port}")

    # Step 2: Re-initialize process group with new rendezvous
    # Use a NEW port so there's no conflict with the old TCPStore
    logger.info(f"[elastic] Rank {rank}: re-initializing process group "
                f"(master={new_master_addr}:{new_master_port})")
    os.environ["MASTER_ADDR"] = new_master_addr
    os.environ["MASTER_PORT"] = new_master_port
    elastic_configure_recovery_nccl_transport()
    device_id = None
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device_id = torch.device(f"cuda:{local_rank}")

    # Create the same explicit, unprefixed store used by replacement startup.
    # Mixing this path with replacement's implicit env:// rendezvous can add a
    # torch-elastic PrefixStore only on one side and make lazy NCCL keys invisible.
    rebuild_timeout = _elastic_rebuild_timeout(args)
    logger.warning("[elastic] Rank %d: rebuild timeout is %s", rank, rebuild_timeout)

    store = elastic_create_rebuild_store(
        new_master_addr,
        new_master_port,
        world_size,
        rank,
        rebuild_timeout,
    )

    init_process_group_kwargs = {
        "backend": "nccl",
        "store": store,
        "world_size": world_size,
        "rank": rank,
        "timeout": rebuild_timeout,
    }
    # Keep rebuilt default NCCL PG lazy by default. Eager device_id init has
    # repeatedly failed when a replacement rank joins from a different physical
    # node because it requires rank 0 to publish the world ncclUniqueId before
    # all ranks have converged on the rebuilt store.
    use_rebuild_device_id = os.environ.get("ELASTIC_REBUILD_INIT_PG_DEVICE_ID", "0") == "1"
    if device_id is not None and use_rebuild_device_id:
        try:
            if "device_id" in signature(dist.init_process_group).parameters:
                init_process_group_kwargs["device_id"] = device_id
        except (TypeError, ValueError):
            pass
    logger.warning(
        "[elastic] Rank %d: rebuild init_process_group device_id enabled=%s device=%s",
        rank,
        use_rebuild_device_id,
        device_id,
    )
    dist.init_process_group(**init_process_group_kwargs)
    try:
        from megatron.training import inprocess_restart

        inprocess_restart.maybe_force_nccl_backend_init(device_id)
    except Exception as exc:
        logger.debug("[elastic] force NCCL backend init skipped/failed: %s", exc)
    elastic_report_recovery_phase(
        "pg_ready",
        pg_device_id_enabled=use_rebuild_device_id,
        pg_device_id=str(device_id) if device_id is not None else None,
    )

    # Step 3: Re-initialize model parallel groups
    logger.info(f"[elastic] Rank {rank}: re-initializing model parallel")
    phase_timeout = _elastic_phase_timeout_seconds(args)
    elastic_report_recovery_phase("mpu_init_start")
    logger.warning(
        "[elastic] Rank %d: waiting for %d ranks to reach mpu_init_start",
        rank,
        world_size,
    )
    if not elastic_wait_for_recovery_phase_count("mpu_init_start", world_size, phase_timeout):
        raise RuntimeError(
            f"[elastic] Not all {world_size} ranks reached mpu_init_start within {phase_timeout}s"
        )
    _initialize_model_parallel_for_rebuild(mpu, args)
    elastic_report_recovery_phase("mpu_init_done")
    elastic_report_recovery_phase("rebind_start")
    _elastic_rebind_model_process_groups(model, optimizer)
    elastic_report_recovery_phase("mpu_ready")
    logger.warning(
        "[elastic] Rank %d: waiting for %d ranks to reach mpu_ready before param sync",
        rank,
        world_size,
    )
    if not elastic_wait_for_recovery_phase_count("mpu_ready", world_size, phase_timeout):
        raise RuntimeError(
            f"[elastic] Not all {world_size} ranks reached mpu_ready within {phase_timeout}s"
        )
    if killed_global_rank >= 0:
        logger.warning(
            "[elastic] Rank %d: waiting for replacement rank %d model_optimizer_ready",
            rank,
            killed_global_rank,
        )
        if not elastic_wait_for_recovery_phase(
            "replacement", killed_global_rank, "model_optimizer_ready", phase_timeout
        ):
            raise RuntimeError(
                f"[elastic] Replacement rank {killed_global_rank} did not reach "
                f"model_optimizer_ready within {phase_timeout}s"
            )

    # Step 4: Synchronize parameters to the new rank (DP peer broadcast).
    # The replacement rank has random/zero weights, so choose a surviving
    # rank in each DP group as the source.
    elastic_report_recovery_phase("param_sync_start")
    _sync_params_to_new_rank(model, optimizer, replacement_rank=killed_global_rank)
    elastic_report_recovery_phase("param_sync_done")
    _elastic_reset_rerun_state_machine(resume_iteration)

    # Step 5: Barrier to ensure all ranks are ready.  Ranks outside the
    # replacement DP group can finish immediately; keep them out of the NCCL
    # default-group barrier until the replacement has completed peer sync.
    _wait_for_replacement_phase_before_global_barrier(
        killed_global_rank, "state_contract_ready", phase_timeout
    )
    _elastic_warmup_rebuild_communicators(killed_global_rank, phase_timeout)
    _elastic_rebuild_final_barrier()
    _elastic_report_and_wait_train_ready(phase_timeout)
    logger.warning(f"[elastic] Rank {rank}: rebuild complete, resuming training")
    elastic_mark_post_rebuild_pending(resume_iteration)

    # Reset pause state
    global _PAUSE_REQUESTED
    with _LOCK:
        _PAUSE_REQUESTED = False

    # Clean up signal files
    fault_dir = os.environ.get("ELASTIC_FAULT_DIR", "/tmp/elastic_faults")
    if int(os.environ.get("LOCAL_RANK", "0")) == 0 and not _launcher_control_enabled():
        for fname in ("rebuild_signal.json", "pause_signal", "fallback_relaunch_signal.json"):
            fpath = os.path.join(fault_dir, fname)
            try:
                os.remove(fpath)
            except OSError:
                pass

    # Restart heartbeat client with new connection
    elastic_client_start()
    return resume_iteration


def elastic_replacement_sync_params(model, optimizer, opt_param_scheduler=None):
    """Called by the REPLACEMENT node after model setup to receive params from DP peers.

    The replacement node has just gone through normal Megatron initialization
    (init_process_group + initialize_model_parallel + setup_model_and_optimizer)
    but with random weights. This function receives the actual weights from
    a surviving DP peer.
    """
    replacement_rank = int(os.environ.get("ELASTIC_REPLACEMENT_RANK", os.environ.get("RANK", "0")))
    replacement_descriptor_info = {
        "descriptor": os.environ.get("ELASTIC_RECOVERY_DESCRIPTOR"),
        "descriptor_sha256": os.environ.get("ELASTIC_RECOVERY_DESCRIPTOR_SHA256"),
        "recovery_epoch": int(os.environ.get("ELASTIC_RECOVERY_EPOCH", "-1")),
        "killed_global_rank": replacement_rank,
        "resume_iteration": int(os.environ.get("ELASTIC_RESUME_ITERATION", "-1")),
        "new_master_port": os.environ.get("MASTER_PORT"),
    }
    try:
        _validate_recovery_descriptor(
            replacement_descriptor_info, dist.get_world_size()
        )
    except RuntimeError as exc:
        elastic_report_recovery_phase("state_contract_error", error=str(exc))
        raise
    model_param_to_name = _build_model_param_name_map(model)
    elastic_report_recovery_phase(
        "state_contract_start",
        recovery_mode=os.environ.get("ELASTIC_MOEGAMBIT_RECOVERY_MODE", "unknown"),
        checkpoint_step=int(os.environ.get("ELASTIC_CHECKPOINT_STEP", "-1")),
        expert_staleness_delta=int(
            os.environ.get("ELASTIC_EXPERT_STALENESS_DELTA", "-1")
        ),
        pg_generation=int(os.environ.get("ELASTIC_PG_GENERATION", "-1")),
    )
    expert_optimizer_summary = _load_expert_optimizer_state_from_checkpoint(
        optimizer, model_param_to_name
    )
    elastic_report_recovery_phase("param_sync_start")
    peer_sync_summary = _sync_params_to_new_rank(
        model,
        optimizer,
        replacement_rank=replacement_rank,
        model_param_to_name=model_param_to_name,
    )
    if peer_sync_summary is None:
        raise RuntimeError(
            "[elastic] state contract failed: replacement did not execute "
            "dense/non-expert peer synchronization"
        )
    if peer_sync_summary["dense_model_params"] <= 0:
        raise RuntimeError(
            "[elastic] state contract failed: no non-expert model params "
            "were received from the current-step peer"
        )
    if peer_sync_summary["checkpoint_expert_model_params"] <= 0:
        raise RuntimeError(
            "[elastic] state contract failed: replacement model contains no "
            "checkpoint-restored expert params"
        )
    elastic_report_recovery_phase("param_sync_done")
    from megatron.training import get_args

    resume_iteration = int(os.environ.get("ELASTIC_RESUME_ITERATION", "-1"))
    aligned_iteration = elastic_align_resume_state(
        get_args(), opt_param_scheduler, resume_iteration
    )
    if aligned_iteration is None or aligned_iteration < 0:
        raise RuntimeError(
            "[elastic] state contract failed: replacement resume state was not aligned"
        )
    elastic_report_recovery_phase(
        "resume_state_applied",
        aligned_iteration=aligned_iteration,
    )
    elastic_report_recovery_phase(
        "state_contract_ready",
        expert_optimizer=expert_optimizer_summary,
        peer_sync=peer_sync_summary,
        aligned_iteration=aligned_iteration,
        two_phase_enabled=os.environ.get("ELASTIC_TWO_PHASE_RECOVERY", "0") == "1",
    )
    _elastic_reset_rerun_state_machine(aligned_iteration)
    phase_timeout = _elastic_phase_timeout_seconds()
    _elastic_warmup_rebuild_communicators(replacement_rank, phase_timeout)
    _elastic_rebuild_final_barrier()
    _elastic_report_and_wait_train_ready(phase_timeout)
    logger.warning("[elastic] Replacement node: param sync complete, joining training loop")
    elastic_mark_post_rebuild_pending(resume_iteration)


def _select_dp_sync_src_rank(dp_group, replacement_rank: int) -> int:
    """Choose a live source rank inside the caller's DP group."""
    group_ranks = list(dist.get_process_group_ranks(dp_group))
    if not group_ranks:
        raise RuntimeError("[elastic] Cannot sync params: empty DP group")

    if replacement_rank >= 0:
        for candidate in group_ranks:
            if candidate != replacement_rank:
                return candidate

    return group_ranks[0]


def _iter_megatron_optimizers(optimizer):
    if optimizer is None:
        return
    chained_optimizers = getattr(optimizer, "chained_optimizers", None)
    if chained_optimizers is not None:
        for inner in chained_optimizers:
            yield inner
    else:
        yield optimizer


def _is_expert_param_name(name: str) -> bool:
    """Best-effort name fallback for EP-local expert weights."""
    return ".mlp.experts." in name or ".local_experts." in name


def _is_expert_model_param(name: str, param) -> bool:
    """Return True for EP-local expert params that must not be DP-broadcast."""
    if hasattr(param, "allreduce"):
        return not getattr(param, "allreduce", True)
    return _is_expert_param_name(name)


def _build_model_param_name_map(model):
    param_to_name = {}
    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            param_to_name[param] = name
    return param_to_name


def _build_optimizer_param_name_map(megatron_optimizer, model_param_to_name):
    param_to_name = {}
    param_to_is_expert = {}

    float16_groups = getattr(megatron_optimizer, "float16_groups", None)
    main_groups = getattr(megatron_optimizer, "fp32_from_float16_groups", None)
    if float16_groups is not None and main_groups is not None:
        for model_group, main_group in zip(float16_groups, main_groups):
            for model_param, main_param in zip(model_group, main_group):
                name = model_param_to_name.get(model_param)
                if name is not None:
                    param_to_name[main_param] = name
                    param_to_is_expert[main_param] = _is_expert_model_param(name, model_param)

    fp32_groups = getattr(megatron_optimizer, "fp32_from_fp32_groups", None)
    if fp32_groups is not None:
        for group in fp32_groups:
            for param in group:
                name = model_param_to_name.get(param)
                if name is not None:
                    param_to_name[param] = name
                    param_to_is_expert[param] = _is_expert_model_param(name, param)

    # DistributedOptimizer replaces full model parameters with local shards in
    # the inner Adam optimizer.  Map those shard objects through Megatron's
    # model/shard group metadata so local-only checkpoint restore and peer
    # overwrite can classify every shard by provenance.
    distributed_group_pairs = (
        ("model_float16_groups", "shard_fp32_from_float16_groups"),
        ("model_float16_groups", "shard_float16_groups"),
        ("model_fp32_groups", "shard_fp32_groups"),
    )
    for model_groups_name, shard_groups_name in distributed_group_pairs:
        model_groups = getattr(megatron_optimizer, model_groups_name, None)
        shard_groups = getattr(megatron_optimizer, shard_groups_name, None)
        if model_groups is None or shard_groups is None:
            continue
        for model_group, shard_group in zip(model_groups, shard_groups):
            for model_param, shard_param in zip(model_group, shard_group):
                if shard_param is None:
                    continue
                name = model_param_to_name.get(model_param)
                if name is not None:
                    param_to_name[shard_param] = name
                    param_to_is_expert[shard_param] = _is_expert_model_param(
                        name, model_param
                    )

    try:
        inner_optimizer = megatron_optimizer.optimizer
    except (AttributeError, AssertionError):
        inner_optimizer = None

    if inner_optimizer is not None:
        for group in getattr(inner_optimizer, "param_groups", []):
            for param in group.get("params", []):
                if param not in param_to_name:
                    name = model_param_to_name.get(param)
                    if name is not None:
                        param_to_name[param] = name
                        param_to_is_expert[param] = _is_expert_model_param(name, param)

    return inner_optimizer, param_to_name, param_to_is_expert


def _optimizer_role_state_summary(optimizer, model_param_to_name, *, expert):
    """Summarize optimizer coverage for one state provenance class."""
    summary = {
        "expected_params": 0,
        "params_with_adam_state": 0,
        "state_tensors": 0,
        "unmapped_params": 0,
    }
    if optimizer is None:
        return summary

    for megatron_optimizer in _iter_megatron_optimizers(optimizer):
        inner_optimizer, param_to_name, param_to_is_expert = (
            _build_optimizer_param_name_map(megatron_optimizer, model_param_to_name)
        )
        if inner_optimizer is None:
            continue
        for group in getattr(inner_optimizer, "param_groups", []):
            for param in group.get("params", []):
                name = param_to_name.get(param)
                if name is None:
                    summary["unmapped_params"] += 1
                    continue
                is_expert = param_to_is_expert.get(
                    param, _is_expert_param_name(name)
                )
                if bool(is_expert) != bool(expert):
                    continue
                summary["expected_params"] += 1
                state = inner_optimizer.state.get(param, {})
                tensor_keys = {
                    key
                    for key, value in state.items()
                    if isinstance(value, torch.Tensor)
                }
                summary["state_tensors"] += len(tensor_keys)
                if {"exp_avg", "exp_avg_sq"}.issubset(tensor_keys):
                    summary["params_with_adam_state"] += 1
    return summary


def _require_optimizer_role_ready(summary, role):
    """Fail closed when the first post-recovery AdamW step would be partial."""
    checkpoint_step = int(os.environ.get("ELASTIC_CHECKPOINT_STEP", "-1"))
    if summary["expected_params"] <= 0:
        raise RuntimeError(
            f"[elastic] state contract failed: no mapped {role} optimizer params"
        )
    if summary["unmapped_params"]:
        raise RuntimeError(
            "[elastic] state contract failed: optimizer metadata mapping is "
            f"incomplete ({summary['unmapped_params']} unmapped params)"
        )
    if (
        checkpoint_step > 0
        and summary["params_with_adam_state"] != summary["expected_params"]
    ):
        raise RuntimeError(
            f"[elastic] state contract failed: {role} Adam state is partial "
            f"({summary['params_with_adam_state']}/"
            f"{summary['expected_params']} params ready)"
        )


def _get_local_distributed_optimizer_checkpoint_name():
    from megatron.training import get_args
    from megatron.training.checkpointing import (
        get_checkpoint_name,
        get_checkpoint_tracker_filename,
        get_distributed_optimizer_checkpoint_name,
        isfile,
        read_metadata,
    )

    args = get_args()
    if not getattr(args, "use_distributed_optimizer", False):
        return None

    load_dir = getattr(args, "load", None)
    if load_dir is None:
        return None

    tracker_filename = get_checkpoint_tracker_filename(load_dir)
    if not isfile(tracker_filename):
        logger.warning(
            "[elastic] Replacement node: no checkpoint tracker at %s; "
            "expert optimizer state will not be loaded",
            tracker_filename,
        )
        return None

    iteration, release = read_metadata(tracker_filename)
    if getattr(args, "ckpt_step", None):
        iteration = args.ckpt_step

    model_checkpoint_name = get_checkpoint_name(load_dir, iteration, release)
    optim_checkpoint_name = get_distributed_optimizer_checkpoint_name(model_checkpoint_name)
    if not isfile(optim_checkpoint_name):
        logger.warning(
            "[elastic] Replacement node: distributed optimizer checkpoint %s "
            "does not exist; expert optimizer state will not be loaded",
            optim_checkpoint_name,
        )
        return None

    return optim_checkpoint_name


def _ensure_distributed_optimizer_tensor_state(megatron_optimizer):
    inner_optimizer = getattr(megatron_optimizer, "optimizer", None)
    if inner_optimizer is None:
        return False

    has_tensor_state = any(
        any(isinstance(val, torch.Tensor) for val in state.values())
        for state in getattr(inner_optimizer, "state", {}).values()
    )
    if has_tensor_state:
        return True

    init_fn = getattr(megatron_optimizer, "_init_optimizer_states_with_dummy_values", None)
    if init_fn is None:
        return False

    init_fn()
    return any(
        any(isinstance(val, torch.Tensor) for val in state.values())
        for state in getattr(inner_optimizer, "state", {}).values()
    )


def _state_dict_for_megatron_optimizer(all_states, state_index):
    if isinstance(all_states, list):
        if state_index >= len(all_states):
            return None, state_index + 1
        return all_states[state_index], state_index + 1
    return all_states, state_index + 1


@torch.no_grad()
def _copy_expert_state_from_dp_zero_world_tensors(
    megatron_optimizer, state_dict, model_param_to_name
):
    if not state_dict:
        return 0, 0, 0
    if not hasattr(megatron_optimizer, "gbuf_ranges"):
        return 0, 0, 0
    if not hasattr(megatron_optimizer, "_set_main_param_and_optimizer_states"):
        return 0, 0, 0

    data_parallel_group = getattr(megatron_optimizer, "data_parallel_group", None)
    if data_parallel_group is None:
        return 0, 0, 0

    data_parallel_world_size = data_parallel_group.size()
    data_parallel_rank = data_parallel_group.rank()
    loaded_params = 0
    loaded_tensors = 0
    skipped_params = 0

    split_if_needed = getattr(megatron_optimizer, "split_state_dict_if_needed", None)
    if split_if_needed is not None:
        split_if_needed(state_dict)

    for gbuf_idx, gbuf_range_maps in enumerate(megatron_optimizer.gbuf_ranges):
        if gbuf_idx not in state_dict:
            continue
        for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
            if dtype not in state_dict[gbuf_idx]:
                continue
            dtype_state = state_dict[gbuf_idx][dtype]
            offset_in_world_tensors = 0

            for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                bucket = megatron_optimizer.buffers[gbuf_idx].buckets[bucket_idx]
                gbuf_world_numel = bucket.grad_data.numel()
                if gbuf_world_numel % data_parallel_world_size != 0:
                    raise RuntimeError(
                        "[elastic] Cannot load expert optimizer state: bucket size "
                        f"{gbuf_world_numel} is not divisible by DP size {data_parallel_world_size}"
                    )

                gbuf_local_numel = gbuf_world_numel // data_parallel_world_size
                gbuf_world_numel_unpadded = bucket.numel_unpadded
                local_start = data_parallel_rank * gbuf_local_numel
                local_end = local_start + gbuf_local_numel
                local_shards = {}

                for key in ("param", "exp_avg", "exp_avg_sq"):
                    world_tensors = dtype_state.get(key)
                    if world_tensors is None:
                        continue
                    start = offset_in_world_tensors
                    end = start + gbuf_world_numel_unpadded
                    if end > world_tensors.numel():
                        raise RuntimeError(
                            "[elastic] Cannot load expert optimizer state: "
                            f"checkpoint tensor {key} for gbuf {gbuf_idx} bucket "
                            f"{bucket_idx} is too short ({world_tensors.numel()} < {end})"
                        )
                    world_tensor = world_tensors[start:end]
                    world_tensor = torch.nn.functional.pad(
                        world_tensor, (0, gbuf_world_numel - gbuf_world_numel_unpadded)
                    )
                    local_shards[key] = world_tensor[local_start:local_end]

                offset_in_world_tensors += gbuf_world_numel_unpadded
                if set(local_shards) != {"param", "exp_avg", "exp_avg_sq"}:
                    skipped_params += len(gbuf_range_map["param_map"])
                    continue

                for model_param, param_range_map in gbuf_range_map["param_map"].items():
                    name = model_param_to_name.get(model_param)
                    if name is None or not _is_expert_model_param(name, model_param):
                        continue

                    gbuf_local_start = param_range_map["gbuf_local"].start
                    gbuf_local_end = param_range_map["gbuf_local"].end
                    tensors = {
                        key: shard[gbuf_local_start:gbuf_local_end]
                        for key, shard in local_shards.items()
                    }
                    megatron_optimizer._set_main_param_and_optimizer_states(model_param, tensors)
                    loaded_params += 1
                    loaded_tensors += len(tensors)

    return loaded_params, loaded_tensors, skipped_params


def _load_expert_optimizer_state_from_checkpoint(optimizer, model_param_to_name):
    if optimizer is None:
        raise RuntimeError("[elastic] state contract failed: optimizer is missing")

    from megatron.training import get_args

    args = get_args()
    if not getattr(args, "use_distributed_optimizer", False):
        # checkpointing.py has already loaded the complete local optimizer
        # shard.  Peer sync will overwrite only the non-expert half at step t.
        summary = _optimizer_role_state_summary(
            optimizer, model_param_to_name, expert=True
        )
        _require_optimizer_role_ready(summary, "expert")
        summary["source"] = "checkpoint_full_local_shard"
        return summary

    optim_checkpoint_name = _get_local_distributed_optimizer_checkpoint_name()
    if optim_checkpoint_name is None:
        raise RuntimeError(
            "[elastic] state contract failed: distributed expert optimizer "
            "checkpoint is unavailable"
        )

    try:
        all_states = torch.load(optim_checkpoint_name, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            "[elastic] failed to load distributed optimizer checkpoint "
            f"{optim_checkpoint_name}"
        ) from exc

    total_loaded_params = 0
    total_loaded_tensors = 0
    total_skipped_params = 0
    unsupported = 0
    state_index = 0

    for megatron_optimizer in _iter_megatron_optimizers(optimizer):
        if not hasattr(megatron_optimizer, "load_parameter_state_from_dp_zero"):
            unsupported += 1
            continue

        state_dict, state_index = _state_dict_for_megatron_optimizer(all_states, state_index)
        if state_dict is None:
            continue

        if not _ensure_distributed_optimizer_tensor_state(megatron_optimizer):
            unsupported += 1
            continue

        loaded_params, loaded_tensors, skipped_params = (
            _copy_expert_state_from_dp_zero_world_tensors(
                megatron_optimizer, state_dict, model_param_to_name
            )
        )
        total_loaded_params += loaded_params
        total_loaded_tensors += loaded_tensors
        total_skipped_params += skipped_params

    logger.warning(
        "[elastic] Replacement node: local expert optimizer load complete "
        "(file=%s, expert_params=%d, tensors=%d, skipped=%d, unsupported=%d)",
        optim_checkpoint_name,
        total_loaded_params,
        total_loaded_tensors,
        total_skipped_params,
        unsupported,
    )
    if unsupported:
        raise RuntimeError(
            "[elastic] state contract failed: distributed expert optimizer "
            f"loader does not support {unsupported} optimizer wrapper(s)"
        )
    summary = _optimizer_role_state_summary(
        optimizer, model_param_to_name, expert=True
    )
    summary.update(
        {
            "source": "checkpoint_distributed_local_only",
            "loaded_params": total_loaded_params,
            "loaded_tensors": total_loaded_tensors,
            "skipped_params": total_skipped_params,
        }
    )
    _require_optimizer_role_ready(summary, "expert")
    if total_loaded_params != summary["expected_params"]:
        raise RuntimeError(
            "[elastic] state contract failed: distributed expert optimizer "
            f"coverage is partial ({total_loaded_params}/"
            f"{summary['expected_params']} params loaded)"
        )
    return summary


class _PeerSyncStream:
    def __init__(self, src_rank: int, dst_rank: int):
        self.src_rank = src_rank
        self.dst_rank = dst_rank
        self.rank = dist.get_rank()
        self.sock = None
        self.server_sock = None
        self.timeout = float(os.environ.get("ELASTIC_PEER_SYNC_TIMEOUT", "900"))
        self.chunk_mb = int(os.environ.get("ELASTIC_PARAM_SYNC_CHUNK_MB", "256"))

    def __enter__(self):
        if self.rank == self.src_rank:
            self._open_source()
        elif self.rank == self.dst_rank:
            self._open_destination()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        for sock in (self.sock, self.server_sock):
            if sock is None:
                continue
            try:
                sock.close()
            except OSError:
                pass
        self.sock = None
        self.server_sock = None

    def _peer_id(self) -> str:
        return ":".join(
            [
                os.environ.get("MASTER_PORT", "0"),
                os.environ.get("ELASTIC_RESUME_ITERATION", "-1"),
                str(self.src_rank),
                str(self.dst_rank),
            ]
        )

    def _open_source(self):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(("0.0.0.0", 0))
        self.server_sock.listen(1)
        self.server_sock.settimeout(self.timeout)
        _, port = self.server_sock.getsockname()
        peer_id = self._peer_id()
        endpoint_msg = {
            "type": "peer_sync_endpoint",
            "node_rank": int(os.environ.get("NODE_RANK", "-1")),
            "rank": self.rank,
            "peer_id": peer_id,
            "port": int(port),
            "src_rank": self.src_rank,
            "dst_rank": self.dst_rank,
        }
        endpoint_msg.update(_elastic_recovery_epoch_payload())
        if not _send_one_shot_to_watcher(endpoint_msg):
            raise RuntimeError(f"[elastic] failed publishing peer sync endpoint {peer_id}")
        logger.warning(
            "[elastic] Rank %d: peer TCP sync listening id=%s port=%d dst=%d",
            self.rank,
            peer_id,
            port,
            self.dst_rank,
        )
        self.sock, addr = self.server_sock.accept()
        self.sock.settimeout(self.timeout)
        logger.warning("[elastic] Rank %d: peer TCP sync accepted %s", self.rank, addr)

    def _open_destination(self):
        peer_id = self._peer_id()
        endpoint = _elastic_wait_for_peer_sync_endpoint(peer_id, self.timeout)
        if endpoint is None:
            raise RuntimeError(f"[elastic] timed out waiting for peer sync endpoint {peer_id}")
        host = endpoint.get("host")
        port = int(endpoint.get("port"))
        last_error = None
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                self.sock = socket.create_connection((host, port), timeout=10.0)
                self.sock.settimeout(self.timeout)
                logger.warning(
                    "[elastic] Rank %d: peer TCP sync connected id=%s endpoint=%s:%d",
                    self.rank,
                    peer_id,
                    host,
                    port,
                )
                return
            except OSError as e:
                last_error = e
                time.sleep(0.5)
        raise RuntimeError(
            f"[elastic] failed connecting to peer sync endpoint {host}:{port}: {last_error}"
        )

    def send_json(self, payload: dict):
        self.send_blob(json.dumps(payload, sort_keys=True).encode("utf-8"))

    def recv_json(self) -> dict:
        return json.loads(self.recv_blob().decode("utf-8"))

    def send_blob(self, payload: bytes):
        if self.sock is None:
            raise RuntimeError("[elastic] peer sync socket is not connected")
        self.sock.sendall(struct.pack("!Q", len(payload)))
        self.sock.sendall(payload)

    def recv_blob(self) -> bytes:
        if self.sock is None:
            raise RuntimeError("[elastic] peer sync socket is not connected")
        header = self._recvall(8)
        (size,) = struct.unpack("!Q", header)
        return self._recvall(size)

    def _recvall(self, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            chunk = self.sock.recv(min(remaining, 8 * 1024 * 1024))
            if not chunk:
                raise RuntimeError("[elastic] peer sync socket closed during transfer")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def send_tensor(self, tensor, label: str):
        if tensor.numel() == 0:
            self.send_json({"label": label, "chunks": 0, "numel": 0})
            return
        flat = tensor.contiguous().view(-1)
        max_elems = max(1, (max(1, self.chunk_mb) * 1024 * 1024) // max(1, tensor.element_size()))
        chunks = (flat.numel() + max_elems - 1) // max_elems
        self.send_json(
            {
                "label": label,
                "chunks": chunks,
                "numel": flat.numel(),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        )
        logger.info(
            "[elastic] Rank %d: peer-tcp-send %s to rank %d "
            "(numel=%d, dtype=%s, chunks=%d, chunk_mb=%d)",
            self.rank,
            label,
            self.dst_rank,
            tensor.numel(),
            tensor.dtype,
            chunks,
            self.chunk_mb,
        )
        for start in range(0, flat.numel(), max_elems):
            cpu_chunk = flat[start : start + max_elems].detach().to("cpu", non_blocking=False).contiguous()
            buffer = io.BytesIO()
            torch.save(cpu_chunk, buffer)
            self.send_blob(buffer.getvalue())

    def recv_tensor_into(self, tensor, label: str):
        meta = self.recv_json()
        if meta.get("label") != label:
            raise RuntimeError(
                f"[elastic] peer tensor label mismatch: expected={label} got={meta.get('label')}"
            )
        if int(meta.get("numel", -1)) != tensor.numel() or meta.get("dtype") != str(tensor.dtype):
            raise RuntimeError(
                f"[elastic] peer tensor metadata mismatch for {label}: "
                f"local numel={tensor.numel()} dtype={tensor.dtype}, remote={meta}"
            )
        if tensor.numel() == 0:
            return
        if tensor.is_contiguous():
            flat = tensor.view(-1)
            needs_copy_back = False
        else:
            flat = tensor.contiguous().view(-1)
            needs_copy_back = True
        offset = 0
        for _ in range(int(meta.get("chunks", 0))):
            payload = self.recv_blob()
            chunk = torch.load(io.BytesIO(payload), map_location="cpu")
            if chunk.dtype != tensor.dtype:
                raise RuntimeError(
                    f"[elastic] peer tensor chunk dtype mismatch for {label}: "
                    f"local={tensor.dtype} remote={chunk.dtype}"
                )
            end = offset + chunk.numel()
            if end > flat.numel():
                raise RuntimeError(f"[elastic] peer tensor chunk overflow for {label}")
            flat[offset:end].copy_(chunk.to(device=flat.device, non_blocking=False))
            offset = end
        if offset != flat.numel():
            raise RuntimeError(
                f"[elastic] peer tensor underflow for {label}: got={offset} expected={flat.numel()}"
            )
        if needs_copy_back:
            tensor.copy_(flat.view_as(tensor))


def _sync_optimizer_state_tensor_peer(val, src_rank, dst_rank, device, peer_stream):
    if peer_stream is not None:
        _sync_tensor_peer_chunked(
            val, src_rank, dst_rank, label="optimizer-state", peer_stream=peer_stream
        )
        return

    if val.is_cuda:
        _sync_tensor_peer_chunked(
            val, src_rank, dst_rank, label="optimizer-state", peer_stream=peer_stream
        )
        return

    if isinstance(device, torch.device) and device.type == "cuda":
        broadcast_device = device
    else:
        broadcast_device = torch.device("cuda", torch.cuda.current_device())

    tmp = val.to(device=broadcast_device, non_blocking=True)
    _sync_tensor_peer_chunked(
        tmp, src_rank, dst_rank, label="optimizer-state", peer_stream=peer_stream
    )
    if dist.get_rank() == dst_rank:
        val.copy_(tmp.to(device=val.device))


def _stable_hash_int(text: str) -> int:
    # FNV-1a 63-bit hash: stable across Python processes and cheap enough per tensor.
    value = 1469598103934665603
    for byte in text.encode("utf-8", errors="replace"):
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return value & ((1 << 63) - 1)


def _dtype_code(dtype) -> int:
    return _stable_hash_int(str(dtype)) & ((1 << 31) - 1)


def _shape_hash(shape) -> int:
    return _stable_hash_int(",".join(str(dim) for dim in shape))


def _sync_tensor_peer_chunked(tensor, src_rank, dst_rank, label: str, peer_stream=None):
    rank = dist.get_rank()
    if rank not in (src_rank, dst_rank):
        return
    if tensor.numel() == 0:
        return
    if peer_stream is None:
        raise RuntimeError("[elastic] peer tensor sync requires a TCP peer stream")
    if rank == src_rank:
        peer_stream.send_tensor(tensor, label)
    else:
        peer_stream.recv_tensor_into(tensor, label)


def _validate_param_peer_manifest(name, param, index, src_rank, dst_rank, peer_stream=None):
    rank = dist.get_rank()
    if rank not in (src_rank, dst_rank):
        return

    local_meta_obj = {
        "index": index,
        "numel": param.numel(),
        "shape": list(param.shape),
        "shape_hash": _shape_hash(param.shape),
        "dtype": str(param.dtype),
        "dtype_code": _dtype_code(param.dtype),
        "name_hash": _stable_hash_int(name),
    }
    if peer_stream is None:
        raise RuntimeError("[elastic] peer manifest sync requires a TCP peer stream")
    if rank == src_rank:
        peer_stream.send_json(local_meta_obj)
        status_obj = peer_stream.recv_json()
        status = int(status_obj.get("status", 1))
    else:
        src_meta_obj = peer_stream.recv_json()
        status = 0 if local_meta_obj == src_meta_obj else 1
        peer_stream.send_json({"status": status})
        if status != 0:
            logger.error(
                "[elastic] Rank %d: dense param manifest mismatch at index=%d "
                "(local name=%s shape=%s dtype=%s meta=%s src_meta=%s)",
                rank,
                index,
                name,
                tuple(param.shape),
                param.dtype,
                local_meta_obj,
                src_meta_obj,
            )

    if status != 0:
        logger.error(
            "[elastic] Rank %d: dense param manifest mismatch at index=%d "
            "(local name=%s shape=%s dtype=%s)",
            rank,
            index,
            name,
            tuple(param.shape),
            param.dtype,
        )
        raise RuntimeError(
            "[elastic] dense param sync manifest mismatch; "
            "replacement and source ranks do not have identical dense parameter order"
        )


def _sync_non_expert_optimizer_state_peer(
    optimizer, model_param_to_name, sync_src_rank, replacement_rank, peer_stream
):
    rank = dist.get_rank()
    main_param_count = 0
    state_tensor_count = 0
    skipped_state_count = 0
    unmapped_count = 0
    params_with_adam_state = 0
    unsupported_count = 0

    if rank not in (sync_src_rank, replacement_rank):
        return None
    if peer_stream is None:
        raise RuntimeError("[elastic] non-expert optimizer sync requires a TCP peer stream")

    for megatron_optimizer in _iter_megatron_optimizers(optimizer):
        inner_optimizer, optim_param_to_name, optim_param_to_is_expert = (
            _build_optimizer_param_name_map(megatron_optimizer, model_param_to_name)
        )
        if inner_optimizer is None:
            unsupported_count += 1
            continue

        for group in getattr(inner_optimizer, "param_groups", []):
            for param in group.get("params", []):
                name = optim_param_to_name.get(param)
                if name is None:
                    unmapped_count += 1
                    continue
                if optim_param_to_is_expert.get(param, _is_expert_param_name(name)):
                    continue

                _validate_param_peer_manifest(
                    f"optimizer:{name}",
                    param.data,
                    main_param_count + 1,
                    sync_src_rank,
                    replacement_rank,
                    peer_stream=peer_stream,
                )
                _sync_optimizer_state_tensor_peer(
                    param.data, sync_src_rank, replacement_rank, param.device, peer_stream
                )
                main_param_count += 1

                state = inner_optimizer.state.get(param, {})
                tensor_state = [
                    (str(key), value)
                    for key, value in state.items()
                    if isinstance(value, torch.Tensor)
                ]
                tensor_state.sort(key=lambda item: item[0])
                scalar_state = {
                    str(key): value
                    for key, value in state.items()
                    if isinstance(value, (bool, int, float, str))
                }
                local_state_manifest = [
                    {
                        "key": key,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                    for key, value in tensor_state
                ]
                if rank == sync_src_rank:
                    peer_stream.send_json(
                        {
                            "state_manifest": local_state_manifest,
                            "scalar_state": scalar_state,
                        }
                    )
                    status_obj = peer_stream.recv_json()
                    manifest_matches = bool(status_obj.get("manifest_matches", False))
                else:
                    src_state_obj = peer_stream.recv_json()
                    manifest_matches = (
                        local_state_manifest == src_state_obj.get("state_manifest", [])
                    )
                    if manifest_matches:
                        for key, value in src_state_obj.get("scalar_state", {}).items():
                            state[key] = value
                    peer_stream.send_json({"manifest_matches": manifest_matches})
                if not manifest_matches:
                    raise RuntimeError(
                        "[elastic] non-expert optimizer state manifest mismatch "
                        f"for {name}: local={local_state_manifest}"
                    )
                if not tensor_state:
                    skipped_state_count += 1
                    continue

                state_keys = {key for key, _ in tensor_state}
                if {"exp_avg", "exp_avg_sq"}.issubset(state_keys):
                    params_with_adam_state += 1
                for key, val in tensor_state:
                    _sync_optimizer_state_tensor_peer(
                        val, sync_src_rank, replacement_rank, param.device, peer_stream
                    )
                    state_tensor_count += 1

    logger.warning(
        "[elastic] Rank %d: non-expert optimizer sync complete "
        "(main_params=%d, state_tensors=%d, skipped=%d, unmapped=%d)",
        rank,
        main_param_count,
        state_tensor_count,
        skipped_state_count,
        unmapped_count,
    )
    summary = {
        "source": "current_step_dense_dp_peer",
        "expected_params": main_param_count,
        "params_with_adam_state": params_with_adam_state,
        "state_tensors": state_tensor_count,
        "skipped_state_params": skipped_state_count,
        "unmapped_params": unmapped_count,
        "unsupported_wrappers": unsupported_count,
    }
    if unsupported_count:
        raise RuntimeError(
            "[elastic] state contract failed: non-expert optimizer peer sync "
            f"does not support {unsupported_count} optimizer wrapper(s)"
        )
    _require_optimizer_role_ready(summary, "non-expert")
    return summary


def _sync_params_to_new_rank(
    model, optimizer, replacement_rank: int = -1, model_param_to_name=None
):
    """Restore dense/non-expert model and optimizer state from a DP peer.

    After group rebuild, the replacement rank restores EP-local expert weights
    and expert optimizer state from its checkpoint shard. Dense/non-expert
    parameters are DP-replicated, so we transfer model weights, main params,
    and tensor optimizer state from a surviving rank within each DP group over
    a dedicated TCP peer stream instead of the rebuilding NCCL process group.
    """
    from megatron.core import parallel_state as mpu

    sync_group = mpu.get_data_parallel_group()
    sync_group_ranks = list(dist.get_process_group_ranks(sync_group))
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    dp_rank = mpu.get_data_parallel_rank()
    if replacement_rank not in sync_group_ranks:
        logger.info(
            "[elastic] Rank %d: skipping param sync for pp_rank=%d; "
            "replacement rank %d is not in DP group %s",
            dist.get_rank(),
            pp_rank,
            replacement_rank,
            sync_group_ranks,
        )
        return None

    sync_src_rank = _select_dp_sync_src_rank(sync_group, replacement_rank)

    rank = dist.get_rank()
    if rank not in (sync_src_rank, replacement_rank):
        logger.info(
            "[elastic] Rank %d: skipping peer param sync for pp_rank=%d, dp_rank=%d; "
            "src=%d replacement=%d",
            rank,
            pp_rank,
            dp_rank,
            sync_src_rank,
            replacement_rank,
        )
        return None

    logger.info(
        f"[elastic] Rank {rank}: syncing dense params with peer transfer "
        f"(pp_rank={pp_rank}, dp_rank={dp_rank}, group={sync_group_ranks}, "
        f"src={sync_src_rank}, replacement={replacement_rank})"
    )

    if model_param_to_name is None:
        model_param_to_name = _build_model_param_name_map(model)
    with _PeerSyncStream(sync_src_rank, replacement_rank) as peer_stream:
        dense_count = 0
        expert_count = 0
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                if _is_expert_model_param(name, param):
                    expert_count += 1
                    continue
                dense_count += 1
                _validate_param_peer_manifest(
                    name,
                    param.data,
                    dense_count,
                    sync_src_rank,
                    replacement_rank,
                    peer_stream=peer_stream,
                )
                if dense_count <= 3 or param.data.numel() * param.data.element_size() >= 128 * 1024 * 1024:
                    logger.info(
                        "[elastic] Rank %d: syncing dense param %d name=%s "
                        "shape=%s dtype=%s",
                        rank,
                        dense_count,
                        name,
                        tuple(param.data.shape),
                        param.data.dtype,
                    )
                _sync_tensor_peer_chunked(
                    param.data,
                    sync_src_rank,
                    replacement_rank,
                    label=f"dense-param:{name}",
                    peer_stream=peer_stream,
                )

        logger.warning(
            "[elastic] Rank %d: dense model param sync complete "
            "(synced=%d, expert_from_ckpt=%d)",
            rank,
            dense_count,
            expert_count,
        )
        optimizer_summary = _sync_non_expert_optimizer_state_peer(
            optimizer, model_param_to_name, sync_src_rank, replacement_rank, peer_stream
        )

    logger.info(f"[elastic] Rank {rank}: param sync complete")
    return {
        "dense_model_params": dense_count,
        "checkpoint_expert_model_params": expert_count,
        "non_expert_optimizer": optimizer_summary,
        "source_rank": sync_src_rank,
        "replacement_rank": replacement_rank,
    }


def is_rebuild_mode() -> bool:
    """Check if this process is a replacement worker in rebuild mode."""
    return os.environ.get("ELASTIC_REBUILD_MODE") == "1"

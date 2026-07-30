#!/usr/bin/env python3
"""Megatron compatibility hot-spare watcher daemon.

Runs on the spare node (NODE_RANK=8). Does NOT join torch.distributed.
Responsibilities:
  1. Accept heartbeat connections from fault-isolated node launchers
  2. Detect node failure via heartbeat timeout
  3. Optionally inject faults: at a given step, PAUSE all → kill one rank
  4. Signal all training ranks to pause at next safe point
  5. Wait for all nodes to confirm paused (destroy_process_group done)
  6. The target node kills the selected local rank
  7. Launch 1 replacement process on spare node with the dead rank
  8. Coordinate group rebuild (all ranks re-init_process_group)

Fault injection flow (safe, no NCCL timeout):
  1. Watcher sees step >= fault_inject_step
  2. Watcher sends PAUSE to ALL training nodes (including target)
  3. All nodes reach safe point → destroy_process_group → send ready_to_rebuild
  4. Target node kills the selected local rank before reporting ready
  5. Watcher receives 8/8 ready_to_rebuild
  6. Watcher launches 1 spare process on this node
  7. Watcher sends rebuild signal to all surviving ranks
  8. All ranks re-init_process_group → sync params → resume

Protocol (TCP, JSON lines):
  - Training → Watcher: {"type": "heartbeat", "node_rank": N, "step": S, "step_tag": T, "train_phase": P}
  - Training → Watcher: {"type": "ready_to_rebuild", "node_rank": N, "step": S, "step_tag": T, "train_phase": P}
  - Training → Watcher: {"type": "nccl_error", "node_rank": N, "error": "..."}
  - Watcher → Training: {"type": "pause", "failed_node": N}
  - Watcher → Training: {"type": "rebuild", "failed_node": N, "new_master_addr": ..., "new_master_port": ..., "killed_rank": R}
  - Watcher → Training: {"type": "kill_rank", "target_node": N, "local_rank": R}
  - Watcher → Training: {"type": "fallback_relaunch", "action": "checkpoint_relaunch|abort_training", "recovery_epoch": E, "reason": "..."}
"""

import argparse
import hashlib
import json
import logging
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[watcher %(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("elastic_watcher")


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return _str_to_bool(value)
    except argparse.ArgumentTypeError:
        log.warning("Invalid %s=%s; using default=%s", name, value, default)
        return default

_PHASE_ORDER = {
    "zero2_memory_quiesce_ready": 5,
    "init_pg_start": 10,
    "rebuild_store_ready": 15,
    "standby_activated": 12,
    "pg_ready": 20,
    "mpu_init_start": 30,
    "mpu_group_start": 32,
    "mpu_group_done": 32,
    "mpu_init_done": 35,
    "rebind_start": 37,
    "mpu_ready": 40,
    "cold_start_deps_ready": 50,
    "checkpoint_loaded": 55,
    "model_optimizer_ready": 60,
    "state_contract_start": 62,
    "state_contract_error": 63,
    "param_sync_start": 70,
    "param_sync_done": 80,
    "zero2_memory_reconfigured": 81,
    "resume_state_applied": 82,
    "state_contract_ready": 83,
    "rerun_state_reset": 84,
    "comm_warmup_start": 85,
    "comm_warmup_done": 88,
    "train_ready": 90,
    "data_ready": 94,
    "train_loop_entered": 100,
    "post_rebuild_iteration_ready": 110,
    "rerun_state_contract_start": 111,
    "rerun_state_contract_ready": 112,
    "rerun_state_contract_error": 113,
    "forward_backward_start": 120,
    "moe_first_collective_start": 122,
    "moe_first_collective_done": 124,
    "moe_first_collective_error": 126,
    "moe_first_collective_timeout": 128,
    "forward_backward_done": 130,
    "optimizer_pg_contract_ready": 135,
    "optimizer_pg_contract_error": 136,
    "optimizer_step_start": 140,
    "optimizer_step_done": 150,
    "optimizer_skipped": 155,
    "train_step_finalize_done": 160,
    "training_log_start": 170,
    "training_log_done": 180,
    "post_step_callbacks_start": 190,
    "post_step_callbacks_done": 200,
    "checkpoint_exit_start": 210,
    "checkpoint_exit_done": 220,
    "post_rebuild_commit_ready": 230,
}


class ElasticWatcher:
    """TCP-based watcher that monitors training nodes and coordinates recovery."""

    def __init__(self, args):
        self.port = args.port
        self.training_nnodes = args.training_nnodes
        self.nproc_per_node = args.nproc_per_node
        self.master_addr = args.master_addr
        self.master_port = args.master_port
        self.fault_dir = Path(args.fault_dir)
        self.heartbeat_timeout = args.heartbeat_timeout
        self.startup_heartbeat_timeout = args.startup_heartbeat_timeout
        self.forward_heartbeat_timeout = args.forward_heartbeat_timeout
        self.checkpoint_heartbeat_timeout = args.checkpoint_heartbeat_timeout
        self.disconnect_grace_timeout = args.disconnect_grace_timeout
        self.recovery_stall_timeout = max(
            0.0,
            float(os.environ.get("ELASTIC_RECOVERY_STALL_TIMEOUT_SECONDS", "0")),
        )
        self.fallback_relaunch = args.fallback_relaunch
        self.fallback_exit_code = args.fallback_exit_code
        self.fallback_restart_standby = args.fallback_restart_standby
        self.recovery_abort_exit_code = getattr(args, "recovery_abort_exit_code", 76)

        # Fault injection config
        self.fault_inject_step = args.fault_inject_step
        self.fault_inject_node = args.fault_inject_node
        self.fault_inject_local_rank = getattr(args, 'fault_inject_local_rank', 0)
        self.fault_injected = False

        # State
        self.running = True
        self.exit_code = 0
        self.node_connections = {}  # node_rank -> socket
        self.launcher_connections = {}  # node_rank -> launcher control socket
        self.last_heartbeat = {}   # node_rank -> timestamp
        self.node_disconnected_at = {}  # node_rank -> TCP EOF/reset timestamp
        self.node_steps = {}       # node_rank -> last reported step
        self.node_step_tags = {}   # node_rank -> FlashRecovery step tag
        self.node_train_phases = {}  # node_rank -> forward_backward / optimizer_step / step_complete
        self.node_rank_states = {}  # node_rank -> per-rank state from launcher agent
        self.node_control_owners = {}
        self.startup_manifests = {}  # attempt -> node_rank -> manifest
        self.startup_connections = {}  # attempt -> node_rank -> socket
        self.startup_rejections = {}  # attempt -> reason
        self.startup_released_attempt = -1
        self.pending_nccl_fallback_nodes = set()
        self.failed_node = None
        self.killed_local_rank = 0
        self.recovery_in_progress = False
        self.rebuild_ready_count = 0
        self.rebuild_ready_nodes = set()
        self.rebuild_ready_steps = {}
        self.rebuild_ready_step_tags = {}
        self.rebuild_ready_phases = {}
        self.rebuild_ready_rank_states = {}
        self.rebuild_ready_quiescence = {}
        self.rebuild_ready_control_owners = {}
        self.rank_quiescence_proof = {"status": "pending"}
        self.recovery_phases = {}
        self.ordinal_barriers = {}
        self.peer_sync_endpoints = {}
        self.recovery_epoch = 0
        self.fallback_initiated = False
        self.rebuild_triggered = False
        self.replacement_ready_event = threading.Event()
        self.standby_proc = None
        self.standby_prearmed = False
        self.standby_prearmed_ready = False
        self.standby_prearmed_epoch = None
        self.standby_prearmed_rank = None
        self.standby_assignment_file = self.fault_dir / "spare_assignment.json"
        self.recovery_descriptor_file = self.fault_dir / "recovery_descriptor.json"
        history_root = Path(os.environ.get("CKPT_DIR", str(self.fault_dir)))
        self.expert_staleness_history_file = (
            history_root / "moegambit_expert_staleness_history.json"
        )
        self.fallback_relaunch_file = self.fault_dir / "fallback_relaunch.json"
        self.recovery_abort_file = self.fault_dir / "recovery_abort.json"
        self.recorded_contract_epochs = set()
        self.lock = threading.Lock()
        self.phase_cv = threading.Condition(self.lock)

        # Server socket
        self.server_sock = None

    def start(self):
        """Main entry point."""
        log.info(f"Starting watcher on port {self.port}")
        log.info(f"Monitoring {self.training_nnodes} training nodes")
        log.info(f"Heartbeat timeout: {self.heartbeat_timeout}s")
        log.info(f"Startup heartbeat timeout: {self.startup_heartbeat_timeout}s")
        log.info(f"Forward/backward heartbeat timeout: {self.forward_heartbeat_timeout}s")
        log.info(f"Checkpoint heartbeat timeout: {self.checkpoint_heartbeat_timeout}s")
        log.info(f"Disconnect grace timeout: {self.disconnect_grace_timeout}s")
        if self.recovery_stall_timeout > 0:
            log.info(
                "Recovery ordinal stall fail-fast: enabled, timeout=%.1fs",
                self.recovery_stall_timeout,
            )
        else:
            log.info("Recovery ordinal stall fail-fast: disabled")
        log.info(
            "Recovery failure policy: fallback_relaunch=%s fallback_exit_code=%s "
            "abort_exit_code=%s restart_standby=%s",
            self.fallback_relaunch,
            self.fallback_exit_code,
            self.recovery_abort_exit_code,
            self.fallback_restart_standby,
        )
        if self.fault_inject_step >= 0:
            log.info(f"Fault injection: kill node {self.fault_inject_node} "
                     f"local_rank {self.fault_inject_local_rank} at step {self.fault_inject_step}")

        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(("0.0.0.0", self.port))
        self.server_sock.listen(self.training_nnodes * self.nproc_per_node + 16)
        self.server_sock.settimeout(1.0)

        # Signal handling
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Start heartbeat checker thread
        checker = threading.Thread(target=self._heartbeat_checker, daemon=True)
        checker.start()

        self._start_standby_worker()

        # Accept loop
        while self.running:
            try:
                conn, addr = self.server_sock.accept()
                conn.settimeout(5.0)
                t = threading.Thread(target=self._handle_connection, args=(conn, addr), daemon=True)
                t.start()
            except socket.timeout:
                continue
            except OSError:
                break

        log.info("Watcher shutting down")
        return self.exit_code

    def _rebuild_master_port(self, recovery_epoch=None):
        if recovery_epoch is None:
            recovery_epoch = self.recovery_epoch
        base = int(
            os.environ.get(
                "ELASTIC_REBUILD_MASTER_PORT_BASE", str(int(self.master_port) + 1)
            )
        )
        return base + max(int(recovery_epoch) - 1, 0)

    def _signal_handler(self, signum, frame):
        log.info(f"Received signal {signum}, shutting down")
        self.running = False
        if self.standby_proc is not None and self.standby_proc.poll() is None:
            try:
                self.standby_proc.terminate()
            except OSError:
                pass

    def _handle_connection(self, conn, addr):
        """Handle a single training node connection."""
        buf = b""
        node_rank = None
        try:
            while self.running:
                try:
                    data = conn.recv(4096)
                    if not data:
                        break
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        msg = json.loads(line.decode())
                        node_rank = self._process_message(msg, conn, addr)
                except socket.timeout:
                    continue
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    log.warning(f"Bad message from {addr}: {e}")
                    continue
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            if node_rank is not None:
                with self.lock:
                    if self.node_connections.get(node_rank) is conn:
                        del self.node_connections[node_rank]
                        self.node_disconnected_at[node_rank] = time.time()
                        log.warning(
                            "Node %s disconnected; allowing %.1fs reconnect grace",
                            node_rank,
                            self.disconnect_grace_timeout,
                        )
                    if self.launcher_connections.get(node_rank) is conn:
                        del self.launcher_connections[node_rank]
            conn.close()

    def _record_startup_manifest_locked(self, node_rank, manifest, conn):
        """Validate one launcher contract and return a release/reject action."""
        try:
            attempt = int(manifest.get("attempt", -1))
            manifest_node = int(manifest.get("node_rank", -1))
            nnodes = int(manifest.get("nnodes", -1))
            nproc = int(manifest.get("nproc_per_node", -1))
            world_size = int(manifest.get("world_size", -1))
        except (TypeError, ValueError):
            attempt = -1
            manifest_node = -1
            nnodes = -1
            nproc = -1
            world_size = -1

        errors = []
        expected_world_size = self.training_nnodes * self.nproc_per_node
        expected = {
            "node_rank": int(node_rank),
            "nnodes": self.training_nnodes,
            "nproc_per_node": self.nproc_per_node,
            "world_size": expected_world_size,
            "master_addr": str(self.master_addr),
            "master_port": str(self.master_port),
        }
        observed = {
            "node_rank": manifest_node,
            "nnodes": nnodes,
            "nproc_per_node": nproc,
            "world_size": world_size,
            "master_addr": str(manifest.get("master_addr", "")),
            "master_port": str(manifest.get("master_port", "")),
        }
        for key, expected_value in expected.items():
            if observed[key] != expected_value:
                errors.append(f"{key}={observed[key]!r} expected={expected_value!r}")
        if attempt < 0:
            errors.append(f"attempt={attempt!r} expected_non_negative")
        if not manifest.get("command_sha256"):
            errors.append("command_sha256 is empty")

        if attempt in self.startup_rejections:
            reason = self.startup_rejections[attempt]
            connections = self.startup_connections.setdefault(attempt, {})
            connections[int(node_rank)] = conn
            return "reject", attempt, list(connections.items()), reason, False
        if errors:
            reason = f"node {node_rank} startup contract mismatch: " + "; ".join(errors)
            self.startup_rejections[attempt] = reason
            connections = self.startup_connections.setdefault(attempt, {})
            connections[int(node_rank)] = conn
            return "reject", attempt, list(connections.items()), reason, False
        if attempt < self.startup_released_attempt:
            reason = (
                f"stale launch attempt={attempt}; "
                f"released_attempt={self.startup_released_attempt}"
            )
            self.startup_rejections[attempt] = reason
            return "reject", attempt, [(int(node_rank), conn)], reason, False
        if attempt == self.startup_released_attempt:
            # Heartbeats from the active attempt can race with fallback shutdown.
            # They are already released and must not be converted into rejections.
            return None

        manifests = self.startup_manifests.setdefault(attempt, {})
        connections = self.startup_connections.setdefault(attempt, {})
        manifests[int(node_rank)] = dict(manifest)
        connections[int(node_rank)] = conn
        if len(manifests) < self.training_nnodes:
            return None

        command_digests = {
            str(item.get("command_sha256", "")) for item in manifests.values()
        }
        if len(command_digests) != 1:
            digest_by_node = {
                rank: str(item.get("command_sha256", ""))[:12]
                for rank, item in sorted(manifests.items())
            }
            reason = f"launcher command mismatch for attempt={attempt}: {digest_by_node}"
            self.startup_rejections[attempt] = reason
            return "reject", attempt, list(connections.items()), reason, False

        restart_standby = self.fallback_initiated and self.fallback_restart_standby
        self.startup_released_attempt = attempt
        self.fallback_initiated = False
        self.recovery_in_progress = False
        self.failed_node = None
        self.rebuild_ready_count = 0
        self.rebuild_ready_nodes = set()
        self.rebuild_ready_steps = {}
        self.rebuild_ready_step_tags = {}
        self.rebuild_ready_phases = {}
        self.rebuild_ready_rank_states = {}
        self.rebuild_ready_quiescence = {}
        self.rebuild_ready_control_owners = {}
        self.rank_quiescence_proof = {"status": "pending"}
        self.recovery_phases = {}
        self.ordinal_barriers = {}
        self.peer_sync_endpoints = {}
        self.rebuild_triggered = False
        self.pending_nccl_fallback_nodes.clear()
        self.node_disconnected_at.clear()
        self.replacement_ready_event.clear()
        self.phase_cv.notify_all()
        return "release", attempt, list(connections.items()), "", restart_standby

    def _complete_startup_action(self, action, attempt, connections, reason, restart_standby):
        payload = {
            "type": "startup_release" if action == "release" else "startup_reject",
            "attempt": attempt,
        }
        if reason:
            payload["reason"] = reason
        encoded = (json.dumps(payload) + "\n").encode()
        for node_rank, conn in connections:
            try:
                conn.sendall(encoded)
            except (BrokenPipeError, OSError) as exc:
                log.warning(
                    "Failed to send %s for attempt=%s to node %s: %s",
                    action,
                    attempt,
                    node_rank,
                    exc,
                )
        if action == "reject":
            log.error("Startup attempt %s rejected: %s", attempt, reason)
            return

        for path in (
            self.fallback_relaunch_file,
            self.recovery_abort_file,
            self.fault_dir / "fallback_relaunch_signal.json",
        ):
            try:
                path.unlink()
            except OSError:
                pass
        log.info(
            "Startup attempt %s released: launchers=%s master=%s:%s world_size=%s",
            attempt,
            sorted(rank for rank, _ in connections),
            self.master_addr,
            self.master_port,
            self.training_nnodes * self.nproc_per_node,
        )
        if restart_standby:
            self._start_standby_worker()

    def _process_message(self, msg, conn, addr=None):
        """Process a message from a training node. Returns node_rank."""
        msg_type = msg.get("type")
        node_rank = msg.get("node_rank")

        if msg_type == "heartbeat":
            startup_action = None
            with self.lock:
                previous_conn = self.node_connections.get(node_rank)
                first_registration = node_rank not in self.last_heartbeat
                self.node_connections[node_rank] = conn
                self.last_heartbeat[node_rank] = time.time()
                self.node_disconnected_at.pop(node_rank, None)
                step = msg.get("step", -1)
                step_tag = msg.get("step_tag", step)
                train_phase = msg.get("train_phase", "unknown")
                self.node_steps[node_rank] = step
                self.node_step_tags[node_rank] = step_tag
                self.node_train_phases[node_rank] = train_phase
                if isinstance(msg.get("rank_states"), dict):
                    self.node_rank_states[node_rank] = msg["rank_states"]
                self.node_control_owners[node_rank] = msg.get(
                    "control_owner", "training_rank"
                )
                if msg.get("control_owner") == "launcher":
                    self.launcher_connections[node_rank] = conn
                connected_nodes = sorted(
                    rank
                    for rank in self.node_connections
                    if isinstance(rank, int) and 0 <= rank < self.training_nnodes
                )
                if (
                    msg.get("control_owner") == "launcher"
                    and isinstance(msg.get("startup_manifest"), dict)
                ):
                    startup_action = self._record_startup_manifest_locked(
                        node_rank, msg["startup_manifest"], conn
                    )

            if first_registration or previous_conn is not conn:
                peer = f"{addr[0]}:{addr[1]}" if addr else "unknown"
                event = "registered" if first_registration else "reconnected"
                log.info(
                    "Node %s %s from %s owner=%s; launchers=%s/%s connected=%s",
                    node_rank,
                    event,
                    peer,
                    msg.get("control_owner", "training_rank"),
                    len(connected_nodes),
                    self.training_nnodes,
                    connected_nodes,
                )

            if startup_action is not None:
                self._complete_startup_action(*startup_action)

            # Check for step-based fault injection
            self._maybe_inject_fault(node_rank, msg.get("step", -1))
            return node_rank

        elif msg_type == "worker_failure":
            local_rank = int(msg.get("local_rank", -1))
            exit_code = int(msg.get("exit_code", 1))
            log.error(
                "WORKER FAILURE: node=%s local_rank=%s global_rank=%s exit_code=%s",
                node_rank,
                local_rank,
                msg.get("global_rank"),
                msg.get("exit_code"),
            )
            with self.lock:
                suppress_failure = self.recovery_in_progress or self.fallback_initiated
            if suppress_failure:
                log.info(
                    "Ignoring cascading worker failure for active recovery/fallback: "
                    "node=%s local_rank=%s exit_code=%s",
                    node_rank,
                    local_rank,
                    exit_code,
                )
            elif local_rank >= 0:
                failure_scope = (
                    "single_rank" if exit_code in {-11, -9, -6} else "software"
                )
                self._handle_fault(
                    node_rank,
                    killed_local_rank=local_rank,
                    failure_scope=failure_scope,
                )
            return node_rank

        elif msg_type == "ready_to_rebuild":
            if self._is_stale_recovery_message(msg, msg_type):
                return node_rank
            should_trigger = False
            with self.lock:
                if node_rank not in self.rebuild_ready_nodes:
                    self.rebuild_ready_nodes.add(node_rank)
                    self.rebuild_ready_count = len(self.rebuild_ready_nodes)
                    self.rebuild_ready_steps[node_rank] = msg.get("step", -1)
                    self.rebuild_ready_step_tags[node_rank] = msg.get(
                        "step_tag", msg.get("step", -1)
                    )
                    self.rebuild_ready_phases[node_rank] = msg.get("train_phase", "unknown")
                    self.rebuild_ready_rank_states[node_rank] = (
                        msg.get("rank_states")
                        if isinstance(msg.get("rank_states"), dict)
                        else {}
                    )
                    self.rebuild_ready_quiescence[node_rank] = (
                        msg.get("quiescence")
                        if isinstance(msg.get("quiescence"), dict)
                        else {}
                    )
                    self.rebuild_ready_control_owners[node_rank] = msg.get(
                        "control_owner", "training_rank"
                    )
                else:
                    log.info(f"Duplicate ready_to_rebuild from node {node_rank}; ignoring")
                total = self.training_nnodes  # All nodes pause (including target)
                log.info(
                    "Node %s ready to rebuild (%s/%s, step=%s, step_tag=%s, phase=%s)",
                    node_rank,
                    self.rebuild_ready_count,
                    total,
                    self.rebuild_ready_steps.get(node_rank, -1),
                    self.rebuild_ready_step_tags.get(node_rank, -1),
                    self.rebuild_ready_phases.get(node_rank, "unknown"),
                )
                if self.rebuild_ready_count >= total and not self.rebuild_triggered:
                    self.rebuild_triggered = True
                    should_trigger = True
            if should_trigger:
                self._trigger_rebuild()
            return node_rank

        elif msg_type == "nccl_error":
            log.warning(
                "Node %s reported NCCL error at step=%s step_tag=%s phase=%s: %s",
                node_rank,
                msg.get("step", -1),
                msg.get("step_tag", msg.get("step", -1)),
                msg.get("train_phase", "unknown"),
                msg.get("error", "?"),
            )
            if not self.recovery_in_progress:
                log.info(
                    "NCCL error report received without a classified worker failure; "
                    "waiting briefly for launcher evidence"
                )
                with self.lock:
                    should_start = node_rank not in self.pending_nccl_fallback_nodes
                    if should_start:
                        self.pending_nccl_fallback_nodes.add(node_rank)
                if should_start:
                    threading.Thread(
                        target=self._defer_unclassified_nccl_fallback,
                        args=(node_rank,),
                        daemon=True,
                    ).start()
            return node_rank

        elif msg_type == "recovery_phase":
            rank = msg.get("rank", "?")
            phase = msg.get("phase", "?")
            role = msg.get("role", "survivor")
            step = msg.get("step", -1)
            key = f"{role}:{rank}"
            if self._is_stale_recovery_message(msg, msg_type):
                return node_rank
            should_fallback = phase in {
                "state_contract_error",
                "moe_first_collective_error",
                "moe_first_collective_timeout",
                "optimizer_pg_contract_error",
                "rerun_state_contract_error",
            }
            record_contract_epoch = None
            with self.lock:
                previous = self.recovery_phases.get(key)
                prev_phase = previous.get("phase") if previous else None
                prev_order = _PHASE_ORDER.get(prev_phase, -1)
                new_order = _PHASE_ORDER.get(phase, -1)
                if previous is not None and new_order < prev_order:
                    log.info(
                        "Ignoring stale recovery phase: role=%s rank=%s "
                        "node=%s step=%s phase=%s current=%s",
                        role, rank, node_rank, step, phase, prev_phase,
                    )
                else:
                    extra = {
                        k: v
                        for k, v in msg.items()
                        if k
                        not in (
                            "type",
                            "node_rank",
                            "rank",
                            "role",
                            "phase",
                            "step",
                        )
                    }
                    self.recovery_phases[key] = {
                        "phase": phase,
                        "step": step,
                        "timestamp": time.time(),
                        "node_rank": node_rank,
                        "role": role,
                        "extra": extra,
                    }
                if (
                    role == "replacement"
                    and phase == "init_pg_start"
                    and msg.get("standby_prearmed") is True
                ):
                    self.standby_prearmed_ready = True
                if role == "replacement" and phase in ("init_pg_start", "pg_ready"):
                    self.replacement_ready_event.set()
                if (
                    phase == "post_rebuild_commit_ready"
                    and self._phase_count_locked("post_rebuild_commit_ready")
                    >= self.training_nnodes * self.nproc_per_node
                ):
                    if self.recovery_epoch not in self.recorded_contract_epochs:
                        self.recorded_contract_epochs.add(self.recovery_epoch)
                        record_contract_epoch = self.recovery_epoch
                    self.recovery_in_progress = False
                    self.failed_node = None
                    self.rebuild_ready_count = 0
                    self.rebuild_ready_nodes = set()
                    self.rebuild_ready_steps = {}
                    self.rebuild_ready_step_tags = {}
                    self.rebuild_ready_phases = {}
                    self.ordinal_barriers = {}
                    log.info(
                        "All training ranks acknowledged the completed post-rebuild "
                        "iteration; recovery contract committed"
                    )
                self.phase_cv.notify_all()
            extra_text = self._format_recovery_phase_extra(msg)
            if extra_text:
                log.info(
                    "Recovery phase: role=%s rank=%s node=%s step=%s phase=%s %s",
                    role, rank, node_rank, step, phase, extra_text,
                )
            else:
                log.info(
                    "Recovery phase: role=%s rank=%s node=%s step=%s phase=%s",
                    role, rank, node_rank, step, phase,
                )
            if record_contract_epoch is not None:
                self._record_successful_hybrid_contract(record_contract_epoch)
            if should_fallback:
                self._initiate_fallback_relaunch(
                    f"recovery_phase:{phase}",
                    {
                        "role": role,
                        "rank": rank,
                        "node_rank": node_rank,
                        "step": step,
                        "phase": phase,
                    },
                )
            return node_rank

        elif msg_type == "wait_phase":
            if self._is_stale_recovery_message(msg, msg_type):
                self._send_json_response(conn, {
                    "type": "wait_phase_result",
                    "role": msg.get("role", "replacement"),
                    "rank": msg.get("rank", "?"),
                    "phase": msg.get("phase", "?"),
                    "ok": False,
                    "stale_epoch": True,
                })
                return node_rank
            role = msg.get("role", "replacement")
            rank = msg.get("rank", "?")
            phase = msg.get("phase", "?")
            timeout = float(msg.get("timeout", 300.0))
            ok = self._wait_for_phase(role, rank, phase, timeout)
            response = json.dumps({
                "type": "wait_phase_result",
                "role": role,
                "rank": rank,
                "phase": phase,
                "ok": ok,
            }) + "\n"
            try:
                conn.sendall(response.encode())
            except OSError:
                pass
            if not ok:
                self._initiate_fallback_relaunch(
                    "wait_phase_timeout",
                    {"role": role, "rank": rank, "phase": phase, "timeout": timeout},
                )
            return node_rank

        elif msg_type == "wait_phase_count":
            if self._is_stale_recovery_message(msg, msg_type):
                self._send_json_response(conn, {
                    "type": "wait_phase_count_result",
                    "phase": msg.get("phase", "?"),
                    "min_count": int(msg.get("min_count", self.training_nnodes * self.nproc_per_node)),
                    "count": 0,
                    "ok": False,
                    "stale_epoch": True,
                })
                return node_rank
            phase = msg.get("phase", "?")
            min_count = int(msg.get("min_count", self.training_nnodes * self.nproc_per_node))
            timeout = float(msg.get("timeout", 300.0))
            count = self._wait_for_phase_count(phase, min_count, timeout)
            missing = []
            pending = []
            unreported = []
            if count < min_count:
                with self.phase_cv:
                    missing, pending, unreported = self._phase_missing_locked(phase, min_count)
            response = json.dumps({
                "type": "wait_phase_count_result",
                "phase": phase,
                "min_count": min_count,
                "count": count,
                "ok": count >= min_count,
                "missing": missing,
                "pending": pending,
                "unreported": unreported,
            }) + "\n"
            try:
                conn.sendall(response.encode())
            except OSError:
                pass
            if count < min_count:
                self._initiate_fallback_relaunch(
                    "wait_phase_count_timeout",
                    {
                        "phase": phase,
                        "min_count": min_count,
                        "count": count,
                        "missing": missing,
                        "pending": pending,
                        "unreported": unreported,
                        "timeout": timeout,
                    },
                )
            return node_rank

        elif msg_type == "ordinal_barrier":
            if self._is_stale_recovery_message(msg, msg_type):
                self._send_json_response(conn, {
                    "type": "ordinal_barrier_result",
                    "barrier_id": msg.get("barrier_id", "?"),
                    "min_count": int(msg.get("min_count", self.training_nnodes * self.nproc_per_node)),
                    "count": 0,
                    "ok": False,
                    "stale_epoch": True,
                })
                return node_rank
            barrier_id = msg.get("barrier_id", "?")
            rank = int(msg.get("rank", -1))
            min_count = int(msg.get("min_count", self.training_nnodes * self.nproc_per_node))
            timeout = float(msg.get("timeout", 300.0))
            count, missing, arrived, manifest_ok = self._wait_for_ordinal_barrier(
                barrier_id,
                rank,
                min_count,
                timeout,
                msg,
            )
            response = json.dumps({
                "type": "ordinal_barrier_result",
                "barrier_id": barrier_id,
                "min_count": min_count,
                "count": count,
                "ok": count >= min_count and manifest_ok,
                "manifest_ok": manifest_ok,
                "missing": missing,
                "arrived": arrived,
            }) + "\n"
            try:
                conn.sendall(response.encode())
            except OSError:
                pass
            if not manifest_ok:
                self._initiate_fallback_relaunch(
                    "ordinal_barrier_manifest_mismatch",
                    {
                        "barrier_id": barrier_id,
                        "rank": rank,
                        "group_desc": msg.get("group_desc"),
                        "group_ranks": msg.get("group_ranks"),
                    },
                )
            elif count < min_count:
                self._initiate_fallback_relaunch(
                    "ordinal_barrier_timeout",
                    {
                        "barrier_id": barrier_id,
                        "rank": rank,
                        "min_count": min_count,
                        "count": count,
                        "missing": missing,
                        "arrived": arrived,
                        "timeout": timeout,
                        "group_desc": msg.get("group_desc"),
                        "group_ranks": msg.get("group_ranks"),
                    },
                )
            return node_rank

        elif msg_type == "peer_sync_endpoint":
            if self._is_stale_recovery_message(msg, msg_type):
                return node_rank
            peer_id = msg.get("peer_id")
            host = msg.get("host") or (addr[0] if addr else None)
            port = msg.get("port")
            if not peer_id or not host or port is None:
                log.warning("Bad peer_sync_endpoint message: %s", msg)
                return node_rank
            with self.phase_cv:
                self.peer_sync_endpoints[peer_id] = {
                    "host": host,
                    "port": int(port),
                    "timestamp": time.time(),
                    "src_rank": msg.get("src_rank"),
                    "dst_rank": msg.get("dst_rank"),
                }
                self.phase_cv.notify_all()
            log.info(
                "Peer sync endpoint: id=%s src=%s dst=%s endpoint=%s:%s",
                peer_id,
                msg.get("src_rank"),
                msg.get("dst_rank"),
                host,
                port,
            )
            return node_rank

        elif msg_type == "wait_peer_sync_endpoint":
            if self._is_stale_recovery_message(msg, msg_type):
                self._send_json_response(conn, {
                    "type": "peer_sync_endpoint_result",
                    "peer_id": msg.get("peer_id"),
                    "ok": False,
                    "endpoint": None,
                    "stale_epoch": True,
                })
                return node_rank
            peer_id = msg.get("peer_id")
            timeout = float(msg.get("timeout", 300.0))
            endpoint = self._wait_for_peer_sync_endpoint(peer_id, timeout)
            response = json.dumps({
                "type": "peer_sync_endpoint_result",
                "peer_id": peer_id,
                "ok": endpoint is not None,
                "endpoint": endpoint,
            }) + "\n"
            try:
                conn.sendall(response.encode())
            except OSError:
                pass
            if endpoint is None:
                self._initiate_fallback_relaunch(
                    "peer_sync_endpoint_timeout",
                    {"peer_id": peer_id, "timeout": timeout},
                )
            return node_rank

        else:
            log.warning(f"Unknown message type: {msg_type}")
            return node_rank

    @staticmethod
    def _send_json_response(conn, payload):
        try:
            conn.sendall((json.dumps(payload) + "\n").encode())
        except OSError:
            pass

    @staticmethod
    def _coerce_epoch(value):
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _is_stale_recovery_message(self, msg, msg_type):
        msg_epoch = self._coerce_epoch(msg.get("recovery_epoch"))
        if msg_epoch is None:
            return False
        with self.lock:
            current_epoch = self.recovery_epoch
            in_recovery = self.recovery_in_progress
        if (
            msg_type == "recovery_phase"
            and msg.get("standby_prearmed") is True
            and not in_recovery
            and msg_epoch == current_epoch + 1
        ):
            return False
        if msg_epoch != current_epoch:
            log.warning(
                "Ignoring stale recovery message type=%s epoch=%s current_epoch=%s "
                "in_recovery=%s",
                msg_type,
                msg_epoch,
                current_epoch,
                in_recovery,
            )
            return True
        return False

    def _wait_for_peer_sync_endpoint(self, peer_id, timeout):
        if not peer_id:
            return None
        deadline = time.time() + timeout
        with self.phase_cv:
            while peer_id not in self.peer_sync_endpoints:
                remaining = deadline - time.time()
                if remaining <= 0:
                    log.warning("wait_peer_sync_endpoint timed out: id=%s", peer_id)
                    return None
                self.phase_cv.wait(timeout=min(remaining, 1.0))
            return dict(self.peer_sync_endpoints[peer_id])

    def _wait_for_ordinal_barrier(self, barrier_id, rank, min_count, timeout, msg):
        deadline = time.time() + timeout
        with self.phase_cv:
            state = self.ordinal_barriers.setdefault(
                barrier_id,
                {
                    "arrived": set(),
                    "meta_by_rank": {},
                    "manifest_by_rank": {},
                    "created": time.time(),
                    "last_progress": time.time(),
                    "min_count": min_count,
                    "mismatch_logged": False,
                    "released_logged": False,
                },
            )
            if rank not in state["arrived"]:
                state["last_progress"] = time.time()
            state["arrived"].add(rank)
            state["meta_by_rank"][rank] = {
                k: v
                for k, v in msg.items()
                if k
                not in (
                    "type",
                    "node_rank",
                    "timeout",
                    "min_count",
                )
            }
            manifest = {
                k: msg.get(k)
                for k in (
                    "group_desc",
                    "group_backend",
                    "group_init_mode",
                    "group_use_local_synchronization",
                    "group_c10d_count",
                    "group_size",
                    "group_ranks",
                    "group_first_rank",
                    "group_last_rank",
                    "group_timeout_seconds",
                    "group_ordinal",
                    "barrier_stage",
                    "state_contract",
                )
                if k in msg
            }
            state["manifest_by_rank"][rank] = json.dumps(manifest, sort_keys=True)
            manifest_ok = len(set(state["manifest_by_rank"].values())) <= 1
            if not manifest_ok and not state["mismatch_logged"]:
                state["mismatch_logged"] = True
                sample = {
                    r: state["manifest_by_rank"][r]
                    for r in sorted(state["manifest_by_rank"])[:8]
                }
                log.warning(
                    "ordinal_barrier manifest mismatch: id=%s sample=%s",
                    barrier_id,
                    sample,
                )
            self.phase_cv.notify_all()

            if not manifest_ok:
                return len(state["arrived"]), [], sorted(state["arrived"]), False

            while len(state["arrived"]) < min_count:
                if self.fallback_initiated:
                    arrived = sorted(state["arrived"])
                    group_ranks = msg.get("group_ranks")
                    if isinstance(group_ranks, list) and len(group_ranks) == min_count:
                        expected = [int(r) for r in group_ranks]
                    else:
                        expected = list(range(min_count))
                    missing = [r for r in expected if r not in state["arrived"]]
                    return len(arrived), missing, arrived, True
                remaining = deadline - time.time()
                if remaining <= 0:
                    arrived = sorted(state["arrived"])
                    group_ranks = msg.get("group_ranks")
                    if isinstance(group_ranks, list) and len(group_ranks) == min_count:
                        expected = [int(r) for r in group_ranks]
                    else:
                        expected = list(range(min_count))
                    missing = [r for r in expected if r not in state["arrived"]]
                    sample = [
                        state["meta_by_rank"].get(r)
                        for r in arrived[:8]
                    ]
                    log.warning(
                        "ordinal_barrier timed out: id=%s count=%s min_count=%s "
                        "missing=%s arrived=%s sample=%s",
                        barrier_id,
                        len(arrived),
                        min_count,
                        missing,
                        arrived,
                        sample,
                    )
                    return len(arrived), missing, arrived, True
                self.phase_cv.wait(timeout=min(remaining, 1.0))

                if len(set(state["manifest_by_rank"].values())) > 1:
                    return (
                        len(state["arrived"]),
                        [],
                        sorted(state["arrived"]),
                        False,
                    )

            arrived = sorted(state["arrived"])
            missing = []
            if not state["released_logged"]:
                state["released_logged"] = True
                meta = state["meta_by_rank"].get(rank, {})
                log.info(
                    "ordinal_barrier released: id=%s count=%s desc=%s ordinal=%s stage=%s",
                    barrier_id,
                    len(arrived),
                    meta.get("group_desc"),
                    meta.get("group_ordinal"),
                    meta.get("barrier_stage"),
                )
            return len(arrived), missing, arrived, True

    def _stalled_ordinal_barrier_locked(self, now=None):
        """Return the oldest recovery barrier that stopped gaining participants."""
        if self.recovery_stall_timeout <= 0:
            return None
        now = time.time() if now is None else float(now)
        stalled = None
        for barrier_id, state in self.ordinal_barriers.items():
            min_count = int(
                state.get("min_count", self.training_nnodes * self.nproc_per_node)
            )
            arrived = sorted(int(rank) for rank in state.get("arrived", set()))
            if not arrived or len(arrived) >= min_count:
                continue
            last_progress = float(
                state.get("last_progress", state.get("created", now))
            )
            elapsed = now - last_progress
            if elapsed < self.recovery_stall_timeout:
                continue
            details = {
                "barrier_id": barrier_id,
                "timeout": self.recovery_stall_timeout,
                "elapsed_without_progress": elapsed,
                "min_count": min_count,
                "count": len(arrived),
                "arrived": arrived,
                "missing": [rank for rank in range(min_count) if rank not in arrived],
            }
            meta_by_rank = state.get("meta_by_rank", {})
            if meta_by_rank:
                details["sample"] = meta_by_rank.get(arrived[0], {})
            if stalled is None or elapsed > stalled["elapsed_without_progress"]:
                stalled = details
        return stalled

    def _phase_reached_locked(self, role, rank, target_phase):
        state = self.recovery_phases.get(f"{role}:{rank}")
        if state is None:
            return False
        have = _PHASE_ORDER.get(state.get("phase"), -1)
        want = _PHASE_ORDER.get(target_phase, 10**9)
        return have >= want

    def _wait_for_phase(self, role, rank, phase, timeout):
        deadline = time.time() + timeout
        with self.phase_cv:
            while not self._phase_reached_locked(role, rank, phase):
                remaining = deadline - time.time()
                if remaining <= 0:
                    log.warning(
                        "wait_phase timed out: role=%s rank=%s phase=%s",
                        role,
                        rank,
                        phase,
                    )
                    return False
                self.phase_cv.wait(timeout=min(remaining, 1.0))
        return True

    def _phase_count_locked(self, target_phase):
        count = 0
        for state in self.recovery_phases.values():
            have = _PHASE_ORDER.get(state.get("phase"), -1)
            want = _PHASE_ORDER.get(target_phase, 10**9)
            if have >= want:
                count += 1
        return count

    def _phase_missing_locked(self, target_phase, expected_count):
        want = _PHASE_ORDER.get(target_phase, 10**9)
        best_by_rank = {}
        reached = set()
        for key, state in self.recovery_phases.items():
            try:
                rank = int(str(key).rsplit(":", 1)[1])
            except (IndexError, TypeError, ValueError):
                continue
            have = _PHASE_ORDER.get(state.get("phase"), -1)
            prev = best_by_rank.get(rank)
            prev_have = _PHASE_ORDER.get(prev.get("phase"), -1) if prev else -1
            if have >= prev_have:
                best_by_rank[rank] = state
            if have >= want:
                reached.add(rank)
        missing = [rank for rank in range(expected_count) if rank not in reached]
        pending = []
        unreported = []
        for rank in missing:
            state = best_by_rank.get(rank)
            if state is None:
                unreported.append(rank)
            else:
                pending.append(
                    {
                        "rank": rank,
                        "role": state.get("role"),
                        "node": state.get("node_rank"),
                        "phase": state.get("phase"),
                        "step": state.get("step"),
                        "extra": state.get("extra"),
                    }
                )
        return missing, pending, unreported

    def _wait_for_phase_count(self, phase, min_count, timeout):
        deadline = time.time() + timeout
        with self.phase_cv:
            count = self._phase_count_locked(phase)
            while count < min_count:
                remaining = deadline - time.time()
                if remaining <= 0:
                    missing, pending, unreported = self._phase_missing_locked(phase, min_count)
                    log.warning(
                        "wait_phase_count timed out: phase=%s count=%s min_count=%s "
                        "missing=%s pending=%s unreported=%s",
                        phase,
                        count,
                        min_count,
                        missing,
                        pending,
                        unreported,
                    )
                    return count
                self.phase_cv.wait(timeout=min(remaining, 1.0))
                count = self._phase_count_locked(phase)
            return count

    def _maybe_inject_fault(self, reporting_node_rank, step):
        """Check if we should inject a fault based on the reported step.

        NEW FLOW: Send PAUSE to ALL nodes first (including target).
        All nodes will pause at the next safe point, destroy their process
        groups, and send ready_to_rebuild.  Only THEN do we kill the target
        rank.  This avoids NCCL timeout issues entirely.
        """
        if self.fault_injected or self.fault_inject_step < 0:
            return
        if step < self.fault_inject_step:
            return

        target = self.fault_inject_node if self.fault_inject_node >= 0 else 0
        self.fault_injected = True

        log.warning(f"FAULT INJECTION: step {step} >= {self.fault_inject_step}, "
                    f"pausing ALL nodes then killing node {target} local_rank {self.fault_inject_local_rank}")

        # Send PAUSE to ALL nodes (including target) — they will all
        # reach the safe point and destroy process groups before any kill.
        self._handle_fault(
            target,
            killed_local_rank=self.fault_inject_local_rank,
            failure_scope="single_rank",
        )

    def _training_started_locked(self):
        """Return True only after all training nodes have entered train-loop heartbeats."""
        if len(self.last_heartbeat) < self.training_nnodes:
            return False
        for node_rank in range(self.training_nnodes):
            if self.node_train_phases.get(node_rank, "startup") == "startup":
                return False
            if self.node_control_owners.get(node_rank) == "launcher":
                if len(self.node_rank_states.get(node_rank, {})) != self.nproc_per_node:
                    return False
        return True

    def _defer_unclassified_nccl_fallback(self, node_rank):
        grace = float(os.environ.get("ELASTIC_NCCL_CLASSIFICATION_GRACE_SECONDS", "5"))
        time.sleep(grace)
        with self.lock:
            self.pending_nccl_fallback_nodes.discard(node_rank)
            if self.recovery_in_progress or self.fallback_initiated:
                return
        log.error(
            "NCCL error on node %s was not accompanied by a fail-stop worker "
            "event within %.1fs; using checkpoint fallback",
            node_rank,
            grace,
        )
        self._handle_fault(
            node_rank,
            killed_local_rank=-1,
            failure_scope="software",
        )

    def _heartbeat_timeout_for_phase_locked(self, phase, training_started):
        """Return a heartbeat timeout that matches the latest reported phase."""
        if not training_started:
            return self.startup_heartbeat_timeout
        if phase in {
            "checkpoint",
            "checkpoint_done",
            "save_checkpoint",
            "save_checkpoint_done",
            "async_checkpoint_finalize",
            "async_checkpoint_finalize_done",
        }:
            return self.checkpoint_heartbeat_timeout
        if phase in {"forward_backward", "iteration_safe_point"}:
            return self.forward_heartbeat_timeout
        return self.heartbeat_timeout

    def _heartbeat_checker(self):
        """Periodically check for heartbeat timeouts."""
        while self.running and not self.last_heartbeat:
            time.sleep(1.0)

        log.info("Heartbeat checker active")

        while self.running:
            time.sleep(2.0)
            with self.lock:
                recovery_in_progress = self.recovery_in_progress
                fallback_initiated = self.fallback_initiated
                stalled_barrier = (
                    self._stalled_ordinal_barrier_locked()
                    if recovery_in_progress and not fallback_initiated
                    else None
                )
            if stalled_barrier is not None:
                log.error(
                    "Recovery ordinal barrier stalled: id=%s count=%s/%s "
                    "elapsed=%.1fs timeout=%.1fs missing=%s",
                    stalled_barrier["barrier_id"],
                    stalled_barrier["count"],
                    stalled_barrier["min_count"],
                    stalled_barrier["elapsed_without_progress"],
                    stalled_barrier["timeout"],
                    stalled_barrier["missing"],
                )
                self._log_recovery_phase_summary("ordinal-barrier-stall")
                self._initiate_fallback_relaunch(
                    "ordinal_barrier_stall_timeout",
                    stalled_barrier,
                )
                continue
            if recovery_in_progress or fallback_initiated:
                continue

            with self.lock:
                startup_released = self.startup_released_attempt >= 0
            if not startup_released:
                continue

            now = time.time()
            timed_out_node = None
            timed_out_elapsed = 0.0
            timed_out_timeout = self.heartbeat_timeout
            timed_out_phase = "unknown"
            timed_out_step = -1
            timed_out_reason = "heartbeat-timeout"
            training_started = False
            with self.lock:
                training_started = self._training_started_locked()
                for node_rank, last_ts in list(self.last_heartbeat.items()):
                    phase = self.node_train_phases.get(node_rank, "startup")
                    step = self.node_steps.get(node_rank, -1)
                    disconnected_at = self.node_disconnected_at.get(node_rank)
                    if disconnected_at is not None:
                        elapsed = now - disconnected_at
                        timeout = self.disconnect_grace_timeout
                        if elapsed > timeout:
                            log.error(
                                "FAULT DETECTED: Node %s disconnected "
                                "(%.1fs > %.1fs grace, phase=%s, step=%s, "
                                "training_started=%s)",
                                node_rank,
                                elapsed,
                                timeout,
                                phase,
                                step,
                                training_started,
                            )
                            timed_out_node = node_rank
                            timed_out_elapsed = elapsed
                            timed_out_timeout = timeout
                            timed_out_phase = phase
                            timed_out_step = step
                            timed_out_reason = "node-disconnected"
                            break
                    timeout = self._heartbeat_timeout_for_phase_locked(phase, training_started)
                    elapsed = now - last_ts
                    if elapsed > timeout:
                        log.error(
                            "FAULT DETECTED: Node %s heartbeat timeout "
                            "(%.1fs > %.1fs, phase=%s, step=%s, training_started=%s)",
                            node_rank,
                            elapsed,
                            timeout,
                            phase,
                            step,
                            training_started,
                        )
                        timed_out_node = node_rank
                        timed_out_elapsed = elapsed
                        timed_out_timeout = timeout
                        timed_out_phase = phase
                        timed_out_step = step
                        break
            if timed_out_node is not None:
                self._log_recovery_phase_summary(
                    f"{timed_out_reason} "
                    f"node={timed_out_node} elapsed={timed_out_elapsed:.1f}s "
                    f"timeout={timed_out_timeout:.1f}s phase={timed_out_phase} "
                    f"step={timed_out_step} training_started={training_started}"
                )
                self._handle_fault(
                    timed_out_node,
                    killed_local_rank=-1,
                    failure_scope="node",
                )

    def _log_recovery_phase_summary(self, reason):
        with self.lock:
            phases = dict(self.recovery_phases)
        if not phases:
            log.warning("Recovery phase summary (%s): no phase reports", reason)
            return
        log.warning("Recovery phase summary (%s):", reason)
        for key in sorted(phases):
            item = phases[key]
            extra_text = self._format_recovery_phase_extra(item.get("extra") or {})
            if extra_text:
                log.warning(
                    "  %s node=%s step=%s phase=%s %s",
                    key,
                    item.get("node_rank"),
                    item.get("step"),
                    item.get("phase"),
                    extra_text,
                )
            else:
                log.warning(
                    "  %s node=%s step=%s phase=%s",
                    key,
                    item.get("node_rank"),
                    item.get("step"),
                    item.get("phase"),
                )

    @staticmethod
    def _format_recovery_phase_extra(data):
        if not data:
            return ""
        keys = (
            "group_desc",
            "group_backend",
            "group_init_mode",
            "group_use_local_synchronization",
            "group_c10d_count",
            "group_size",
            "group_ranks",
            "group_first_rank",
            "group_last_rank",
            "group_representative_rank",
            "group_report_rank",
            "group_timeout_seconds",
            "replacement_rank",
            "recovery_mode",
            "checkpoint_step",
            "expert_staleness_delta",
            "pg_generation",
            "aligned_iteration",
            "two_phase_enabled",
            "expert_optimizer",
            "peer_sync",
        )
        parts = []
        for key in keys:
            if key in data and data.get(key) is not None:
                parts.append(f"{key}={data.get(key)}")
        return " ".join(parts)

    @staticmethod
    def _coerce_step(value, default=-1):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _select_resume_iteration(self, ready_steps, ready_step_tags, ready_phases):
        """Pick a FlashRecovery-style resume iteration from paused node tags."""
        coerced_steps = {
            node: self._coerce_step(step)
            for node, step in ready_steps.items()
        }
        coerced_tags = {
            node: self._coerce_step(ready_step_tags.get(node, coerced_steps.get(node, -1)))
            for node in ready_steps
        }

        optimizer_pending_nodes = sorted(
            node for node, tag in coerced_tags.items()
            if tag == -1 or ready_phases.get(node) == "optimizer_step"
        )
        if optimizer_pending_nodes:
            valid_steps = [step for step in coerced_steps.values() if step >= 0]
            resume_iteration = min(valid_steps) if valid_steps else -1
            log.warning(
                "Optimizer step was in flight on ready nodes %s; selecting "
                "conservative resume_iteration=%s from ready steps %s",
                optimizer_pending_nodes,
                resume_iteration,
                coerced_steps,
            )
            return resume_iteration

        valid_tags = [tag for tag in coerced_tags.values() if tag >= 0]
        if valid_tags:
            if len(set(valid_tags)) > 1:
                log.warning(
                    "Ready nodes reported mixed step tags %s; selecting "
                    "minimum resume_iteration=%s",
                    coerced_tags,
                    min(valid_tags),
                )
            return min(valid_tags)

        valid_steps = [step for step in coerced_steps.values() if step >= 0]
        return min(valid_steps) if valid_steps else -1

    def _select_fallback_resume_iteration(self, ready_steps, ready_step_tags, ready_phases):
        if ready_steps:
            return self._select_resume_iteration(ready_steps, ready_step_tags, ready_phases)
        with self.lock:
            node_steps = dict(self.node_steps)
            node_step_tags = dict(self.node_step_tags)
            node_phases = dict(self.node_train_phases)
        return self._select_resume_iteration(node_steps, node_step_tags, node_phases)

    def _build_rank_quiescence_proof(self, resume_iteration, killed_global_rank):
        """Prove that every survivor stopped at one committed state version."""
        with self.lock:
            owners = dict(self.rebuild_ready_control_owners)
            per_node = dict(self.rebuild_ready_rank_states)
            node_summaries = dict(self.rebuild_ready_quiescence)

        launcher_nodes = sorted(
            node for node, owner in owners.items() if owner == "launcher"
        )
        if len(launcher_nodes) != self.training_nnodes:
            return {
                "status": "legacy_unverified",
                "required": False,
                "reason": "launcher_control_plane_not_active_on_all_nodes",
                "launcher_nodes": launcher_nodes,
                "expected_nodes": self.training_nnodes,
            }

        rank_states = {}
        duplicate_ranks = []
        for node, states in per_node.items():
            for rank_text, state in states.items():
                try:
                    rank = int(rank_text)
                except (TypeError, ValueError):
                    continue
                if rank in rank_states:
                    duplicate_ranks.append(rank)
                if isinstance(state, dict):
                    rank_states[rank] = dict(state)

        world_size = self.training_nnodes * self.nproc_per_node
        expected = set(range(world_size))
        expected.discard(killed_global_rank)
        observed = set(rank_states)
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        versions = sorted({int(state.get("step", -1)) for state in rank_states.values()})
        step_tags = sorted({int(state.get("step_tag", -1)) for state in rank_states.values()})
        phases = sorted({str(state.get("train_phase", "unknown")) for state in rank_states.values()})
        invalid_nodes = sorted(
            int(node)
            for node, summary in node_summaries.items()
            if not bool(summary.get("valid", False))
        )
        safe_phases = {"iteration_safe_point", "step_complete", "optimizer_skipped"}
        reasons = []
        if missing:
            reasons.append("missing_survivor_rank_states")
        if unexpected:
            reasons.append("unexpected_rank_states")
        if duplicate_ranks:
            reasons.append("duplicate_rank_ownership")
        if invalid_nodes:
            reasons.append("node_quiescence_invalid")
        if versions != [int(resume_iteration)]:
            reasons.append("state_version_mismatch")
        if step_tags != [int(resume_iteration)]:
            reasons.append("step_tag_mismatch")
        if not set(phases).issubset(safe_phases):
            reasons.append("unsafe_training_phase")

        return {
            "status": "verified" if not reasons else "invalid",
            "required": True,
            "reason": "committed_rank_quorum" if not reasons else ",".join(reasons),
            "resume_iteration": int(resume_iteration),
            "expected_survivors": len(expected),
            "observed_survivors": len(observed),
            "missing_ranks": missing,
            "unexpected_ranks": unexpected,
            "duplicate_ranks": sorted(set(duplicate_ranks)),
            "invalid_nodes": invalid_nodes,
            "state_versions": versions,
            "step_tags": step_tags,
            "phases": phases,
            "launcher_nodes": launcher_nodes,
        }

    def _write_json_atomic(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True, indent=2)
        os.replace(tmp_path, path)

    @staticmethod
    def _env_int(name, default):
        try:
            return int(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            log.warning("Invalid integer env %s=%s; using %s", name, os.environ.get(name), default)
            return default

    @staticmethod
    def _env_flag(name, default=False):
        value = os.environ.get(name)
        if value is None:
            return default
        return str(value).strip().lower() not in {"", "0", "false", "no", "off"}

    def _parallel_layout_from_env(self):
        world_size = self.training_nnodes * self.nproc_per_node
        tp_size = max(1, self._env_int("TP_SIZE", 1))
        pp_size = max(1, self._env_int("PP_SIZE", 1))
        cp_size = max(1, self._env_int("CP_SIZE", self._env_int("CONTEXT_PARALLEL_SIZE", 1)))
        ep_size = max(1, self._env_int("EP_SIZE", 1))
        expert_tp_size = max(1, self._env_int("EXPERT_TP_SIZE", tp_size))
        order = os.environ.get(
            "MEGATRON_PARALLEL_ORDER",
            "tp-cp-ep-pp-dp" if self._env_flag("USE_TP_PP_DP_MAPPING", False) else "tp-cp-ep-dp-pp",
        )

        dense_model_size = max(1, tp_size * cp_size * pp_size)
        dense_dp_size = world_size // dense_model_size if world_size % dense_model_size == 0 else 1
        expert_model_size = max(1, expert_tp_size * ep_size * pp_size)
        expert_dp_size = world_size // expert_model_size if world_size % expert_model_size == 0 else 1
        return {
            "world_size": world_size,
            "order": order,
            "dense": {
                "tp": tp_size,
                "cp": cp_size,
                "ep": 1,
                "dp": dense_dp_size,
                "pp": pp_size,
            },
            "expert": {
                "tp": expert_tp_size,
                "cp": 1,
                "ep": ep_size,
                "dp": expert_dp_size,
                "pp": pp_size,
            },
        }

    @staticmethod
    def _ordered_tokens(order, sizes):
        tokens = [token for token in order.split("-") if token]
        for token in sizes:
            if token not in tokens:
                tokens.append(token)
        return tokens

    def _rank_identity(self, rank, sizes, order):
        tokens = self._ordered_tokens(order, sizes)
        stride = 1
        identity = {}
        for token in tokens:
            size = max(1, int(sizes.get(token, 1)))
            identity[token] = (rank // stride) % size
            stride *= size
        return identity

    def _rank_from_identity(self, identity, sizes, order):
        tokens = self._ordered_tokens(order, sizes)
        stride = 1
        rank = 0
        for token in tokens:
            size = max(1, int(sizes.get(token, 1)))
            rank += int(identity.get(token, 0)) * stride
            stride *= size
        return rank

    def _same_partition_dp_candidates(self, rank, sizes, order, killed_global_rank):
        identity = self._rank_identity(rank, sizes, order)
        dp_size = max(1, int(sizes.get("dp", 1)))
        if dp_size <= 1:
            return []

        candidates = []
        for offset in range(1, dp_size):
            peer_identity = dict(identity)
            peer_identity["dp"] = (identity.get("dp", 0) + offset) % dp_size
            peer_rank = self._rank_from_identity(peer_identity, sizes, order)
            if 0 <= peer_rank < self.training_nnodes * self.nproc_per_node:
                if peer_rank != killed_global_rank:
                    candidates.append(peer_rank)
        return candidates

    def _latest_checkpoint_iteration(self):
        """Read checkpoint step c without entering Megatron collectives."""
        checkpoint_dir_text = os.environ.get("CKPT_DIR", "")
        if not checkpoint_dir_text:
            return -1
        checkpoint_dir = Path(checkpoint_dir_text)
        if not checkpoint_dir.exists():
            return -1

        tracker = checkpoint_dir / "latest_checkpointed_iteration.txt"
        if tracker.is_file():
            try:
                return int(tracker.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                log.warning("Cannot parse checkpoint tracker %s", tracker)

        iterations = []
        try:
            for path in checkpoint_dir.glob("iter_*"):
                try:
                    iterations.append(int(path.name.split("iter_", 1)[1]))
                except (IndexError, ValueError):
                    continue
        except OSError:
            return -1
        return max(iterations, default=-1)

    def _load_expert_staleness_history(self):
        try:
            with self.expert_staleness_history_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError, TypeError):
            return []
        events = payload.get("events", []) if isinstance(payload, dict) else []
        return [event for event in events if isinstance(event, dict)]

    def _evaluate_moegambit_contract(self, resume_iteration, peer_available):
        """Evaluate paper contract R2 before any replacement process starts."""
        checkpoint_step = self._latest_checkpoint_iteration()
        current_step = int(resume_iteration)
        gap = (
            current_step - checkpoint_step
            if current_step >= 0 and checkpoint_step >= 0
            else -1
        )
        min_gap = self._env_int("MOEGAMBIT_DELTA_TIME_MIN_GAP", 32)
        max_gap = self._env_int("MOEGAMBIT_MAX_SINGLE_GAP", 192)
        window_steps = self._env_int("MOEGAMBIT_EXPOSURE_WINDOW_STEPS", 20000)
        try:
            phi_max = float(
                os.environ.get("MOEGAMBIT_MAX_EXPERT_STALENESS_DENSITY", "0.1")
            )
        except (TypeError, ValueError):
            phi_max = 0.1
        num_experts = max(1, self._env_int("MOEGAMBIT_NUM_EXPERTS", 128))
        ep_size = max(1, self._env_int("EP_SIZE", 1))
        affected_experts = max(
            1,
            self._env_int(
                "MOEGAMBIT_AFFECTED_EXPERTS",
                (num_experts + ep_size - 1) // ep_size,
            ),
        )

        window_start = max(0, current_step - window_steps)
        live_history = [
            event
            for event in self._load_expert_staleness_history()
            if window_start <= int(event.get("step", -1)) < current_step
        ]
        debt_before = sum(
            int(
                event.get(
                    "expert_iteration_debt",
                    int(event.get("gap", 0))
                    * max(1, int(event.get("num_affected_experts", 1))),
                )
            )
            for event in live_history
        )
        projected_debt = debt_before + affected_experts * max(0, gap)
        denominator = num_experts * window_steps
        projected_density = projected_debt / denominator if denominator > 0 else float("inf")

        if not peer_available:
            reason = "no_peer"
        elif checkpoint_step < 0:
            reason = "no_checkpoint"
        elif current_step < 0:
            reason = "no_safe_point"
        elif gap < min_gap:
            reason = "small_gap"
        elif gap > max_gap:
            reason = "large_gap"
        elif projected_density > phi_max:
            reason = "high_debt"
        else:
            reason = "admit"

        return {
            "name": "moegambit_r2_expert_staleness_density",
            "admitted": reason == "admit",
            "reason": reason,
            "peer_available": bool(peer_available),
            "current_step": current_step,
            "checkpoint_step": checkpoint_step,
            "expert_staleness_delta": gap,
            "delta_time_min_gap": min_gap,
            "max_single_gap": max_gap,
            "exposure_window_steps": window_steps,
            "num_experts": num_experts,
            "num_affected_experts": affected_experts,
            "window_expert_iteration_debt_before": debt_before,
            "projected_expert_iteration_debt": projected_debt,
            "projected_expert_staleness_density": projected_density,
            "max_expert_staleness_density": phi_max,
            "two_phase_enabled": self._env_flag("ELASTIC_TWO_PHASE_RECOVERY", False),
        }

    def _record_successful_hybrid_contract(self, recovery_epoch):
        """Commit R2 debt only after the first post-rebuild step succeeds."""
        try:
            with self.recovery_descriptor_file.open("r", encoding="utf-8") as f:
                descriptor = json.load(f)
        except (OSError, ValueError, TypeError) as exc:
            log.warning("Cannot record recovery contract epoch=%s: %s", recovery_epoch, exc)
            return

        contract = descriptor.get("contract", {}).get("R2", {})
        if (
            descriptor.get("recovery_epoch") != recovery_epoch
            or not contract.get("admitted", False)
        ):
            return

        event = {
            "recovery_epoch": recovery_epoch,
            "step": int(contract["current_step"]),
            "checkpoint_step": int(contract["checkpoint_step"]),
            "gap": int(contract["expert_staleness_delta"]),
            "num_affected_experts": int(contract["num_affected_experts"]),
        }
        event["expert_iteration_debt"] = (
            event["gap"] * event["num_affected_experts"]
        )
        window_start = max(
            0, event["step"] - int(contract["exposure_window_steps"])
        )
        history = [
            item
            for item in self._load_expert_staleness_history()
            if int(item.get("recovery_epoch", -1)) != recovery_epoch
            and int(item.get("step", -1)) >= window_start
        ]
        history.append(event)
        self.expert_staleness_history_file.parent.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(
            self.expert_staleness_history_file,
            {"version": 1, "events": history},
        )
        log.info(
            "Committed hybrid staleness debt: epoch=%s step=%s gap=%s "
            "affected_experts=%s debt=%s",
            recovery_epoch,
            event["step"],
            event["gap"],
            event["num_affected_experts"],
            event["expert_iteration_debt"],
        )

    def _truncate_staleness_history_for_restart(self, checkpoint_step):
        """Drop debt from hybrid updates rolled back by checkpoint relaunch."""
        if checkpoint_step < 0:
            return
        history = self._load_expert_staleness_history()
        retained = [
            event
            for event in history
            if int(event.get("step", -1)) <= checkpoint_step
        ]
        if len(retained) == len(history):
            return
        self._write_json_atomic(
            self.expert_staleness_history_file,
            {"version": 1, "events": retained},
        )
        log.info(
            "Checkpoint relaunch at step %s removed %s rolled-back debt event(s)",
            checkpoint_step,
            len(history) - len(retained),
        )

    def _build_recovery_descriptor(
        self,
        failed_node,
        killed_local_rank,
        recovery_epoch,
        resume_iteration=-1,
    ):
        killed_global_rank = (
            failed_node * self.nproc_per_node + killed_local_rank
            if failed_node is not None and killed_local_rank >= 0
            else -1
        )
        physical_spare_node = int(os.environ.get("NODE_RANK", str(self.training_nnodes)))
        parallel_layout = self._parallel_layout_from_env()
        dense_sizes = parallel_layout["dense"]
        expert_sizes = parallel_layout["expert"]
        order = parallel_layout["order"]
        dense_peer_candidates = []
        expert_peer_candidates = []
        if killed_global_rank >= 0:
            dense_peer_candidates = self._same_partition_dp_candidates(
                killed_global_rank,
                dense_sizes,
                order,
                killed_global_rank,
            )
            expert_peer_candidates = self._same_partition_dp_candidates(
                killed_global_rank,
                expert_sizes,
                order,
                killed_global_rank,
            )
        r2_contract = self._evaluate_moegambit_contract(
            resume_iteration,
            bool(dense_peer_candidates),
        )
        quiescence_proof = dict(self.rank_quiescence_proof)
        hot_swap_feasible = bool(r2_contract["admitted"])
        if quiescence_proof.get("status") == "invalid":
            hot_swap_feasible = False
        rank_table = []
        for node in range(self.training_nnodes):
            for local_rank in range(self.nproc_per_node):
                rank = node * self.nproc_per_node + local_rank
                entry = {
                    "logical_rank": rank,
                    "logical_node": node,
                    "logical_local_rank": local_rank,
                    "physical_node": node,
                    "physical_local_rank": local_rank,
                    "role": "survivor",
                    "dense_identity": self._rank_identity(rank, dense_sizes, order),
                    "expert_identity": self._rank_identity(rank, expert_sizes, order),
                }
                if rank == killed_global_rank:
                    entry.update({
                        "physical_node": physical_spare_node,
                        "physical_local_rank": 0,
                        "role": "replacement",
                    })
                rank_table.append(entry)

        return {
            "version": 3,
            "type": "recovery_descriptor",
            "timestamp": time.time(),
            "recovery_epoch": recovery_epoch,
            "failed_node": failed_node,
            "killed_local_rank": killed_local_rank,
            "killed_global_rank": killed_global_rank,
            "resume_iteration": resume_iteration,
            "version_semantics": "intentional_mixed_version",
            "checkpoint_step": r2_contract["checkpoint_step"],
            "expert_staleness_delta": r2_contract["expert_staleness_delta"],
            "checkpoint_dir": os.environ.get("CKPT_DIR", ""),
            "safe_point": {
                "resume_semantics": "last_completed_step",
                "discard_interrupted_step": True,
                "step_atomicity": "optimizer_step_must_not_commit_partial_progress",
                "rank_quiescence": quiescence_proof,
            },
            "topology": {
                "training_nnodes": self.training_nnodes,
                "nproc_per_node": self.nproc_per_node,
                "world_size": self.training_nnodes * self.nproc_per_node,
                "master_addr": self.master_addr,
                "master_port": self.master_port,
                "rebuild_master_port": str(self._rebuild_master_port(recovery_epoch)),
                "parallel_order": order,
                "parallel_layout": parallel_layout,
            },
            "rank_table": rank_table,
            "failed_shard": {
                "logical_rank": killed_global_rank,
                "dense_identity": (
                    self._rank_identity(killed_global_rank, dense_sizes, order)
                    if killed_global_rank >= 0 else {}
                ),
                "expert_identity": (
                    self._rank_identity(killed_global_rank, expert_sizes, order)
                    if killed_global_rank >= 0 else {}
                ),
                "dense_peer_candidates": dense_peer_candidates,
                "expert_peer_candidates": expert_peer_candidates,
            },
            "state_sources": {
                "non_expert_parameters": {
                    "kind": "peer_broadcast",
                    "version_step": resume_iteration,
                    "source_match": "same_dense_tp_cp_pp_identity_different_dp",
                    "candidate_ranks": dense_peer_candidates,
                },
                "non_expert_optimizer": {
                    "kind": "peer_broadcast",
                    "version_step": resume_iteration,
                    "source_match": "same_dense_tp_cp_pp_identity_different_dp",
                    "candidate_ranks": dense_peer_candidates,
                },
                "expert_parameters": {
                    "kind": "checkpoint_shard",
                    "version_step": r2_contract["checkpoint_step"],
                },
                "expert_optimizer": {
                    "kind": "checkpoint_shard",
                    "version_step": r2_contract["checkpoint_step"],
                },
                "runtime_metadata": "recompute_from_descriptor",
            },
            "contract": {
                "R1": {
                    "safe_point": "last_completed_step",
                    "discard_interrupted_step": True,
                    "optimizer_commit_guard": "required",
                },
                "R2": r2_contract,
                "R3": {
                    "state_machine": [
                        "state_contract_start",
                        "param_sync_start",
                        "param_sync_done",
                        "resume_state_applied",
                        "state_contract_ready",
                        "rerun_state_reset",
                        "train_ready",
                        "train_step_finalize_done",
                        "training_log_start",
                        "training_log_done",
                        "post_step_callbacks_start",
                        "post_step_callbacks_done",
                        "checkpoint_exit_start",
                        "checkpoint_exit_done",
                        "post_rebuild_commit_ready",
                    ],
                    "debt_commit_point": "post_rebuild_commit_ready",
                },
            },
            "recovery": {
                "epoch_serialized": True,
                "hot_swap_feasible": hot_swap_feasible,
                "fallback_action": "checkpoint_relaunch",
                "fallback_exit_code": self.fallback_exit_code,
                "normal_path_intrusion": (
                    "disabled_without_launcher_flag; local_unix_state_tags_when_enabled"
                ),
                "state_validation": "fail_closed_before_train_ready",
                "rank_version_validation": quiescence_proof.get("status", "pending"),
                "optimizer_requirement": (
                    "all_expert_and_non_expert_state_ready_before_train_ready"
                    if not r2_contract["two_phase_enabled"]
                    else "expert_weights_ready_optimizer_commit_guarded"
                ),
            },
            "invariants": {
                "I1_topology_consistency": "all ranks consume the same descriptor epoch",
                "I2_shard_completeness": "one replacement owns the failed logical rank",
                "I3_optimizer_availability": (
                    "non_expert from DP peer and expert from checkpoint, "
                    "validated before train_ready"
                ),
                "I4_step_atomicity": "resume from last completed iteration",
                "I5_control_plane_isolation": "watcher connection owned by node launcher",
            },
        }

    def _write_recovery_descriptor(
        self,
        failed_node,
        killed_local_rank,
        recovery_epoch,
        resume_iteration=-1,
    ):
        descriptor = self._build_recovery_descriptor(
            failed_node,
            killed_local_rank,
            recovery_epoch,
            resume_iteration=resume_iteration,
        )
        self._write_json_atomic(self.recovery_descriptor_file, descriptor)
        epoch_path = self.fault_dir / f"recovery_descriptor_epoch_{recovery_epoch}.json"
        self._write_json_atomic(epoch_path, descriptor)
        log.info(
            "Recovery descriptor written: epoch=%s failed_node=%s killed_local_rank=%s "
            "resume_iteration=%s path=%s",
            recovery_epoch,
            failed_node,
            killed_local_rank,
            resume_iteration,
            self.recovery_descriptor_file,
        )
        r2 = descriptor.get("contract", {}).get("R2", {})
        log.info(
            "MoEGambit R2: decision=%s reason=%s t=%s c=%s delta=%s "
            "affected=%s/%s debt_before=%s projected_phi=%.8f phi_max=%s",
            "hybrid" if r2.get("admitted") else "restart",
            r2.get("reason"),
            r2.get("current_step"),
            r2.get("checkpoint_step"),
            r2.get("expert_staleness_delta"),
            r2.get("num_affected_experts"),
            r2.get("num_experts"),
            r2.get("window_expert_iteration_debt_before"),
            float(r2.get("projected_expert_staleness_density", 0.0)),
            r2.get("max_expert_staleness_density"),
        )
        return descriptor

    def _terminate_spare_process(self, reason):
        proc = self.standby_proc
        if proc is None or proc.poll() is not None:
            self.standby_proc = None
            self.standby_prearmed = False
            self.standby_prearmed_ready = False
            self.standby_prearmed_epoch = None
            self.standby_prearmed_rank = None
            return
        log.warning(
            "Terminating spare/replacement pid=%s due to recovery shutdown (%s)",
            proc.pid,
            reason,
        )
        try:
            proc.terminate()
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
        except OSError:
            pass
        self.standby_proc = None
        self.standby_prearmed = False
        self.standby_prearmed_ready = False
        self.standby_prearmed_epoch = None
        self.standby_prearmed_rank = None

    def _reset_after_fallback(self):
        with self.lock:
            self.recovery_in_progress = False
            self.failed_node = None
            self.rebuild_ready_count = 0
            self.rebuild_ready_nodes = set()
            self.rebuild_ready_steps = {}
            self.rebuild_ready_step_tags = {}
            self.rebuild_ready_phases = {}
            self.rebuild_ready_rank_states = {}
            self.rebuild_ready_quiescence = {}
            self.rebuild_ready_control_owners = {}
            self.rank_quiescence_proof = {"status": "pending"}
            self.recovery_phases = {}
            self.ordinal_barriers = {}
            self.peer_sync_endpoints = {}
            self.rebuild_triggered = False
            self.replacement_ready_event.clear()
            self.phase_cv.notify_all()

    def _shutdown_connections_locked(self):
        """Return one control connection per node, preferring its launcher."""
        node_ranks = set(self.node_connections) | set(self.launcher_connections)
        return [
            (
                node_rank,
                self.launcher_connections.get(
                    node_rank, self.node_connections.get(node_rank)
                ),
            )
            for node_rank in sorted(node_ranks)
            if self.launcher_connections.get(
                node_rank, self.node_connections.get(node_rank)
            )
            is not None
        ]

    def _initiate_fallback_relaunch(self, reason, details=None):
        if not self.fallback_relaunch:
            self._initiate_recovery_abort(reason, details)
            return

        with self.lock:
            if self.fallback_initiated:
                log.info("Fallback relaunch already initiated; ignoring reason=%s", reason)
                return
            if not self.recovery_in_progress:
                log.info(
                    "Ignoring fallback relaunch outside recovery: reason=%s details=%s",
                    reason,
                    details,
                )
                return
            self.fallback_initiated = True
            recovery_epoch = self.recovery_epoch
            failed_node = self.failed_node
            killed_local_rank = self.killed_local_rank
            ready_steps = dict(self.rebuild_ready_steps)
            ready_step_tags = dict(self.rebuild_ready_step_tags)
            ready_phases = dict(self.rebuild_ready_phases)
            node_steps = dict(self.node_steps)
            node_step_tags = dict(self.node_step_tags)
            node_phases = dict(self.node_train_phases)
            recovery_phases = dict(self.recovery_phases)
            connections = self._shutdown_connections_locked()

        resume_iteration = self._select_fallback_resume_iteration(
            ready_steps,
            ready_step_tags,
            ready_phases,
        )
        self._truncate_staleness_history_for_restart(
            self._latest_checkpoint_iteration()
        )
        request = {
            "type": "fallback_relaunch",
            "action": "checkpoint_relaunch",
            "reason": reason,
            "details": details or {},
            "timestamp": time.time(),
            "recovery_epoch": recovery_epoch,
            "failed_node": failed_node,
            "killed_local_rank": killed_local_rank,
            "killed_global_rank": (
                failed_node * self.nproc_per_node + killed_local_rank
                if failed_node is not None and killed_local_rank >= 0
                else -1
            ),
            "resume_iteration": resume_iteration,
            "checkpoint_dir": os.environ.get("CKPT_DIR", ""),
            "exit_code": self.fallback_exit_code,
            "descriptor": str(self.recovery_descriptor_file),
            "ready_steps": ready_steps,
            "ready_step_tags": ready_step_tags,
            "ready_phases": ready_phases,
            "last_steps": node_steps,
            "last_step_tags": node_step_tags,
            "last_phases": node_phases,
            "recovery_phases": recovery_phases,
        }
        self._write_json_atomic(self.fallback_relaunch_file, request)
        epoch_path = self.fault_dir / f"fallback_relaunch_epoch_{recovery_epoch}.json"
        self._write_json_atomic(epoch_path, request)
        log.error(
            "Fallback relaunch requested: epoch=%s reason=%s resume_iteration=%s "
            "checkpoint_dir=%s manifest=%s",
            recovery_epoch,
            reason,
            resume_iteration,
            request["checkpoint_dir"],
            self.fallback_relaunch_file,
        )

        payload = json.dumps(request) + "\n"
        for node_rank, conn in connections:
            try:
                conn.sendall(payload.encode())
                log.warning("Sent fallback relaunch to node %s", node_rank)
            except (BrokenPipeError, OSError) as exc:
                log.warning("Failed to send fallback relaunch to node %s: %s", node_rank, exc)

        self._terminate_spare_process(reason)
        self._reset_after_fallback()
        if self.fallback_restart_standby:
            log.info(
                "Warm standby restart deferred until every launcher joins the next attempt"
            )

    def _initiate_recovery_abort(self, reason, details=None):
        """Terminate the whole training job after an unrecoverable recovery error."""
        with self.lock:
            if self.fallback_initiated:
                log.info("Recovery termination already initiated; ignoring reason=%s", reason)
                return
            if not self.recovery_in_progress:
                log.info(
                    "Ignoring recovery abort outside recovery: reason=%s details=%s",
                    reason,
                    details,
                )
                return
            # Reuse the existing shutdown latch so cascading worker failures do
            # not trigger a second fault epoch while launchers are exiting.
            self.fallback_initiated = True
            recovery_epoch = self.recovery_epoch
            failed_node = self.failed_node
            killed_local_rank = self.killed_local_rank
            connections = self._shutdown_connections_locked()
            recovery_phases = dict(self.recovery_phases)
            node_steps = dict(self.node_steps)
            node_step_tags = dict(self.node_step_tags)
            node_phases = dict(self.node_train_phases)
            self.phase_cv.notify_all()

        request = {
            # Keep the established transport message and signal filename; the
            # action and distinct exit code make this a terminal abort.
            "type": "fallback_relaunch",
            "action": "abort_training",
            "reason": reason,
            "details": details or {},
            "timestamp": time.time(),
            "recovery_epoch": recovery_epoch,
            "failed_node": failed_node,
            "killed_local_rank": killed_local_rank,
            "killed_global_rank": (
                failed_node * self.nproc_per_node + killed_local_rank
                if failed_node is not None and killed_local_rank >= 0
                else -1
            ),
            "exit_code": self.recovery_abort_exit_code,
            "descriptor": str(self.recovery_descriptor_file),
            "last_steps": node_steps,
            "last_step_tags": node_step_tags,
            "last_phases": node_phases,
            "recovery_phases": recovery_phases,
        }
        self._write_json_atomic(self.recovery_abort_file, request)
        epoch_path = self.fault_dir / f"recovery_abort_epoch_{recovery_epoch}.json"
        self._write_json_atomic(epoch_path, request)
        log.error(
            "Recovery failed; aborting all training launchers without relaunch: "
            "epoch=%s reason=%s exit_code=%s manifest=%s",
            recovery_epoch,
            reason,
            self.recovery_abort_exit_code,
            self.recovery_abort_file,
        )

        payload = (json.dumps(request) + "\n").encode()
        for node_rank, conn in connections:
            try:
                conn.sendall(payload)
                log.warning("Sent terminal recovery abort to node %s", node_rank)
            except (BrokenPipeError, OSError) as exc:
                log.warning("Failed to send terminal recovery abort to node %s: %s", node_rank, exc)

        self._terminate_spare_process(reason)
        self.exit_code = self.recovery_abort_exit_code
        self.running = False

    def _handle_fault(
        self,
        failed_node_rank,
        killed_local_rank=None,
        failure_scope="single_rank",
    ):
        """Handle a detected node failure (or planned fault injection).

        Sends PAUSE to ALL connected nodes.  For fault injection, this
        includes the target node — all nodes pause together, destroy their
        process groups, then the target rank is killed.
        """
        with self.lock:
            if self.fallback_initiated:
                log.info(
                    "Ignoring fault while checkpoint fallback is active: "
                    "node=%s local_rank=%s scope=%s",
                    failed_node_rank,
                    killed_local_rank,
                    failure_scope,
                )
                return
            if self.recovery_in_progress:
                log.info(
                    "Ignoring duplicate fault during recovery epoch=%s: "
                    "node=%s local_rank=%s scope=%s",
                    self.recovery_epoch,
                    failed_node_rank,
                    killed_local_rank,
                    failure_scope,
                )
                return
            self.recovery_epoch += 1
            recovery_epoch = self.recovery_epoch
            self.recovery_in_progress = True
            self.failed_node = failed_node_rank
            if killed_local_rank is None:
                killed_local_rank = self.fault_inject_local_rank
            self.killed_local_rank = killed_local_rank
            self.rebuild_ready_count = 0
            self.rebuild_ready_nodes = set()
            self.rebuild_ready_steps = {}
            self.rebuild_ready_step_tags = {}
            self.rebuild_ready_phases = {}
            self.rebuild_ready_rank_states = {}
            self.rebuild_ready_quiescence = {}
            self.rebuild_ready_control_owners = {}
            self.rank_quiescence_proof = {"status": "pending"}
            self.recovery_phases = {}
            self.ordinal_barriers = {}
            self.peer_sync_endpoints = {}
            self.rebuild_triggered = False
            connections = list(self.node_connections.items())

        if failure_scope != "single_rank":
            fallback_reason = (
                "node_failure_requires_full_relaunch"
                if failure_scope == "node"
                else "software_failure_requires_external_fallback"
            )
            log.error(
                "%s failure detected for node %s; single-rank hot swap is "
                "not admissible, requesting checkpoint relaunch",
                failure_scope,
                failed_node_rank,
            )
            self._write_recovery_descriptor(
                failed_node_rank,
                -1,
                recovery_epoch,
                resume_iteration=-1,
            )
            self._initiate_fallback_relaunch(
                fallback_reason,
                {
                    "failed_node": failed_node_rank,
                    "killed_local_rank": killed_local_rank,
                    "failure_scope": failure_scope,
                },
            )
            return

        self._write_recovery_descriptor(
            failed_node_rank,
            killed_local_rank,
            recovery_epoch,
            resume_iteration=-1,
        )

        # Write fault file (backup signal mechanism)
        fault_file = self.fault_dir / "latest"
        fault_file.write_text(json.dumps({
            "failed_node": failed_node_rank,
            "killed_local_rank": killed_local_rank,
            "recovery_epoch": recovery_epoch,
            "timestamp": time.time(),
            "action": "rebuild",
        }))

        log.info(f"Sending PAUSE signal to ALL {self.training_nnodes} nodes "
                 f"(target_node={failed_node_rank})")

        # Send pause signal to ALL connected nodes (including target)
        pause_msg = json.dumps({
            "type": "pause",
            "failed_node": failed_node_rank,
            "killed_local_rank": killed_local_rank,
            "recovery_epoch": recovery_epoch,
            "descriptor": str(self.recovery_descriptor_file),
        }) + "\n"
        for nr, conn in connections:
            try:
                conn.sendall(pause_msg.encode())
            except (BrokenPipeError, OSError) as e:
                log.warning(f"Failed to send pause to node {nr}: {e}")

        threading.Thread(
            target=self._watch_quiescence_timeout,
            args=(recovery_epoch,),
            daemon=True,
        ).start()

    def _watch_quiescence_timeout(self, recovery_epoch):
        timeout = float(os.environ.get("ELASTIC_QUIESCENCE_TIMEOUT_SECONDS", "300"))
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                if (
                    not self.recovery_in_progress
                    or self.recovery_epoch != recovery_epoch
                    or self.rebuild_triggered
                ):
                    return
            time.sleep(min(1.0, max(deadline - time.time(), 0.0)))

        with self.lock:
            if (
                not self.recovery_in_progress
                or self.recovery_epoch != recovery_epoch
                or self.rebuild_triggered
            ):
                return
            ready_nodes = sorted(self.rebuild_ready_nodes)
            missing_nodes = sorted(set(range(self.training_nnodes)) - set(ready_nodes))
        log.error(
            "Recovery epoch %s quiescence timeout after %.1fs: ready=%s missing=%s",
            recovery_epoch,
            timeout,
            ready_nodes,
            missing_nodes,
        )
        self._initiate_fallback_relaunch(
            "rank_quiescence_timeout",
            {
                "timeout": timeout,
                "ready_nodes": ready_nodes,
                "missing_nodes": missing_nodes,
            },
        )

    def _trigger_rebuild(self):
        """All nodes are paused and have destroyed process groups.

        The target rank has already been killed by its fault-isolated launcher
        after all local survivors reported a committed safe point.

        Now:
        1. Launch 1 spare process on this node
        2. Send rebuild signal to all surviving ranks
        """
        with self.lock:
            failed_node = self.failed_node
            killed_local_rank = self.killed_local_rank
            recovery_epoch = self.recovery_epoch
            ready_steps = dict(self.rebuild_ready_steps)
            ready_step_tags = dict(self.rebuild_ready_step_tags)
            ready_phases = dict(self.rebuild_ready_phases)

        if failed_node is None:
            log.warning("Rebuild trigger requested but failed_node is not set")
            return

        log.info(f"All {self.training_nnodes} nodes ready. "
                 f"Target rank already killed (node {failed_node} "
                 f"local_rank {killed_local_rank}). Launching spare...")
        resume_iteration = self._select_resume_iteration(
            ready_steps, ready_step_tags, ready_phases
        )
        killed_global_rank = failed_node * self.nproc_per_node + killed_local_rank
        self.rank_quiescence_proof = self._build_rank_quiescence_proof(
            resume_iteration, killed_global_rank
        )
        log.info("Rank quiescence proof: %s", self.rank_quiescence_proof)
        descriptor = self._write_recovery_descriptor(
            failed_node,
            killed_local_rank,
            recovery_epoch,
            resume_iteration=resume_iteration,
        )
        if not descriptor.get("recovery", {}).get("hot_swap_feasible", True):
            r2_contract = descriptor.get("contract", {}).get("R2", {})
            quiescence = descriptor.get("safe_point", {}).get("rank_quiescence", {})
            rejection_reason = (
                f"rank_quiescence:{quiescence.get('reason')}"
                if quiescence.get("status") == "invalid"
                else f"moegambit_contract:{r2_contract.get('reason', 'infeasible')}"
            )
            log.error(
                "Hot-swap recovery is not admitted for rank %s: reason=%s "
                "quiescence=%s gap=%s projected_density=%s. "
                "Falling back to checkpoint relaunch.",
                failed_node * self.nproc_per_node + killed_local_rank,
                rejection_reason,
                quiescence.get("status", "unknown"),
                r2_contract.get("expert_staleness_delta"),
                r2_contract.get("projected_expert_staleness_density"),
            )
            self._initiate_fallback_relaunch(
                rejection_reason,
                {
                    "failed_node": failed_node,
                    "killed_local_rank": killed_local_rank,
                    "failed_shard": descriptor.get("failed_shard", {}),
                    "contract": r2_contract,
                },
            )
            return
        log.info(
            "Elastic resume iteration selected: %s from ready steps=%s "
            "step_tags=%s phases=%s",
            resume_iteration,
            ready_steps,
            ready_step_tags,
            ready_phases,
        )

        # Step 1: Launch 1 spare process on this node
        self.replacement_ready_event.clear()
        spare_proc = self._launch_spare_worker(
            failed_node, killed_local_rank, resume_iteration
        )

        replacement_start_timeout = float(
            os.environ.get("ELASTIC_REPLACEMENT_START_TIMEOUT", "300")
        )
        log.info(
            "Waiting up to %.1fs for replacement to reach init_pg_start...",
            replacement_start_timeout,
        )
        if not self.replacement_ready_event.wait(timeout=replacement_start_timeout):
            rc = spare_proc.poll()
            if rc is None:
                log.error(
                    "Replacement did not reach init_pg_start within %.1fs; "
                    "process is still running with pid=%s. Not sending rebuild "
                    "signal to avoid deadlocking survivor ranks.",
                    replacement_start_timeout,
                    spare_proc.pid,
                )
                self._initiate_fallback_relaunch(
                    "replacement_start_timeout",
                    {
                        "pid": spare_proc.pid,
                        "timeout": replacement_start_timeout,
                        "failed_node": failed_node,
                        "killed_local_rank": killed_local_rank,
                    },
                )
            else:
                log.error(
                    "Replacement exited before init_pg_start with code %s. "
                    "Not sending rebuild signal to avoid deadlocking survivor ranks.",
                    rc,
                )
                self._log_recovery_phase_summary("replacement-start-failed")
                self._initiate_fallback_relaunch(
                    "replacement_start_exit",
                    {
                        "pid": spare_proc.pid,
                        "exit_code": rc,
                        "failed_node": failed_node,
                        "killed_local_rank": killed_local_rank,
                    },
                )
            return
        log.info("Replacement reached init_pg_start; broadcasting rebuild signal")

        # Step 2: Signal all nodes to rebuild
        r2_contract = descriptor.get("contract", {}).get("R2", {})
        descriptor_sha256 = hashlib.sha256(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        rebuild_msg = json.dumps({
            "type": "rebuild",
            "failed_node": failed_node,
            "killed_local_rank": killed_local_rank,
            "killed_global_rank": killed_global_rank,
            "recovery_epoch": recovery_epoch,
            "descriptor": str(self.recovery_descriptor_file),
            "descriptor_data": descriptor,
            "descriptor_sha256": descriptor_sha256,
            "resume_iteration": resume_iteration,
            "resume_ready_steps": ready_steps,
            "resume_step_tags": ready_step_tags,
            "resume_phases": ready_phases,
            "checkpoint_step": r2_contract.get("checkpoint_step", -1),
            "expert_staleness_delta": r2_contract.get("expert_staleness_delta", -1),
            "recovery_mode": "intentional_mixed_version",
            "two_phase_enabled": r2_contract.get("two_phase_enabled", False),
            "rank_quiescence": self.rank_quiescence_proof,
            "new_master_addr": self.master_addr,
            "new_master_port": str(self._rebuild_master_port(recovery_epoch)),
        }) + "\n"

        with self.lock:
            connections = list(self.node_connections.items())

        for nr, conn in connections:
            try:
                conn.sendall(rebuild_msg.encode())
            except (BrokenPipeError, OSError) as e:
                log.warning(f"Failed to send rebuild to node {nr}: {e}")

        log.info("Rebuild signal sent to all nodes. Waiting for training to resume...")
        with self.lock:
            self.rebuild_ready_count = 0
            self.rebuild_ready_nodes = set()
            self.rebuild_ready_steps = {}
            self.rebuild_ready_step_tags = {}
            self.rebuild_ready_phases = {}

    def _build_spare_env(
        self,
        failed_node,
        killed_local_rank,
        resume_iteration=-1,
        recovery_epoch=None,
    ):
        killed_global_rank = failed_node * self.nproc_per_node + killed_local_rank
        physical_node_rank = os.environ.get("NODE_RANK", str(self.training_nnodes))
        if recovery_epoch is None:
            recovery_epoch = self.recovery_epoch

        env = os.environ.copy()
        # The replacement uses our standalone TCPStore, not torchrun's agent
        # rendezvous.  An inherited agent-store flag adds a private PrefixStore
        # on replacement only and makes survivor NCCL bootstrap keys invisible.
        for key in (
            "TORCHELASTIC_USE_AGENT_STORE",
            "TORCHELASTIC_RUN_ID",
            "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_MAX_RESTARTS",
            "TORCHELASTIC_ERROR_FILE",
            "TORCHELASTIC_ROLE",
            "TORCHELASTIC_ROLE_RANK",
            "TORCHELASTIC_ROLE_WORLD_SIZE",
        ):
            env.pop(key, None)
        env["ELASTIC_LOGICAL_NODE_RANK"] = str(failed_node)
        env["ELASTIC_PHYSICAL_NODE_RANK"] = str(physical_node_rank)
        env["NODE_RANK"] = str(failed_node)
        env["LOCAL_RANK"] = "0"  # Only 1 GPU visible, so local device is always 0
        env["LOCAL_WORLD_SIZE"] = "1"
        env["GROUP_RANK"] = str(physical_node_rank)
        env["RANK"] = str(killed_global_rank)
        env["ELASTIC_REPLACEMENT_RANK"] = str(killed_global_rank)
        env["ELASTIC_RESUME_ITERATION"] = str(resume_iteration)
        env["ELASTIC_RECOVERY_EPOCH"] = str(recovery_epoch)
        env["ELASTIC_PG_GENERATION"] = str(recovery_epoch)
        env["ELASTIC_RECOVERY_DESCRIPTOR"] = str(self.recovery_descriptor_file)
        try:
            with self.recovery_descriptor_file.open("r", encoding="utf-8") as f:
                descriptor = json.load(f)
            r2_contract = descriptor.get("contract", {}).get("R2", {})
            env["ELASTIC_RECOVERY_DESCRIPTOR_SHA256"] = hashlib.sha256(
                json.dumps(
                    descriptor, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
        except (OSError, ValueError, TypeError):
            r2_contract = {}
        env["ELASTIC_CHECKPOINT_STEP"] = str(r2_contract.get("checkpoint_step", -1))
        env["ELASTIC_EXPERT_STALENESS_DELTA"] = str(
            r2_contract.get("expert_staleness_delta", -1)
        )
        env["ELASTIC_MOEGAMBIT_RECOVERY_MODE"] = "intentional_mixed_version"
        env["ELASTIC_TWO_PHASE_RECOVERY"] = (
            "1" if r2_contract.get("two_phase_enabled", False) else "0"
        )
        env["MASTER_ADDR"] = self.master_addr
        env["MASTER_PORT"] = str(self._rebuild_master_port(recovery_epoch))
        env["ELASTIC_WATCHER_ADDR"] = os.environ.get(
            "ELASTIC_REPLACEMENT_WATCHER_ADDR", "127.0.0.1"
        )
        env["ELASTIC_WATCHER_PORT"] = str(self.port)
        env["ELASTIC_REBUILD_MODE"] = "1"
        env["ELASTIC_SELECTIVE_GROUP_REBUILD"] = os.environ.get(
            "ELASTIC_SELECTIVE_GROUP_REBUILD",
            "1",
        )
        env["NNODES"] = str(self.training_nnodes)
        env["ELASTIC_TRAINING_NPROC_PER_NODE"] = str(self.nproc_per_node)
        env["WORLD_SIZE"] = str(self.training_nnodes * self.nproc_per_node)
        env["PYTHONUNBUFFERED"] = "1"
        env["DISTRIBUTED_TIMEOUT_MINUTES"] = os.environ.get(
            "DISTRIBUTED_TIMEOUT_MINUTES",
            os.environ.get("ELASTIC_REBUILD_TIMEOUT_MINUTES", "10"),
        )
        env["ELASTIC_REBUILD_TIMEOUT_MINUTES"] = os.environ.get(
            "ELASTIC_REBUILD_TIMEOUT_MINUTES",
            env["DISTRIBUTED_TIMEOUT_MINUTES"],
        )
        phase_timeout = os.environ.get(
            "ELASTIC_PHASE_TIMEOUT_SECONDS",
            os.environ.get("ELASTIC_REBUILD_PHASE_TIMEOUT"),
        )
        if not phase_timeout:
            try:
                phase_timeout = str(int(env["DISTRIBUTED_TIMEOUT_MINUTES"]) * 60 + 120)
            except (TypeError, ValueError):
                phase_timeout = "720"
        env["ELASTIC_PHASE_TIMEOUT_SECONDS"] = phase_timeout
        env["ELASTIC_REBUILD_PHASE_TIMEOUT"] = os.environ.get(
            "ELASTIC_REBUILD_PHASE_TIMEOUT",
            phase_timeout,
        )
        env["ELASTIC_TRACE_REPLACEMENT_GROUP_MEMBERS"] = os.environ.get(
            "ELASTIC_TRACE_REPLACEMENT_GROUP_MEMBERS",
            "1",
        )
        env["ELASTIC_MPU_GROUP_ORDINAL_BARRIER"] = os.environ.get(
            "ELASTIC_MPU_GROUP_ORDINAL_BARRIER",
            "1",
        )
        env["ELASTIC_MPU_GROUP_ORDINAL_TIMEOUT_SECONDS"] = os.environ.get(
            "ELASTIC_MPU_GROUP_ORDINAL_TIMEOUT_SECONDS",
            env["ELASTIC_PHASE_TIMEOUT_SECONDS"],
        )
        env["ELASTIC_INIT_PG_DEVICE_ID"] = os.environ.get(
            "ELASTIC_INIT_PG_DEVICE_ID",
            "0",
        )
        env["ELASTIC_REBUILD_INIT_PG_DEVICE_ID"] = os.environ.get(
            "ELASTIC_REBUILD_INIT_PG_DEVICE_ID",
            "0",
        )
        env["ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER"] = os.environ.get(
            "ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER",
            "1",
        )
        env["ELASTIC_MOE_FIRST_COLLECTIVE_FAIL_FAST"] = os.environ.get(
            "ELASTIC_MOE_FIRST_COLLECTIVE_FAIL_FAST",
            "0",
        )
        env["ELASTIC_MOE_FIRST_COLLECTIVE_TIMEOUT"] = os.environ.get(
            "ELASTIC_MOE_FIRST_COLLECTIVE_TIMEOUT",
            "70",
        )
        env["ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER_TIMEOUT"] = os.environ.get(
            "ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER_TIMEOUT",
            env["ELASTIC_PHASE_TIMEOUT_SECONDS"],
        )
        env["ELASTIC_RECOVERY_NCCL_SOCKET_ONLY"] = os.environ.get(
            "ELASTIC_RECOVERY_NCCL_SOCKET_ONLY",
            "1",
        )
        env["ELASTIC_RECOVERY_NCCL_DEBUG"] = os.environ.get(
            "ELASTIC_RECOVERY_NCCL_DEBUG",
            "WARN",
        )
        if os.environ.get("ELASTIC_RECOVERY_NCCL_SOCKET_IFNAME"):
            env["ELASTIC_RECOVERY_NCCL_SOCKET_IFNAME"] = os.environ[
                "ELASTIC_RECOVERY_NCCL_SOCKET_IFNAME"
            ]
        # Use the specific GPU that corresponds to the killed local_rank
        env["CUDA_VISIBLE_DEVICES"] = str(killed_local_rank)
        return env

    def _script_cmd(self):
        repository = Path(__file__).resolve().parents[4]
        return ["bash", str(repository / "run_spare_single_rank.sh")]

    def _log_spare_output(self, proc, label):
        for line in iter(proc.stdout.readline, b""):
            log.info(f"[{label}] {line.decode().rstrip()}")
        proc.wait()
        log.info(f"{label} exited with code {proc.returncode}")
        if proc.returncode != 0:
            self._log_recovery_phase_summary(f"{label}-exit")
            self._initiate_fallback_relaunch(
                f"{label}-exit",
                {"pid": proc.pid, "exit_code": proc.returncode},
            )

    def _start_standby_worker(self):
        if os.environ.get("ELASTIC_PRESTART_SPARE", "1") != "1":
            log.info("Warm standby disabled (ELASTIC_PRESTART_SPARE != 1)")
            return

        if self.standby_proc is not None and self.standby_proc.poll() is None:
            return

        try:
            self.standby_assignment_file.unlink()
        except OSError:
            pass

        env = os.environ.copy()
        env["ELASTIC_SPARE_ASSIGNMENT_FILE"] = str(self.standby_assignment_file)
        env["ELASTIC_WATCHER_ADDR"] = os.environ.get(
            "ELASTIC_REPLACEMENT_WATCHER_ADDR", "127.0.0.1"
        )
        env["ELASTIC_WATCHER_PORT"] = str(self.port)
        env["PYTHONUNBUFFERED"] = "1"

        prearm_planned = (
            os.environ.get("ELASTIC_PREARM_PLANNED_SPARE", "1") == "1"
            and self.fault_inject_step >= 0
            and self.fault_inject_node >= 0
            and self.fault_inject_local_rank >= 0
        )
        if prearm_planned:
            next_epoch = self.recovery_epoch + 1
            env = self._build_spare_env(
                self.fault_inject_node,
                self.fault_inject_local_rank,
                resume_iteration=-1,
                recovery_epoch=next_epoch,
            )
            env["ELASTIC_PREARMED_STANDBY"] = "1"
            env.pop("ELASTIC_STANDBY_MODE", None)
            env["ELASTIC_SPARE_ASSIGNMENT_FILE"] = str(
                self.standby_assignment_file
            )
            standby_store_timeout_minutes = os.environ.get(
                "ELASTIC_STANDBY_STORE_TIMEOUT_MINUTES",
                "1440",
            )
            env["ELASTIC_STANDBY_ASSIGNMENT_TIMEOUT_SECONDS"] = os.environ.get(
                "ELASTIC_STANDBY_ASSIGNMENT_TIMEOUT_SECONDS",
                str(float(standby_store_timeout_minutes) * 60.0),
            )
            env["ELASTIC_REBUILD_TIMEOUT_MINUTES"] = standby_store_timeout_minutes
            self.standby_prearmed = True
            self.standby_prearmed_ready = False
            self.standby_prearmed_epoch = next_epoch
            self.standby_prearmed_rank = int(env["RANK"])
        else:
            env["ELASTIC_STANDBY_MODE"] = "1"
            self.standby_prearmed = False
            self.standby_prearmed_ready = False
            self.standby_prearmed_epoch = None
            self.standby_prearmed_rank = None

        proc = subprocess.Popen(
            self._script_cmd(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.standby_proc = proc
        if prearm_planned:
            log.info(
                "Prearmed GPU standby pid=%s target_rank=%s epoch=%s "
                "store=%s:%s gpu=%s",
                proc.pid,
                env["RANK"],
                next_epoch,
                env["MASTER_ADDR"],
                env["MASTER_PORT"],
                env["CUDA_VISIBLE_DEVICES"],
            )
        else:
            log.info(
                "Warm standby worker pid=%s started; waiting on %s",
                proc.pid,
                self.standby_assignment_file,
            )
        t = threading.Thread(
            target=self._log_spare_output,
            args=(proc, "spare-standby"),
            daemon=True,
        )
        t.start()

    def _activate_standby_worker(self, env):
        if self.standby_proc is None or self.standby_proc.poll() is not None:
            self.standby_prearmed = False
            self.standby_prearmed_ready = False
            self.standby_prearmed_epoch = None
            self.standby_prearmed_rank = None
            return None
        if self.standby_prearmed and (
            int(env["RANK"]) != self.standby_prearmed_rank
            or int(env["ELASTIC_RECOVERY_EPOCH"]) != self.standby_prearmed_epoch
        ):
            log.warning(
                "Prearmed standby role does not match failure: armed_rank=%s "
                "armed_epoch=%s actual_rank=%s actual_epoch=%s; using cold launch",
                self.standby_prearmed_rank,
                self.standby_prearmed_epoch,
                env["RANK"],
                env["ELASTIC_RECOVERY_EPOCH"],
            )
            self._terminate_spare_process("prearmed-role-mismatch")
            return None

        assignment = {
            key: value
            for key, value in env.items()
            if key
            in {
                "NODE_RANK",
                "ELASTIC_LOGICAL_NODE_RANK",
                "ELASTIC_PHYSICAL_NODE_RANK",
                "LOCAL_RANK",
                "LOCAL_WORLD_SIZE",
                "GROUP_RANK",
                "RANK",
                "ELASTIC_REPLACEMENT_RANK",
                "ELASTIC_RESUME_ITERATION",
                "ELASTIC_RECOVERY_EPOCH",
                "ELASTIC_PG_GENERATION",
                "ELASTIC_RECOVERY_DESCRIPTOR",
                "ELASTIC_RECOVERY_DESCRIPTOR_SHA256",
                "ELASTIC_CHECKPOINT_STEP",
                "ELASTIC_EXPERT_STALENESS_DELTA",
                "ELASTIC_MOEGAMBIT_RECOVERY_MODE",
                "ELASTIC_TWO_PHASE_RECOVERY",
                "MASTER_ADDR",
                "MASTER_PORT",
                "ELASTIC_WATCHER_ADDR",
                "ELASTIC_WATCHER_PORT",
                "ELASTIC_REBUILD_MODE",
                "ELASTIC_SELECTIVE_GROUP_REBUILD",
                "NNODES",
                "ELASTIC_TRAINING_NPROC_PER_NODE",
                "WORLD_SIZE",
                "PYTHONUNBUFFERED",
                "DISTRIBUTED_TIMEOUT_MINUTES",
                "ELASTIC_REBUILD_TIMEOUT_MINUTES",
                "ELASTIC_PHASE_TIMEOUT_SECONDS",
                "ELASTIC_REBUILD_PHASE_TIMEOUT",
                "ELASTIC_TRACE_REPLACEMENT_GROUP_MEMBERS",
                "ELASTIC_MPU_GROUP_ORDINAL_BARRIER",
                "ELASTIC_MPU_GROUP_ORDINAL_TIMEOUT_SECONDS",
                "ELASTIC_INIT_PG_DEVICE_ID",
                "ELASTIC_REBUILD_INIT_PG_DEVICE_ID",
                "ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER",
                "ELASTIC_MOE_FIRST_COLLECTIVE_FAIL_FAST",
                "ELASTIC_MOE_FIRST_COLLECTIVE_TIMEOUT",
                "ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER_TIMEOUT",
                "ELASTIC_RECOVERY_NCCL_SOCKET_ONLY",
                "ELASTIC_RECOVERY_NCCL_DEBUG",
                "ELASTIC_RECOVERY_NCCL_SOCKET_IFNAME",
                "CUDA_VISIBLE_DEVICES",
            }
        }
        self.fault_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.standby_assignment_file.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(assignment, f)
        os.replace(tmp_path, self.standby_assignment_file)
        if self.standby_prearmed and self.standby_prearmed_ready:
            self.replacement_ready_event.set()
        log.info(
            "Activated %s standby pid=%s for replacement rank=%s",
            "prearmed GPU" if self.standby_prearmed else "warm shell",
            self.standby_proc.pid,
            env["RANK"],
        )
        return self.standby_proc

    def _launch_spare_worker(self, failed_node, killed_local_rank, resume_iteration=-1):
        """Launch or activate 1 replacement process on this spare node.

        The replacement process takes over the killed rank's global position.
        """
        killed_global_rank = failed_node * self.nproc_per_node + killed_local_rank
        log.info(f"Launching spare: replacing global_rank={killed_global_rank} "
                 f"(node {failed_node}, local_rank {killed_local_rank})")

        env = self._build_spare_env(failed_node, killed_local_rank, resume_iteration)
        proc = self._activate_standby_worker(env)
        if proc is not None:
            return proc

        # Launch a single training process (not elastic_launcher with 8 workers)
        proc = subprocess.Popen(
            self._script_cmd(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log.info(
            "Spare worker pid=%s started (watcher_addr=%s:%s)",
            proc.pid,
            env["ELASTIC_WATCHER_ADDR"],
            env["ELASTIC_WATCHER_PORT"],
        )

        # Log output in background
        t = threading.Thread(target=self._log_spare_output, args=(proc, "spare-worker"), daemon=True)
        t.start()

        return proc


def main(argv=None):
    parser = argparse.ArgumentParser(description="Elastic Watcher for hot-spare recovery")
    parser.add_argument("--port", type=int, default=20200)
    parser.add_argument("--training-nnodes", type=int, default=8)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=str, default="20115")
    parser.add_argument("--fault-dir", type=str, default="/tmp/elastic_faults")
    parser.add_argument("--heartbeat-timeout", type=float, default=30.0,
                        help="Seconds without heartbeat before declaring node dead")
    parser.add_argument(
        "--startup-heartbeat-timeout",
        type=float,
        default=float(os.environ.get("ELASTIC_STARTUP_HEARTBEAT_TIMEOUT", "600")),
        help="Seconds without heartbeat tolerated before all nodes enter training",
    )
    parser.add_argument(
        "--forward-heartbeat-timeout",
        type=float,
        default=float(os.environ.get("ELASTIC_FORWARD_HEARTBEAT_TIMEOUT", "180")),
        help="Seconds without heartbeat tolerated during a long forward/backward step",
    )
    parser.add_argument(
        "--checkpoint-heartbeat-timeout",
        type=float,
        default=float(os.environ.get("ELASTIC_CHECKPOINT_HEARTBEAT_TIMEOUT", "900")),
        help="Seconds without heartbeat tolerated while saving or finalizing checkpoints",
    )
    parser.add_argument(
        "--disconnect-grace-timeout",
        type=float,
        default=float(os.environ.get("ELASTIC_DISCONNECT_GRACE_TIMEOUT", "15")),
        help="Seconds to wait for a node to reconnect after TCP EOF/reset",
    )
    parser.add_argument(
        "--fallback-relaunch",
        type=_str_to_bool,
        default=_env_bool("ELASTIC_FALLBACK_RELAUNCH", False),
        help="When true, failed recovery epochs ask training nodes to relaunch from checkpoint",
    )
    parser.add_argument(
        "--fallback-exit-code",
        type=int,
        default=int(os.environ.get("ELASTIC_FALLBACK_EXIT_CODE", "75")),
        help="Exit code requested for local fallback relaunch exits",
    )
    parser.add_argument(
        "--recovery-abort-exit-code",
        type=int,
        default=int(os.environ.get("ELASTIC_RECOVERY_ABORT_EXIT_CODE", "76")),
        help="Terminal exit code requested when recovery fails and fallback is disabled",
    )
    parser.add_argument(
        "--fallback-restart-standby",
        type=_str_to_bool,
        default=_env_bool("ELASTIC_FALLBACK_RESTART_STANDBY", False),
        help="Restart a warm standby process after fallback relaunch is requested",
    )
    # Fault injection args
    parser.add_argument("--fault-inject-step", type=int, default=-1,
                        help="Step at which to inject a fault (-1 = disabled)")
    parser.add_argument("--fault-inject-node", type=int, default=0,
                        help="Node rank to kill for fault injection (default: 0)")
    parser.add_argument("--fault-inject-local-rank", type=int, default=0,
                        help="Local rank to kill on the target node (default: 0)")
    args = parser.parse_args(argv)

    watcher = ElasticWatcher(args)
    return watcher.start()


if __name__ == "__main__":
    raise SystemExit(main())

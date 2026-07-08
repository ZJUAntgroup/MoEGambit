#!/usr/bin/env python3
"""
elastic_watcher.py — Hot-spare watcher daemon (FlashRecovery style).

Runs on the spare node (NODE_RANK=8). Does NOT join torch.distributed.
Responsibilities:
  1. Accept heartbeat connections from training rank 0 (one per node)
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
"""

import argparse
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

_PHASE_ORDER = {
    "init_pg_start": 10,
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
    "param_sync_start": 70,
    "param_sync_done": 80,
    "comm_warmup_start": 85,
    "comm_warmup_done": 88,
    "train_ready": 90,
    "resume_state_applied": 92,
    "data_ready": 94,
    "train_loop_entered": 100,
    "post_rebuild_iteration_ready": 110,
    "post_rebuild_comm_warmup_start": 112,
    "post_rebuild_comm_warmup_done": 114,
    "post_rebuild_moe_comm_warmup_start": 112,
    "post_rebuild_moe_comm_warmup_done": 114,
    "forward_backward_start": 120,
    "moe_first_collective_start": 122,
    "moe_first_collective_done": 124,
    "moe_first_collective_error": 126,
    "moe_first_collective_timeout": 128,
    "forward_backward_done": 130,
    "optimizer_step_start": 140,
    "optimizer_step_done": 150,
    "optimizer_skipped": 155,
    "post_rebuild_step_complete": 160,
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

        # Fault injection config
        self.fault_inject_step = args.fault_inject_step
        self.fault_inject_node = args.fault_inject_node
        self.fault_inject_local_rank = getattr(args, 'fault_inject_local_rank', 0)
        self.fault_injected = False

        # State
        self.running = True
        self.node_connections = {}  # node_rank -> socket
        self.last_heartbeat = {}   # node_rank -> timestamp
        self.node_disconnected_at = {}  # node_rank -> TCP EOF/reset timestamp
        self.node_steps = {}       # node_rank -> last reported step
        self.node_step_tags = {}   # node_rank -> FlashRecovery step tag
        self.node_train_phases = {}  # node_rank -> forward_backward / optimizer_step / step_complete
        self.failed_node = None
        self.killed_local_rank = 0
        self.recovery_in_progress = False
        self.rebuild_ready_count = 0
        self.rebuild_ready_nodes = set()
        self.rebuild_ready_steps = {}
        self.rebuild_ready_step_tags = {}
        self.rebuild_ready_phases = {}
        self.recovery_phases = {}
        self.ordinal_barriers = {}
        self.peer_sync_endpoints = {}
        self.rebuild_triggered = False
        self.replacement_ready_event = threading.Event()
        self.standby_proc = None
        self.standby_assignment_file = self.fault_dir / "spare_assignment.json"
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
            conn.close()

    def _process_message(self, msg, conn, addr=None):
        """Process a message from a training node. Returns node_rank."""
        msg_type = msg.get("type")
        node_rank = msg.get("node_rank")

        if msg_type == "heartbeat":
            with self.lock:
                self.node_connections[node_rank] = conn
                self.last_heartbeat[node_rank] = time.time()
                self.node_disconnected_at.pop(node_rank, None)
                step = msg.get("step", -1)
                step_tag = msg.get("step_tag", step)
                train_phase = msg.get("train_phase", "unknown")
                self.node_steps[node_rank] = step
                self.node_step_tags[node_rank] = step_tag
                self.node_train_phases[node_rank] = train_phase

            # Check for step-based fault injection
            self._maybe_inject_fault(node_rank, msg.get("step", -1))
            return node_rank

        elif msg_type == "ready_to_rebuild":
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
                log.info("NCCL error report received but no recovery in progress yet — "
                         "waiting for heartbeat timeout to confirm")
            return node_rank

        elif msg_type == "recovery_phase":
            rank = msg.get("rank", "?")
            phase = msg.get("phase", "?")
            role = msg.get("role", "survivor")
            step = msg.get("step", -1)
            key = f"{role}:{rank}"
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
                if role == "replacement" and phase in ("init_pg_start", "pg_ready"):
                    self.replacement_ready_event.set()
                if (
                    phase == "train_ready"
                    and self._phase_count_locked("train_ready")
                    >= self.training_nnodes * self.nproc_per_node
                ):
                    self.recovery_in_progress = False
                    self.failed_node = None
                    self.rebuild_ready_count = 0
                    self.rebuild_ready_nodes = set()
                    self.rebuild_ready_steps = {}
                    self.rebuild_ready_step_tags = {}
                    self.rebuild_ready_phases = {}
                    self.ordinal_barriers = {}
                    log.info("All training ranks reached train_ready; recovery complete")
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
            return node_rank

        elif msg_type == "wait_phase":
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
            return node_rank

        elif msg_type == "wait_phase_count":
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
            return node_rank

        elif msg_type == "ordinal_barrier":
            barrier_id = msg.get("barrier_id", "?")
            rank = int(msg.get("rank", -1))
            min_count = int(msg.get("min_count", self.training_nnodes * self.nproc_per_node))
            timeout = float(msg.get("timeout", 300.0))
            count, missing, arrived = self._wait_for_ordinal_barrier(
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
                "ok": count >= min_count,
                "missing": missing,
                "arrived": arrived,
            }) + "\n"
            try:
                conn.sendall(response.encode())
            except OSError:
                pass
            return node_rank

        elif msg_type == "peer_sync_endpoint":
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
            return node_rank

        else:
            log.warning(f"Unknown message type: {msg_type}")
            return node_rank

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
                    "mismatch_logged": False,
                    "released_logged": False,
                },
            )
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
                    "group_size",
                    "group_ranks",
                    "group_first_rank",
                    "group_last_rank",
                    "group_timeout_seconds",
                    "group_ordinal",
                    "barrier_stage",
                )
                if k in msg
            }
            state["manifest_by_rank"][rank] = json.dumps(manifest, sort_keys=True)
            if (
                len(set(state["manifest_by_rank"].values())) > 1
                and not state["mismatch_logged"]
            ):
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

            while len(state["arrived"]) < min_count:
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
                    return len(arrived), missing, arrived
                self.phase_cv.wait(timeout=min(remaining, 1.0))

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
            return len(arrived), missing, arrived

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
        self._handle_fault(target)

    def _training_started_locked(self):
        """Return True only after all training nodes have entered train-loop heartbeats."""
        if len(self.last_heartbeat) < self.training_nnodes:
            return False
        for node_rank in range(self.training_nnodes):
            if self.node_train_phases.get(node_rank, "startup") == "startup":
                return False
        return True

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
            if self.recovery_in_progress:
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
                self._handle_fault(timed_out_node)

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
            "group_size",
            "group_ranks",
            "group_first_rank",
            "group_last_rank",
            "group_representative_rank",
            "group_report_rank",
            "group_timeout_seconds",
            "replacement_rank",
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

    def _handle_fault(self, failed_node_rank):
        """Handle a detected node failure (or planned fault injection).

        Sends PAUSE to ALL connected nodes.  For fault injection, this
        includes the target node — all nodes pause together, destroy their
        process groups, then the target rank is killed.
        """
        with self.lock:
            self.recovery_in_progress = True
            self.failed_node = failed_node_rank
            killed_local_rank = self.fault_inject_local_rank
            self.killed_local_rank = killed_local_rank
            self.rebuild_ready_count = 0
            self.rebuild_ready_nodes = set()
            self.rebuild_ready_steps = {}
            self.rebuild_ready_step_tags = {}
            self.rebuild_ready_phases = {}
            self.recovery_phases = {}
            self.ordinal_barriers = {}
            self.peer_sync_endpoints = {}
            self.rebuild_triggered = False
            connections = list(self.node_connections.items())

        # Write fault file (backup signal mechanism)
        fault_file = self.fault_dir / "latest"
        fault_file.write_text(json.dumps({
            "failed_node": failed_node_rank,
            "killed_local_rank": killed_local_rank,
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
        }) + "\n"
        for nr, conn in connections:
            try:
                conn.sendall(pause_msg.encode())
            except (BrokenPipeError, OSError) as e:
                log.warning(f"Failed to send pause to node {nr}: {e}")

    def _trigger_rebuild(self):
        """All nodes are paused and have destroyed process groups.

        The target rank has already been killed by the target node itself
        (in elastic_do_rebuild, before sending ready_to_rebuild).

        Now:
        1. Launch 1 spare process on this node
        2. Send rebuild signal to all surviving ranks
        """
        with self.lock:
            failed_node = self.failed_node
            killed_local_rank = self.killed_local_rank
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
            else:
                log.error(
                    "Replacement exited before init_pg_start with code %s. "
                    "Not sending rebuild signal to avoid deadlocking survivor ranks.",
                    rc,
                )
                self._log_recovery_phase_summary("replacement-start-failed")
            return
        log.info("Replacement reached init_pg_start; broadcasting rebuild signal")

        # Step 2: Signal all nodes to rebuild
        killed_global_rank = failed_node * self.nproc_per_node + killed_local_rank
        rebuild_msg = json.dumps({
            "type": "rebuild",
            "failed_node": failed_node,
            "killed_local_rank": killed_local_rank,
            "killed_global_rank": killed_global_rank,
            "resume_iteration": resume_iteration,
            "resume_ready_steps": ready_steps,
            "resume_step_tags": ready_step_tags,
            "resume_phases": ready_phases,
            "new_master_addr": self.master_addr,
            "new_master_port": str(int(self.master_port) + 1),
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

    def _build_spare_env(self, failed_node, killed_local_rank, resume_iteration=-1):
        killed_global_rank = failed_node * self.nproc_per_node + killed_local_rank
        physical_node_rank = os.environ.get("NODE_RANK", str(self.training_nnodes))

        env = os.environ.copy()
        env["ELASTIC_LOGICAL_NODE_RANK"] = str(failed_node)
        env["ELASTIC_PHYSICAL_NODE_RANK"] = str(physical_node_rank)
        env["NODE_RANK"] = str(failed_node)
        env["LOCAL_RANK"] = "0"  # Only 1 GPU visible, so local device is always 0
        env["LOCAL_WORLD_SIZE"] = "1"
        env["GROUP_RANK"] = str(physical_node_rank)
        env["RANK"] = str(killed_global_rank)
        env["ELASTIC_REPLACEMENT_RANK"] = str(killed_global_rank)
        env["ELASTIC_RESUME_ITERATION"] = str(resume_iteration)
        env["MASTER_ADDR"] = self.master_addr
        env["MASTER_PORT"] = str(int(self.master_port) + 1)  # Rebuild uses new port
        env["ELASTIC_WATCHER_ADDR"] = os.environ.get(
            "ELASTIC_REPLACEMENT_WATCHER_ADDR", "127.0.0.1"
        )
        env["ELASTIC_WATCHER_PORT"] = str(self.port)
        env["ELASTIC_REBUILD_MODE"] = "1"
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
        env["ELASTIC_POST_REBUILD_COMM_WARMUP"] = os.environ.get(
            "ELASTIC_POST_REBUILD_COMM_WARMUP",
            "0",
        )
        env["ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER"] = os.environ.get(
            "ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER",
            "1",
        )
        env["ELASTIC_MOE_FIRST_COLLECTIVE_WARMUP"] = os.environ.get(
            "ELASTIC_MOE_FIRST_COLLECTIVE_WARMUP",
            "0",
        )
        env["ELASTIC_MOE_FIRST_COLLECTIVE_FAIL_FAST"] = os.environ.get(
            "ELASTIC_MOE_FIRST_COLLECTIVE_FAIL_FAST",
            "1",
        )
        env["ELASTIC_MOE_FIRST_COLLECTIVE_TIMEOUT"] = os.environ.get(
            "ELASTIC_MOE_FIRST_COLLECTIVE_TIMEOUT",
            "180",
        )
        env["ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER_TIMEOUT"] = os.environ.get(
            "ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER_TIMEOUT",
            env["ELASTIC_PHASE_TIMEOUT_SECONDS"],
        )
        # Use the specific GPU that corresponds to the killed local_rank
        env["CUDA_VISIBLE_DEVICES"] = str(killed_local_rank)
        return env

    def _script_cmd(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return ["bash", os.path.join(script_dir, "run_spare_single_rank.sh")]

    def _log_spare_output(self, proc, label):
        for line in iter(proc.stdout.readline, b""):
            log.info(f"[{label}] {line.decode().rstrip()}")
        proc.wait()
        log.info(f"{label} exited with code {proc.returncode}")
        if proc.returncode != 0:
            self._log_recovery_phase_summary(f"{label}-exit")

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
        env["ELASTIC_STANDBY_MODE"] = "1"
        env["ELASTIC_SPARE_ASSIGNMENT_FILE"] = str(self.standby_assignment_file)
        env["ELASTIC_WATCHER_ADDR"] = os.environ.get(
            "ELASTIC_REPLACEMENT_WATCHER_ADDR", "127.0.0.1"
        )
        env["ELASTIC_WATCHER_PORT"] = str(self.port)
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            self._script_cmd(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.standby_proc = proc
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
                "MASTER_ADDR",
                "MASTER_PORT",
                "ELASTIC_WATCHER_ADDR",
                "ELASTIC_WATCHER_PORT",
                "ELASTIC_REBUILD_MODE",
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
                "ELASTIC_POST_REBUILD_COMM_WARMUP",
                "ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER",
                "ELASTIC_MOE_FIRST_COLLECTIVE_WARMUP",
                "ELASTIC_MOE_FIRST_COLLECTIVE_FAIL_FAST",
                "ELASTIC_MOE_FIRST_COLLECTIVE_TIMEOUT",
                "ELASTIC_MOE_FIRST_COLLECTIVE_BARRIER_TIMEOUT",
                "CUDA_VISIBLE_DEVICES",
            }
        }
        self.fault_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.standby_assignment_file.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(assignment, f)
        os.replace(tmp_path, self.standby_assignment_file)
        log.info(
            "Activated warm standby pid=%s for replacement rank=%s",
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


def main():
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
    # Fault injection args
    parser.add_argument("--fault-inject-step", type=int, default=-1,
                        help="Step at which to inject a fault (-1 = disabled)")
    parser.add_argument("--fault-inject-node", type=int, default=0,
                        help="Node rank to kill for fault injection (default: 0)")
    parser.add_argument("--fault-inject-local-rank", type=int, default=0,
                        help="Local rank to kill on the target node (default: 0)")
    args = parser.parse_args()

    watcher = ElasticWatcher(args)
    watcher.start()


if __name__ == "__main__":
    main()

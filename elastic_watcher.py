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
  6. Kill the target rank on the target node
  7. Launch 1 replacement process on spare node with the dead rank
  8. Coordinate group rebuild (all ranks re-init_process_group)

Fault injection flow (safe, no NCCL timeout):
  1. Watcher sees step >= fault_inject_step
  2. Watcher sends PAUSE to ALL training nodes (including target)
  3. All nodes reach safe point → destroy_process_group → send ready_to_rebuild
  4. Watcher receives 8/8 ready_to_rebuild
  5. Watcher sends kill_rank to target node (kills local_rank 0 only)
  6. Watcher launches 1 spare process on this node
  7. Watcher sends rebuild signal to all surviving ranks
  8. All ranks re-init_process_group → sync params → resume

Protocol (TCP, JSON lines):
  - Training → Watcher: {"type": "heartbeat", "node_rank": N, "step": S}
  - Training → Watcher: {"type": "ready_to_rebuild", "node_rank": N}
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

        # Fault injection config
        self.fault_inject_step = args.fault_inject_step
        self.fault_inject_node = args.fault_inject_node
        self.fault_inject_local_rank = getattr(args, 'fault_inject_local_rank', 0)
        self.fault_injected = False

        # State
        self.running = True
        self.node_connections = {}  # node_rank -> socket
        self.last_heartbeat = {}   # node_rank -> timestamp
        self.node_steps = {}       # node_rank -> last reported step
        self.failed_node = None
        self.killed_local_rank = 0
        self.recovery_in_progress = False
        self.rebuild_ready_count = 0
        self.lock = threading.Lock()

        # Server socket
        self.server_sock = None

    def start(self):
        """Main entry point."""
        log.info(f"Starting watcher on port {self.port}")
        log.info(f"Monitoring {self.training_nnodes} training nodes")
        log.info(f"Heartbeat timeout: {self.heartbeat_timeout}s")
        if self.fault_inject_step >= 0:
            log.info(f"Fault injection: kill node {self.fault_inject_node} "
                     f"local_rank {self.fault_inject_local_rank} at step {self.fault_inject_step}")

        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(("0.0.0.0", self.port))
        self.server_sock.listen(self.training_nnodes + 4)
        self.server_sock.settimeout(1.0)

        # Signal handling
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Start heartbeat checker thread
        checker = threading.Thread(target=self._heartbeat_checker, daemon=True)
        checker.start()

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
                        node_rank = self._process_message(msg, conn)
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
                        log.warning(f"Node {node_rank} disconnected")
            conn.close()

    def _process_message(self, msg, conn):
        """Process a message from a training node. Returns node_rank."""
        msg_type = msg.get("type")
        node_rank = msg.get("node_rank")

        if msg_type == "heartbeat":
            with self.lock:
                self.node_connections[node_rank] = conn
                self.last_heartbeat[node_rank] = time.time()
                step = msg.get("step", -1)
                self.node_steps[node_rank] = step

            # Check for step-based fault injection
            self._maybe_inject_fault(node_rank, msg.get("step", -1))
            return node_rank

        elif msg_type == "ready_to_rebuild":
            with self.lock:
                self.rebuild_ready_count += 1
                total = self.training_nnodes  # All nodes pause (including target)
                log.info(f"Node {node_rank} ready to rebuild "
                         f"({self.rebuild_ready_count}/{total})")
                if self.rebuild_ready_count >= total:
                    self._trigger_rebuild()
            return node_rank

        elif msg_type == "nccl_error":
            log.warning(f"Node {node_rank} reported NCCL error: {msg.get('error', '?')}")
            if not self.recovery_in_progress:
                log.info("NCCL error report received but no recovery in progress yet — "
                         "waiting for heartbeat timeout to confirm")
            return node_rank

        else:
            log.warning(f"Unknown message type: {msg_type}")
            return node_rank

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
            with self.lock:
                for node_rank, last_ts in list(self.last_heartbeat.items()):
                    if now - last_ts > self.heartbeat_timeout:
                        log.error(f"FAULT DETECTED: Node {node_rank} heartbeat timeout "
                                  f"({now - last_ts:.1f}s > {self.heartbeat_timeout}s)")
                        self._handle_fault(node_rank)
                        break

    def _handle_fault(self, failed_node_rank):
        """Handle a detected node failure (or planned fault injection).

        Sends PAUSE to ALL connected nodes.  For fault injection, this
        includes the target node — all nodes pause together, destroy their
        process groups, then the target rank is killed.
        """
        self.recovery_in_progress = True
        self.failed_node = failed_node_rank
        self.killed_local_rank = self.fault_inject_local_rank
        self.rebuild_ready_count = 0

        # Write fault file (backup signal mechanism)
        fault_file = self.fault_dir / "latest"
        fault_file.write_text(json.dumps({
            "failed_node": failed_node_rank,
            "killed_local_rank": self.killed_local_rank,
            "timestamp": time.time(),
            "action": "rebuild",
        }))

        log.info(f"Sending PAUSE signal to ALL {self.training_nnodes} nodes "
                 f"(target_node={failed_node_rank})")

        # Send pause signal to ALL connected nodes (including target)
        pause_msg = json.dumps({
            "type": "pause",
            "failed_node": failed_node_rank,
            "killed_local_rank": self.killed_local_rank,
        }) + "\n"
        with self.lock:
            for nr, conn in list(self.node_connections.items()):
                try:
                    conn.sendall(pause_msg.encode())
                except (BrokenPipeError, OSError) as e:
                    log.warning(f"Failed to send pause to node {nr}: {e}")

    def _trigger_rebuild(self):
        """All nodes are paused and have destroyed process groups.

        Now:
        1. Send kill_rank to target node (kills one worker process)
        2. Launch 1 spare process on this node
        3. Send rebuild signal to all surviving ranks
        """
        log.info(f"All {self.training_nnodes} nodes ready. "
                 f"Killing rank on node {self.failed_node} local_rank {self.killed_local_rank}...")

        # Step 1: Kill the target rank on the target node
        self._kill_target_rank()

        # Brief delay for the kill to take effect
        time.sleep(1.0)

        # Step 2: Launch 1 spare process on this node
        log.info("Launching spare worker process...")
        spare_proc = self._launch_spare_worker()

        # Give the spare worker a moment to start
        time.sleep(3.0)

        # Step 3: Signal all nodes to rebuild
        # The killed rank's global rank = failed_node * nproc_per_node + killed_local_rank
        killed_global_rank = self.failed_node * self.nproc_per_node + self.killed_local_rank
        rebuild_msg = json.dumps({
            "type": "rebuild",
            "failed_node": self.failed_node,
            "killed_local_rank": self.killed_local_rank,
            "killed_global_rank": killed_global_rank,
            "new_master_addr": self.master_addr,
            "new_master_port": str(int(self.master_port) + 1),
        }) + "\n"

        with self.lock:
            for nr, conn in list(self.node_connections.items()):
                try:
                    conn.sendall(rebuild_msg.encode())
                except (BrokenPipeError, OSError) as e:
                    log.warning(f"Failed to send rebuild to node {nr}: {e}")

        log.info("Rebuild signal sent to all nodes. Waiting for training to resume...")
        self.recovery_in_progress = False
        self.failed_node = None

    def _kill_target_rank(self):
        """Send kill_rank command to the target node.

        The target node's elastic_client will kill only the specified
        local_rank worker process (not the entire node).
        """
        target = self.failed_node
        with self.lock:
            target_conn = self.node_connections.get(target)
            if target_conn is not None:
                kill_msg = json.dumps({
                    "type": "kill_rank",
                    "target_node": target,
                    "local_rank": self.killed_local_rank,
                }) + "\n"
                try:
                    target_conn.sendall(kill_msg.encode())
                    log.info(f"kill_rank sent to node {target} (local_rank={self.killed_local_rank})")
                except (BrokenPipeError, OSError) as e:
                    log.warning(f"Failed to send kill_rank to node {target}: {e}")
            else:
                log.warning(f"Node {target} not connected — may already be dead")

    def _launch_spare_worker(self):
        """Launch 1 replacement process on this spare node.

        The replacement process takes over the killed rank's global position.
        """
        killed_global_rank = self.failed_node * self.nproc_per_node + self.killed_local_rank
        log.info(f"Launching spare: replacing global_rank={killed_global_rank} "
                 f"(node {self.failed_node}, local_rank {self.killed_local_rank})")

        env = os.environ.copy()
        env["NODE_RANK"] = str(self.failed_node)
        env["LOCAL_RANK"] = str(self.killed_local_rank)
        env["RANK"] = str(killed_global_rank)
        env["MASTER_PORT"] = str(int(self.master_port) + 1)  # Rebuild uses new port
        env["ELASTIC_REBUILD_MODE"] = "1"
        env["NNODES"] = str(self.training_nnodes)
        env["WORLD_SIZE"] = str(self.training_nnodes * self.nproc_per_node)
        # Use the specific GPU that corresponds to the killed local_rank
        env["CUDA_VISIBLE_DEVICES"] = str(self.killed_local_rank)

        # Launch a single training process (not elastic_launcher with 8 workers)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = [
            "bash", os.path.join(script_dir, "run_spare_single_rank.sh"),
        ]

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Log output in background
        def _log_output(p):
            for line in iter(p.stdout.readline, b""):
                log.info(f"[spare-worker] {line.decode().rstrip()}")
            p.wait()
            log.info(f"Spare worker exited with code {p.returncode}")

        t = threading.Thread(target=_log_output, args=(proc,), daemon=True)
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

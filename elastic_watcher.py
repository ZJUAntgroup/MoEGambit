#!/usr/bin/env python3
"""
elastic_watcher.py — Hot-spare watcher daemon (FlashRecovery style).

Runs on the spare node (NODE_RANK=8). Does NOT join torch.distributed.
Responsibilities:
  1. Accept heartbeat connections from training rank 0 (one per node)
  2. Detect node failure via heartbeat timeout
  3. Signal surviving training ranks to pause at next safe point
  4. Launch torchrun on this spare node with the failed node's NODE_RANK
  5. Coordinate group rebuild (all nodes re-init_process_group)

Protocol (TCP, JSON lines):
  - Training → Watcher: {"type": "heartbeat", "node_rank": N, "step": S}
  - Training → Watcher: {"type": "ready_to_rebuild"}  (after pause)
  - Watcher → Training: {"type": "pause", "failed_node": N}
  - Watcher → Training: {"type": "rebuild", "new_master_addr": ..., "new_master_port": ...}
"""

import argparse
import json
import logging
import os
import select
import signal
import socket
import subprocess
import sys
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

        # State
        self.running = True
        self.node_connections = {}  # node_rank -> socket
        self.last_heartbeat = {}   # node_rank -> timestamp
        self.node_steps = {}       # node_rank -> last reported step
        self.failed_node = None
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
                self.node_steps[node_rank] = msg.get("step", -1)
            return node_rank

        elif msg_type == "ready_to_rebuild":
            with self.lock:
                self.rebuild_ready_count += 1
                log.info(f"Node {node_rank} ready to rebuild "
                         f"({self.rebuild_ready_count}/{self.training_nnodes - 1} surviving)")
                if self.rebuild_ready_count >= self.training_nnodes - 1:
                    self._trigger_rebuild()
            return node_rank

        else:
            log.warning(f"Unknown message type: {msg_type}")
            return node_rank

    def _heartbeat_checker(self):
        """Periodically check for heartbeat timeouts."""
        # Wait for at least one node to connect before checking
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
        """Handle a detected node failure."""
        self.recovery_in_progress = True
        self.failed_node = failed_node_rank
        self.rebuild_ready_count = 0

        # Write fault file (backup signal mechanism)
        fault_file = self.fault_dir / "latest"
        fault_file.write_text(json.dumps({
            "failed_node": failed_node_rank,
            "timestamp": time.time(),
            "action": "rebuild",
        }))

        log.info(f"Sending PAUSE signal to surviving nodes (failed={failed_node_rank})")

        # Send pause signal to all connected nodes
        pause_msg = json.dumps({"type": "pause", "failed_node": failed_node_rank}) + "\n"
        for nr, conn in list(self.node_connections.items()):
            if nr == failed_node_rank:
                continue
            try:
                conn.sendall(pause_msg.encode())
            except (BrokenPipeError, OSError) as e:
                log.warning(f"Failed to send pause to node {nr}: {e}")

    def _trigger_rebuild(self):
        """All surviving nodes are paused. Launch spare and signal rebuild."""
        log.info("All surviving nodes ready. Launching spare worker...")

        # The spare node will launch torchrun with the failed node's NODE_RANK
        # This replaces the failed node in the world
        spare_proc = self._launch_spare_worker(self.failed_node)

        # Give the spare worker a moment to start
        time.sleep(3.0)

        # Signal all surviving nodes to rebuild
        # Use the same MASTER_ADDR/PORT — the rendezvous will be re-done
        rebuild_msg = json.dumps({
            "type": "rebuild",
            "failed_node": self.failed_node,
            "new_master_addr": self.master_addr,
            "new_master_port": str(int(self.master_port) + 1),  # Use a different port for rebuild
        }) + "\n"

        with self.lock:
            for nr, conn in list(self.node_connections.items()):
                if nr == self.failed_node:
                    continue
                try:
                    conn.sendall(rebuild_msg.encode())
                except (BrokenPipeError, OSError) as e:
                    log.warning(f"Failed to send rebuild to node {nr}: {e}")

        log.info("Rebuild signal sent. Waiting for training to resume...")
        # Recovery complete — reset state for next potential fault
        # (In practice, after one replacement the spare pool is exhausted)
        self.recovery_in_progress = False
        self.failed_node = None

    def _launch_spare_worker(self, target_node_rank):
        """Launch elastic_launcher.py on this spare node, taking over the failed node's rank.

        Uses our custom launcher (not torchrun) so that:
        - Workers won't be auto-killed on peer failure
        - We can do destroy_process_group + re-init_process_group for hot rebuild
        """
        log.info(f"Launching elastic_launcher with NODE_RANK={target_node_rank} on spare node")

        env = os.environ.copy()
        env["NODE_RANK"] = str(target_node_rank)
        env["MASTER_PORT"] = str(int(self.master_port) + 1)  # Rebuild uses new port
        env["ELASTIC_REBUILD_MODE"] = "1"  # Signal: this is a replacement worker
        env["NNODES"] = str(self.training_nnodes)

        # Re-run the same launch script with modified NODE_RANK and ELASTIC_REBUILD_MODE
        # The script detects ELASTIC_REBUILD_MODE and the training code enters rebuild path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = [
            "bash", os.path.join(script_dir, "run_moe64_hotspare.sh"),
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
    args = parser.parse_args()

    watcher = ElasticWatcher(args)
    watcher.start()


if __name__ == "__main__":
    main()

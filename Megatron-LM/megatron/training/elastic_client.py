"""
elastic_client.py — Training-side client for hot-spare elastic recovery.

Each training node's rank 0 (local_rank=0) runs a background heartbeat thread
that connects to the elastic_watcher on the spare node. When the watcher
detects a fault and sends a "pause" signal, all ranks on the node pause at
the next safe point, then coordinate a group rebuild.

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
import logging
import os
import signal as signal_module
import socket
import subprocess
import threading
import time
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Module state
_CLIENT: Optional["ElasticClient"] = None
_PAUSE_REQUESTED = False
_REBUILD_INFO: Optional[dict] = None
_LOCK = threading.Lock()


class ElasticClient:
    """Background heartbeat client that communicates with the watcher."""

    def __init__(self, watcher_addr: str, watcher_port: int, node_rank: int):
        self.watcher_addr = watcher_addr
        self.watcher_port = watcher_port
        self.node_rank = node_rank
        self.sock: Optional[socket.socket] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.step = 0

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

    def update_step(self, step: int):
        self.step = step

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
        global _PAUSE_REQUESTED, _REBUILD_INFO

        if not self._connect():
            return

        while self.running:
            # Send heartbeat
            self._send({
                "type": "heartbeat",
                "node_rank": self.node_rank,
                "step": self.step,
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
        })


def _write_pause_signal(msg=None):
    """Write the pause signal file so all local ranks can see it."""
    fault_dir = os.environ.get("ELASTIC_FAULT_DIR", "/tmp/elastic_faults")
    pause_file = os.path.join(fault_dir, "pause_signal")
    try:
        os.makedirs(fault_dir, exist_ok=True)
        content = json.dumps(msg) if msg else "1"
        with open(pause_file, "w") as f:
            f.write(content)
        logger.info("[elastic] Wrote pause signal file: %s", pause_file)
    except OSError as e:
        logger.warning("[elastic] Failed to write pause file: %s", e)


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


def elastic_client_update_step(step: int):
    """Update the current training step (for heartbeat reporting)."""
    if _CLIENT is not None:
        _CLIENT.update_step(step)


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


def elastic_report_recovery_phase(phase: str, **extra):
    """Report a rebuild/replacement milestone to the watcher for diagnostics."""
    if not os.environ.get("ELASTIC_WATCHER_ADDR"):
        return

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
                return False
            response = json.loads(data.split(b"\n", 1)[0].decode())
            return bool(response.get("ok"))
    except (OSError, ValueError, json.JSONDecodeError) as e:
        logger.warning(
            "[elastic] Failed waiting for recovery phase role=%s rank=%s phase=%s: %s",
            role,
            rank,
            phase,
            e,
        )
        return False


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


def _initialize_model_parallel_for_rebuild(mpu, args):
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
        distributed_timeout_minutes=_elastic_rebuild_timeout_minutes(args),
        nccl_communicator_config_path=getattr(args, "nccl_communicator_config_path", None),
        order="tp-cp-ep-dp-pp"
        if not getattr(args, "use_tp_pp_dp_mapping", False)
        else "tp-cp-ep-pp-dp",
        create_gloo_process_groups=False,
        high_priority_stream_groups=getattr(args, "high_priority_stream_groups", None),
        sharp_enabled_group=getattr(args, "sharp_enabled_group", None),
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

    # Notify watcher (best-effort — the watcher may already know via
    # heartbeat timeout, but sending explicit notification is faster)
    if _CLIENT is not None:
        _CLIENT._send({
            "type": "nccl_error",
            "node_rank": _CLIENT.node_rank,
            "error": str(exception)[:200],
        })


def elastic_wait_for_rebuild_signal() -> dict:
    """Block until the watcher sends the rebuild signal.

    Called after all surviving nodes have paused and notified the watcher.
    Returns the rebuild info dict with new_master_addr, new_master_port, etc.

    NOTE: This is called AFTER destroy_process_group, so we cannot use
    dist.broadcast. Only local_rank 0 has the TCP connection; other local
    ranks on the same node get the info via shared memory / file.
    """
    global _REBUILD_INFO

    # Notify watcher that we're ready
    if _CLIENT is not None:
        _CLIENT.send_ready_to_rebuild()

    # Wait for rebuild signal (only local_rank 0 gets it via TCP)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    fault_dir = os.environ.get("ELASTIC_FAULT_DIR", "/tmp/elastic_faults")
    rebuild_file = os.path.join(fault_dir, "rebuild_signal.json")

    if local_rank == 0:
        while True:
            with _LOCK:
                if _REBUILD_INFO is not None:
                    info = _REBUILD_INFO
                    _REBUILD_INFO = None
                    break
            time.sleep(0.5)

        # Write to shared file so other local ranks can read it
        with open(rebuild_file, "w") as f:
            json.dump(info, f)
    else:
        # Other local ranks wait for the file
        while True:
            if os.path.exists(rebuild_file):
                try:
                    with open(rebuild_file, "r") as f:
                        info = json.load(f)
                    break
                except (json.JSONDecodeError, OSError):
                    pass
            time.sleep(0.5)

    return info


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

    # CRITICAL: Destroy process groups FIRST, before waiting for rebuild signal.
    # The NCCL watchdog runs in a C++ background thread and will SIGABRT
    # the process if it detects a timeout on any process group — even while
    # Python is blocked waiting for the rebuild signal.  Destroying the
    # groups stops the watchdog immediately.
    logger.info(f"[elastic] Rank {rank}: destroying process groups (stop watchdog)")
    try:
        mpu.destroy_model_parallel()
    except Exception as e:
        logger.warning(f"[elastic] destroy_model_parallel failed (expected): {e}")
    try:
        dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"[elastic] destroy_process_group failed (expected): {e}")

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

    if local_rank == 0:
        # Read pause signal to get target info
        killed_local_rank = -1
        failed_node_from_pause = -1
        try:
            with open(pause_file, "r") as f:
                pause_info = json.loads(f.read())
                failed_node_from_pause = pause_info.get("failed_node", -1)
                killed_local_rank = pause_info.get("killed_local_rank", -1)
        except (OSError, json.JSONDecodeError, ValueError):
            pass

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
    resume_iteration = rebuild_info.get("resume_iteration")
    if resume_iteration is not None:
        os.environ["ELASTIC_RESUME_ITERATION"] = str(resume_iteration)

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

    # Create a new TCPStore for rendezvous
    is_master = (rank == 0)
    rebuild_timeout = _elastic_rebuild_timeout(args)
    logger.warning("[elastic] Rank %d: rebuild timeout is %s", rank, rebuild_timeout)

    store = dist.TCPStore(
        host_name=new_master_addr,
        port=int(new_master_port),
        world_size=world_size,
        is_master=is_master,
        timeout=rebuild_timeout,
    )

    dist.init_process_group(
        backend="nccl",
        store=store,
        world_size=world_size,
        rank=rank,
        timeout=rebuild_timeout,
    )
    elastic_report_recovery_phase("pg_ready")

    # Step 3: Re-initialize model parallel groups
    logger.info(f"[elastic] Rank {rank}: re-initializing model parallel")
    _initialize_model_parallel_for_rebuild(mpu, args)
    elastic_report_recovery_phase("mpu_ready")
    phase_timeout = float(os.environ.get("ELASTIC_PHASE_TIMEOUT_SECONDS", "300"))
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

    # Step 5: Barrier to ensure all ranks are ready
    _elastic_barrier("rebuild-final")
    elastic_report_recovery_phase("train_ready")
    logger.warning(f"[elastic] Rank {rank}: rebuild complete, resuming training")

    # Reset pause state
    global _PAUSE_REQUESTED
    with _LOCK:
        _PAUSE_REQUESTED = False

    # Clean up signal files
    fault_dir = os.environ.get("ELASTIC_FAULT_DIR", "/tmp/elastic_faults")
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        for fname in ("rebuild_signal.json", "pause_signal"):
            fpath = os.path.join(fault_dir, fname)
            try:
                os.remove(fpath)
            except OSError:
                pass

    # Restart heartbeat client with new connection
    elastic_client_start()


def elastic_replacement_sync_params(model, optimizer):
    """Called by the REPLACEMENT node after model setup to receive params from DP peers.

    The replacement node has just gone through normal Megatron initialization
    (init_process_group + initialize_model_parallel + setup_model_and_optimizer)
    but with random weights. This function receives the actual weights from
    a surviving DP peer.
    """
    replacement_rank = int(os.environ.get("ELASTIC_REPLACEMENT_RANK", os.environ.get("RANK", "0")))
    model_param_to_name = _build_model_param_name_map(model)
    _load_expert_optimizer_state_from_checkpoint(optimizer, model_param_to_name)
    elastic_report_recovery_phase("param_sync_start")
    _sync_params_to_new_rank(
        model,
        optimizer,
        replacement_rank=replacement_rank,
        model_param_to_name=model_param_to_name,
    )
    elastic_report_recovery_phase("param_sync_done")
    _elastic_barrier("rebuild-final")
    elastic_report_recovery_phase("train_ready")
    logger.warning("[elastic] Replacement node: param sync complete, joining training loop")


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
    """Return True for EP-local expert weights that must not be DP-broadcast."""
    return ".mlp.experts." in name or ".local_experts." in name


def _build_model_param_name_map(model):
    param_to_name = {}
    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            param_to_name[param] = name
    return param_to_name


def _build_optimizer_param_name_map(megatron_optimizer, model_param_to_name):
    param_to_name = {}

    float16_groups = getattr(megatron_optimizer, "float16_groups", None)
    main_groups = getattr(megatron_optimizer, "fp32_from_float16_groups", None)
    if float16_groups is not None and main_groups is not None:
        for model_group, main_group in zip(float16_groups, main_groups):
            for model_param, main_param in zip(model_group, main_group):
                name = model_param_to_name.get(model_param)
                if name is not None:
                    param_to_name[main_param] = name

    fp32_groups = getattr(megatron_optimizer, "fp32_from_fp32_groups", None)
    if fp32_groups is not None:
        for group in fp32_groups:
            for param in group:
                name = model_param_to_name.get(param)
                if name is not None:
                    param_to_name[param] = name

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

    return inner_optimizer, param_to_name


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
                    if name is None or not _is_expert_param_name(name):
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
        return

    optim_checkpoint_name = _get_local_distributed_optimizer_checkpoint_name()
    if optim_checkpoint_name is None:
        return

    try:
        all_states = torch.load(optim_checkpoint_name, map_location="cpu")
    except Exception:
        logger.exception(
            "[elastic] Replacement node: failed to load distributed optimizer "
            "checkpoint %s",
            optim_checkpoint_name,
        )
        return

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

        try:
            loaded_params, loaded_tensors, skipped_params = (
                _copy_expert_state_from_dp_zero_world_tensors(
                    megatron_optimizer, state_dict, model_param_to_name
                )
            )
        except Exception:
            logger.exception(
                "[elastic] Replacement node: local expert optimizer load "
                "failed for optimizer %s; continuing with model-weight "
                "checkpoint state and peer-synced non-expert optimizer state",
                type(megatron_optimizer).__name__,
            )
            continue
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


def _broadcast_optimizer_state_tensor(val, src_rank, group, device):
    if val.is_cuda:
        dist.broadcast(val, src=src_rank, group=group)
        return

    if isinstance(device, torch.device) and device.type == "cuda":
        broadcast_device = device
    else:
        broadcast_device = torch.device("cuda", torch.cuda.current_device())

    tmp = val.to(device=broadcast_device, non_blocking=True)
    dist.broadcast(tmp, src=src_rank, group=group)
    val.copy_(tmp.to(device=val.device))


def _sync_non_expert_optimizer_state(optimizer, model_param_to_name, sync_src_rank, dp_group):
    rank = dist.get_rank()
    main_param_count = 0
    state_tensor_count = 0
    skipped_state_count = 0
    unmapped_count = 0

    for megatron_optimizer in _iter_megatron_optimizers(optimizer):
        inner_optimizer, optim_param_to_name = _build_optimizer_param_name_map(
            megatron_optimizer, model_param_to_name
        )
        if inner_optimizer is None:
            logger.warning(
                "[elastic] Rank %d: skipping optimizer-state sync for unsupported "
                "optimizer wrapper %s",
                rank,
                type(megatron_optimizer).__name__,
            )
            continue

        for group in getattr(inner_optimizer, "param_groups", []):
            for param in group.get("params", []):
                name = optim_param_to_name.get(param)
                if name is None:
                    unmapped_count += 1
                    continue
                if _is_expert_param_name(name):
                    continue

                _broadcast_optimizer_state_tensor(
                    param.data, sync_src_rank, dp_group, param.device
                )
                main_param_count += 1

                state = inner_optimizer.state.get(param, None)
                has_tensor_state = int(
                    state is not None and any(
                        isinstance(val, torch.Tensor) for val in state.values()
                    )
                )
                reduce_device = (
                    param.device
                    if isinstance(param.device, torch.device) and param.device.type == "cuda"
                    else torch.device("cuda", torch.cuda.current_device())
                )
                has_tensor_state_tensor = torch.tensor(
                    [has_tensor_state], dtype=torch.int32, device=reduce_device
                )
                dist.all_reduce(
                    has_tensor_state_tensor,
                    op=dist.ReduceOp.MIN,
                    group=dp_group,
                )
                if has_tensor_state_tensor.item() == 0:
                    skipped_state_count += 1
                    continue

                for _, val in state.items():
                    if isinstance(val, torch.Tensor):
                        _broadcast_optimizer_state_tensor(
                            val, sync_src_rank, dp_group, param.device
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


def _sync_params_to_new_rank(
    model, optimizer, replacement_rank: int = -1, model_param_to_name=None
):
    """Broadcast dense/non-expert model and optimizer state from a DP peer.

    After group rebuild, the replacement rank restores EP-local expert weights
    and expert optimizer state from its checkpoint shard. Dense/non-expert
    parameters are DP-replicated, so we broadcast model weights, main params,
    and tensor optimizer state from a surviving rank within each DP group.
    The source must be a *global* rank that belongs to the provided process
    group.
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
        return

    sync_src_rank = _select_dp_sync_src_rank(sync_group, replacement_rank)

    rank = dist.get_rank()
    logger.info(
        f"[elastic] Rank {rank}: syncing dense params "
        f"(pp_rank={pp_rank}, dp_rank={dp_rank}, group={sync_group_ranks}, "
        f"src={sync_src_rank}, replacement={replacement_rank})"
    )

    if model_param_to_name is None:
        model_param_to_name = _build_model_param_name_map(model)
    dense_count = 0
    expert_count = 0
    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            if _is_expert_param_name(name):
                expert_count += 1
                continue
            dense_count += 1
            dist.broadcast(param.data, src=sync_src_rank, group=sync_group)

    logger.warning(
        "[elastic] Rank %d: dense model param sync complete "
        "(broadcast=%d, expert_from_ckpt=%d)",
        rank,
        dense_count,
        expert_count,
    )
    _sync_non_expert_optimizer_state(optimizer, model_param_to_name, sync_src_rank, sync_group)

    logger.info(f"[elastic] Rank {rank}: param sync complete")


def is_rebuild_mode() -> bool:
    """Check if this process is a replacement worker in rebuild mode."""
    return os.environ.get("ELASTIC_REBUILD_MODE") == "1"

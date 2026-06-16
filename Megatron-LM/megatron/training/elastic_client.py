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

    logger.warning(f"[elastic] Rank {rank}: rebuild signal received. "
                   f"Failed node={failed_node}, killed_rank={killed_global_rank}, "
                   f"new master={new_master_addr}:{new_master_port}")

    # Step 2: Re-initialize process group with new rendezvous
    # Use a NEW port so there's no conflict with the old TCPStore
    logger.info(f"[elastic] Rank {rank}: re-initializing process group "
                f"(master={new_master_addr}:{new_master_port})")
    os.environ["MASTER_ADDR"] = new_master_addr
    os.environ["MASTER_PORT"] = new_master_port

    # Create a new TCPStore for rendezvous
    is_master = (rank == 0)
    store = dist.TCPStore(
        host_name=new_master_addr,
        port=int(new_master_port),
        world_size=world_size,
        is_master=is_master,
        timeout=timedelta(minutes=10),
    )

    dist.init_process_group(
        backend="nccl",
        store=store,
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=10),
    )

    # Step 3: Re-initialize model parallel groups
    logger.info(f"[elastic] Rank {rank}: re-initializing model parallel")
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=getattr(args, 'virtual_pipeline_model_parallel_size', None),
        expert_model_parallel_size=getattr(args, 'expert_model_parallel_size', 1),
    )

    # Step 4: Synchronize parameters to the new rank (DP peer broadcast).
    # The replacement rank has random/zero weights, so choose a surviving
    # rank in each DP group as the source.
    _sync_params_to_new_rank(model, optimizer, replacement_rank=killed_global_rank)

    # Step 5: Barrier to ensure all ranks are ready
    dist.barrier()
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
    _sync_params_to_new_rank(model, optimizer, replacement_rank=replacement_rank)
    dist.barrier()
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


def _get_single_inner_optimizer_for_state_sync(optimizer):
    """Return a single torch optimizer when optimizer-state sync is safe."""
    if optimizer is None:
        return None

    chained_optimizers = getattr(optimizer, "chained_optimizers", None)
    if chained_optimizers is not None:
        if len(chained_optimizers) != 1:
            logger.warning(
                "[elastic] Skipping optimizer-state sync for ChainedOptimizer "
                "with %d inner optimizers",
                len(chained_optimizers),
            )
            return None
        optimizer = chained_optimizers[0]

    try:
        inner_optimizer = optimizer.optimizer
    except (AttributeError, AssertionError) as e:
        logger.warning("[elastic] Skipping optimizer-state sync: %s", e)
        return None

    if not hasattr(inner_optimizer, "param_groups") or not hasattr(inner_optimizer, "state"):
        logger.warning("[elastic] Skipping optimizer-state sync: unsupported optimizer type")
        return None

    return inner_optimizer


def _sync_params_to_new_rank(model, optimizer, replacement_rank: int = -1):
    """Broadcast model parameters from a surviving DP peer to all DP peers.

    After group rebuild, the replacement rank has random/zero weights.
    We broadcast from a surviving rank within each DP group so the new rank
    gets the correct parameters.  The source must be a *global* rank that
    belongs to the provided process group.

    Optimizer state sync is best-effort.  For multi-optimizer MoE chains we
    skip it because the chain wrapper does not expose a single safe underlying
    optimizer.
    """
    from megatron.core import parallel_state as mpu

    dp_rank = mpu.get_data_parallel_rank()
    dp_group = mpu.get_data_parallel_group()
    sync_src_rank = _select_dp_sync_src_rank(dp_group, replacement_rank)

    rank = dist.get_rank()
    logger.info(
        f"[elastic] Rank {rank}: syncing params "
        f"(dp_rank={dp_rank}, src={sync_src_rank}, replacement={replacement_rank})"
    )

    # Broadcast model parameters
    for model_chunk in model:
        for param in model_chunk.parameters():
            dist.broadcast(param.data, src=sync_src_rank, group=dp_group)

    # Broadcast optimizer state (momentum, variance) for smooth continuation
    # when every rank has matching tensor state.  A replacement worker that did
    # not load a checkpoint may have empty optimizer state; in that case all
    # ranks skip optimizer-state sync for that parameter to keep the collective
    # schedule consistent.
    inner_opt = _get_single_inner_optimizer_for_state_sync(optimizer)
    if inner_opt is not None:
        skipped_optimizer_state = False
        for group in inner_opt.param_groups:
            for p in group['params']:
                state = inner_opt.state.get(p, None)
                has_tensor_state = int(
                    state is not None and any(isinstance(val, torch.Tensor) for val in state.values())
                )
                has_tensor_state_tensor = torch.tensor(
                    [has_tensor_state], dtype=torch.int32, device=p.device
                )
                dist.all_reduce(
                    has_tensor_state_tensor,
                    op=dist.ReduceOp.MIN,
                    group=dp_group,
                )
                if has_tensor_state_tensor.item() == 0:
                    skipped_optimizer_state = True
                    continue

                for key, val in state.items():
                    if isinstance(val, torch.Tensor):
                        dist.broadcast(val, src=sync_src_rank, group=dp_group)

        if skipped_optimizer_state:
            logger.warning(
                "[elastic] Rank %d: skipped some optimizer-state sync because "
                "at least one DP peer had no tensor state",
                rank,
            )

    logger.info(f"[elastic] Rank {rank}: param sync complete")


def is_rebuild_mode() -> bool:
    """Check if this process is a replacement worker in rebuild mode."""
    return os.environ.get("ELASTIC_REBUILD_MODE") == "1"

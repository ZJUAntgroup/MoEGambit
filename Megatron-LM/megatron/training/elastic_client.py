"""
elastic_client.py — Training-side client for hot-spare elastic recovery.

Each training node's rank 0 (local_rank=0) runs a background heartbeat thread
that connects to the elastic_watcher on the spare node. When the watcher
detects a fault and sends a "pause" signal, all ranks on the node pause at
the next safe point (bsr_before_iteration), then coordinate a group rebuild.

Architecture:
  - Surviving nodes: pause at safe point → destroy_process_group →
    re-init_process_group (new port) → re-initialize_model_parallel →
    broadcast params to new rank → resume training
  - Replacement node: starts fresh via elastic_launcher → init_process_group
    (same new port) → initialize_model_parallel → receive params from DP peer
    → join training loop

The key insight: both surviving and replacement nodes call init_process_group
with the SAME new MASTER_PORT, so they rendezvous together. The replacement
node goes through Megatron's normal initialization path but with
ELASTIC_REBUILD_MODE=1, which tells pretrain() to skip checkpoint loading
and instead receive params from DP peers after model setup.

Usage in training code:
    from megatron.training.elastic_client import (
        elastic_client_start,
        elastic_check_pause,
        elastic_do_rebuild,
    )
"""

import json
import logging
import os
import socket
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
                    logger.warning(f"[elastic] PAUSE signal received! "
                                   f"Failed node: {msg.get('failed_node')}")
                elif msg_type == "rebuild":
                    with _LOCK:
                        _REBUILD_INFO = msg
                    logger.info(f"[elastic] REBUILD signal received: {msg}")

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

    All ranks must call this collectively — local_rank 0 has the actual
    signal, and broadcasts it to other local ranks via all_reduce on
    the default process group.
    """
    global _PAUSE_REQUESTED

    # If no watcher configured, never pause
    if not os.environ.get("ELASTIC_WATCHER_ADDR"):
        return False

    # Use a tensor to broadcast the pause signal across all ranks
    # (local_rank 0 has the TCP connection, others need to know too)
    pause_flag = torch.tensor([1 if _PAUSE_REQUESTED else 0],
                              dtype=torch.int32, device="cuda")
    dist.all_reduce(pause_flag, op=dist.ReduceOp.MAX)

    if pause_flag.item() > 0:
        with _LOCK:
            _PAUSE_REQUESTED = True
        return True
    return False


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

    if local_rank == 0:
        while True:
            with _LOCK:
                if _REBUILD_INFO is not None:
                    info = _REBUILD_INFO
                    _REBUILD_INFO = None
                    break
            time.sleep(0.5)

        # Write to shared file so other local ranks can read it
        import tempfile
        rebuild_file = os.path.join(
            os.environ.get("ELASTIC_FAULT_DIR", "/tmp/elastic_faults"),
            "rebuild_signal.json"
        )
        with open(rebuild_file, "w") as f:
            json.dump(info, f)
    else:
        # Other local ranks wait for the file
        rebuild_file = os.path.join(
            os.environ.get("ELASTIC_FAULT_DIR", "/tmp/elastic_faults"),
            "rebuild_signal.json"
        )
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
    """Execute the full group rebuild sequence (SURVIVING nodes only).

    Called when elastic_check_pause() returns True.
    This function:
      1. Pauses training (we're already at a safe point)
      2. Notifies watcher we're ready
      3. Waits for rebuild signal (spare node has started)
      4. Destroys current process groups
      5. Re-initializes process groups (with spare node replacing failed node)
      6. Re-initializes model parallel groups
      7. Broadcasts params to new rank (from DP peer)
      8. Returns to training loop

    The REPLACEMENT node goes through pretrain() normally with
    ELASTIC_REBUILD_MODE=1, which makes it:
      - init_process_group with the new port (rendezvous with surviving nodes)
      - initialize_model_parallel normally
      - Skip checkpoint loading
      - Receive params from DP peer via broadcast
      - Join the training loop
    """
    from megatron.core import parallel_state as mpu
    from megatron.training.global_vars import get_args

    args = get_args()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logger.warning(f"[elastic] Rank {rank}: entering rebuild sequence")

    # Step 1: Wait for rebuild signal from watcher
    # This blocks until watcher confirms spare node is launching
    rebuild_info = elastic_wait_for_rebuild_signal()
    failed_node = rebuild_info.get("failed_node", -1)
    new_master_addr = rebuild_info.get("new_master_addr", os.environ.get("MASTER_ADDR"))
    new_master_port = rebuild_info.get("new_master_port", os.environ.get("MASTER_PORT"))

    logger.warning(f"[elastic] Rank {rank}: rebuild signal received. "
                   f"Failed node={failed_node}, new master={new_master_addr}:{new_master_port}")

    # Step 2: Destroy all process groups
    logger.info(f"[elastic] Rank {rank}: destroying process groups")
    mpu.destroy_model_parallel()
    dist.destroy_process_group()

    # Brief sleep to ensure port is released
    time.sleep(2.0)

    # Step 3: Re-initialize process group with new rendezvous
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

    # Step 4: Re-initialize model parallel groups
    logger.info(f"[elastic] Rank {rank}: re-initializing model parallel")
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=getattr(args, 'virtual_pipeline_model_parallel_size', None),
        expert_model_parallel_size=getattr(args, 'expert_model_parallel_size', 1),
    )

    # Step 5: Synchronize parameters to the new rank (DP peer broadcast)
    # The new rank (on spare node) has random/zero weights.
    # Broadcast from DP rank 0 to all DP peers for each model chunk.
    _sync_params_to_new_rank(model, optimizer)

    # Step 6: Barrier to ensure all ranks are ready
    dist.barrier()
    logger.warning(f"[elastic] Rank {rank}: rebuild complete, resuming training")

    # Reset pause state
    global _PAUSE_REQUESTED
    with _LOCK:
        _PAUSE_REQUESTED = False

    # Clean up rebuild signal file
    rebuild_file = os.path.join(
        os.environ.get("ELASTIC_FAULT_DIR", "/tmp/elastic_faults"),
        "rebuild_signal.json"
    )
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        try:
            os.remove(rebuild_file)
        except OSError:
            pass

    # Restart heartbeat client with new connection
    elastic_client_start()


def elastic_replacement_sync_params(model, optimizer):
    """Called by the REPLACEMENT node after model setup to receive params from DP peers.

    The replacement node has just gone through normal Megatron initialization
    (init_process_group + initialize_model_parallel + setup_model_and_optimizer)
    but with random weights. This function receives the actual weights from
    the DP rank 0 peer.
    """
    _sync_params_to_new_rank(model, optimizer)
    dist.barrier()
    logger.warning("[elastic] Replacement node: param sync complete, joining training loop")


def _sync_params_to_new_rank(model, optimizer):
    """Broadcast model parameters from DP rank 0 to all DP peers.

    After group rebuild, the replacement rank has random/zero weights.
    We broadcast from DP rank 0 within each DP group so the new rank
    gets the correct parameters.

    This also syncs optimizer state (momentum, variance) for RPO=0.
    """
    from megatron.core import parallel_state as mpu

    dp_rank = mpu.get_data_parallel_rank()
    dp_group = mpu.get_data_parallel_group()

    rank = dist.get_rank()
    logger.info(f"[elastic] Rank {rank}: syncing params (dp_rank={dp_rank})")

    # Broadcast model parameters
    for model_chunk in model:
        for param in model_chunk.parameters():
            dist.broadcast(param.data, src=0, group=dp_group)

    # Broadcast optimizer state (momentum, variance) for smooth continuation
    # Without this, the replacement rank would have zero momentum → training spike
    if optimizer is not None and hasattr(optimizer, 'optimizer'):
        inner_opt = optimizer.optimizer
        for group in inner_opt.param_groups:
            for p in group['params']:
                if p in inner_opt.state:
                    state = inner_opt.state[p]
                    for key, val in state.items():
                        if isinstance(val, torch.Tensor):
                            dist.broadcast(val, src=0, group=dp_group)

    logger.info(f"[elastic] Rank {rank}: param sync complete")


def is_rebuild_mode() -> bool:
    """Check if this process is a replacement worker in rebuild mode."""
    return os.environ.get("ELASTIC_REBUILD_MODE") == "1"

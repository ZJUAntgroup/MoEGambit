#!/usr/bin/env python3
# =============================================================================
# elastic_launcher.py
#
# Custom process manager that replaces torchrun for hot-spare node replacement
# with hybrid recovery support.
#
# Key difference from torchrun:
#   - torchrun kills ALL ranks when one fails → no hybrid recovery possible
#   - This launcher keeps surviving ranks alive, replaces only the failed rank,
#     then coordinates NCCL group rebuild + hybrid recovery
#
# Architecture:
#   ┌─────────────────────────────────────────────────────────────────┐
#   │                    Elastic Launcher (this file)                  │
#   │                                                                 │
#   │  ┌──────────┐  ┌──────────┐       ┌──────────┐  ┌──────────┐  │
#   │  │  rank 0  │  │  rank 1  │  ...  │  rank N  │  │  spare   │  │
#   │  │ (proc 0) │  │ (proc 1) │       │ (proc N) │  │  (idle)  │  │
#   │  └────┬─────┘  └────┬─────┘       └────┬─────┘  └────┬─────┘  │
#   │       │              │                  │              │        │
#   │       └──────────────┴──────────────────┴──────────────┘        │
#   │                          │                                      │
#   │                    TCPStore (signaling)                          │
#   │                                                                 │
#   │  Fault detection: poll process exit codes                       │
#   │  Recovery: launch replacement on spare node with same rank      │
#   │  Coordination: signal all ranks to rebuild NCCL groups          │
#   └─────────────────────────────────────────────────────────────────┘
#
# Protocol for rank replacement:
#   1. Launcher detects rank X process exited (non-zero)
#   2. Launcher writes to TCPStore: "fault/rank" = X, "fault/epoch" = N
#   3. Surviving ranks poll TCPStore in their training loop (at safe points)
#   4. Surviving ranks enter "group_rebuild" barrier
#   5. Launcher starts new process on spare node with RANK=X
#   6. New process calls init_process_group (all ranks must participate)
#      → Actually: we use a DIFFERENT approach (see below)
#
# NCCL Group Rebuild Strategy:
#   Since init_process_group can only be called once, we use:
#   - The DEFAULT process group (world) is initialized with world_size=N+S
#     where S = number of spare slots (spare ranks exist but are idle)
#   - Spare ranks call init_process_group but then enter a standby loop
#   - On fault: spare rank is "activated" — it takes over the failed rank's
#     role in all sub-groups (EP, DP, TP, etc.)
#   - Sub-groups are rebuilt via destroy + new_group (already implemented
#     in group_rebuild.py / safe_point_group_repair.py)
#
# Wait — this is the approach we tried before (spare ranks in torchrun world)
# and it failed because new_group() is collective across ALL ranks in world.
#
# REVISED Strategy (the one that actually works):
#   - Use init_process_group with world_size = training_ranks + spare_ranks
#   - ALL ranks (including spares) participate in init_process_group
#   - Spares do NOT join training sub-groups (EP, DP, TP, PP)
#   - Spares enter a standby loop, polling TCPStore for activation signal
#   - On fault: the spare rank is activated, joins rebuilt sub-groups
#   - Sub-group rebuild uses new_group with use_local_synchronization=True
#     so only the ranks IN the new group need to participate
#   - The failed rank's process is dead, but since we use
#     use_local_synchronization=True for sub-groups, the dead rank doesn't
#     block new_group creation
#
# The key insight: new_group(ranks, use_local_synchronization=True) only
# requires the ranks listed in `ranks` to call it — NOT all ranks in the
# world. This is what makes single-rank replacement possible!
#
# Implementation:
#   - This launcher spawns N+S processes (N training + S spare)
#   - Training ranks proceed normally through Megatron training
#   - Spare ranks enter standby after init_process_group
#   - On fault detection, launcher signals via TCPStore
#   - Spare rank wakes up, participates in group rebuild
#   - Training ranks detect fault at safe point, rebuild groups
#   - New rank walks hybrid recovery path (dense from peer, expert from ckpt)
#
# =============================================================================

import argparse
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="[elastic %(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("elastic_launcher")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LauncherConfig:
    """Configuration for the elastic launcher."""

    # Script to run
    training_script: str = "./Megatron-LM/pretrain_gpt.py"
    script_args: List[str] = field(default_factory=list)

    # Topology
    nnodes: int = 8
    nproc_per_node: int = 8
    num_spares: int = 8  # one spare node = 8 GPUs

    # Node addresses (training nodes + spare nodes)
    # For single-machine testing, all are localhost
    train_nodes: List[str] = field(default_factory=list)
    spare_nodes: List[str] = field(default_factory=list)

    # Network
    master_addr: str = "127.0.0.1"
    master_port: int = 20115
    store_port: int = 20116  # separate port for signaling TCPStore

    # Fault tolerance
    max_replacements: int = 8  # max number of rank replacements
    health_check_interval: float = 2.0  # seconds between health checks
    safe_point_timeout: float = 300.0  # max wait for safe point

    # Environment
    env_vars: Dict[str, str] = field(default_factory=dict)

    @property
    def world_size(self) -> int:
        """Total world size including spare ranks."""
        return self.nnodes * self.nproc_per_node + self.num_spares

    @property
    def training_world_size(self) -> int:
        """Number of training ranks (excluding spares)."""
        return self.nnodes * self.nproc_per_node

    @property
    def spare_rank_start(self) -> int:
        """First spare rank index."""
        return self.training_world_size


# =============================================================================
# Process Manager
# =============================================================================

class RankProcess:
    """Manages a single rank's process."""

    def __init__(self, rank: int, local_rank: int, node: str, is_spare: bool = False):
        self.rank = rank
        self.local_rank = local_rank
        self.node = node
        self.is_spare = is_spare
        self.process: Optional[subprocess.Popen] = None
        self.is_alive = False
        self.exit_code: Optional[int] = None
        self.activated = False  # for spare ranks: True once activated

    def start(self, config: LauncherConfig) -> None:
        """Start the rank process."""
        env = os.environ.copy()
        env.update(config.env_vars)

        # Standard distributed env vars (same as torchrun sets)
        env["RANK"] = str(self.rank)
        env["LOCAL_RANK"] = str(self.local_rank)
        env["WORLD_SIZE"] = str(config.world_size)
        env["LOCAL_WORLD_SIZE"] = str(config.nproc_per_node)
        env["MASTER_ADDR"] = config.master_addr
        env["MASTER_PORT"] = str(config.master_port)

        # Custom env vars for elastic launcher coordination
        env["ELASTIC_LAUNCHER"] = "1"
        env["ELASTIC_STORE_PORT"] = str(config.store_port)
        env["ELASTIC_TRAINING_WORLD_SIZE"] = str(config.training_world_size)
        env["ELASTIC_NUM_SPARES"] = str(config.num_spares)
        env["ELASTIC_IS_SPARE"] = "1" if self.is_spare else "0"
        env["ELASTIC_SPARE_RANK_START"] = str(config.spare_rank_start)

        # GPU assignment: in single-node mode (all localhost), don't restrict
        # CUDA_VISIBLE_DEVICES since multiple ranks share the same GPUs.
        # The training script uses torch.cuda.set_device(local_rank) internally.
        # In multi-node mode, each node has dedicated GPUs.
        if self.node not in ("localhost", "127.0.0.1"):
            env["CUDA_VISIBLE_DEVICES"] = str(self.local_rank)
        # else: leave CUDA_VISIBLE_DEVICES unset (all GPUs visible)

        cmd = [sys.executable, "-u", config.training_script] + config.script_args

        if self.node == "localhost" or self.node == "127.0.0.1":
            # Local launch — inherit stdout/stderr so output goes to spare.log
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=None,  # inherit parent's stdout
                stderr=None,  # inherit parent's stderr
            )
        else:
            # Remote launch via SSH
            env_str = " ".join(f"{k}={v}" for k, v in env.items()
                              if k.startswith(("RANK", "LOCAL", "WORLD", "MASTER",
                                              "ELASTIC", "CUDA", "NCCL", "PYTHON",
                                              "HF_", "TRANSFORM", "TORCH", "BSR",
                                              "PYTORCH")))
            remote_cmd = f"cd {os.getcwd()} && {env_str} {' '.join(cmd)}"
            self.process = subprocess.Popen(
                ["ssh", "-o", "StrictHostKeyChecking=no", self.node, remote_cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

        self.is_alive = True
        self.exit_code = None
        logger.info(
            "Started rank %d (local_rank=%d, node=%s, spare=%s, pid=%d)",
            self.rank, self.local_rank, self.node, self.is_spare,
            self.process.pid,
        )

    def poll(self) -> Optional[int]:
        """Check if process is still running. Returns exit code if done."""
        if self.process is None:
            return None
        rc = self.process.poll()
        if rc is not None:
            self.is_alive = False
            self.exit_code = rc
        return rc

    def kill(self) -> None:
        """Kill the process."""
        if self.process and self.is_alive:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
            self.is_alive = False


class ElasticLauncher:
    """Main launcher that manages all rank processes."""

    def __init__(self, config: LauncherConfig):
        self.config = config
        self.ranks: Dict[int, RankProcess] = {}
        self.replacements_made = 0
        self._stop_event = threading.Event()
        self._store_thread: Optional[threading.Thread] = None

    def launch_all(self) -> None:
        """Launch all training + spare rank processes."""
        cfg = self.config

        # Launch training ranks
        for node_idx in range(cfg.nnodes):
            node = cfg.train_nodes[node_idx] if cfg.train_nodes else "localhost"
            for local_rank in range(cfg.nproc_per_node):
                rank = node_idx * cfg.nproc_per_node + local_rank
                rp = RankProcess(rank, local_rank, node, is_spare=False)
                rp.start(cfg)
                self.ranks[rank] = rp

        # Launch spare ranks
        spare_node_idx = 0
        for i in range(cfg.num_spares):
            spare_rank = cfg.spare_rank_start + i
            local_rank = i % cfg.nproc_per_node
            if cfg.spare_nodes:
                node = cfg.spare_nodes[spare_node_idx]
                if (i + 1) % cfg.nproc_per_node == 0:
                    spare_node_idx += 1
            else:
                node = "localhost"
            rp = RankProcess(spare_rank, local_rank, node, is_spare=True)
            rp.start(cfg)
            self.ranks[spare_rank] = rp

        logger.info(
            "Launched %d training ranks + %d spare ranks (world_size=%d)",
            cfg.training_world_size, cfg.num_spares, cfg.world_size,
        )

    def monitor_loop(self) -> int:
        """Main monitoring loop. Returns final exit code."""
        cfg = self.config

        while not self._stop_event.is_set():
            time.sleep(cfg.health_check_interval)

            # Check all training ranks
            for rank in range(cfg.training_world_size):
                rp = self.ranks.get(rank)
                if rp is None:
                    continue

                rc = rp.poll()
                if rc is None:
                    continue  # still running

                if rc == 0:
                    # Normal exit — training complete
                    logger.info("Rank %d exited normally (rc=0)", rank)
                    self._stop_event.set()
                    return 0

                # Rank failed!
                logger.warning(
                    "FAULT DETECTED: rank %d exited with rc=%d", rank, rc
                )

                if self.replacements_made >= cfg.max_replacements:
                    logger.error("Max replacements (%d) reached, aborting",
                                cfg.max_replacements)
                    self._cleanup_all()
                    return rc

                # Signal fault to surviving ranks via a file-based protocol
                # (TCPStore would require the store to be in this process)
                self._signal_fault(rank)
                self.replacements_made += 1

            # Check if all training ranks are dead (catastrophic failure)
            alive_training = sum(
                1 for r in range(cfg.training_world_size)
                if self.ranks.get(r) and self.ranks[r].is_alive
            )
            if alive_training == 0:
                logger.error("All training ranks are dead, aborting")
                return 1

        return 0

    def _signal_fault(self, failed_rank: int) -> None:
        """Signal a fault to all surviving ranks.

        Writes a fault file that training ranks poll at safe points.
        The file contains JSON with fault details.
        """
        fault_dir = Path(os.environ.get("ELASTIC_FAULT_DIR", "/tmp/elastic_faults"))
        fault_dir.mkdir(parents=True, exist_ok=True)

        # Find next available spare
        spare_rank = self._find_available_spare()
        if spare_rank is None:
            logger.error("No available spare ranks!")
            return

        fault_info = {
            "failed_rank": failed_rank,
            "spare_rank": spare_rank,
            "epoch": self.replacements_made,
            "timestamp": time.time(),
        }

        fault_file = fault_dir / f"fault_{self.replacements_made:04d}.json"
        fault_file.write_text(json.dumps(fault_info))

        # Also write "latest" pointer
        (fault_dir / "latest").write_text(json.dumps(fault_info))

        logger.info(
            "Signaled fault: rank %d → spare %d (epoch %d), file=%s",
            failed_rank, spare_rank, self.replacements_made, fault_file,
        )

    def _find_available_spare(self) -> Optional[int]:
        """Find the next available (alive, not yet activated) spare rank."""
        for rank in range(self.config.spare_rank_start,
                         self.config.spare_rank_start + self.config.num_spares):
            rp = self.ranks.get(rank)
            if rp and rp.is_alive and not rp.activated:
                rp.activated = True
                return rank
        return None

    def _cleanup_all(self) -> None:
        """Kill all remaining processes."""
        for rp in self.ranks.values():
            rp.kill()

    def run(self) -> int:
        """Main entry point: launch, monitor, return exit code."""
        try:
            self.launch_all()
            return self.monitor_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted, cleaning up...")
            self._cleanup_all()
            return 130
        finally:
            self._cleanup_all()


# =============================================================================
# Training-side integration (imported by the training script)
# =============================================================================

def is_elastic_launcher() -> bool:
    """Check if running under the elastic launcher."""
    return os.environ.get("ELASTIC_LAUNCHER") == "1"


def is_spare_rank() -> bool:
    """Check if this rank is a spare (standby) rank."""
    return os.environ.get("ELASTIC_IS_SPARE") == "1"


def get_training_world_size() -> int:
    """Get the number of training ranks (excluding spares)."""
    return int(os.environ.get("ELASTIC_TRAINING_WORLD_SIZE", "0"))


def get_spare_rank_start() -> int:
    """Get the first spare rank index."""
    return int(os.environ.get("ELASTIC_SPARE_RANK_START", "0"))


def poll_fault_signal() -> Optional[Dict]:
    """Poll for a fault signal from the launcher.

    Called by training ranks at safe points (iteration boundaries).
    Returns fault info dict if a new fault was signaled, None otherwise.
    """
    fault_dir = Path(os.environ.get("ELASTIC_FAULT_DIR", "/tmp/elastic_faults"))
    latest_file = fault_dir / "latest"

    if not latest_file.exists():
        return None

    try:
        info = json.loads(latest_file.read_text())
        # Check if we've already processed this fault
        processed_file = fault_dir / f"processed_{info['epoch']}"
        if processed_file.exists():
            return None
        return info
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def ack_fault_processed(epoch: int) -> None:
    """Acknowledge that a fault has been processed by this rank."""
    fault_dir = Path(os.environ.get("ELASTIC_FAULT_DIR", "/tmp/elastic_faults"))
    fault_dir.mkdir(parents=True, exist_ok=True)
    (fault_dir / f"processed_{epoch}").touch()


# =============================================================================
# Spare rank standby loop
# =============================================================================

def spare_rank_standby_loop() -> Optional[Dict]:
    """Standby loop for spare ranks.

    Spare ranks call this after init_process_group. They block here
    until activated by the launcher (fault signal with their rank as spare).

    Returns:
        Fault info dict when activated, containing 'failed_rank' and 'spare_rank'.
        Returns None if the training completes without needing this spare.
    """
    my_rank = int(os.environ.get("RANK", "-1"))
    fault_dir = Path(os.environ.get("ELASTIC_FAULT_DIR", "/tmp/elastic_faults"))

    logger.info("Spare rank %d entering standby loop...", my_rank)

    while True:
        time.sleep(1.0)

        latest_file = fault_dir / "latest"
        if not latest_file.exists():
            continue

        try:
            info = json.loads(latest_file.read_text())
            if info.get("spare_rank") == my_rank:
                logger.info(
                    "Spare rank %d ACTIVATED! Taking over for failed rank %d",
                    my_rank, info["failed_rank"],
                )
                return info
        except (json.JSONDecodeError, KeyError, OSError):
            continue

        # Check if training is done (all training ranks exited normally)
        done_file = fault_dir / "training_complete"
        if done_file.exists():
            logger.info("Spare rank %d: training complete, exiting standby", my_rank)
            return None


# =============================================================================
# CLI entry point
# =============================================================================

def parse_args() -> LauncherConfig:
    """Parse command line arguments into LauncherConfig."""
    parser = argparse.ArgumentParser(
        description="Elastic launcher for hot-spare node replacement",
    )
    parser.add_argument("--nnodes", type=int, default=8)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--num-spares", type=int, default=8)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=20115)
    parser.add_argument("--store-port", type=int, default=20116)
    parser.add_argument("--train-nodes", type=str, default="",
                       help="Comma-separated training node addresses")
    parser.add_argument("--spare-nodes", type=str, default="",
                       help="Comma-separated spare node addresses")
    parser.add_argument("--max-replacements", type=int, default=8)
    parser.add_argument("--health-check-interval", type=float, default=2.0)
    parser.add_argument("training_script", type=str)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    config = LauncherConfig(
        training_script=args.training_script,
        script_args=args.script_args,
        nnodes=args.nnodes,
        nproc_per_node=args.nproc_per_node,
        num_spares=args.num_spares,
        master_addr=args.master_addr,
        master_port=args.master_port,
        store_port=args.store_port,
        train_nodes=args.train_nodes.split(",") if args.train_nodes else [],
        spare_nodes=args.spare_nodes.split(",") if args.spare_nodes else [],
        max_replacements=args.max_replacements,
        health_check_interval=args.health_check_interval,
    )

    return config


def main():
    config = parse_args()

    logger.info("=" * 60)
    logger.info("Elastic Launcher for Hot-Spare Node Replacement")
    logger.info("=" * 60)
    logger.info("Training world size: %d (%d nodes x %d GPUs)",
               config.training_world_size, config.nnodes, config.nproc_per_node)
    logger.info("Spare ranks: %d (ranks %d-%d)",
               config.num_spares, config.spare_rank_start,
               config.spare_rank_start + config.num_spares - 1)
    logger.info("Total world size: %d", config.world_size)
    logger.info("Master: %s:%d", config.master_addr, config.master_port)
    logger.info("=" * 60)

    launcher = ElasticLauncher(config)
    rc = launcher.run()

    logger.info("Launcher exiting with rc=%d (replacements made: %d)",
               rc, launcher.replacements_made)
    sys.exit(rc)


if __name__ == "__main__":
    main()

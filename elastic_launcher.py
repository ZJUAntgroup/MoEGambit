#!/usr/bin/env python3
"""
elastic_launcher.py — Lightweight process launcher for hot-spare elastic training.

Replaces torchrun for scenarios where we need:
  - Worker processes to SURVIVE peer failures (torchrun kills all on any exit)
  - Ability to destroy_process_group + re-init_process_group without port conflicts
  - Full control over process lifecycle for hot rebuild

This launcher:
  1. Forks N worker processes (one per GPU on this node)
  2. Sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, LOCAL_WORLD_SIZE
  3. Does NOT monitor/restart workers — if a worker dies, it stays dead
  4. The elastic_watcher handles fault detection and recovery coordination

Usage:
    python elastic_launcher.py \
        --nproc-per-node 8 \
        --nnodes 8 \
        --node-rank 0 \
        --master-addr 10.0.0.1 \
        --master-port 20115 \
        -- python ./Megatron-LM/pretrain_gpt.py [training args...]

Environment variables set for each worker:
    RANK            = node_rank * nproc_per_node + local_rank
    LOCAL_RANK      = 0..nproc_per_node-1
    WORLD_SIZE      = nnodes * nproc_per_node
    LOCAL_WORLD_SIZE = nproc_per_node
    MASTER_ADDR     = master address for init_process_group
    MASTER_PORT     = master port for init_process_group
    NODE_RANK       = this node's rank
    GROUP_RANK      = same as NODE_RANK (for compatibility)
"""

import argparse
import os
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="Lightweight launcher for elastic hot-spare training",
        usage="%(prog)s [launcher args] -- command [command args]",
    )
    parser.add_argument("--nproc-per-node", type=int, required=True,
                        help="Number of worker processes per node (GPUs per node)")
    parser.add_argument("--nnodes", type=int, required=True,
                        help="Total number of training nodes")
    parser.add_argument("--node-rank", type=int, required=True,
                        help="Rank of this node (0-based)")
    parser.add_argument("--master-addr", type=str, required=True,
                        help="Master address for torch.distributed rendezvous")
    parser.add_argument("--master-port", type=str, required=True,
                        help="Master port for torch.distributed rendezvous")

    # Split on '--' to separate launcher args from training command
    argv = sys.argv[1:]
    if "--" in argv:
        sep_idx = argv.index("--")
        launcher_args = argv[:sep_idx]
        cmd_args = argv[sep_idx + 1:]
    else:
        launcher_args = argv
        cmd_args = []

    args = parser.parse_args(launcher_args)

    if not cmd_args:
        parser.error("No command specified after '--'")

    nproc = args.nproc_per_node
    nnodes = args.nnodes
    node_rank = args.node_rank
    world_size = nnodes * nproc

    # Create a new process group so that kill_node can kill the launcher
    # and all workers without affecting the calling shell (nohup, etc.).
    try:
        os.setpgrp()
    except OSError:
        pass

    print(f"[launcher] Starting {nproc} workers on node {node_rank}/{nnodes} "
          f"(world_size={world_size})", flush=True)
    print(f"[launcher] Master: {args.master_addr}:{args.master_port}", flush=True)
    print(f"[launcher] PID={os.getpid()}, PGID={os.getpgrp()}", flush=True)
    print(f"[launcher] Command: {' '.join(cmd_args)}", flush=True)

    # Fork worker processes
    processes = []
    fault_dir = os.environ.get("ELASTIC_FAULT_DIR", "/tmp/elastic_faults")
    os.makedirs(fault_dir, exist_ok=True)

    for local_rank in range(nproc):
        global_rank = node_rank * nproc + local_rank

        env = os.environ.copy()
        env["RANK"] = str(global_rank)
        env["LOCAL_RANK"] = str(local_rank)
        env["WORLD_SIZE"] = str(world_size)
        env["LOCAL_WORLD_SIZE"] = str(nproc)
        env["MASTER_ADDR"] = args.master_addr
        env["MASTER_PORT"] = args.master_port
        env["NODE_RANK"] = str(node_rank)
        env["GROUP_RANK"] = str(node_rank)
        # CUDA device selection
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(nproc)))

        proc = subprocess.Popen(
            cmd_args,
            env=env,
            # Each worker inherits stdout/stderr (they'll interleave, but that's fine for now)
        )
        processes.append((local_rank, global_rank, proc))
        print(f"[launcher] Started worker local_rank={local_rank} "
              f"global_rank={global_rank} pid={proc.pid}", flush=True)

        # Write PID file so elastic_client can kill individual workers
        pid_file = os.path.join(fault_dir, f"worker_pid_{local_rank}")
        with open(pid_file, "w") as f:
            f.write(str(proc.pid))

    # Wait for all workers to finish
    # Unlike torchrun, we do NOT kill other workers when one exits.
    # We just wait for all to finish and report exit codes.
    exit_codes = {}
    remaining = set(range(nproc))

    def _sigterm_handler(signum, frame):
        """Forward SIGTERM to all workers."""
        print(f"[launcher] Received signal {signum}, forwarding to workers...", flush=True)
        for lr, gr, proc in processes:
            if proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGTERM)
                except OSError:
                    pass

    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)

    while remaining:
        for local_rank, global_rank, proc in processes:
            if local_rank not in remaining:
                continue
            ret = proc.poll()
            if ret is not None:
                remaining.discard(local_rank)
                exit_codes[local_rank] = ret
                if ret != 0:
                    print(f"[launcher] Worker local_rank={local_rank} "
                          f"(global_rank={global_rank}) exited with code {ret}",
                          flush=True)
                else:
                    print(f"[launcher] Worker local_rank={local_rank} "
                          f"(global_rank={global_rank}) exited normally",
                          flush=True)
        if remaining:
            time.sleep(0.5)

    # Report summary
    failed = {lr: rc for lr, rc in exit_codes.items() if rc != 0}
    if failed:
        print(f"[launcher] {len(failed)}/{nproc} workers failed: {failed}", flush=True)
        sys.exit(1)
    else:
        print(f"[launcher] All {nproc} workers completed successfully", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()

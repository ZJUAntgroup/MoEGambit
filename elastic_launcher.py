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
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time


_TORCHELASTIC_ENV_VARS = (
    "TORCHELASTIC_USE_AGENT_STORE",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_ERROR_FILE",
    "TORCHELASTIC_ROLE",
    "TORCHELASTIC_ROLE_RANK",
    "TORCHELASTIC_ROLE_WORLD_SIZE",
)


def _clean_worker_env(env):
    """Remove torchrun rendezvous state inherited by this standalone launcher."""
    removed = []
    for key in _TORCHELASTIC_ENV_VARS:
        if key in env:
            removed.append(key)
            env.pop(key, None)
    return removed


def _atomic_write_json(path, payload):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


class LauncherControlAgent:
    """Fault-isolated watcher client owned by the node launcher.

    Training ranks publish small state records over a local Unix datagram
    socket.  The launcher keeps the remote control connection alive even when
    local_rank 0 (or any other worker) fails.
    """

    def __init__(
        self,
        *,
        watcher_addr,
        watcher_port,
        node_rank,
        nproc,
        fault_dir,
        status_socket_path,
        kill_worker,
        worker_health,
        startup_manifest,
        heartbeat_interval=1.0,
    ):
        self.watcher_addr = watcher_addr
        self.watcher_port = int(watcher_port)
        self.node_rank = int(node_rank)
        self.nproc = int(nproc)
        self.fault_dir = fault_dir
        self.status_socket_path = status_socket_path
        self.kill_worker = kill_worker
        self.worker_health = worker_health
        self.startup_manifest = dict(startup_manifest)
        self.heartbeat_interval = float(heartbeat_interval)
        self.running = False
        self.rank_states = {}
        self.ready_ranks = set()
        self.consumed_ranks = set()
        self.pause_info = None
        self.ready_epoch_sent = None
        self.target_killed_epoch = None
        self.reported_worker_failures = set()
        self.startup_release_event = threading.Event()
        self.startup_error = None
        self.lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.watcher_sock = None
        self.status_sock = None
        self.threads = []

    def start(self):
        os.makedirs(self.fault_dir, exist_ok=True)
        try:
            os.unlink(self.status_socket_path)
        except FileNotFoundError:
            pass
        self.status_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.status_sock.bind(self.status_socket_path)
        self.status_sock.settimeout(0.5)
        self.running = True
        self.threads = [
            threading.Thread(target=self._status_loop, daemon=True),
            threading.Thread(target=self._watcher_loop, daemon=True),
        ]
        for thread in self.threads:
            thread.start()
        print(
            "[launcher-control] started "
            f"node={self.node_rank} watcher={self.watcher_addr}:{self.watcher_port} "
            f"socket={self.status_socket_path}",
            flush=True,
        )

    def stop(self):
        self.running = False
        for sock in (self.status_sock, self.watcher_sock):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        try:
            os.unlink(self.status_socket_path)
        except OSError:
            pass

    def wait_for_startup_release(self, timeout):
        """Wait until every launcher presents the same distributed contract."""
        attempt = self.startup_manifest["attempt"]
        print(
            "[launcher-control] waiting for startup quorum "
            f"attempt={attempt} timeout={timeout:.1f}s",
            flush=True,
        )
        if not self.startup_release_event.wait(timeout):
            raise RuntimeError(
                f"startup quorum timed out after {timeout:.1f}s for attempt={attempt}"
            )
        if self.startup_error:
            raise RuntimeError(self.startup_error)
        print(
            f"[launcher-control] startup quorum released attempt={attempt}",
            flush=True,
        )

    def _connect_watcher(self):
        while self.running:
            try:
                sock = socket.create_connection(
                    (self.watcher_addr, self.watcher_port), timeout=5.0
                )
                sock.settimeout(0.5)
                with self.send_lock:
                    old_sock = self.watcher_sock
                    self.watcher_sock = sock
                if old_sock is not None:
                    try:
                        old_sock.close()
                    except OSError:
                        pass
                print(
                    "[launcher-control] connected to watcher "
                    f"{self.watcher_addr}:{self.watcher_port}",
                    flush=True,
                )
                return sock
            except OSError as exc:
                print(
                    f"[launcher-control] watcher connect failed: {exc}; retrying",
                    flush=True,
                )
                time.sleep(2.0)
        return None

    def _send_watcher(self, payload):
        data = (json.dumps(payload, sort_keys=True) + "\n").encode()
        with self.send_lock:
            sock = self.watcher_sock
            if sock is None:
                return False
            try:
                sock.sendall(data)
                return True
            except OSError:
                try:
                    sock.close()
                except OSError:
                    pass
                if self.watcher_sock is sock:
                    self.watcher_sock = None
                return False

    def _status_loop(self):
        while self.running:
            try:
                data = self.status_sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode())
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            self._process_local_status(msg)

    def _process_local_status(self, msg):
        event = msg.get("event", "rank_state")
        rank = int(msg.get("rank", -1))
        if rank < 0:
            return

        state = {
            "rank": rank,
            "local_rank": int(msg.get("local_rank", -1)),
            "pid": int(msg.get("pid", -1)),
            "step": int(msg.get("step", -1)),
            "step_tag": int(msg.get("step_tag", msg.get("step", -1))),
            "train_phase": str(msg.get("train_phase", "unknown")),
            "recovery_epoch": int(msg.get("recovery_epoch", 0) or 0),
            "updated_at": time.time(),
        }
        with self.lock:
            self.rank_states[rank] = state
            if event == "rebuild_ready":
                self.ready_ranks.add(rank)
            elif event == "rebuild_consumed":
                self.consumed_ranks.add(rank)

        if event == "nccl_error":
            self._send_watcher({
                "type": "nccl_error",
                "node_rank": self.node_rank,
                **state,
                "error": str(msg.get("error", ""))[:200],
                "control_owner": "launcher",
            })
        if event == "rebuild_ready":
            self._maybe_send_node_ready()
        elif event == "rebuild_consumed":
            self._maybe_cleanup_epoch_files()

    def _expected_ranks_locked(self):
        expected = {
            self.node_rank * self.nproc + local_rank
            for local_rank in range(self.nproc)
        }
        pause = self.pause_info or {}
        if int(pause.get("failed_node", -1)) == self.node_rank:
            target_local_rank = int(pause.get("killed_local_rank", -1))
            expected.discard(self.node_rank * self.nproc + target_local_rank)
        return expected

    @staticmethod
    def _quiescence_summary(states):
        if not states:
            return {
                "valid": False,
                "reason": "no_rank_states",
                "resume_iteration": -1,
                "step_tag": -1,
                "phase": "unknown",
            }
        steps = {int(state.get("step", -1)) for state in states.values()}
        tags = {int(state.get("step_tag", -1)) for state in states.values()}
        phases = {str(state.get("train_phase", "unknown")) for state in states.values()}
        valid_phases = {"iteration_safe_point", "step_complete", "optimizer_skipped"}
        valid = len(steps) == 1 and len(tags) == 1 and tags != {-1} and phases <= valid_phases
        reason = "committed_rank_quorum" if valid else "rank_state_version_mismatch"
        return {
            "valid": valid,
            "reason": reason,
            "resume_iteration": min(steps),
            "step_tag": min(tags),
            "phase": next(iter(phases)) if len(phases) == 1 else "mixed",
        }

    def _maybe_send_node_ready(self):
        with self.lock:
            pause = dict(self.pause_info or {})
            if not pause:
                return
            epoch = int(pause.get("recovery_epoch", 0) or 0)
            if self.ready_epoch_sent == epoch:
                return
            expected = self._expected_ranks_locked()
            if not expected.issubset(self.ready_ranks):
                return
            states = {
                str(rank): dict(self.rank_states[rank])
                for rank in sorted(expected)
                if rank in self.rank_states
            }
            summary = self._quiescence_summary(states)
            target_local_rank = (
                int(pause.get("killed_local_rank", -1))
                if int(pause.get("failed_node", -1)) == self.node_rank
                else -1
            )
            should_kill = target_local_rank >= 0 and self.target_killed_epoch != epoch
            if should_kill:
                self.target_killed_epoch = epoch
            self.ready_epoch_sent = epoch

        if should_kill:
            self.kill_worker(target_local_rank)
        payload = {
            "type": "ready_to_rebuild",
            "node_rank": self.node_rank,
            "recovery_epoch": epoch,
            "step": summary["resume_iteration"],
            "step_tag": summary["step_tag"],
            "train_phase": summary["phase"],
            "rank_states": states,
            "quiescence": summary,
            "control_owner": "launcher",
        }
        if not self._send_watcher(payload):
            with self.lock:
                if self.ready_epoch_sent == epoch:
                    self.ready_epoch_sent = None
            return
        print(
            "[launcher-control] node quiesced "
            f"epoch={epoch} ranks={len(states)}/{len(expected)} "
            f"valid={summary['valid']} step={summary['resume_iteration']}",
            flush=True,
        )

    def _maybe_cleanup_epoch_files(self):
        with self.lock:
            if not self.pause_info:
                return
            expected = self._expected_ranks_locked()
            if not expected.issubset(self.consumed_ranks):
                return
            self.pause_info = None
            self.ready_ranks.clear()
            self.consumed_ranks.clear()
        for name in ("pause_signal", "rebuild_signal.json"):
            try:
                os.remove(os.path.join(self.fault_dir, name))
            except OSError:
                pass

    def _heartbeat_payload(self):
        with self.lock:
            states = {str(rank): dict(state) for rank, state in sorted(self.rank_states.items())}
        summary = self._quiescence_summary(states)
        if not states:
            summary.update({
                "resume_iteration": -1,
                "step_tag": -1,
                "phase": "startup",
            })
        return {
            "type": "heartbeat",
            "node_rank": self.node_rank,
            "step": summary["resume_iteration"],
            "step_tag": summary["step_tag"],
            "train_phase": summary["phase"],
            "rank_states": states,
            "launcher_pid": os.getpid(),
            "workers": self.worker_health(),
            "control_owner": "launcher",
            "startup_manifest": self.startup_manifest,
        }

    def _report_worker_failures(self):
        with self.lock:
            recovery_active = self.pause_info is not None
        if recovery_active:
            return
        for worker in self.worker_health():
            returncode = worker.get("returncode")
            local_rank = int(worker.get("local_rank", -1))
            if returncode is None or int(returncode) == 0 or local_rank < 0:
                continue
            marker = (local_rank, int(returncode))
            if marker in self.reported_worker_failures:
                continue
            payload = {
                "type": "worker_failure",
                "node_rank": self.node_rank,
                "local_rank": local_rank,
                "global_rank": self.node_rank * self.nproc + local_rank,
                "exit_code": int(returncode),
                "rank_state": self.rank_states.get(
                    self.node_rank * self.nproc + local_rank, {}
                ),
                "control_owner": "launcher",
            }
            if self._send_watcher(payload):
                self.reported_worker_failures.add(marker)
                print(
                    "[launcher-control] reported worker failure "
                    f"local_rank={local_rank} exit_code={returncode}",
                    flush=True,
                )

    def _handle_watcher_command(self, msg):
        msg_type = msg.get("type")
        if msg_type == "pause":
            with self.lock:
                self.pause_info = dict(msg)
                self.ready_ranks.clear()
                self.consumed_ranks.clear()
                self.ready_epoch_sent = None
            _atomic_write_json(os.path.join(self.fault_dir, "pause_signal"), msg)
            print(
                "[launcher-control] pause received "
                f"epoch={msg.get('recovery_epoch')} failed_node={msg.get('failed_node')} "
                f"local_rank={msg.get('killed_local_rank')}",
                flush=True,
            )
        elif msg_type == "rebuild":
            local_msg = dict(msg)
            descriptor_data = local_msg.get("descriptor_data")
            if isinstance(descriptor_data, dict):
                epoch = int(local_msg.get("recovery_epoch", 0) or 0)
                descriptor_path = os.path.join(
                    self.fault_dir, f"recovery_descriptor_epoch_{epoch}.json"
                )
                _atomic_write_json(descriptor_path, descriptor_data)
                local_msg["descriptor"] = descriptor_path
            _atomic_write_json(
                os.path.join(self.fault_dir, "rebuild_signal.json"), local_msg
            )
        elif msg_type == "fallback_relaunch":
            _atomic_write_json(
                os.path.join(self.fault_dir, "fallback_relaunch_signal.json"), msg
            )
        elif msg_type == "kill_rank" and int(msg.get("target_node", -1)) == self.node_rank:
            self.kill_worker(int(msg.get("local_rank", -1)))
        elif msg_type in {"startup_release", "startup_reject"}:
            attempt = int(msg.get("attempt", -1))
            if attempt != int(self.startup_manifest["attempt"]):
                return
            if msg_type == "startup_reject":
                self.startup_error = str(msg.get("reason", "startup contract rejected"))
            self.startup_release_event.set()

    def _watcher_loop(self):
        recv_buf = b""
        next_heartbeat = 0.0
        while self.running:
            sock = self.watcher_sock
            if sock is None:
                sock = self._connect_watcher()
                recv_buf = b""
                next_heartbeat = 0.0
                if sock is None:
                    break

            now = time.time()
            if now >= next_heartbeat:
                if not self._send_watcher(self._heartbeat_payload()):
                    continue
                next_heartbeat = now + self.heartbeat_interval
                self._report_worker_failures()
                self._maybe_send_node_ready()

            try:
                data = sock.recv(65536)
                if not data:
                    with self.send_lock:
                        if self.watcher_sock is sock:
                            self.watcher_sock = None
                    try:
                        sock.close()
                    except OSError:
                        pass
                    continue
                recv_buf += data
                while b"\n" in recv_buf:
                    line, recv_buf = recv_buf.split(b"\n", 1)
                    if line:
                        self._handle_watcher_command(json.loads(line.decode()))
            except socket.timeout:
                continue
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                with self.send_lock:
                    if self.watcher_sock is sock:
                        self.watcher_sock = None


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
    launch_attempt = int(os.environ.get("ELASTIC_LAUNCH_ATTEMPT", "0"))
    command_digest = hashlib.sha256("\0".join(cmd_args).encode()).hexdigest()
    startup_manifest = {
        "attempt": launch_attempt,
        "node_rank": node_rank,
        "nnodes": nnodes,
        "nproc_per_node": nproc,
        "world_size": world_size,
        "master_addr": args.master_addr,
        "master_port": str(args.master_port),
        "command_sha256": command_digest,
    }

    # Create a new process group so that kill_node can kill the launcher
    # and all workers without affecting the calling shell (nohup, etc.).
    try:
        os.setpgrp()
    except OSError:
        pass

    print(f"[launcher] Starting {nproc} workers on node {node_rank}/{nnodes} "
          f"(world_size={world_size})", flush=True)
    print(f"[launcher] Master: {args.master_addr}:{args.master_port}", flush=True)
    print(
        f"[launcher] Launch attempt={launch_attempt} contract_sha256={command_digest[:12]}",
        flush=True,
    )
    print(f"[launcher] PID={os.getpid()}, PGID={os.getpgrp()}", flush=True)
    print(f"[launcher] Command: {' '.join(cmd_args)}", flush=True)
    inherited_torchelastic = [key for key in _TORCHELASTIC_ENV_VARS if key in os.environ]
    if inherited_torchelastic:
        print(
            "[launcher] Clearing inherited torchrun rendezvous env: "
            + ",".join(inherited_torchelastic),
            flush=True,
        )

    # Fork worker processes
    processes = []
    fault_dir = os.environ.get("ELASTIC_FAULT_DIR", "/tmp/elastic_faults")
    os.makedirs(fault_dir, exist_ok=True)
    fallback_signal_file = os.path.join(fault_dir, "fallback_relaunch_signal.json")
    control_agent = None

    def _read_fallback_relaunch_request():
        path = fallback_signal_file
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data.setdefault("path", path)
                return data
        except (OSError, json.JSONDecodeError, ValueError):
            return {"path": path}
        return {"path": path}

    def _fallback_exit_code(request):
        try:
            return int((request or {}).get("exit_code", os.environ.get("ELASTIC_FALLBACK_EXIT_CODE", "75")))
        except (TypeError, ValueError):
            return 75

    def _terminate_workers(sig=signal.SIGTERM):
        for lr, gr, proc in processes:
            if proc.poll() is None:
                try:
                    proc.send_signal(sig)
                except OSError:
                    pass

    def _kill_worker(local_rank):
        for lr, gr, proc in list(processes):
            if lr != local_rank:
                continue
            if proc.poll() is None:
                print(
                    "[launcher-control] killing worker "
                    f"local_rank={lr} global_rank={gr} pid={proc.pid}",
                    flush=True,
                )
                try:
                    proc.kill()
                except OSError:
                    pass
            return
        print(
            f"[launcher-control] cannot find worker local_rank={local_rank}",
            flush=True,
        )

    def _worker_health():
        return [
            {
                "local_rank": lr,
                "global_rank": gr,
                "pid": proc.pid,
                "returncode": proc.poll(),
            }
            for lr, gr, proc in list(processes)
        ]

    watcher_addr = os.environ.get("ELASTIC_WATCHER_ADDR")
    watcher_port = os.environ.get("ELASTIC_WATCHER_PORT")
    launcher_control_enabled = (
        os.environ.get("ELASTIC_LAUNCHER_CONTROL_PLANE", "1") != "0"
        and bool(watcher_addr)
        and bool(watcher_port)
    )
    status_socket_path = os.path.join(fault_dir, f"launcher_control_{node_rank}.sock")
    if launcher_control_enabled:
        control_agent = LauncherControlAgent(
            watcher_addr=watcher_addr,
            watcher_port=watcher_port,
            node_rank=node_rank,
            nproc=nproc,
            fault_dir=fault_dir,
            status_socket_path=status_socket_path,
            kill_worker=_kill_worker,
            worker_health=_worker_health,
            startup_manifest=startup_manifest,
            heartbeat_interval=float(
                os.environ.get("ELASTIC_LAUNCHER_HEARTBEAT_INTERVAL", "1.0")
            ),
        )
        control_agent.start()
        try:
            control_agent.wait_for_startup_release(
                float(os.environ.get("ELASTIC_WATCHER_STARTUP_TIMEOUT_SECONDS", "600"))
            )
        except (RuntimeError, ValueError) as exc:
            print(f"[launcher-control] startup rejected: {exc}", flush=True)
            control_agent.stop()
            sys.exit(70)

    for local_rank in range(nproc):
        global_rank = node_rank * nproc + local_rank

        env = os.environ.copy()
        _clean_worker_env(env)
        env["RANK"] = str(global_rank)
        env["LOCAL_RANK"] = str(local_rank)
        env["WORLD_SIZE"] = str(world_size)
        env["LOCAL_WORLD_SIZE"] = str(nproc)
        env["MASTER_ADDR"] = args.master_addr
        env["MASTER_PORT"] = args.master_port
        env["NODE_RANK"] = str(node_rank)
        env["GROUP_RANK"] = str(node_rank)
        if launcher_control_enabled:
            env["ELASTIC_LAUNCHER_CONTROL_PLANE"] = "1"
            env["ELASTIC_LAUNCHER_CONTROL_SOCKET"] = status_socket_path
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
        _terminate_workers(signal.SIGTERM)

    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)

    while remaining:
        fallback_request = _read_fallback_relaunch_request()
        if fallback_request is not None:
            exit_code = _fallback_exit_code(fallback_request)
            print(
                "[launcher] Fallback relaunch requested "
                f"(exit_code={exit_code}, request={fallback_request}); terminating workers",
                flush=True,
            )
            _terminate_workers(signal.SIGTERM)
            deadline = time.time() + 15.0
            while time.time() < deadline:
                if all(proc.poll() is not None for _, _, proc in processes):
                    break
                time.sleep(0.5)
            _terminate_workers(signal.SIGKILL)
            if control_agent is not None:
                control_agent.stop()
            sys.exit(exit_code)

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

    if control_agent is not None:
        control_agent.stop()

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

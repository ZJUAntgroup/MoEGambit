"""Persistent CPU-cache supervisor and actual two-process-generation restart."""
import fcntl
import hashlib
import json
from multiprocessing.connection import Listener, Client
import os
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

from moc_timing_launch import atomic_json, AlreadyStarted
from moc_e2e_plan import manifest


def cache_request(address, message):
    # Local socket is private to this launcher. Standard pickle copies tensor
    # storage; no worker-lifetime shared-memory handles survive by accident.
    with Client(address, family="AF_UNIX") as connection:
        connection.send_bytes(pickle.dumps(message, protocol=5))
        result = pickle.loads(connection.recv_bytes())
    if not result["ok"]:
        raise RuntimeError(result["error"])
    return result.get("value")


class CacheServer:
    def __init__(self, address, ranks):
        self.address, self.ranks = address, set(ranks)
        self.state = {}
        self.lock = threading.Lock()
        self.listener = Listener(address, family="AF_UNIX")
        os.chmod(address, 0o600)
        threading.Thread(target=self.accept, daemon=True).start()

    def accept(self):
        while True:
            try:
                connection = self.listener.accept()
            except (OSError, EOFError):
                return
            threading.Thread(target=self.handle, args=(connection,), daemon=True).start()

    def handle(self, connection):
        with connection:
            try:
                message = pickle.loads(connection.recv_bytes())
                rank = message["rank"]
                if rank not in self.ranks:
                    raise ValueError("rank is not local to this supervisor")
                with self.lock:
                    if message["op"] == "put":
                        if rank in self.state:
                            raise ValueError("one immutable checkpoint generation per short job")
                        self.state[rank] = message["payload"]
                        value = None
                    elif message["op"] == "get":
                        value = self.state[rank]
                    else:
                        raise ValueError("unknown cache operation")
                response = {"ok": True, "value": value}
            except Exception:
                response = {"ok": False, "error": traceback.format_exc()}
            connection.send_bytes(pickle.dumps(response, protocol=5))

    def discard_failed(self, ranks):
        with self.lock:
            for rank in ranks:
                self.state.pop(rank, None)

    def clear(self):
        with self.lock:
            self.state.clear()

    def close(self):
        self.listener.close()
        Path(self.address).unlink(missing_ok=True)


def wait(root, predicate, description, timeout=7200):
    deadline = time.monotonic() + timeout
    while not predicate():
        failures = list((root / "orchestrator").glob("FAILED.node_*.json"))
        if failures:
            raise RuntimeError(f"another node failed; inspect {failures[0]}")
        if time.monotonic() > deadline:
            raise TimeoutError(description)
        time.sleep(0.5)


def resource_snapshot(root, scratch):
    values = {"time": time.time(), "disk": {str(path): dict(zip(("total", "used", "free"), shutil.disk_usage(path)))
                                             for path in (root, scratch)}}
    for filename in ("/proc/meminfo", "/sys/fs/cgroup/memory.events", "/sys/fs/cgroup/memory.current",
                     "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.failcnt",
                     "/sys/fs/cgroup/memory/memory.oom_control"):
        try:
            values[filename] = Path(filename).read_text()
        except OSError:
            pass
    return values


def phase_command(command, cfg, job, phase, job_index):
    """Keep the 10k LR schedule and stop through Megatron's native exit hook."""
    if phase not in ("prefix", "resume"):
        raise ValueError("unknown training phase")
    load, step = Path(cfg["base_dir"]) / "ckpt", cfg["step"]
    if phase == "resume" and job["arm"] == "full_sync_native":
        load, step = Path(cfg["scratch"]) / job["id"] / "native", cfg["checkpoint_step"]
    stop = cfg["failure_step"] if phase == "prefix" else cfg["endpoint"]
    port = cfg["master_port"] + job_index * 2 + (phase == "resume")
    if port > 65535:
        raise ValueError("MASTER_PORT range exceeds 65535")
    argv = [f"--master_port={port}" if arg.startswith("--master_port=") else arg for arg in command]
    return argv + ["--load", str(load), "--ckpt-step", str(step), "--exit-interval", str(stop)]



def worker_failure_details(phase_dir, node, orchestrator=None):
    """Keep a bounded worker error excerpt in the launcher's own failure log."""
    phase_dir = Path(phase_dir)
    paths = [(phase_dir / f"node_{node}.log", 8192)]
    errors = sorted((phase_dir / "worker_errors").glob("FAILED.rank_*.json"))
    paths += [(path, 6144) for path in errors[:3]]
    if orchestrator is not None:
        paths += [(path, 4096) for path in sorted(Path(orchestrator).glob("FAILED.node_*.json"))[:1]]
    parts = []
    for path, limit in paths:
        try:
            with path.open('rb') as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - limit))
                content = stream.read(limit).decode('utf-8', errors='replace')
            parts.append(f"--- {path} (last {limit} bytes) ---\n{content}")
        except OSError as error:
            parts.append(f"--- {path} ---\nCannot read diagnostic: {error}")
    return "\n".join(parts)


def run():
    cfg = manifest()
    root, scratch = Path(cfg["result_dir"]), Path(cfg["scratch"])
    node = int(os.environ["NODE_RANK"])
    if not 0 <= node < cfg["nnodes"]:
        raise ValueError("invalid NODE_RANK")
    root.mkdir(parents=True, exist_ok=True)
    (root / "orchestrator").mkdir(exist_ok=True)
    lock = (root / "orchestrator" / f"node_{node}.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise AlreadyStarted("a launcher is already active on this node")
    started = root / "orchestrator" / f"started.node_{node}.json"
    if started.exists() or (root / "COMPLETE.json").exists():
        raise AlreadyStarted("use a new RUN_ID; incomplete timing jobs cannot be silently reused")
    if node == 0:
        atomic_json(root / "plan.json", cfg)
    wait(root, lambda: (root / "plan.json").is_file(), "waiting for node-0 plan", 300)
    if json.loads((root / "plan.json").read_text()) != cfg:
        raise ValueError("all nodes must have identical configuration and source hashes")
    scratch.mkdir(parents=True, exist_ok=True)
    claim = scratch / "moc_e2e_owner.json"
    if node == 0:
        if not claim.exists() and any(scratch.iterdir()):
            raise ValueError("scratch is not empty: refuse to claim or clean unrelated checkpoints")
        if claim.exists() and json.loads(claim.read_text()) != cfg:
            raise ValueError("scratch belongs to another run")
        atomic_json(claim, cfg)
    wait(root, claim.is_file, "waiting for scratch ownership", 300)
    if json.loads(claim.read_text()) != cfg:
        raise ValueError("scratch ownership/configuration mismatch")
    for directory in (root, scratch):
        probe = directory / f".probe.{node}.{os.getpid()}"
        try:
            with probe.open("xb") as output:
                output.write(b"0" * 1048576)
                output.flush()
                os.fsync(output.fileno())
        finally:
            probe.unlink(missing_ok=True)
    command = sys.argv[2:]
    if len(sys.argv) < 3 or sys.argv[1] != "--":
        raise ValueError("expected -- torchrun ...")
    atomic_json(started, {"pid": os.getpid(), "time": time.time(), "source": cfg["source_revision"]})
    local_ranks = range(node * cfg["per_node"], (node + 1) * cfg["per_node"])
    socket_id = hashlib.sha256(f"{root}:{node}:{os.getpid()}".encode()).hexdigest()[:20]
    address = f"/tmp/moc-e2e-{socket_id}.sock"
    cache = CacheServer(address, local_ranks)
    process = None
    def terminate(signum, frame):
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        raise RuntimeError(f"supervisor received signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        for job_index, job in enumerate(cfg["jobs"]):
            job_root = root / job["id"]
            job_root.mkdir(exist_ok=True)
            for phase in ("prefix", "resume"):
                if phase == "resume":
                    cache.discard_failed(job["failed_ranks"])
                ready = root / "orchestrator" / f"{job['id']}.{phase}.ready.node_{node}.json"
                atomic_json(ready, {"time": time.time()})
                wait(root, lambda: all((root / "orchestrator" / f"{job['id']}.{phase}.ready.node_{i}.json").exists()
                                      for i in range(cfg["nnodes"])), f"waiting for all {phase} launchers")
                env = os.environ.copy()
                env.update(MOC_E2E_JOB=job["id"], MOC_E2E_PHASE=phase, MOC_E2E_CACHE_SOCKET=address)
                error_dir = job_root / phase / "worker_errors"
                error_dir.mkdir(parents=True, exist_ok=True)
                atomic_json(job_root / phase / f"resources.node_{node}.start.json", resource_snapshot(root, scratch))
                env["TORCH_NCCL_DEBUG_INFO_TEMP_FILE"] = str(error_dir / f"nccl.node_{node}.")
                argv = phase_command(command, cfg, job, phase, job_index)
                print(f"[moc-e2e] node={node} job={job['id']} phase={phase}; logs={job_root / phase}", flush=True)
                with (job_root / phase / f"node_{node}.log").open("a", buffering=1) as log:
                    log.write(json.dumps({"command": argv, "started": time.time()}) + "\n")
                    process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                               env=env, start_new_session=True)
                    deadline = time.monotonic() + 7200
                    while process.poll() is None:
                        if list((root / "orchestrator").glob("FAILED.node_*.json")):
                            raise RuntimeError("another node failed during torchrun\n" +
                                               worker_failure_details(job_root / phase, node, root / "orchestrator"))
                        if time.monotonic() > deadline:
                            raise TimeoutError("torchrun exceeded the two-hour per-phase deadline\n" +
                                               worker_failure_details(job_root / phase, node))
                        time.sleep(1)
                    atomic_json(job_root / phase / f"resources.node_{node}.end.json", resource_snapshot(root, scratch))
                    if process.returncode:
                        raise RuntimeError(f"torchrun exited {process.returncode}; see {job_root / phase / f'node_{node}.log'}\n" +
                                           worker_failure_details(job_root / phase, node))
                atomic_json(root / "orchestrator" / f"{job['id']}.{phase}.DONE.node_{node}.json",
                            {"time": time.time(), "returncode": 0, "torchrun_pid": process.pid})
                wait(root, lambda: all((root / "orchestrator" / f"{job['id']}.{phase}.DONE.node_{i}.json").exists()
                                      for i in range(cfg["nnodes"])), f"waiting for all {phase} exits")
            cache.clear()
            cleanup = root / "orchestrator" / f"{job['id']}.cleaned.json"
            if node == 0:
                from moc_e2e_summarize import validate_job
                # Persist result before deleting only this claimed job's scratch.
                atomic_json(job_root / "result.json", validate_job(cfg, job, job_root))
                if not cfg["keep_scratch"]:
                    if json.loads(claim.read_text()) != cfg:
                        raise RuntimeError("scratch claim changed; refuse cleanup")
                    shutil.rmtree(scratch / job["id"])
                atomic_json(cleanup, {"kept": cfg["keep_scratch"]})
            wait(root, cleanup.is_file, "waiting for validation and scratch cleanup")
        if node == 0:
            from moc_e2e_summarize import summarize
            summarize(root)
        atomic_json(root / "orchestrator" / f"DONE.node_{node}.json", {"time": time.time()})
        print(f"[moc-e2e] complete node={node}; results={root}", flush=True)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        cache.close()


if __name__ == "__main__":
    try:
        run()
    except AlreadyStarted:
        raise
    except BaseException:
        error = {"node_rank": os.environ.get("NODE_RANK"), "time": time.time(), "traceback": traceback.format_exc()}
        print(json.dumps(error), file=sys.stderr, flush=True)
        try:
            root = Path(manifest()["result_dir"]) / "orchestrator"
            root.mkdir(parents=True, exist_ok=True)
            atomic_json(root / f"FAILED.node_{os.environ.get('NODE_RANK', 'unknown')}.json", error)
        except Exception as exception:
            print(f"cannot save failure JSON: {exception}", file=sys.stderr, flush=True)
        raise

"""Per-node lock, durable diagnostics, and one torchrun per node."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from moc_timing_plan import Plan


class AlreadyStarted(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with tmp.open("w") as f:
            json.dump(value, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    plan = Plan.from_env()
    rank = int(os.environ["NODE_RANK"])
    if not 0 <= rank < plan.nnodes:
        raise ValueError("invalid NODE_RANK")
    root = Path(plan.result_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "orchestrator").mkdir(exist_ok=True)
    lock = (root / "orchestrator" / f"node_{rank}.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise AlreadyStarted(f"node {rank} already has a launcher for {plan.run_id}")
    if (root / "COMPLETE.json").exists() or (root / "orchestrator" / f"started.node_{rank}.json").exists():
        raise AlreadyStarted("this RUN_ID has already started; use a NEW RUN_ID to avoid mixed timings")
    config = root / "plan.json"
    manifest = plan.manifest()
    if rank == 0:
        atomic_json(config, manifest)
    deadline = time.monotonic() + 180
    while not config.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("node 0 did not publish plan.json")
        time.sleep(1)
    if json.loads(config.read_text()) != manifest:
        raise ValueError("configuration differs between nodes")
    scratch = Path(plan.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    # Each launcher independently probes both filesystems before distributed work.
    for directory in (scratch, root):
        probe = directory / f".probe.node_{rank}.{os.getpid()}"
        try:
            with probe.open("xb") as f:
                f.write(b"0" * 1048576)
                f.flush()
                os.fsync(f.fileno())
        finally:
            probe.unlink(missing_ok=True)
    marker = scratch / "moc_timing_owner.json"
    if rank == 0:
        if not marker.exists() and any(not p.name.startswith(".probe.") for p in scratch.iterdir()):
            raise ValueError("unclaimed scratch directory is nonempty; use a new MOC_CKPT_ROOT")
        if marker.exists() and json.loads(marker.read_text()) != manifest:
            raise ValueError("scratch is claimed by another run")
        atomic_json(marker, manifest)
    while not marker.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("scratch owner marker did not appear")
        time.sleep(1)
    if json.loads(marker.read_text()) != manifest:
        raise ValueError("scratch owner marker differs")
    commands = sys.argv[2:]
    if not commands or sys.argv[1] != "--":
        raise ValueError("expected -- torchrun ...")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = {"node_rank": rank, "pid": os.getpid(), "source": commit,
              "command": commands, "started_at": time.time(), "plan": manifest}
    atomic_json(root / "orchestrator" / f"started.node_{rank}.json", record)
    worker_errors = root / "worker_errors"
    worker_errors.mkdir(exist_ok=True)
    os.environ["TORCH_NCCL_DEBUG_INFO_TEMP_FILE"] = str(worker_errors / f"nccl.node_{rank}.")
    print(f"[moc-timing] node={rank} results={root} scratch={scratch}", flush=True)
    process = None
    try:
        with (root / f"node_{rank}.log").open("a", buffering=1) as log:
            log.write(json.dumps(record) + "\n")
            process = subprocess.Popen(commands, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            def terminate(signum, frame):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                raise RuntimeError(f"launcher received signal {signum}")
            signal.signal(signal.SIGTERM, terminate)
            signal.signal(signal.SIGINT, terminate)
            code = process.wait()
        record.update(returncode=code, ended_at=time.time())
        if code:
            raise RuntimeError(f"torchrun exited {code}; inspect {root}/node_{rank}.log and worker_errors/")
        atomic_json(root / "orchestrator" / f"DONE.node_{rank}.json", record)
        print(f"[moc-timing] node={rank} finished; results={root}", flush=True)
    except BaseException:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        record.update(traceback=traceback.format_exc(), ended_at=time.time())
        try:
            atomic_json(root / "orchestrator" / f"FAILED.node_{rank}.json", record)
        except OSError as e:
            print(f"[moc-timing] cannot save failure to /personal: {e}; {record}", file=sys.stderr)
        raise


if __name__ == "__main__":
    try:
        main()
    except AlreadyStarted:
        # A duplicate invocation must not mark an active experiment as FAILED.
        raise
    except BaseException:
        error = {"node_rank": os.environ.get("NODE_RANK"), "time": time.time(),
                 "traceback": traceback.format_exc()}
        try:
            root = Path(Plan.from_env().result_dir) / "orchestrator"
            root.mkdir(parents=True, exist_ok=True)
            atomic_json(root / f"FAILED.node_{os.environ.get('NODE_RANK', 'unknown')}.json", error)
        except Exception as e:
            print(f"[moc-timing] preflight failure: {error}; cannot write failure JSON: {e}", file=sys.stderr)
        raise

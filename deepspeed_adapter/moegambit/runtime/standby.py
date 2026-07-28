"""Resident local GPU workers for node-level hot-spare recovery.

The launcher starts one process per local GPU before a failure.  Each worker
imports the training workload, initializes its CUDA context, and calls the
optional ``prepare_standby`` hook.  Activation updates the distributed
environment in-place and enters the workload without replacing the process.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _redirect_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o644,
    )
    try:
        os.dup2(descriptor, sys.stdout.fileno())
        os.dup2(descriptor, sys.stderr.fileno())
    finally:
        if descriptor not in {sys.stdout.fileno(), sys.stderr.fileno()}:
            os.close(descriptor)


def _load_workload(path: Path, local_rank: int) -> ModuleType:
    module_name = f"_moegambit_standby_workload_{os.getpid()}_{local_rank}"
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot import standby workload from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def _training_arguments(
    arguments: Sequence[str], local_rank: int
) -> list[str]:
    result = list(arguments)
    if not any(
        item == "--local-rank"
        or item == "--local_rank"
        or item.startswith("--local-rank=")
        or item.startswith("--local_rank=")
        for item in result
    ):
        result.append(f"--local_rank={local_rank}")
    return result


def _call_main(module: ModuleType, arguments: Sequence[str]) -> int:
    main = getattr(module, "main", None)
    if not callable(main):
        raise AttributeError("resident standby workload has no callable main")
    sys.argv = [str(module.__file__), *arguments]
    if inspect.signature(main).parameters:
        result = main(arguments)
    else:
        result = main()
    return int(result or 0)


def _write_workload_failure(module: ModuleType, exc: BaseException) -> None:
    writer = getattr(module, "write_fatal_artifact", None)
    if callable(writer):
        try:
            writer(exc)
        except Exception as artifact_exc:
            print(
                "FATAL_ARTIFACT_WRITE_FAILED "
                f"{type(artifact_exc).__name__}: {artifact_exc}",
                flush=True,
            )


def _worker(args: argparse.Namespace, training_args: Sequence[str]) -> int:
    control_dir = Path(args.control_dir)
    status_path = (
        control_dir
        / f"status_{args.session_id}_{args.local_rank}.json"
    )

    def write_status(phase: str, **payload: Any) -> None:
        _atomic_json(
            status_path,
            {
                "session_id": args.session_id,
                "local_rank": args.local_rank,
                "pid": os.getpid(),
                "phase": phase,
                "time": time.time(),
                **payload,
            },
        )

    standby_log = (
        Path(args.log_dir)
        / f"{args.session_id}_local_rank{args.local_rank}.log"
    )
    _redirect_output(standby_log)
    os.environ["LOCAL_RANK"] = str(args.local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(args.num_workers)
    started = time.time()
    module: ModuleType | None = None

    try:
        write_status("importing")
        module = _load_workload(
            Path(args.training_script), args.local_rank
        )
        import torch

        torch.cuda.set_device(args.local_rank)
        probe = torch.empty(1, device=f"cuda:{args.local_rank}")
        del probe
        torch.cuda.synchronize(args.local_rank)
        write_status("cuda_ready")

        details: dict[str, Any] = {}
        prepare = getattr(module, "prepare_standby", None)
        worker_arguments = _training_arguments(
            training_args, args.local_rank
        )
        if callable(prepare):
            write_status("preparing_workload")
            prepared = prepare(
                worker_arguments,
                local_rank=args.local_rank,
                logical_node=args.expected_logical_node,
                world_size=args.world_size,
            )
            if isinstance(prepared, Mapping):
                details.update(prepared)

        details.update(
            {
                "allocated_gib": round(
                    torch.cuda.memory_allocated(args.local_rank)
                    / (1024**3),
                    3,
                ),
                "reserved_gib": round(
                    torch.cuda.memory_reserved(args.local_rank)
                    / (1024**3),
                    3,
                ),
            }
        )
        _atomic_json(
            control_dir
            / f"ready_{args.session_id}_{args.local_rank}.json",
            {
                "session_id": args.session_id,
                "local_rank": args.local_rank,
                "pid": os.getpid(),
                "ready_at": time.time(),
                "warmup_seconds": time.time() - started,
                **details,
            },
        )
        write_status("ready", **details)
        print(
            "STANDBY_READY "
            f"local_rank={args.local_rank} pid={os.getpid()} "
            f"allocated_gib={details['allocated_gib']:.3f} "
            f"warmup_s={time.time() - started:.2f}",
            flush=True,
        )

        activation_path = (
            control_dir / f"activate_{args.session_id}.json"
        )
        activation: Mapping[str, Any] | None = None
        while activation is None:
            activation = _read_json(activation_path)
            if (
                activation is not None
                and activation.get("session_id") != args.session_id
            ):
                activation = None
            if activation is None:
                time.sleep(0.1)

        environment = activation.get("environment", {})
        if not isinstance(environment, dict):
            raise ValueError("standby activation environment is not a map")
        os.environ.update(
            {str(key): str(value) for key, value in environment.items()}
        )
        logical_node = int(activation["logical_node"])
        rank = logical_node * args.num_workers + args.local_rank
        os.environ.update(
            {
                "RANK": str(rank),
                "LOCAL_RANK": str(args.local_rank),
                "LOCAL_WORLD_SIZE": str(args.num_workers),
                "LOCAL_SIZE": str(args.num_workers),
                "CROSS_RANK": str(logical_node),
                "CROSS_SIZE": str(args.world_size // args.num_workers),
                "WORLD_SIZE": str(args.world_size),
            }
        )
        write_status(
            "activated",
            rank=rank,
            epoch=int(activation["epoch"]),
        )

        rank_log_dir = Path(str(activation["rank_log_dir"]))
        rank_log = (
            rank_log_dir
            / (
                time.strftime("%Y%m%d%H%M%S", time.localtime())
                + f"_standby_pid{os.getpid()}_rank{rank}.log"
            )
        )
        _redirect_output(rank_log)
        print(
            "STANDBY_ACTIVATED "
            f"local_rank={args.local_rank} rank={rank} "
            f"epoch={os.environ.get('MOEGAMBIT_RECOVERY_EPOCH')} "
            f"master={os.environ.get('MASTER_ADDR')}:"
            f"{os.environ.get('MASTER_PORT')}",
            flush=True,
        )
        activate = getattr(module, "activate_standby", None)
        if callable(activate):
            activated = activate(
                worker_arguments,
                local_rank=args.local_rank,
                logical_node=logical_node,
                rank=rank,
                epoch=int(activation["epoch"]),
            )
            activation_details = (
                dict(activated)
                if isinstance(activated, Mapping)
                else {}
            )
            write_status(
                "activation_prepared",
                rank=rank,
                epoch=int(activation["epoch"]),
                **activation_details,
            )
        return _call_main(module, worker_arguments)
    except BaseException as exc:
        try:
            write_status(
                "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        except OSError:
            pass
        if module is not None:
            _write_workload_failure(module, exc)
        print(
            f"FATAL {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        return int(exc.code) if isinstance(exc, SystemExit) else 1


def _terminate_children(children: Sequence[subprocess.Popen]) -> None:
    for child in children:
        if child.poll() is None:
            child.terminate()
    deadline = time.monotonic() + 20.0
    for child in children:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            child.kill()
    for child in children:
        if child.poll() is None:
            child.wait(timeout=5.0)


def _launcher(args: argparse.Namespace, training_args: Sequence[str]) -> int:
    control_dir = Path(args.control_dir)
    control_dir.mkdir(parents=True, exist_ok=True)
    children: list[subprocess.Popen] = []
    stopping = False

    def stop(_signum=None, _frame=None) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        _terminate_children(children)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    for local_rank in range(args.num_workers):
        command = [
            sys.executable,
            "-u",
            "-m",
            "moegambit.runtime.standby",
            "--mode",
            "worker",
            "--control-dir",
            args.control_dir,
            "--session-id",
            args.session_id,
            "--num-workers",
            str(args.num_workers),
            "--local-rank",
            str(local_rank),
            "--expected-logical-node",
            str(args.expected_logical_node),
            "--world-size",
            str(args.world_size),
            "--training-script",
            args.training_script,
            "--log-dir",
            args.log_dir,
            "--",
            *training_args,
        ]
        children.append(subprocess.Popen(command, env=os.environ.copy()))

    ready_logged = False
    try:
        while not stopping:
            return_codes = [child.poll() for child in children]
            failed = [
                code
                for code in return_codes
                if code is not None and code != 0
            ]
            if failed:
                _terminate_children(children)
                return int(failed[0])
            if all(code == 0 for code in return_codes):
                return 0

            if not ready_logged:
                ready = [
                    _read_json(
                        control_dir
                        / f"ready_{args.session_id}_{rank}.json"
                    )
                    for rank in range(args.num_workers)
                ]
                if all(value is not None for value in ready):
                    allocated = sum(
                        float(value.get("allocated_gib", 0.0))
                        for value in ready
                        if value is not None
                    )
                    elapsed = max(
                        float(value.get("warmup_seconds", 0.0))
                        for value in ready
                        if value is not None
                    )
                    print(
                        "resident standby READY "
                        f"workers={args.num_workers}/{args.num_workers} "
                        f"allocated_gib_total={allocated:.2f} "
                        f"warmup_s={elapsed:.2f}",
                        flush=True,
                    )
                    ready_logged = True
            time.sleep(0.2)
    finally:
        if stopping:
            _terminate_children(children)
    return 143


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MoEGambit resident local standby workers"
    )
    parser.add_argument(
        "--mode", choices=("launcher", "worker"), required=True
    )
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--expected-logical-node", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--training-script", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    training_args = list(args.training_args)
    if training_args and training_args[0] == "--":
        training_args.pop(0)
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if args.world_size <= 0 or args.world_size % args.num_workers:
        raise ValueError("--world-size must be divisible by --num-workers")
    if args.mode == "worker":
        if not 0 <= args.local_rank < args.num_workers:
            raise ValueError("--local-rank is outside the local worker set")
        return _worker(args, training_args)
    return _launcher(args, training_args)


if __name__ == "__main__":
    raise SystemExit(main())

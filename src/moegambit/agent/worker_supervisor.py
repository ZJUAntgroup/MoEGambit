"""Node-local worker lifecycle ownership for static-rank replacement."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence

from ..errors import ContractViolation, RecoveryTimeout

__all__ = ["WorkerSpec", "WorkerHandle", "WorkerSupervisor"]


@dataclass(frozen=True)
class WorkerSpec:
    logical_rank: int
    local_rank: int
    argv: Sequence[str]
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None

    def __post_init__(self) -> None:
        if self.logical_rank < 0 or self.local_rank < 0:
            raise ValueError("worker ranks must be non-negative")
        if isinstance(self.argv, str) or not self.argv:
            raise ValueError("worker argv must be a non-empty sequence")
        if any(not isinstance(item, str) or not item for item in self.argv):
            raise ValueError("worker argv must contain non-empty strings")
        for key, value in self.env.items():
            if (
                not isinstance(key, str)
                or not isinstance(value, str)
                or not key
                or "\x00" in key + value
            ):
                raise ValueError("worker environment must contain valid strings")


@dataclass
class WorkerHandle:
    spec: WorkerSpec
    process: Any
    generation: int
    started_at: float

    @property
    def pid(self) -> int:
        return int(self.process.pid)

    @property
    def running(self) -> bool:
        return self.process.poll() is None


class WorkerSupervisor:
    """Own one process per logical rank and never invoke a shell."""

    def __init__(
        self,
        *,
        process_factory: Callable[..., Any] = subprocess.Popen,
        base_env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._factory = process_factory
        self._base_env = dict(os.environ if base_env is None else base_env)
        self._workers: MutableMapping[int, WorkerHandle] = {}
        self._generations: MutableMapping[int, int] = {}
        self._lock = threading.RLock()

    def start(self, spec: WorkerSpec, *, replacement: bool = False) -> WorkerHandle:
        with self._lock:
            current = self._workers.get(spec.logical_rank)
            if current is not None and current.running:
                raise ContractViolation(
                    f"logical rank {spec.logical_rank} already has a live worker"
                )
            generation = self._generations.get(spec.logical_rank, -1) + 1
            env = dict(self._base_env)
            env.update(spec.env)
            env.update(
                {
                    "RANK": str(spec.logical_rank),
                    "LOCAL_RANK": str(spec.local_rank),
                    "MOEGAMBIT_WORKER_GENERATION": str(generation),
                    "MOEGAMBIT_REPLACEMENT": "1" if replacement else "0",
                }
            )
            process = self._factory(
                list(spec.argv),
                env=env,
                cwd=spec.cwd,
                shell=False,
                start_new_session=True,
            )
            handle = WorkerHandle(spec, process, generation, time.monotonic())
            self._workers[spec.logical_rank] = handle
            self._generations[spec.logical_rank] = generation
            return handle

    def get(self, logical_rank: int) -> Optional[WorkerHandle]:
        with self._lock:
            return self._workers.get(int(logical_rank))

    def terminate(self, logical_rank: int, *, timeout_s: float = 10.0) -> None:
        if timeout_s <= 0:
            raise ValueError("worker termination timeout must be positive")
        with self._lock:
            handle = self._workers.get(int(logical_rank))
        if handle is None or not handle.running:
            return
        try:
            os.killpg(handle.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            handle.process.terminate()
        try:
            handle.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(handle.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                handle.process.kill()
            try:
                handle.process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired as exc:
                raise RecoveryTimeout(
                    f"worker rank {logical_rank} did not exit after SIGKILL"
                ) from exc

    def replace(
        self,
        logical_rank: int,
        spec: WorkerSpec,
        *,
        timeout_s: float = 10.0,
    ) -> WorkerHandle:
        if int(logical_rank) != spec.logical_rank:
            raise ContractViolation("replacement must retain the same logical rank")
        self.terminate(logical_rank, timeout_s=timeout_s)
        return self.start(spec, replacement=True)

    def terminate_all(self, *, timeout_s: float = 10.0) -> None:
        with self._lock:
            ranks = tuple(self._workers)
        for rank in ranks:
            self.terminate(rank, timeout_s=timeout_s)

    def snapshot(self) -> Mapping[int, Mapping[str, object]]:
        with self._lock:
            return {
                rank: {
                    "pid": handle.pid,
                    "generation": handle.generation,
                    "running": handle.running,
                    "returncode": handle.process.poll(),
                    "local_rank": handle.spec.local_rank,
                }
                for rank, handle in sorted(self._workers.items())
            }

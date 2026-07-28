"""Engine-neutral ordering contract for torch.distributed group creation.

Torch is imported only when ``TorchDistributedProtocol`` is instantiated, so
the package's policy and controller layers remain usable in a plain Python
process.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterable, Mapping, Protocol

from moegambit.runtime.protocol import WireMessage
from moegambit.runtime.watcher_client import WatcherClient


@dataclass(frozen=True)
class GroupSpec:
    ordinal: int
    name: str
    ranks: tuple[int, ...]
    backend: str = "nccl"

    def as_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "name": self.name,
            "ranks": list(self.ranks),
            "backend": self.backend,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GroupManifest:
    groups: tuple[GroupSpec, ...]

    @classmethod
    def build(cls, groups: Iterable[GroupSpec]) -> "GroupManifest":
        ordered = tuple(sorted(groups, key=lambda group: group.ordinal))
        ordinals = [group.ordinal for group in ordered]
        if ordinals != list(range(len(ordered))):
            raise ValueError("group ordinals must be contiguous and start at zero")
        names = [group.name for group in ordered]
        if len(names) != len(set(names)):
            raise ValueError("group names must be unique")
        return cls(ordered)

    @property
    def fingerprint(self) -> str:
        payload = [group.as_dict() for group in self.groups]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class OrdinalBarrier(Protocol):
    def wait(
        self, phase: str, group: GroupSpec, manifest_fingerprint: str
    ) -> None:
        ...


class NoopOrdinalBarrier:
    def wait(
        self, phase: str, group: GroupSpec, manifest_fingerprint: str
    ) -> None:
        del phase, group, manifest_fingerprint


class WatcherOrdinalBarrier:
    """Align all ranks through an out-of-band watcher control plane."""

    def __init__(
        self,
        client: WatcherClient,
        rank: int,
        world_size: int,
        *,
        run_id: str | None = None,
        physical_node: int | None = None,
        recovery_epoch: int | None = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.client = client
        self.rank = rank
        self.world_size = world_size
        self.run_id = run_id
        self.physical_node = physical_node
        self.recovery_epoch = recovery_epoch
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(
        cls,
        rank: int,
        world_size: int,
        *,
        timeout_seconds: float,
    ) -> "WatcherOrdinalBarrier":
        from moegambit.runtime.watcher_client import WatcherEndpoint

        host = os.environ.get("MOEGAMBIT_HOT_SPARE_COORDINATOR_ADDR")
        port = os.environ.get("MOEGAMBIT_HOT_SPARE_COORDINATOR_PORT")
        run_id = os.environ.get("MOEGAMBIT_HOT_SPARE_RUN_ID")
        if not host or not port or not run_id:
            raise RuntimeError(
                "recovery group ordering requires the hot-spare "
                "coordinator endpoint and run id"
            )
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        logical_node = rank // local_world_size
        physical_node = int(
            os.environ.get(
                "MOEGAMBIT_PHYSICAL_NODE_RANK", str(logical_node)
            )
        )
        recovery_epoch = int(
            os.environ.get(
                "MOEGAMBIT_RECOVERY_EPOCH",
                os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"),
            )
        )
        return cls(
            WatcherClient(
                WatcherEndpoint(
                    host,
                    int(port),
                    timeout=max(10.0, timeout_seconds + 10.0),
                )
            ),
            rank,
            world_size,
            run_id=run_id,
            physical_node=physical_node,
            recovery_epoch=recovery_epoch,
            timeout_seconds=timeout_seconds,
        )

    def wait(
        self, phase: str, group: GroupSpec, manifest_fingerprint: str
    ) -> None:
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        message = WireMessage(
            "group_ordinal_barrier",
            {
                "run_id": self.run_id,
                "physical_node": self.physical_node,
                "logical_node": self.rank // local_world_size,
                "epoch": self.recovery_epoch,
                "phase": phase,
                "ordinal": group.ordinal,
                "group": group.name,
                "ranks": list(group.ranks),
                "backend": group.backend,
                "rank": self.rank,
                "world_size": self.world_size,
                "manifest": manifest_fingerprint,
                "timeout_seconds": self.timeout_seconds,
            },
        )
        deadline = time.monotonic() + self.timeout_seconds + 10.0
        last_error: BaseException | None = None
        while True:
            try:
                response = self.client.request(message)
                break
            except (ConnectionError, OSError, TimeoutError) as exc:
                last_error = exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "group ordinal watcher request timed out: "
                        f"group={group.name} phase={phase} "
                        f"rank={self.rank}"
                    ) from last_error
                time.sleep(min(0.2, remaining))
        if response.kind == "error":
            raise RuntimeError(
                "watcher rejected group ordinal barrier: "
                f"{dict(response.payload)}"
            )
        if response.kind not in {"ack", "group_ordinal_ready"}:
            raise RuntimeError(
                "watcher rejected group ordinal barrier: "
                f"{response.kind} {dict(response.payload)}"
            )


def wait_for_recovery_group_barrier(
    name: str,
    *,
    ordinal: int,
    rank: int,
    world_size: int,
    timeout_seconds: float,
    phase: str = "ready",
) -> None:
    """Wait out of band without using the retiring or new NCCL world."""
    group = GroupSpec(
        ordinal=ordinal,
        name=name,
        ranks=tuple(range(world_size)),
        backend="control",
    )
    barrier = WatcherOrdinalBarrier.from_environment(
        rank,
        world_size,
        timeout_seconds=timeout_seconds,
    )
    barrier.wait(phase, group, group.fingerprint)


class TorchDistributedProtocol:
    """Create a manifest in the same ordinal order on every global rank."""

    def __init__(self, distributed: Any | None = None) -> None:
        self.distributed = distributed or importlib.import_module(
            "torch.distributed"
        )

    def create_groups(
        self,
        manifest: GroupManifest,
        *,
        barrier: OrdinalBarrier | None = None,
        timeout_seconds: float | None = None,
    ) -> Mapping[str, Any]:
        dist = self.distributed
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")
        synchronizer = barrier or NoopOrdinalBarrier()
        fingerprint = manifest.fingerprint
        created: dict[str, Any] = {}
        timeout = (
            timedelta(seconds=timeout_seconds)
            if timeout_seconds is not None
            else None
        )

        for group in manifest.groups:
            synchronizer.wait("start", group, fingerprint)
            kwargs: dict[str, object] = {
                "ranks": list(group.ranks),
                "backend": group.backend,
            }
            if timeout is not None:
                kwargs["timeout"] = timeout
            created[group.name] = dist.new_group(**kwargs)
            synchronizer.wait("done", group, fingerprint)
        return created

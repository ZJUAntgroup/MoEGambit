"""Engine-neutral ordering contract for torch.distributed group creation.

Torch is imported only when ``TorchDistributedProtocol`` is instantiated, so
the package's policy and controller layers remain usable in a plain Python
process.
"""

from __future__ import annotations

import hashlib
import importlib
import json
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

    def __init__(self, client: WatcherClient, rank: int, world_size: int) -> None:
        self.client = client
        self.rank = rank
        self.world_size = world_size

    def wait(
        self, phase: str, group: GroupSpec, manifest_fingerprint: str
    ) -> None:
        response = self.client.request(
            WireMessage(
                "group_ordinal_barrier",
                {
                    "phase": phase,
                    "ordinal": group.ordinal,
                    "group": group.name,
                    "ranks": list(group.ranks),
                    "rank": self.rank,
                    "world_size": self.world_size,
                    "manifest": manifest_fingerprint,
                },
            )
        )
        if response.kind not in {"ack", "group_ordinal_ready"}:
            raise RuntimeError(
                "watcher rejected group ordinal barrier: "
                f"{response.kind} {dict(response.payload)}"
            )


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

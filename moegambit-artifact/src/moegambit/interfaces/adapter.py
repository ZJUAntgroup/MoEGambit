"""Protocol implemented by training-engine integration packages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Protocol, Sequence, runtime_checkable

from moegambit.core.contracts import FeatureSwitches


@dataclass(frozen=True)
class AdapterCapabilities:
    hot_swap: bool
    zero2: bool
    legacy_watcher: bool = False


@dataclass(frozen=True)
class LaunchRequest:
    command: tuple[str, ...]
    environment: Mapping[str, str]
    features: FeatureSwitches
    cwd: Optional[Path] = None


@dataclass(frozen=True)
class PreparedLaunch:
    command: tuple[str, ...]
    environment: Mapping[str, str]
    cwd: Optional[Path] = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class EngineAdapter(Protocol):
    name: str
    capabilities: AdapterCapabilities

    def probe(self, command: Sequence[str]) -> bool:
        ...

    def prepare_launch(self, request: LaunchRequest) -> PreparedLaunch:
        ...

    def watcher_command(
        self, features: FeatureSwitches, arguments: Sequence[str]
    ) -> Optional[tuple[str, ...]]:
        ...

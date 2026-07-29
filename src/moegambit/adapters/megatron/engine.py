"""Framework-discovery adapter for launching Megatron workloads."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Sequence

from ...interfaces import (
    AdapterCapabilities,
    LaunchRequest,
    PreparedLaunch,
)


def _prepend_path(environment: dict[str, str], path: Path) -> None:
    current = environment.get("PYTHONPATH", "")
    values = [item for item in current.split(os.pathsep) if item]
    path_text = str(path)
    if path_text in values:
        values.remove(path_text)
    environment["PYTHONPATH"] = os.pathsep.join([path_text, *values])


class MegatronEngineAdapter:
    """Prepare a command without importing Megatron or torch."""

    name = "megatron"
    capabilities = AdapterCapabilities(
        hot_swap=True,
        zero2=True,
        legacy_watcher=True,
        watcher_required=True,
    )

    def probe(self, command: Sequence[str]) -> bool:
        text = " ".join(command).lower()
        return "megatron" in text or "pretrain_" in text

    def prepare_launch(self, request: LaunchRequest) -> PreparedLaunch:
        environment = dict(request.environment)
        repository = Path(__file__).resolve().parents[4]
        _prepend_path(environment, repository / "Megatron-LM")
        _prepend_path(environment, repository / "src")
        environment.update(
            {
                "MOEGAMBIT_MEGATRON_ADAPTER": "1",
                "ELASTIC_HOT_SWAP_ENABLED": (
                    "1" if request.features.hot_swap else "0"
                ),
                "ELASTIC_ZERO2_MEMORY_REPLICATION": (
                    "1" if request.features.zero2 else "0"
                ),
            }
        )
        return PreparedLaunch(
            command=tuple(request.command),
            environment=environment,
            cwd=request.cwd,
            metadata={
                "engine": self.name,
                "integration": "training-hooks",
                "hot_swap": request.features.hot_swap,
                "zero2": request.features.zero2,
            },
        )

    def watcher_command(
        self, features, arguments: Sequence[str]
    ) -> tuple[str, ...] | None:
        if not features.hot_swap:
            return None
        repository = Path(__file__).resolve().parents[4]
        return (
            sys.executable,
            str(repository / "elastic_watcher.py"),
            *tuple(arguments),
        )


__all__ = ["MegatronEngineAdapter"]

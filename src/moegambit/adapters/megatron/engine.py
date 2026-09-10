"""Framework-discovery adapter for launching Megatron workloads."""

from __future__ import annotations

import sys
from typing import Sequence

from ...interfaces import (
    AdapterCapabilities,
    LaunchRequest,
    PreparedLaunch,
)
from ...runtime.launch_environment import prepend_python_path, repository_root


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
        repository = repository_root(environment)
        if repository is not None:
            prepend_python_path(environment, repository / "Megatron-LM")
            prepend_python_path(environment, repository / "src")
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
        return (
            sys.executable,
            "-m",
            "moegambit.adapters.megatron.compat_watcher",
            *tuple(arguments),
        )


__all__ = ["MegatronEngineAdapter"]

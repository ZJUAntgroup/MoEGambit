"""Megatron-LM adapter for the MoEGambit runtime."""

from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Sequence

from moegambit.interfaces import (
    AdapterCapabilities,
    LaunchRequest,
    PreparedLaunch,
)


_BOOLEAN_HOT_SWAP_FLAGS = {
    "--moe-moegambit-hot-spare-pool",
    "--moe-moegambit-recovery-controller",
}
_VALUED_HOT_SWAP_FLAGS = {"--moe-moegambit-num-hot-spares"}


def _remove_flags(
    command: Sequence[str],
    boolean_flags: set[str],
    valued_flags: set[str],
) -> tuple[str, ...]:
    result: list[str] = []
    skip_next = False
    for item in command:
        if skip_next:
            skip_next = False
            continue
        if item in boolean_flags:
            continue
        if item in valued_flags:
            skip_next = True
            continue
        if any(item.startswith(flag + "=") for flag in valued_flags):
            continue
        result.append(item)
    return tuple(result)


class MegatronAdapter:
    name = "megatron"
    capabilities = AdapterCapabilities(
        hot_swap=True, zero2=True, legacy_watcher=True
    )

    def probe(self, command: Sequence[str]) -> bool:
        text = " ".join(command).lower()
        return "megatron" in text or "pretrain_gpt.py" in text

    def prepare_launch(self, request: LaunchRequest) -> PreparedLaunch:
        command = tuple(request.command)
        if not request.features.hot_swap:
            command = _remove_flags(
                command, _BOOLEAN_HOT_SWAP_FLAGS, _VALUED_HOT_SWAP_FLAGS
            )
        if request.features.zero2 and "--use-distributed-optimizer" not in command:
            command += ("--use-distributed-optimizer",)

        environment = dict(request.environment)
        environment.update(
            {
                "MOEGAMBIT_MEGATRON_ADAPTER": "1",
                "ELASTIC_ZERO2_USE_DISTRIBUTED_OPTIMIZER": (
                    "1" if request.features.zero2 else "0"
                ),
            }
        )
        return PreparedLaunch(
            command=command,
            environment=environment,
            cwd=request.cwd,
            metadata={
                "engine": self.name,
                "hot_swap": request.features.hot_swap,
                "zero2": request.features.zero2,
            },
        )

    def watcher_command(
        self, features, arguments: Sequence[str]
    ) -> tuple[str, ...] | None:
        if not features.hot_swap:
            return None
        backend = find_spec("elastic.elastic_watcher")
        legacy = Path(backend.origin) if backend and backend.origin else None
        if legacy is None:
            repository = Path(__file__).resolve().parents[2]
            legacy = repository / "src" / "elastic" / "elastic_watcher.py"
        if not legacy.exists():
            raise FileNotFoundError(
                "Megatron watcher backend is not bundled; install an adapter "
                f"that provides it or restore {legacy}"
            )
        return (sys.executable, str(legacy), *arguments)


__all__ = ["MegatronAdapter"]

"""DeepSpeed adapter boundary.

The generic runtime owns feature switches and the control plane. A DeepSpeed
integration can consume the projected environment without importing Megatron.
"""

from __future__ import annotations

from typing import Sequence

from moegambit.interfaces import (
    AdapterCapabilities,
    LaunchRequest,
    PreparedLaunch,
)


class DeepSpeedAdapter:
    name = "deepspeed"
    capabilities = AdapterCapabilities(hot_swap=True, zero2=True)

    def probe(self, command: Sequence[str]) -> bool:
        return any("deepspeed" in item.lower() for item in command)

    def prepare_launch(self, request: LaunchRequest) -> PreparedLaunch:
        environment = dict(request.environment)
        environment.update(
            {
                "MOEGAMBIT_DEEPSPEED_ADAPTER": "1",
                "DEEPSPEED_MOEGAMBIT_HOT_SWAP": (
                    "1" if request.features.hot_swap else "0"
                ),
                "DEEPSPEED_MOEGAMBIT_ZERO2": (
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
                "integration": "environment-hooks",
            },
        )

    def watcher_command(self, features, arguments: Sequence[str]):
        del features, arguments
        return None


__all__ = ["DeepSpeedAdapter"]

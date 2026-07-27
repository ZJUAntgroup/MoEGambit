"""Process lifecycle owned by the engine-neutral runtime."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from typing import Mapping, Sequence

from moegambit.interfaces import LaunchRequest, PreparedLaunch
from moegambit.runtime.config import RuntimeConfig
from moegambit.runtime.discovery import load_adapter


def prepare_launch(
    command: Sequence[str],
    config: RuntimeConfig,
    *,
    base_environment: Mapping[str, str] | None = None,
) -> PreparedLaunch:
    if not command:
        raise ValueError("training command cannot be empty")
    adapter = load_adapter(config.adapter, command)
    if config.features.hot_swap and not adapter.capabilities.hot_swap:
        raise ValueError(f"adapter {adapter.name!r} does not support hot swap")
    if config.features.zero2 and not adapter.capabilities.zero2:
        raise ValueError(
            f"adapter {adapter.name!r} does not support ZeRO-2 replication"
        )
    projected = replace(config, adapter=adapter.name).project_environment(
        base_environment
    )
    request = LaunchRequest(
        command=tuple(command),
        environment=projected,
        features=config.features,
    )
    return adapter.prepare_launch(request)


def run(command: Sequence[str], config: RuntimeConfig) -> int:
    prepared = prepare_launch(command, config)
    completed = subprocess.run(
        prepared.command,
        env=dict(prepared.environment),
        cwd=prepared.cwd,
        check=False,
    )
    return int(completed.returncode)

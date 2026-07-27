"""Runtime configuration and compatibility environment projection."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

from moegambit.core.contracts import FeatureSwitches


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_ONE_SHOT_RECOVERY_ENV = {
    "ELASTIC_REBUILD_MODE",
    "ELASTIC_REPLACEMENT_RANK",
    "ELASTIC_RESUME_ITERATION",
    "ELASTIC_RECOVERY_EPOCH",
    "ELASTIC_RECOVERY_DESCRIPTOR",
    "ELASTIC_RECOVERY_DESCRIPTOR_SHA256",
    "ELASTIC_PG_GENERATION",
    "ELASTIC_POST_REBUILD_PENDING",
    "ELASTIC_POST_REBUILD_TRACE_ACTIVE",
    "ELASTIC_POST_REBUILD_TRACE_TOKEN",
    "ELASTIC_POST_REBUILD_TRACE_ITERATION",
}


def parse_bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    raise ValueError(f"{name} must be one of 0/1, false/true, no/yes, off/on")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    return default if value is None else parse_bool(value, name=name)


@dataclass(frozen=True)
class RuntimeConfig:
    adapter: str = "auto"
    features: FeatureSwitches = field(default_factory=FeatureSwitches)
    environment: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, adapter: str = "auto") -> "RuntimeConfig":
        hot_swap = env_bool(
            "MOEGAMBIT_HOT_SWAP",
            env_bool("ELASTIC_HOT_SWAP_ENABLED", False),
        )
        zero2 = env_bool(
            "MOEGAMBIT_ZERO2",
            env_bool("ELASTIC_ZERO2_MEMORY_REPLICATION", False),
        )
        return cls(
            adapter=os.environ.get("MOEGAMBIT_ENGINE_ADAPTER", adapter),
            features=FeatureSwitches(hot_swap=hot_swap, zero2=zero2),
        )

    def project_environment(
        self, base: Mapping[str, str] | None = None
    ) -> dict[str, str]:
        env = dict(os.environ if base is None else base)
        env.update(self.environment)
        if not self.features.hot_swap:
            for name in _ONE_SHOT_RECOVERY_ENV:
                env.pop(name, None)
        enabled = lambda value: "1" if value else "0"
        env.update(
            {
                "MOEGAMBIT_ENGINE_ADAPTER": self.adapter,
                "MOEGAMBIT_HOT_SWAP": enabled(self.features.hot_swap),
                "MOEGAMBIT_ZERO2": enabled(self.features.zero2),
                "ELASTIC_HOT_SWAP_ENABLED": enabled(self.features.hot_swap),
                "ELASTIC_ZERO2_MEMORY_REPLICATION": enabled(
                    self.features.zero2
                ),
            }
        )
        return env

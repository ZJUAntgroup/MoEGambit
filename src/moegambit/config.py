"""Typed configuration replacing scattered environment variables.

Design doc section 16.1.  Resolution order, highest priority first::

    explicit Python argument > CLI > config file > MOEGAMBIT_* env > default

Legacy ``ELASTIC_*`` variables are still read by the compatibility loader for
one release cycle and emit a single de-duplicated deprecation warning.  All
parsing lives here rather than being spread across the runtime.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Tuple

__all__ = [
    "Endpoint",
    "FaultModel",
    "FallbackMode",
    "ControlStoreConfig",
    "ReplicationConfig",
    "SecurityConfig",
    "RuntimeConfig",
]

_ENV_PREFIX = "MOEGAMBIT_"
_LEGACY_PREFIX = "ELASTIC_"

#: Legacy variables already warned about, so we nag at most once per process.
_warned_legacy: set = set()


class FallbackMode(str):
    """What to do when the in-process fast path is not available."""

    CHECKPOINT_RELAUNCH = "checkpoint_relaunch"
    ABORT = "abort"


@dataclass(frozen=True)
class Endpoint:
    host: str = "127.0.0.1"
    port: int = 20200

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.host}:{self.port}"


@dataclass(frozen=True)
class FaultModel:
    """Which failures we claim to handle.

    Deliberately narrow: MoEGambit's fast path assumes *fail-stop*.  Byzantine
    or silently-corrupting ranks are out of scope and must not be silently
    routed into the hot-replacement path.
    """

    fail_stop: bool = True
    inject_step: int = -1
    inject_node: int = -1
    inject_local_rank: int = -1

    @property
    def injection_enabled(self) -> bool:
        return self.inject_step >= 0


@dataclass(frozen=True)
class ReplicationConfig:
    """In-memory optimizer replication (the PHOENIX-style ring)."""

    enabled: bool = False
    port: int = 20300
    connect_timeout_s: float = 30.0
    max_retries: int = 3


@dataclass(frozen=True)
class ControlStoreConfig:
    """Persistent watcher state used to freeze recovery epochs."""

    backend: str = "sqlite"
    path: str = ".moegambit/control.db"
    poll_interval_s: float = 0.05
    busy_timeout_s: float = 5.0


@dataclass(frozen=True)
class SecurityConfig:
    """Design doc 11.3.  Defaults assume a trusted training network.

    ``bind_host`` deliberately defaults to loopback rather than ``0.0.0.0``:
    the control plane speaks unauthenticated JSON and must not be exposed by
    accident.  Operators opt in to a real NIC explicitly.
    """

    bind_host: str = "127.0.0.1"
    job_token: Optional[str] = None
    require_token: bool = False
    max_message_bytes: int = 1 << 20


@dataclass(frozen=True)
class RuntimeConfig:
    """Top-level configuration handed to :func:`moegambit.initialize`."""

    enabled: bool = False
    framework: str = "generic_ddp"
    job_id: str = "default"
    attempt_id: str = "0"
    watcher: Endpoint = field(default_factory=Endpoint)
    fault_model: FaultModel = field(default_factory=FaultModel)
    recovery_timeout_s: float = 600.0
    subgroup_timeout_s: float = 70.0
    fallback: str = FallbackMode.ABORT
    optimizer_replication: ReplicationConfig = field(default_factory=ReplicationConfig)
    control_store: ControlStoreConfig = field(default_factory=ControlStoreConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    # ---- construction -------------------------------------------------

    @classmethod
    def from_env(
        cls,
        env: Optional[Mapping[str, str]] = None,
        *,
        allow_legacy: bool = True,
    ) -> "RuntimeConfig":
        """Build a config from ``MOEGAMBIT_*`` (falling back to ``ELASTIC_*``)."""
        source = os.environ if env is None else env

        def lookup(name: str, *legacy_names: str) -> Optional[str]:
            value = source.get(_ENV_PREFIX + name)
            if value is not None:
                return value
            if not allow_legacy:
                return None
            candidates = legacy_names or (name,)
            for candidate in candidates:
                legacy_name = (
                    candidate
                    if candidate.startswith(_LEGACY_PREFIX)
                    else _LEGACY_PREFIX + candidate
                )
                value = source.get(legacy_name)
                if value is None:
                    continue
                if legacy_name not in _warned_legacy:
                    _warned_legacy.add(legacy_name)
                    warnings.warn(
                        f"{legacy_name} is deprecated; use {_ENV_PREFIX + name} instead. "
                        "Legacy ELASTIC_* variables are honoured for one release cycle.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                return value
            return None

        def as_bool(value: Optional[str], default: bool) -> bool:
            if value is None:
                return default
            return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

        def as_float(value: Optional[str], default: float) -> float:
            try:
                return default if value is None else float(value)
            except (TypeError, ValueError):
                return default

        def as_int(value: Optional[str], default: int) -> int:
            try:
                return default if value is None else int(value)
            except (TypeError, ValueError):
                return default

        watcher = Endpoint(
            host=lookup("WATCHER_HOST", "WATCHER_ADDR") or Endpoint.host,
            port=as_int(lookup("WATCHER_PORT"), Endpoint.port),
        )
        fault_model = FaultModel(
            inject_step=as_int(lookup("FAULT_INJECT_STEP"), -1),
            inject_node=as_int(lookup("FAULT_INJECT_NODE"), -1),
            inject_local_rank=as_int(lookup("FAULT_INJECT_LOCAL_RANK"), -1),
        )
        replication = ReplicationConfig(
            enabled=as_bool(
                lookup(
                    "OPTIMIZER_REPLICATION",
                    "ELASTIC_ZERO2_MEMORY_REPLICATION",
                    "ELASTIC_ZERO2_MEMORY_CHECKPOINT",
                ),
                False,
            ),
            port=as_int(
                lookup("OPTIMIZER_REPLICATION_PORT", "ELASTIC_ZERO2_PORT"),
                20300,
            ),
        )
        security = SecurityConfig(
            bind_host=lookup("BIND_HOST") or SecurityConfig.bind_host,
            job_token=lookup("JOB_TOKEN"),
            require_token=as_bool(lookup("REQUIRE_TOKEN"), False),
            max_message_bytes=as_int(
                lookup("MAX_MESSAGE_BYTES", "ELASTIC_WATCHER_MAX_MESSAGE_BYTES"),
                SecurityConfig.max_message_bytes,
            ),
        )
        control_store = ControlStoreConfig(
            backend=(lookup("CONTROL_STORE_BACKEND") or "sqlite").strip().lower(),
            path=lookup("CONTROL_STORE_PATH") or ControlStoreConfig.path,
            poll_interval_s=as_float(
                lookup("CONTROL_STORE_POLL_SECONDS"),
                ControlStoreConfig.poll_interval_s,
            ),
            busy_timeout_s=as_float(
                lookup("CONTROL_STORE_BUSY_TIMEOUT_SECONDS"),
                ControlStoreConfig.busy_timeout_s,
            ),
        )
        fallback = (
            FallbackMode.CHECKPOINT_RELAUNCH
            if as_bool(lookup("FALLBACK_RELAUNCH"), False)
            else FallbackMode.ABORT
        )
        return cls(
            enabled=as_bool(lookup("ENABLED", "ELASTIC_ENABLED"), False),
            framework=lookup("FRAMEWORK") or "generic_ddp",
            job_id=(
                lookup("JOB_ID", "ELASTIC_JOB_ID")
                or source.get("TORCHELASTIC_RUN_ID")
                or "default"
            ),
            attempt_id=(
                lookup("ATTEMPT_ID", "ELASTIC_ATTEMPT_ID")
                or source.get("TORCHELASTIC_RESTART_COUNT")
                or "0"
            ),
            watcher=watcher,
            fault_model=fault_model,
            recovery_timeout_s=as_float(lookup("RECOVERY_TIMEOUT_SECONDS"), 600.0),
            subgroup_timeout_s=as_float(lookup("SUBGROUP_TIMEOUT"), 70.0),
            fallback=fallback,
            optimizer_replication=replication,
            control_store=control_store,
            security=security,
        )

    def merged_with(self, **overrides: object) -> "RuntimeConfig":
        """Return a copy with explicit Python overrides applied (top priority)."""
        return replace(self, **overrides)  # type: ignore[arg-type]

    # ---- validation ---------------------------------------------------

    def validate(self) -> Tuple[str, ...]:
        """Return human-readable problems; empty tuple means usable.

        Returns findings instead of raising so ``moegambit doctor`` can report
        every problem at once rather than one per run.
        """
        problems = []
        if self.watcher.port <= 0 or self.watcher.port > 65535:
            problems.append(f"watcher port out of range: {self.watcher.port}")
        if self.recovery_timeout_s <= 0:
            problems.append("recovery_timeout_s must be positive")
        if self.subgroup_timeout_s <= 0:
            problems.append("subgroup_timeout_s must be positive")
        if self.security.require_token and not self.security.job_token:
            problems.append("require_token is set but no job_token was provided")
        if self.security.max_message_bytes <= 0:
            problems.append("max_message_bytes must be positive")
        if self.security.bind_host.strip() in ("", "0.0.0.0", "::"):
            problems.append(
                "bind_host must name an explicit interface, not a wildcard address"
            )
        if self.control_store.backend not in ("sqlite", "memory"):
            problems.append(
                f"unknown control_store backend: {self.control_store.backend}"
            )
        if self.control_store.backend == "sqlite" and not self.control_store.path.strip():
            problems.append("SQLite control_store requires a non-empty path")
        if self.control_store.poll_interval_s <= 0:
            problems.append("control_store poll_interval_s must be positive")
        if self.control_store.busy_timeout_s <= 0:
            problems.append("control_store busy_timeout_s must be positive")
        if self.fault_model.injection_enabled and self.fault_model.inject_node < 0:
            problems.append("fault injection enabled without a target node")
        if self.fallback not in (FallbackMode.CHECKPOINT_RELAUNCH, FallbackMode.ABORT):
            problems.append(f"unknown fallback mode: {self.fallback}")
        if not str(self.job_id).strip():
            problems.append("job_id must be non-empty")
        if not str(self.attempt_id).strip():
            problems.append("attempt_id must be non-empty")
        return tuple(problems)

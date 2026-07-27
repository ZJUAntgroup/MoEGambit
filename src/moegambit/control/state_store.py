"""Epoch-aware control-plane state storage."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, runtime_checkable

from ..errors import ContractViolation, RecoveryTimeout

__all__ = ["ControlStore", "InMemoryControlStore"]


@runtime_checkable
class ControlStore(Protocol):
    def put_if_epoch(self, key: str, value: Any, epoch: int) -> bool: ...

    def get(self, key: str) -> Any: ...

    def wait(
        self,
        predicates: Mapping[str, Callable[[Any], bool]],
        timeout: float,
    ) -> Mapping[str, Any]: ...

    def compare_and_set(
        self,
        key: str,
        expected: Any,
        value: Any,
        epoch: int,
    ) -> bool: ...


@dataclass(frozen=True)
class _Entry:
    epoch: int
    value: Any


class InMemoryControlStore:
    """Thread-safe store whose values can only move to a newer epoch."""

    def __init__(self) -> None:
        self._entries: Dict[str, _Entry] = {}
        self._condition = threading.Condition()

    @staticmethod
    def _validate(key: str, epoch: int) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("control-store key must be non-empty")
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("control-store epoch must be non-negative")

    def put_if_epoch(self, key: str, value: Any, epoch: int) -> bool:
        self._validate(key, epoch)
        with self._condition:
            current = self._entries.get(key)
            if current is not None and epoch < current.epoch:
                return False
            self._entries[key] = _Entry(epoch, value)
            self._condition.notify_all()
            return True

    def get(self, key: str) -> Any:
        with self._condition:
            entry = self._entries.get(key)
            return None if entry is None else entry.value

    def get_epoch(self, key: str) -> Optional[int]:
        with self._condition:
            entry = self._entries.get(key)
            return None if entry is None else entry.epoch

    def wait(
        self,
        predicates: Mapping[str, Callable[[Any], bool]],
        timeout: float,
    ) -> Mapping[str, Any]:
        if not predicates:
            raise ValueError("control-store wait needs at least one predicate")
        if timeout <= 0:
            raise ValueError("control-store wait timeout must be positive")
        deadline = time.monotonic() + float(timeout)
        with self._condition:
            while True:
                values = {
                    key: (
                        None
                        if self._entries.get(key) is None
                        else self._entries[key].value
                    )
                    for key in predicates
                }
                if all(
                    predicate(values[key])
                    for key, predicate in predicates.items()
                ):
                    return values
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RecoveryTimeout(
                        "control-store predicates were not satisfied before timeout"
                    )
                self._condition.wait(remaining)

    def compare_and_set(
        self,
        key: str,
        expected: Any,
        value: Any,
        epoch: int,
    ) -> bool:
        self._validate(key, epoch)
        with self._condition:
            current = self._entries.get(key)
            if current is not None and epoch < current.epoch:
                return False
            current_value = None if current is None else current.value
            if current_value != expected:
                return False
            self._entries[key] = _Entry(epoch, value)
            self._condition.notify_all()
            return True

    def require_same_epoch(self, *keys: str) -> int:
        if not keys:
            raise ValueError("at least one control-store key is required")
        with self._condition:
            entries = [self._entries.get(key) for key in keys]
            if any(entry is None for entry in entries):
                raise ContractViolation(
                    "control-store values do not belong to one complete epoch"
                )
            epochs = {entry.epoch for entry in entries if entry is not None}
        if len(epochs) != 1:
            raise ContractViolation(
                "control-store values do not belong to one complete epoch"
            )
        return next(iter(epochs))

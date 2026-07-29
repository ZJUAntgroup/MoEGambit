"""Epoch-aware control-plane state storage."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, runtime_checkable

from ..errors import ContractViolation, RecoveryTimeout

__all__ = ["ControlStore", "InMemoryControlStore", "SQLiteControlStore"]


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

    def get_epoch(self, key: str) -> Optional[int]: ...

    def require_same_epoch(self, *keys: str) -> int: ...


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
            if current is not None:
                if epoch < current.epoch:
                    return False
                if epoch == current.epoch:
                    # Same-epoch writes are idempotent, never last-writer-wins.
                    # A different value at one epoch is a split-brain signal.
                    return current.value == value
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


class SQLiteControlStore:
    """Process-safe persistent store with monotonic epochs and CAS.

    SQLite is the built-in external-store implementation: it survives watcher
    restart and coordinates multiple local watcher processes without adding a
    runtime dependency.  Deployments may implement :class:`ControlStore`
    against a platform KV/etcd service when cross-host failover is required.
    """

    def __init__(
        self,
        path: str,
        *,
        poll_interval_s: float = 0.05,
        busy_timeout_s: float = 5.0,
    ) -> None:
        if not isinstance(path, str) or not path.strip() or path == ":memory:":
            raise ValueError(
                "SQLiteControlStore requires a non-empty persistent file path"
            )
        if poll_interval_s <= 0 or busy_timeout_s <= 0:
            raise ValueError("SQLite control-store timeouts must be positive")
        self.path = str(Path(path).expanduser().resolve())
        self.poll_interval_s = float(poll_interval_s)
        self.busy_timeout_s = float(busy_timeout_s)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @staticmethod
    def _validate(key: str, epoch: int) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("control-store key must be non-empty")
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("control-store epoch must be non-negative")

    @staticmethod
    def _encode(value: Any) -> str:
        try:
            return json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"control-store values must be canonical JSON: {exc}"
            ) from exc

    @staticmethod
    def _decode(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ContractViolation(
                "control-store contains invalid persisted JSON"
            ) from exc

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_s,
            isolation_level=None,
        )
        connection.execute(
            f"PRAGMA busy_timeout={int(self.busy_timeout_s * 1000)}"
        )
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS control_entries (
                    key TEXT PRIMARY KEY,
                    epoch INTEGER NOT NULL CHECK(epoch >= 0),
                    value_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def put_if_epoch(self, key: str, value: Any, epoch: int) -> bool:
        self._validate(key, epoch)
        encoded = self._encode(value)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT epoch, value_json FROM control_entries WHERE key = ?",
                (key,),
            ).fetchone()
            if row is not None:
                current_epoch, current_value = int(row[0]), str(row[1])
                if epoch < current_epoch:
                    connection.execute("ROLLBACK")
                    return False
                if epoch == current_epoch:
                    connection.execute("ROLLBACK")
                    return current_value == encoded
            connection.execute(
                """
                INSERT INTO control_entries(key, epoch, value_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    epoch = excluded.epoch,
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (key, epoch, encoded, time.time()),
            )
            connection.execute("COMMIT")
            return True
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def get(self, key: str) -> Any:
        if not isinstance(key, str) or not key:
            raise ValueError("control-store key must be non-empty")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value_json FROM control_entries WHERE key = ?", (key,)
            ).fetchone()
        return None if row is None else self._decode(str(row[0]))

    def get_epoch(self, key: str) -> Optional[int]:
        if not isinstance(key, str) or not key:
            raise ValueError("control-store key must be non-empty")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT epoch FROM control_entries WHERE key = ?", (key,)
            ).fetchone()
        return None if row is None else int(row[0])

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
        while True:
            values = {key: self.get(key) for key in predicates}
            if all(
                predicate(values[key]) for key, predicate in predicates.items()
            ):
                return values
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RecoveryTimeout(
                    "control-store predicates were not satisfied before timeout"
                )
            time.sleep(min(self.poll_interval_s, remaining))

    def compare_and_set(
        self,
        key: str,
        expected: Any,
        value: Any,
        epoch: int,
    ) -> bool:
        self._validate(key, epoch)
        expected_json = None if expected is None else self._encode(expected)
        value_json = self._encode(value)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT epoch, value_json FROM control_entries WHERE key = ?",
                (key,),
            ).fetchone()
            current_epoch = None if row is None else int(row[0])
            current_json = None if row is None else str(row[1])
            if current_epoch is not None and epoch < current_epoch:
                connection.execute("ROLLBACK")
                return False
            if current_json != expected_json:
                connection.execute("ROLLBACK")
                return False
            connection.execute(
                """
                INSERT INTO control_entries(key, epoch, value_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    epoch = excluded.epoch,
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (key, epoch, value_json, time.time()),
            )
            connection.execute("COMMIT")
            return True
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def require_same_epoch(self, *keys: str) -> int:
        if not keys:
            raise ValueError("at least one control-store key is required")
        placeholders = ",".join("?" for _ in keys)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT key, epoch FROM control_entries WHERE key IN ({placeholders})",
                tuple(keys),
            ).fetchall()
        if len(rows) != len(set(keys)):
            raise ContractViolation(
                "control-store values do not belong to one complete epoch"
            )
        epochs = {int(row[1]) for row in rows}
        if len(epochs) != 1:
            raise ContractViolation(
                "control-store values do not belong to one complete epoch"
            )
        return next(iter(epochs))

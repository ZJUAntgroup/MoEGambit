"""Framework-neutral atomic checkpoint commit records."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "CHECKPOINT_COMMIT_NAME",
    "atomic_write_json",
    "atomic_write_text",
    "load_checkpoint_commit",
    "publish_checkpoint_commit",
]


CHECKPOINT_COMMIT_NAME = ".moegambit-complete.json"


def _replace_and_sync(temporary: Path, target: Path) -> None:
    os.replace(temporary, target)
    directory_fd = os.open(str(target.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_and_sync(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(dict(value), sort_keys=True) + "\n",
    )


def publish_checkpoint_commit(
    checkpoint_dir: Path,
    *,
    framework: str,
    step: int,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    record = {
        "format_version": 1,
        "framework": str(framework),
        "step": int(step),
        "metadata": dict(metadata or {}),
    }
    target = checkpoint_dir / CHECKPOINT_COMMIT_NAME
    atomic_write_json(target, record)
    return target


def load_checkpoint_commit(
    checkpoint_dir: Path,
    *,
    framework: str | None = None,
    step: int | None = None,
) -> dict[str, Any]:
    path = Path(checkpoint_dir) / CHECKPOINT_COMMIT_NAME
    record = json.loads(path.read_text(encoding="utf-8"))
    if int(record.get("format_version", -1)) != 1:
        raise RuntimeError(f"unsupported checkpoint commit record: {path}")
    if framework is not None and record.get("framework") != framework:
        raise RuntimeError(
            "checkpoint framework mismatch: "
            f"{record.get('framework')!r} != {framework!r}"
        )
    if step is not None and int(record.get("step", -1)) != int(step):
        raise RuntimeError(
            "checkpoint step mismatch: "
            f"{record.get('step')!r} != {int(step)}"
        )
    return record

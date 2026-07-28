"""Filesystem protocol between a hot-spare agent and survivor workers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


PROTOCOL_VERSION = 1


def handoff_root() -> Path:
    value = os.environ.get("MOEGAMBIT_RECOVERY_HANDOFF_DIR", "").strip()
    if not value:
        raise RuntimeError(
            "MOEGAMBIT_RECOVERY_HANDOFF_DIR is required for mixed-version "
            "DeepSpeed recovery"
        )
    return Path(value)


def request_path(root: Path, epoch: int) -> Path:
    return root / f"request_epoch_{int(epoch)}.json"


def rank_state_path(root: Path, epoch: int, rank: int) -> Path:
    return root / f"epoch_{int(epoch)}_rank_{int(rank)}.pt"


def rank_ready_path(root: Path, epoch: int, rank: int) -> Path:
    return root / f"epoch_{int(epoch)}_rank_{int(rank)}.ready.json"


def rank_error_path(root: Path, epoch: int, rank: int) -> Path:
    return root / f"epoch_{int(epoch)}_rank_{int(rank)}.error.json"


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"handoff payload is not an object: {path}")
    return value

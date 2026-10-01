"""Strict JSON and atomic report publication."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError(f"non-finite JSON value: {value}")))


def write_json(path: str | Path, value: Any, *, exclusive: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = canonical(value)  # Validate before creating a temporary file.
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(name, target)  # Publish a complete file without overwriting another writer.
        else:
            os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value

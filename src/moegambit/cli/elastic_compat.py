"""Adapter dispatch for the historical elastic launcher filenames."""

from __future__ import annotations

import os
import sys
from typing import Sequence

__all__ = ["launcher_main", "watcher_main"]


def _extract_adapter(
    arguments: Sequence[str],
) -> tuple[str, list[str]]:
    values = list(arguments)
    selected = os.environ.get("MOEGAMBIT_ADAPTER", "megatron")
    forwarded: list[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value == "--":
            forwarded.extend(values[index:])
            break
        if value == "--adapter":
            if index + 1 >= len(values):
                raise SystemExit("--adapter requires a value")
            selected = values[index + 1]
            index += 2
            continue
        if value.startswith("--adapter="):
            selected = value.split("=", 1)[1]
            index += 1
            continue
        forwarded.append(value)
        index += 1
    normalized = selected.strip().lower()
    if normalized not in {"megatron", "deepspeed"}:
        raise SystemExit(
            "--adapter must be either 'megatron' or 'deepspeed'"
        )
    return normalized, forwarded


def launcher_main(arguments: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if arguments is None else arguments)
    options = values[:values.index("--")] if "--" in values else values
    if "--moegambit-runtime" in options:
        from .launch import main as runtime_main

        return int(
            runtime_main(
                [value for value in options if value != "--moegambit-runtime"]
                + values[len(options):]
            )
        )
    adapter, forwarded = _extract_adapter(values)
    if adapter == "deepspeed":
        from ..adapters.deepspeed.compat import launcher_main as selected_main
    else:
        from ..adapters.megatron.compat_launcher import main as selected_main
    return int(selected_main(forwarded))


def watcher_main(arguments: Sequence[str] | None = None) -> int:
    adapter, forwarded = _extract_adapter(
        sys.argv[1:] if arguments is None else arguments
    )
    if adapter == "deepspeed":
        from ..adapters.deepspeed.compat import watcher_main as selected_main
    else:
        from ..adapters.megatron.compat_watcher import main as selected_main
    return int(selected_main(forwarded))

"""Compatibility entry points for the shared elastic launcher names."""

from __future__ import annotations

from typing import Sequence

from ...runtime.hot_spare import main as hot_spare_main

__all__ = ["launcher_main", "watcher_main"]


def _with_mode(arguments: Sequence[str], expected: str) -> list[str]:
    values = list(arguments)
    selected = None
    options = values[:values.index("--")] if "--" in values else values
    for index, value in enumerate(options):
        if value == "--mode":
            if index + 1 >= len(options):
                raise SystemExit("--mode requires a value")
            selected = values[index + 1]
            break
        if value.startswith("--mode="):
            selected = value.split("=", 1)[1]
            break
    if selected is not None and selected != expected:
        raise SystemExit(
            f"this entry point requires --mode {expected}; got {selected}"
        )
    if selected is None:
        values[:0] = ["--mode", expected]
    return values


def launcher_main(arguments: Sequence[str]) -> int:
    """Run an active DeepSpeed node through the common hot-spare agent."""

    return int(hot_spare_main(_with_mode(arguments, "agent")))


def watcher_main(arguments: Sequence[str]) -> int:
    """Run the DeepSpeed spare node's coordinator and resident agent."""

    return int(hot_spare_main(_with_mode(arguments, "coordinator-agent")))

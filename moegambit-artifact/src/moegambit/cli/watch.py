"""moegambit-watch entry point."""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import Sequence

from moegambit.cli.common import add_runtime_options, runtime_config
from moegambit.runtime.config import RuntimeConfig
from moegambit.runtime.discovery import load_adapter
from moegambit.runtime.protocol import WireMessage
from moegambit.runtime.watcher import WatcherRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the watcher selected by an engine adapter"
    )
    add_runtime_options(parser)
    parser.set_defaults(adapter="megatron")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20200)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def _arguments(value: Sequence[str]) -> tuple[str, ...]:
    return tuple(value[1:] if value and value[0] == "--" else value)


def _has_option(arguments: Sequence[str], name: str) -> bool:
    return name in arguments or any(
        argument.startswith(name + "=") for argument in arguments
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = runtime_config(args)
    adapter = load_adapter(args.adapter)
    if not config.features.hot_swap:
        print("MoEGambit hot swap is disabled; no watcher was started.")
        return 0
    watcher_arguments = _arguments(args.arguments)
    if (
        adapter.capabilities.legacy_watcher
        and not _has_option(watcher_arguments, "--port")
    ):
        watcher_arguments = ("--port", str(args.port), *watcher_arguments)
    command = adapter.watcher_command(config.features, watcher_arguments)
    if command is not None:
        env = RuntimeConfig(
            adapter=adapter.name, features=config.features
        ).project_environment(os.environ)
        return int(subprocess.run(command, env=env, check=False).returncode)

    def handle(message: WireMessage) -> WireMessage:
        return WireMessage(
            "ack",
            {
                "received": message.kind,
                "adapter": adapter.name,
                "features": config.features.as_dict(),
            },
        )

    WatcherRuntime(args.host, args.port, handle).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

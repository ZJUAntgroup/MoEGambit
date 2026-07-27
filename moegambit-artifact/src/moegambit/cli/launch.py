"""moegambit-launch entry point."""

from __future__ import annotations

import argparse
import json
import shlex
from typing import Sequence

from moegambit.cli.common import add_runtime_options, runtime_config
from moegambit.runtime.launcher import prepare_launch, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a training engine through the MoEGambit runtime"
    )
    add_runtime_options(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the prepared command and feature environment",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="training command after --",
    )
    return parser


def _command(value: Sequence[str]) -> tuple[str, ...]:
    return tuple(value[1:] if value and value[0] == "--" else value)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = _command(args.command)
    if not command:
        raise SystemExit("a training command is required after --")
    config = runtime_config(args)
    if args.dry_run:
        prepared = prepare_launch(command, config)
        interesting = {
            key: value
            for key, value in prepared.environment.items()
            if key.startswith(("MOEGAMBIT_", "ELASTIC_ZERO2", "ELASTIC_HOT"))
        }
        print(
            json.dumps(
                {
                    "command": shlex.join(prepared.command),
                    "features": config.features.as_dict(),
                    "environment": interesting,
                    "metadata": dict(prepared.metadata),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return run(command, config)


if __name__ == "__main__":
    raise SystemExit(main())

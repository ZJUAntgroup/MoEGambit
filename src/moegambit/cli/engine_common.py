"""Shared CLI option handling."""

from __future__ import annotations

import argparse

from moegambit.core.contracts import FeatureSwitches
from moegambit.runtime.config import RuntimeConfig


def add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--adapter",
        default="auto",
        help="engine adapter name, or auto to infer from the command",
    )
    parser.add_argument(
        "--hot-swap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable in-place rank replacement",
    )
    parser.add_argument(
        "--zero2",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable ZeRO-2 optimizer memory replication",
    )


def runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    inherited = RuntimeConfig.from_env(adapter=args.adapter)
    return RuntimeConfig(
        adapter=args.adapter,
        features=FeatureSwitches(
            hot_swap=(
                inherited.features.hot_swap
                if args.hot_swap is None
                else args.hot_swap
            ),
            zero2=(
                inherited.features.zero2 if args.zero2 is None else args.zero2
            ),
        ),
    )

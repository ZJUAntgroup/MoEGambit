"""CLI for the framework-neutral node agent."""

from __future__ import annotations

import argparse
import os
import signal
import sys
from typing import Optional, Sequence

from ..agent.node_agent import NodeAgent, NodeLaunchSpec

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch fault-isolated distributed training workers",
        usage="%(prog)s [launcher args] -- command [command args]",
    )
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    return parser


def _split_argv(argv: Sequence[str]) -> tuple:
    values = list(argv)
    if "--" not in values:
        return values, []
    index = values.index("--")
    return values[:index], values[index + 1 :]


def main(argv: Optional[Sequence[str]] = None) -> int:
    launcher_argv, command = _split_argv(
        sys.argv[1:] if argv is None else argv
    )
    parser = build_parser()
    args = parser.parse_args(launcher_argv)
    if not command:
        parser.error("a training command is required after '--'")
    spec = NodeLaunchSpec(
        nnodes=args.nnodes,
        nproc_per_node=args.nproc_per_node,
        node_rank=args.node_rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
        argv=tuple(command),
        env=os.environ,
    )
    agent = NodeAgent(spec)

    def stop(signum: int, frame: object) -> None:
        del signum, frame
        agent.terminate_all()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    agent.start_all()
    exit_codes = agent.wait()
    return 0 if all(code == 0 for code in exit_codes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

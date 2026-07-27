"""CLI for the framework-neutral node agent."""

from __future__ import annotations

import argparse
import os
import signal
import sys
from typing import Optional, Sequence

from ..agent.node_agent import NodeAgent, NodeLaunchSpec
from ..config import RuntimeConfig
from ..runtime.client import ControlClient, ControlClientConfig

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
    parser.add_argument("--watcher-host")
    parser.add_argument("--watcher-port", type=int)
    parser.add_argument("--job-id")
    parser.add_argument("--attempt-id")
    parser.add_argument("--job-token")
    parser.add_argument(
        "--checkpoint-arg",
        action="append",
        default=[],
        help=(
            "argument appended on checkpoint relaunch; supports "
            "{checkpoint_locator}, {checkpoint_step}, {next_attempt_id}"
        ),
    )
    parser.add_argument("--control-poll-seconds", type=float, default=0.5)
    parser.add_argument("--relaunch-grace-seconds", type=float, default=30.0)
    parser.add_argument("--max-relaunches", type=int, default=3)
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
    config = RuntimeConfig.from_env()
    job_id = args.job_id or config.job_id
    attempt_id = args.attempt_id or config.attempt_id
    spec = NodeLaunchSpec(
        nnodes=args.nnodes,
        nproc_per_node=args.nproc_per_node,
        node_rank=args.node_rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
        argv=tuple(command),
        env=os.environ,
        job_id=job_id,
        attempt_id=attempt_id,
        checkpoint_argv_template=tuple(args.checkpoint_arg),
    )
    agent = NodeAgent(spec)

    def stop(signum: int, frame: object) -> None:
        del signum, frame
        agent.terminate_all()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    agent.start_all()
    watcher_host = args.watcher_host or (
        config.watcher.host if config.enabled else None
    )
    if watcher_host:
        watcher_port = args.watcher_port or config.watcher.port
        client = ControlClient(
            ControlClientConfig(
                host=watcher_host,
                port=int(watcher_port),
                job_id=job_id,
                attempt_id=attempt_id,
                sender={
                    "role": "node_agent",
                    "node_rank": args.node_rank,
                },
                job_token=args.job_token or config.security.job_token,
                request_timeout_s=config.recovery_timeout_s,
                max_message_bytes=config.security.max_message_bytes,
            )
        )
        exit_codes = agent.run_with_control(
            client,
            poll_interval_s=args.control_poll_seconds,
            relaunch_grace_s=args.relaunch_grace_seconds,
            max_relaunches=args.max_relaunches,
        )
    else:
        exit_codes = agent.wait()
    return 0 if all(code == 0 for code in exit_codes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

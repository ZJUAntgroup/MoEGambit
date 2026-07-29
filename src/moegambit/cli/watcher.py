"""CLI for the versioned MoEGambit watcher/control service."""

from __future__ import annotations

import argparse
import signal
import threading
from typing import Optional, Sequence

from ..config import RuntimeConfig
from ..control.service import RecoveryCoordinatorService
from ..control.state_store import InMemoryControlStore, SQLiteControlStore
from ..control.watcher import ControlRequestProcessor, ControlServer
from ..policy import MoeHybridPolicy, PeerOrCheckpointPolicy

__all__ = ["build_parser", "main"]


def build_parser(config: Optional[RuntimeConfig] = None) -> argparse.ArgumentParser:
    resolved = config or RuntimeConfig.from_env()
    parser = argparse.ArgumentParser(
        description="Run the authenticated MoEGambit recovery coordinator"
    )
    parser.add_argument("--bind-host", default=resolved.security.bind_host)
    parser.add_argument("--port", type=int, default=resolved.watcher.port)
    parser.add_argument("--job-token", default=resolved.security.job_token)
    parser.add_argument(
        "--require-token",
        action="store_true",
        default=resolved.security.require_token,
    )
    parser.add_argument(
        "--max-message-bytes",
        type=int,
        default=resolved.security.max_message_bytes,
    )
    parser.add_argument(
        "--control-store",
        choices=("sqlite", "memory"),
        default=resolved.control_store.backend,
    )
    parser.add_argument(
        "--control-store-path", default=resolved.control_store.path
    )
    parser.add_argument(
        "--policy", choices=("peer", "moe-hybrid"), default="peer"
    )
    parser.add_argument("--rendezvous-host", required=True)
    parser.add_argument("--rendezvous-port", type=int, required=True)
    parser.add_argument("--rendezvous-prefix", default="moegambit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = RuntimeConfig.from_env()
    args = build_parser(config).parse_args(argv)
    if not 0 < int(args.rendezvous_port) <= 65535:
        raise SystemExit("--rendezvous-port must be between 1 and 65535")
    if args.control_store == "sqlite":
        control_store = SQLiteControlStore(
            args.control_store_path,
            poll_interval_s=config.control_store.poll_interval_s,
            busy_timeout_s=config.control_store.busy_timeout_s,
        )
    else:
        control_store = InMemoryControlStore()
    policy = MoeHybridPolicy() if args.policy == "moe-hybrid" else PeerOrCheckpointPolicy()
    service = RecoveryCoordinatorService(
        store_provider=lambda payload: {
            "host": args.rendezvous_host,
            "port": int(args.rendezvous_port),
            "prefix": (
                f"{args.rendezvous_prefix}/epoch-"
                f"{int(payload['recovery_epoch'])}"
            ),
            "timeout_s": config.recovery_timeout_s,
        },
        control_store=control_store,
        policy=policy,
    )
    processor = ControlRequestProcessor(
        service,
        job_token=args.job_token,
        require_token=bool(args.require_token),
        max_message_bytes=int(args.max_message_bytes),
    )
    server = ControlServer(args.bind_host, int(args.port), processor)
    stopped = threading.Event()

    def stop(_signum: int, _frame: object) -> None:
        stopped.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with server:
        host, port = server.address
        print(f"MoEGambit watcher listening on {host}:{port}", flush=True)
        stopped.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

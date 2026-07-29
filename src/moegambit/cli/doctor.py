"""Read-only deployment preflight for the canonical MoEGambit package."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .. import PACKAGE_ROLE, __version__
from ..adapters.registry import available_adapters
from ..config import RuntimeConfig

__all__ = ["run_checks", "build_parser", "main"]


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def run_checks(
    config: RuntimeConfig,
    *,
    framework: Optional[str] = None,
    role: str = "worker",
) -> Mapping[str, Any]:
    if role not in ("worker", "watcher"):
        raise ValueError("doctor role must be 'worker' or 'watcher'")
    errors = list(config.validate())
    warnings = []
    selected = framework or config.framework
    adapters = available_adapters()
    if role == "worker" and selected not in adapters:
        errors.append(
            f"framework adapter {selected!r} is unavailable; choices={adapters}"
        )

    requirements = []
    if role == "worker" and selected == "generic_ddp":
        requirements.append("torch")
    elif role == "worker" and selected == "megatron":
        requirements.extend(("torch", "megatron"))
    missing = [name for name in requirements if importlib.util.find_spec(name) is None]
    if missing:
        errors.append(
            f"framework {selected!r} is missing runtime modules: {', '.join(missing)}"
        )

    if config.control_store.backend == "sqlite":
        database = Path(config.control_store.path).expanduser().resolve()
        parent = _nearest_existing_parent(database.parent)
        if not parent.is_dir() or not os.access(str(parent), os.W_OK):
            errors.append(
                f"SQLite control-store parent is not writable: {database.parent}"
            )
        if database.exists() and not os.access(str(database), os.R_OK | os.W_OK):
            errors.append(
                f"SQLite control-store file is not readable and writable: {database}"
            )
        warnings.append(
            "SQLite persists watcher state on one host; multi-host watcher HA "
            "requires a shared CAS-capable ControlStore implementation"
        )
    if not config.security.require_token:
        warnings.append(
            "control authentication is optional; keep the watcher on a trusted interface"
        )

    return {
        "ok": not errors,
        "package_role": PACKAGE_ROLE,
        "version": __version__,
        "framework": selected,
        "role": role,
        "available_adapters": list(adapters),
        "errors": errors,
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight a MoEGambit deployment")
    parser.add_argument("--framework")
    parser.add_argument("--role", choices=("worker", "watcher"), default="worker")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_checks(
        RuntimeConfig.from_env(), framework=args.framework, role=args.role
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        status = "OK" if result["ok"] else "FAILED"
        print(f"MoEGambit doctor: {status}")
        print(f"framework: {result['framework']}")
        for item in result["errors"]:
            print(f"ERROR: {item}")
        for item in result["warnings"]:
            print(f"WARN: {item}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

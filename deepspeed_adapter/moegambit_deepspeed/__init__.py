"""DeepSpeed adapter for elastic relaunch and ZeRO-2 host replication."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from moegambit.interfaces import (
    AdapterCapabilities,
    LaunchRequest,
    PreparedLaunch,
)


def _has_option(command: Sequence[str], name: str) -> bool:
    return name in command or any(item.startswith(name + "=") for item in command)


def _runner_option_index(command: Sequence[str]) -> int | None:
    if command and Path(command[0]).name in {"deepspeed", "deepspeed.exe"}:
        return 1
    if (
        len(command) >= 3
        and command[1] == "-m"
        and command[2] == "deepspeed.launcher.runner"
    ):
        return 3
    return None


def _insert_option(
    command: tuple[str, ...], index: int, option: str
) -> tuple[str, ...]:
    return (*command[:index], option, *command[index:])


def _prepend_path(environment: dict[str, str], path: Path) -> None:
    current = environment.get("PYTHONPATH", "")
    values = [item for item in current.split(os.pathsep) if item]
    path_text = str(path)
    if path_text in values:
        values.remove(path_text)
    environment["PYTHONPATH"] = os.pathsep.join([path_text, *values])


def _enabled(environment: dict[str, str], name: str) -> bool:
    return environment.get(name, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class DeepSpeedAdapter:
    name = "deepspeed"
    capabilities = AdapterCapabilities(
        hot_swap=True,
        zero2=True,
        watcher_required=False,
    )

    def probe(self, command: Sequence[str]) -> bool:
        text = " ".join(command).lower()
        return "deepspeed" in text

    def prepare_launch(self, request: LaunchRequest) -> PreparedLaunch:
        command = tuple(request.command)
        environment = dict(request.environment)
        repository = Path(__file__).resolve().parents[2]
        adapter_root = Path(__file__).resolve().parents[1]
        candidates = (
            repository / "DeepSpeed",
            repository / "src" / "DeepSpeed",
        )
        vendored = next((path for path in candidates if path.is_dir()), None)
        if vendored is not None:
            _prepend_path(environment, vendored)
            _prepend_path(environment, adapter_root)
            environment["MOEGAMBIT_DEEPSPEED_ROOT"] = str(vendored)

        strategy = "disabled"
        if request.features.hot_swap:
            application_checkpoint = _enabled(
                environment,
                "MOEGAMBIT_DEEPSPEED_APPLICATION_CHECKPOINT",
            )
            checkpoint_dir = environment.get(
                "MOEGAMBIT_DEEPSPEED_CHECKPOINT_DIR"
            )
            try:
                checkpoint_interval = int(
                    environment.get(
                        "MOEGAMBIT_DEEPSPEED_CHECKPOINT_INTERVAL", "0"
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    "MOEGAMBIT_DEEPSPEED_CHECKPOINT_INTERVAL must be an integer"
                ) from exc
            if (
                not application_checkpoint
                and (not checkpoint_dir or checkpoint_interval <= 0)
            ):
                raise ValueError(
                    "DeepSpeed hot swap needs durable recovery state: set "
                    "MOEGAMBIT_DEEPSPEED_CHECKPOINT_DIR and a positive "
                    "MOEGAMBIT_DEEPSPEED_CHECKPOINT_INTERVAL, or set "
                    "MOEGAMBIT_DEEPSPEED_APPLICATION_CHECKPOINT=1 when the "
                    "training application saves and restores checkpoints"
                )
            option_index = _runner_option_index(command)
            external = environment.get(
                "MOEGAMBIT_DEEPSPEED_EXTERNAL_ELASTIC", "0"
            ).lower() in {"1", "true", "yes", "on"}
            if option_index is None and not external:
                raise ValueError(
                    "DeepSpeed hot swap requires the `deepspeed` launcher, "
                    "`python -m deepspeed.launcher.runner`, or "
                    "MOEGAMBIT_DEEPSPEED_EXTERNAL_ELASTIC=1"
                )
            if option_index is not None and not external:
                options: list[str] = []
                if not _has_option(command, "--elastic_training"):
                    options.append("--elastic_training")
                master_addr = environment.get("MASTER_ADDR")
                if master_addr and not _has_option(command, "--master_addr"):
                    options.append(f"--master_addr={master_addr}")
                min_nodes = environment.get("MOEGAMBIT_DEEPSPEED_MIN_NODES")
                if min_nodes and not _has_option(command, "--min_elastic_nodes"):
                    options.append(f"--min_elastic_nodes={min_nodes}")
                max_nodes = environment.get("MOEGAMBIT_DEEPSPEED_MAX_NODES")
                if max_nodes and not _has_option(command, "--max_elastic_nodes"):
                    options.append(f"--max_elastic_nodes={max_nodes}")
                for option in reversed(options):
                    command = _insert_option(command, option_index, option)
            strategy = (
                environment.get(
                    "MOEGAMBIT_DEEPSPEED_RECOVERY_STRATEGY",
                    "node_hot_spare_epoch_relaunch",
                )
                if external
                else "torch_elastic_checkpoint_relaunch"
            )

        environment.update(
            {
                "MOEGAMBIT_DEEPSPEED_ADAPTER": "1",
                "DEEPSPEED_MOEGAMBIT_HOT_SWAP": (
                    "1" if request.features.hot_swap else "0"
                ),
                "DEEPSPEED_MOEGAMBIT_ZERO2": (
                    "1" if request.features.zero2 else "0"
                ),
                "MOEGAMBIT_DEEPSPEED_RECOVERY_STRATEGY": strategy,
            }
        )
        return PreparedLaunch(
            command=command,
            environment=environment,
            cwd=request.cwd,
            metadata={
                "engine": self.name,
                "integration": "engine-hooks",
                "recovery_strategy": strategy,
                "hot_swap": request.features.hot_swap,
                "zero2": request.features.zero2,
            },
        )

    def watcher_command(self, features, arguments: Sequence[str]):
        del features, arguments
        return None


from moegambit_deepspeed.integration import attach_engine

__all__ = ["DeepSpeedAdapter", "attach_engine"]

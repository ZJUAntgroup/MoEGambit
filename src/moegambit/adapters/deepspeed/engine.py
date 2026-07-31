"""DeepSpeed adapter for rank-granular hot swap and ZeRO-2 replication."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

from moegambit.interfaces import (
    AdapterCapabilities,
    LaunchRequest,
    PreparedLaunch,
)
from moegambit.runtime.launch_environment import prepend_python_path


def _enabled(environment: dict[str, str], name: str) -> bool:
    return environment.get(name, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class DeepSpeedEngineAdapter:
    name = "deepspeed"
    capabilities = AdapterCapabilities(
        hot_swap=True,
        zero2=True,
        watcher_required=True,
    )

    def probe(self, command: Sequence[str]) -> bool:
        text = " ".join(command).lower()
        return "deepspeed" in text

    def prepare_launch(self, request: LaunchRequest) -> PreparedLaunch:
        command = tuple(request.command)
        environment = dict(request.environment)
        repository = Path(__file__).resolve().parents[4]
        candidates = (
            repository / "DeepSpeed",
            repository / "src" / "DeepSpeed",
        )
        vendored = next((path for path in candidates if path.is_dir()), None)
        if vendored is not None:
            prepend_python_path(environment, vendored)
            prepend_python_path(environment, repository / "src")
            environment["MOEGAMBIT_DEEPSPEED_ROOT"] = str(vendored)

        strategy = "disabled"
        if request.features.hot_swap:
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
            if not checkpoint_dir or checkpoint_interval <= 0:
                raise ValueError(
                    "DeepSpeed hot swap needs durable recovery state: set "
                    "MOEGAMBIT_DEEPSPEED_CHECKPOINT_DIR and a positive "
                    "MOEGAMBIT_DEEPSPEED_CHECKPOINT_INTERVAL"
                )
            if not _enabled(
                environment, "MOEGAMBIT_DEEPSPEED_EXTERNAL_COORDINATOR"
            ):
                raise ValueError(
                    "rank-granular DeepSpeed hot swap requires "
                    "MOEGAMBIT_DEEPSPEED_EXTERNAL_COORDINATOR=1 and the "
                    "MoEGambit hot-spare coordinator"
                )
            requested_strategy = environment.get(
                "MOEGAMBIT_DEEPSPEED_RECOVERY_STRATEGY",
                "rank_in_process_hybrid",
            )
            if requested_strategy != "rank_in_process_hybrid":
                raise ValueError(
                    "DeepSpeed hot swap supports only "
                    "rank_in_process_hybrid"
                )
            strategy = "rank_in_process_hybrid"

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
                "MOEGAMBIT_DEEPSPEED_INPROCESS_RECOVERY": (
                    "1" if request.features.hot_swap else "0"
                ),
                "MOEGAMBIT_DEEPSPEED_HYBRID_RESTORE": (
                    "1" if request.features.hot_swap else "0"
                ),
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
        del features
        return (
            sys.executable,
            "-m",
            "moegambit.runtime.hot_spare",
            "--mode",
            "coordinator-agent",
            *tuple(arguments),
        )


from .integration import attach_engine

# Compatibility for callers that used the pre-unification class name.
DeepSpeedAdapter = DeepSpeedEngineAdapter

__all__ = [
    "DeepSpeedAdapter",
    "DeepSpeedEngineAdapter",
    "attach_engine",
]

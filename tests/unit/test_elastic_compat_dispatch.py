"""Adapter dispatch tests for the historical elastic script names."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from moegambit.adapters.deepspeed.compat import _with_mode
from moegambit.adapters.deepspeed.engine import DeepSpeedEngineAdapter
from moegambit.cli.elastic_compat import launcher_main, watcher_main
from moegambit.core.contracts import FeatureSwitches


def test_megatron_remains_the_default_launcher_adapter():
    with patch(
        "moegambit.adapters.megatron.compat_launcher.main",
        return_value=7,
    ) as selected:
        assert launcher_main(["--legacy-option"]) == 7

    selected.assert_called_once_with(["--legacy-option"])


def test_deepspeed_launcher_selects_active_agent_mode():
    arguments = [
        "--coordinator-host",
        "10.0.0.9",
        "--coordinator-port",
        "20221",
    ]
    with patch(
        "moegambit.adapters.deepspeed.compat.hot_spare_main",
        return_value=0,
    ) as selected:
        assert (
            launcher_main(["--adapter", "deepspeed", *arguments])
            == 0
        )

    assert selected.call_args.args[0] == ["--mode", "agent", *arguments]


def test_deepspeed_watcher_selects_coordinator_agent_mode():
    arguments = [
        "--coordinator-host",
        "10.0.0.9",
        "--coordinator-port",
        "20221",
    ]
    with patch(
        "moegambit.adapters.deepspeed.compat.hot_spare_main",
        return_value=0,
    ) as selected:
        assert (
            watcher_main(["--adapter=deepspeed", *arguments])
            == 0
        )

    assert selected.call_args.args[0] == [
        "--mode",
        "coordinator-agent",
        *arguments,
    ]


def test_framework_entrypoints_reject_conflicting_deepspeed_modes():
    with pytest.raises(SystemExit, match="requires --mode agent"):
        _with_mode(["--mode", "coordinator-agent"], "agent")
    with pytest.raises(SystemExit, match="requires --mode coordinator-agent"):
        _with_mode(["--mode=agent"], "coordinator-agent")


def test_unknown_adapter_is_rejected_before_launch():
    with pytest.raises(SystemExit, match="megatron.*deepspeed"):
        launcher_main(["--adapter", "unknown", "--help"])


def test_deepspeed_engine_adapter_selects_common_hot_spare_watcher():
    adapter = DeepSpeedEngineAdapter()
    command = adapter.watcher_command(
        FeatureSwitches(hot_swap=True),
        ("--coordinator-host", "10.0.0.9"),
    )

    assert adapter.capabilities.watcher_required
    assert command[1:5] == (
        "-m",
        "moegambit.runtime.hot_spare",
        "--mode",
        "coordinator-agent",
    )

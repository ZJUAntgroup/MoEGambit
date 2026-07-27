"""Release preflight remains read-only and dependency-light."""

from __future__ import annotations

from moegambit.cli.doctor import run_checks
from moegambit.config import ControlStoreConfig, RuntimeConfig


def test_doctor_reports_missing_optional_framework_without_importing_it(tmp_path):
    config = RuntimeConfig(
        framework="not-installed",
        control_store=ControlStoreConfig(path=str(tmp_path / "control.db")),
    )

    result = run_checks(config)

    assert result["ok"] is False
    assert any("unavailable" in item for item in result["errors"])


def test_doctor_describes_sqlite_ha_boundary(tmp_path):
    config = RuntimeConfig(
        framework="generic_ddp",
        control_store=ControlStoreConfig(path=str(tmp_path / "control.db")),
    )

    result = run_checks(config)

    assert any("multi-host watcher HA" in item for item in result["warnings"])


def test_watcher_role_does_not_require_training_framework_modules(tmp_path):
    config = RuntimeConfig(
        framework="generic_ddp",
        control_store=ControlStoreConfig(path=str(tmp_path / "control.db")),
    )

    result = run_checks(config, role="watcher")

    assert not any("torch" in item for item in result["errors"])
    assert result["role"] == "watcher"

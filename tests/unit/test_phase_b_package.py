"""Phase B package, import-discipline, configuration, and capability tests."""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest


SRC_ROOT = Path(__file__).parents[2] / "src"
REPOSITORY_ROOT = SRC_ROOT.parent

import moegambit  # noqa: E402
from moegambit.capabilities import AdapterCapabilities  # noqa: E402
from moegambit.config import FallbackMode, RuntimeConfig  # noqa: E402
from moegambit.errors import MoEGambitError  # noqa: E402


def test_public_package_exports_phase_b_contracts_lazily():
    assert moegambit.__version__ == "0.1.0.dev0"
    assert moegambit.PROTOCOL_VERSION == 1
    assert moegambit.PACKAGE_ROLE == "core"
    assert moegambit.AdapterCapabilities.__name__ == AdapterCapabilities.__name__
    assert moegambit.AdapterCapabilities.__module__ == "moegambit.capabilities"
    assert moegambit.RuntimeConfig.__name__ == RuntimeConfig.__name__
    assert moegambit.RuntimeConfig.__module__ == "moegambit.config"
    assert moegambit.MoEGambitError.__name__ == MoEGambitError.__name__
    assert moegambit.MoEGambitError.__module__ == "moegambit.errors"


def test_clean_import_does_not_load_optional_frameworks():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(SRC_ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    code = """
import sys
import moegambit
_ = moegambit.RuntimeConfig
_ = moegambit.AdapterCapabilities
_ = moegambit.MoEGambitError
forbidden = ('torch', 'megatron', 'deepspeed')
assert not any(
    name == prefix or name.startswith(prefix + '.')
    for name in sys.modules
    for prefix in forbidden
)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_core_and_deepspeed_plugin_share_one_package_root():
    plugin_root = SRC_ROOT.parent / "deepspeed_adapter"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(SRC_ROOT), str(plugin_root)]
    )
    code = """
from pathlib import Path
import moegambit
import moegambit_deepspeed
assert Path(moegambit.__file__).resolve().parent == Path(
    r'%s'
).resolve()
assert moegambit_deepspeed.DeepSpeedAdapter.name == 'deepspeed'
""" % (SRC_ROOT / "moegambit")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=SRC_ROOT.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_runtime_config_reads_new_names_and_control_identity():
    config = RuntimeConfig.from_env(
        {
            "MOEGAMBIT_ENABLED": "1",
            "MOEGAMBIT_FRAMEWORK": "custom",
            "MOEGAMBIT_JOB_ID": "job-7",
            "MOEGAMBIT_ATTEMPT_ID": "attempt-3",
            "MOEGAMBIT_WATCHER_HOST": "10.1.2.3",
            "MOEGAMBIT_WATCHER_PORT": "24000",
            "MOEGAMBIT_BIND_HOST": "10.1.2.3",
            "MOEGAMBIT_REQUIRE_TOKEN": "1",
            "MOEGAMBIT_JOB_TOKEN": "secret",
            "MOEGAMBIT_OPTIMIZER_REPLICATION": "1",
            "MOEGAMBIT_CONTROL_STORE_BACKEND": "sqlite",
            "MOEGAMBIT_CONTROL_STORE_PATH": "/tmp/moegambit-test-control.db",
        }
    )

    assert config.enabled
    assert config.framework == "custom"
    assert config.job_id == "job-7"
    assert config.attempt_id == "attempt-3"
    assert str(config.watcher) == "10.1.2.3:24000"
    assert config.security.require_token
    assert config.optimizer_replication.enabled
    assert config.control_store.backend == "sqlite"
    assert config.control_store.path == "/tmp/moegambit-test-control.db"
    assert config.validate() == ()


def test_actual_legacy_environment_names_remain_compatible():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        config = RuntimeConfig.from_env(
            {
                "ELASTIC_ENABLED": "1",
                "ELASTIC_ZERO2_MEMORY_REPLICATION": "1",
            }
        )

    assert config.enabled
    assert config.optimizer_replication.enabled
    assert any(item.category is DeprecationWarning for item in captured)


def test_runtime_config_is_fail_closed_and_reports_all_validation_errors():
    config = RuntimeConfig.from_env(
        {
            "MOEGAMBIT_BIND_HOST": "0.0.0.0",
            "MOEGAMBIT_REQUIRE_TOKEN": "1",
            "MOEGAMBIT_RECOVERY_TIMEOUT_SECONDS": "0",
            "MOEGAMBIT_MAX_MESSAGE_BYTES": "0",
        }
    )

    assert config.fallback == FallbackMode.ABORT
    findings = config.validate()
    assert any("wildcard" in item for item in findings)
    assert any("job_token" in item for item in findings)
    assert any("recovery_timeout_s" in item for item in findings)
    assert any("max_message_bytes" in item for item in findings)


def test_explicit_python_overrides_have_highest_priority():
    config = RuntimeConfig.from_env({"MOEGAMBIT_ENABLED": "0"})

    overridden = config.merged_with(enabled=True, framework="megatron")

    assert overridden.enabled
    assert overridden.framework == "megatron"
    assert not config.enabled


def test_capability_digest_is_stable_and_sensitive():
    first = AdapterCapabilities(
        full_group_rebuild=True,
        supported_zero_stages=frozenset({0}),
        supported_parallel_axes=frozenset({"dp"}),
    )
    equal = AdapterCapabilities(
        supported_parallel_axes=frozenset({"dp"}),
        supported_zero_stages=frozenset({0}),
        full_group_rebuild=True,
    )
    different = AdapterCapabilities(
        full_group_rebuild=True,
        peer_parameter_restore=True,
        supported_zero_stages=frozenset({0}),
        supported_parallel_axes=frozenset({"dp"}),
    )

    assert first.digest() == equal.digest()
    assert first.digest() != different.digest()


def test_unknown_public_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        getattr(moegambit, "not_a_public_contract")


def test_repository_legal_files_are_present_and_packaged():
    license_text = (REPOSITORY_ROOT / "LICENSE").read_text(encoding="utf-8")
    legal_text = (REPOSITORY_ROOT / "LEGAL.md").read_text(encoding="utf-8")
    project_metadata = (REPOSITORY_ROOT / "pyproject.toml").read_text(
        encoding="utf-8"
    )

    assert "Apache License" in license_text
    assert "DeepSpeed/" in legal_text
    assert "Megatron-LM/" in legal_text
    assert 'license-files = ["LICENSE", "LEGAL.md"]' in project_metadata

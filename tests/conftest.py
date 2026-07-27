"""Process-level isolation for the two temporary ``moegambit`` packages.

Switching ``sys.modules`` between tests is not safe: test modules retain class
objects imported during collection, and production code can never rely on
pytest hooks.  The default ``core`` profile therefore exposes only ``src`` and
does not collect the DeepSpeed-owned suite.  That suite can be run explicitly
in a fresh interpreter with the ``deepspeed`` profile, which exposes only
``deepspeed_adapter``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEEPSPEED_SOURCE_ROOT = REPOSITORY_ROOT / "deepspeed_adapter"
PROFILE_ENV = "MOEGAMBIT_TEST_PACKAGE_PROFILE"
CORE_PROFILE = "core"
DEEPSPEED_PROFILE = "deepspeed"
DEEPSPEED_TEST = "test_deepspeed_real_adapter.py"

_profile = os.environ.get(PROFILE_ENV, CORE_PROFILE).strip().lower()
if _profile not in (CORE_PROFILE, DEEPSPEED_PROFILE):
    raise pytest.UsageError(
        f"{PROFILE_ENV} must be {CORE_PROFILE!r} or {DEEPSPEED_PROFILE!r}, "
        f"not {_profile!r}"
    )


def _resolved_path(value: str) -> Path:
    return Path(value or os.getcwd()).resolve()


def _activate_source_root(source_root: Path) -> None:
    excluded = {RUNTIME_SOURCE_ROOT.resolve(), DEEPSPEED_SOURCE_ROOT.resolve()}
    retained = []
    for value in sys.path:
        try:
            if _resolved_path(value) in excluded:
                continue
        except (OSError, RuntimeError):
            pass
        retained.append(value)
    sys.path[:] = [str(source_root), *retained]


_active_source_root = (
    RUNTIME_SOURCE_ROOT if _profile == CORE_PROFILE else DEEPSPEED_SOURCE_ROOT
)
_activate_source_root(_active_source_root)


def pytest_ignore_collect(collection_path: Path, config):
    del config
    test_name = collection_path.name
    if not test_name.startswith("test") or collection_path.suffix != ".py":
        return None
    if _profile == CORE_PROFILE:
        return True if test_name == DEEPSPEED_TEST else None
    return True if test_name != DEEPSPEED_TEST else None


def pytest_sessionstart(session) -> None:
    del session
    spec = importlib.util.find_spec("moegambit")
    expected = (_active_source_root / "moegambit" / "__init__.py").resolve()
    if spec is None or spec.origin is None:
        raise pytest.UsageError(
            f"the {_profile} test profile cannot resolve the moegambit package"
        )
    actual = Path(spec.origin).resolve()
    if actual != expected:
        raise pytest.UsageError(
            f"the {_profile} test profile resolved moegambit from {actual}; "
            f"expected {expected}"
        )

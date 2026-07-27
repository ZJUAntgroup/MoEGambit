"""Test-process isolation for the two temporary ``moegambit`` package roots.

The runtime branch introduces the canonical package under ``src/`` while the
existing DeepSpeed work still carries a private package under
``deepspeed_adapter/``.  Until the future integration decision removes that
duplication, tests must not let one implementation remain cached while testing
the other.  This hook changes no production import behaviour and is narrowly
scoped to the two affected test groups.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEEPSPEED_SOURCE_ROOT = REPOSITORY_ROOT / "deepspeed_adapter"

_active_source_root: Optional[Path] = None


def _purge_moegambit_modules() -> None:
    prefixes = ("moegambit", "moegambit_deepspeed", "moegambit_megatron")
    for module_name in tuple(sys.modules):
        if any(
            module_name == prefix or module_name.startswith(prefix + ".")
            for prefix in prefixes
        ):
            sys.modules.pop(module_name, None)


def _activate_source_root(source_root: Path) -> None:
    global _active_source_root
    if _active_source_root == source_root:
        return

    _purge_moegambit_modules()
    source_strings = {str(RUNTIME_SOURCE_ROOT), str(DEEPSPEED_SOURCE_ROOT)}
    sys.path[:] = [item for item in sys.path if item not in source_strings]
    sys.path.insert(0, str(source_root))
    _active_source_root = source_root


def pytest_runtest_setup(item) -> None:
    test_name = Path(str(item.path)).name
    if test_name == "test_deepspeed_real_adapter.py":
        _activate_source_root(DEEPSPEED_SOURCE_ROOT)
    elif test_name.startswith("test_phase_b_"):
        _activate_source_root(RUNTIME_SOURCE_ROOT)

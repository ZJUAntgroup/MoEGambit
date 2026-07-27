"""Fail-closed checks for ambiguous top-level package roots.

The repository temporarily contains both the canonical package under ``src``
and a DeepSpeed-owned compatibility package under ``deepspeed_adapter``.  If
both roots are importable, Python silently chooses whichever appears first on
``sys.path``.  That is unsafe because the two packages expose different public
contracts under the same name.

This module cannot make the competing package disappear, but it prevents the
canonical package from continuing after it detects that ambiguous setup.
Launchers and tests must still place exactly one package root on ``sys.path``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

__all__ = [
    "PackageIdentityError",
    "find_package_initializers",
    "assert_unambiguous_package",
]


class PackageIdentityError(ImportError):
    """More than one implementation of the same top-level package is visible."""


def _search_root(value: str) -> Optional[Path]:
    try:
        return Path(value or os.getcwd()).resolve()
    except (OSError, RuntimeError):
        return None


def find_package_initializers(
    package: str = "moegambit",
    search_path: Optional[Iterable[str]] = None,
) -> Tuple[Path, ...]:
    """Return every filesystem ``__init__.py`` visible for ``package``.

    Paths are resolved and de-duplicated so a repeated ``PYTHONPATH`` entry is
    not mistaken for a second implementation.
    """

    parts = tuple(package.split("."))
    initializers = []
    seen = set()
    for value in sys.path if search_path is None else search_path:
        root = _search_root(value)
        if root is None:
            continue
        candidate = root.joinpath(*parts, "__init__.py")
        try:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        initializers.append(resolved)
    return tuple(initializers)


def assert_unambiguous_package(current_file: str) -> None:
    """Reject an environment exposing another top-level ``moegambit``.

    Import ambiguity must be repaired before any framework or distributed
    state is initialized.  Continuing would make later failures dependent on
    path ordering and could select an incompatible runtime implementation.
    """

    active = Path(current_file).resolve()
    competing = tuple(
        path for path in find_package_initializers() if path != active
    )
    if not competing:
        return
    locations = ", ".join(str(path) for path in competing)
    raise PackageIdentityError(
        "multiple top-level 'moegambit' packages are importable; "
        f"active={active}; competing={locations}. "
        "Do not place both 'src' and 'deepspeed_adapter' on PYTHONPATH. "
        "Use an isolated core or DeepSpeed package profile."
    )

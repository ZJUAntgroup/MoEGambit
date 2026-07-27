"""Lazy discovery and construction of framework adapters.

Phase B provides the registry contract but intentionally registers no built-in
framework: Megatron and Generic DDP land in later phases, while DeepSpeed is
outside this branch's current scope.
"""

from __future__ import annotations

from importlib import metadata
from typing import Any, Callable, Dict, Iterable, Tuple

from ..errors import AdapterUnsupportedError

__all__ = ["register_adapter", "available_adapters", "get_adapter"]

_ENTRY_POINT_GROUP = "moegambit.frameworks"
_REGISTERED: Dict[str, Callable[..., Any]] = {}


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("adapter name must be a non-empty string")
    if name != name.strip():
        raise ValueError("adapter name must not contain surrounding whitespace")
    return name


def register_adapter(name: str, factory: Callable[..., Any]) -> Callable[..., Any]:
    """Register one process-local adapter factory."""

    validated = _validate_name(name)
    if not callable(factory) and not callable(getattr(factory, "build", None)):
        raise TypeError("adapter factory must be callable or expose build(**kwargs)")
    _REGISTERED[validated] = factory
    return factory


def _entry_points() -> Tuple[Any, ...]:
    try:
        discovered = metadata.entry_points()
    except Exception:
        return ()
    if hasattr(discovered, "select"):
        return tuple(discovered.select(group=_ENTRY_POINT_GROUP))
    if isinstance(discovered, dict):
        return tuple(discovered.get(_ENTRY_POINT_GROUP, ()))
    return tuple(
        item
        for item in discovered
        if getattr(item, "group", None) == _ENTRY_POINT_GROUP
    )


def _entry_points_named(name: str) -> Iterable[Any]:
    return (item for item in _entry_points() if item.name == name)


def available_adapters() -> Tuple[str, ...]:
    names = set(_REGISTERED)
    names.update(item.name for item in _entry_points())
    return tuple(sorted(names))


def _build(factory: Any, kwargs: Dict[str, Any]) -> Any:
    builder = getattr(factory, "build", None)
    if callable(builder):
        return builder(**kwargs)
    if callable(factory):
        return factory(**kwargs)
    raise AdapterUnsupportedError(
        "adapter entry point must be callable or expose build(**kwargs)"
    )


def get_adapter(name: str, **kwargs: Any) -> Any:
    validated = _validate_name(name)
    if validated in _REGISTERED:
        factory = _REGISTERED[validated]
    else:
        matches = tuple(_entry_points_named(validated))
        if not matches:
            available = ", ".join(available_adapters()) or "(none)"
            raise AdapterUnsupportedError(
                f"unknown framework adapter {validated!r}; "
                f"available adapters: {available}"
            )
        try:
            factory = matches[0].load()
        except Exception as exc:
            raise AdapterUnsupportedError(
                f"could not load framework adapter {validated!r}: {exc}"
            ) from exc

    adapter = _build(factory, dict(kwargs))
    if adapter is None:
        raise AdapterUnsupportedError(
            f"framework adapter factory {validated!r} returned None"
        )
    return adapter

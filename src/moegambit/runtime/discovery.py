"""Engine adapter discovery through Python entry points."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points
from typing import Iterable, Mapping, Sequence

from moegambit.interfaces import EngineAdapter


ENTRY_POINT_GROUP = "moegambit.engine_adapters"
_BUILTINS: Mapping[str, str] = {
    "megatron": "moegambit.adapters.megatron.engine:MegatronEngineAdapter",
    "deepspeed": (
        "moegambit.adapters.deepspeed.engine:DeepSpeedEngineAdapter"
    ),
}


def _load_target(target: str) -> type:
    module_name, object_name = target.split(":", 1)
    return getattr(import_module(module_name), object_name)


def _adapter_entry_points() -> Iterable:
    discovered = entry_points()
    if hasattr(discovered, "select"):
        return discovered.select(group=ENTRY_POINT_GROUP)
    return discovered.get(ENTRY_POINT_GROUP, ())


def discover_adapters() -> dict[str, EngineAdapter]:
    adapters: dict[str, EngineAdapter] = {}
    for name, target in _BUILTINS.items():
        try:
            adapters[name] = _load_target(target)()
        except ImportError:
            # Framework plugins are optional. Explicit selection reports the
            # remaining available adapters in ``load_adapter``.
            continue
    for item in _adapter_entry_points():
        adapters[item.name] = item.load()()
    return adapters


def load_adapter(name: str, command: Sequence[str] = ()) -> EngineAdapter:
    adapters = discover_adapters()
    if name != "auto":
        try:
            return adapters[name]
        except KeyError as exc:
            available = ", ".join(sorted(adapters)) or "none"
            raise LookupError(
                f"unknown engine adapter {name!r}; available: {available}"
            ) from exc
    matches = [adapter for adapter in adapters.values() if adapter.probe(command)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        available = ", ".join(sorted(adapters)) or "none"
        raise LookupError(
            "could not infer an engine adapter from the command; "
            f"select --adapter explicitly ({available})"
        )
    names = ", ".join(sorted(adapter.name for adapter in matches))
    raise LookupError(f"multiple engine adapters matched the command: {names}")

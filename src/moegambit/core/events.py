"""Small synchronous event bus used by the recovery controller."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, DefaultDict, Mapping


@dataclass(frozen=True)
class RuntimeEvent:
    name: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


EventHandler = Callable[[RuntimeEvent], None]


class EventBus:
    def __init__(self) -> None:
        self._handlers: DefaultDict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(self, name: str, handler: EventHandler) -> Callable[[], None]:
        self._handlers[name].append(handler)

        def unsubscribe() -> None:
            self._handlers[name].remove(handler)

        return unsubscribe

    def emit(self, event: RuntimeEvent) -> None:
        for handler in tuple(self._handlers.get(event.name, ())):
            handler(event)
        for handler in tuple(self._handlers.get("*", ())):
            handler(event)

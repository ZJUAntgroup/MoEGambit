"""List installed engine adapter plugins."""

from __future__ import annotations

import json
from typing import Sequence

from moegambit.runtime.discovery import discover_adapters


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    payload = {
        name: {
            "hot_swap": adapter.capabilities.hot_swap,
            "zero2": adapter.capabilities.zero2,
            "legacy_watcher": adapter.capabilities.legacy_watcher,
            "watcher_required": adapter.capabilities.watcher_required,
        }
        for name, adapter in sorted(discover_adapters().items())
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

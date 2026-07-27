"""Generic JSON-line watcher service.

Engine adapters can replace this service with an engine-specific watcher
command while preserving the same CLI and feature configuration.
"""

from __future__ import annotations

import socketserver
from dataclasses import dataclass, field
from typing import Callable

from moegambit.runtime.protocol import WireMessage


MessageHandler = Callable[[WireMessage], WireMessage | None]


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline()
        if not raw:
            return
        response = self.server.runtime_handler(WireMessage.decode(raw))
        if response is not None:
            self.wfile.write(response.encode())


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, runtime_handler: MessageHandler):
        self.runtime_handler = runtime_handler
        super().__init__(address, _RequestHandler)


@dataclass
class WatcherRuntime:
    host: str
    port: int
    handler: MessageHandler
    _server: _Server | None = field(default=None, init=False, repr=False)

    def serve_forever(self) -> None:
        self._server = _Server((self.host, self.port), self.handler)
        try:
            self._server.serve_forever()
        finally:
            self._server.server_close()

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()

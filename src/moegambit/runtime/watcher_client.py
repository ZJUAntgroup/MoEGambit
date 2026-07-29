"""Reusable watcher client independent of a training engine."""

from __future__ import annotations

import socket
from dataclasses import dataclass

from moegambit.runtime.protocol import WireMessage


@dataclass(frozen=True)
class WatcherEndpoint:
    host: str
    port: int
    timeout: float = 10.0


class WatcherClient:
    def __init__(self, endpoint: WatcherEndpoint) -> None:
        self.endpoint = endpoint

    def request(self, message: WireMessage) -> WireMessage:
        with socket.create_connection(
            (self.endpoint.host, self.endpoint.port),
            timeout=self.endpoint.timeout,
        ) as connection:
            connection.sendall(message.encode())
            stream = connection.makefile("rb")
            response = stream.readline()
        if not response:
            raise ConnectionError("watcher closed without a response")
        return WireMessage.decode(response)

    def notify(self, message: WireMessage) -> None:
        with socket.create_connection(
            (self.endpoint.host, self.endpoint.port),
            timeout=self.endpoint.timeout,
        ) as connection:
            connection.sendall(message.encode())

"""Async WebSocket client with sender/receiver role handshake."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from common import msgpack_numpy
import websockets
import websockets.exceptions

logger = logging.getLogger(__name__)


class RoleWebsocketClient:
    """Minimal async client: connect, announce role, send/recv msgpack frames."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        *,
        retry_interval: float = 5.0,
    ) -> None:
        self.server_url = f"ws://{host}:{port}"
        self._retry_interval = retry_interval
        self._packer = msgpack_numpy.Packer()
        self._ws = None
        self._connected = False
        self._role: str | None = None

    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    async def connect(self, role: str, *, max_retries: int | None = None) -> None:
        if role not in ("sender", "receiver"):
            raise ValueError(f"role must be 'sender' or 'receiver', got {role!r}")

        retries = 0
        while True:
            try:
                logger.info("Connecting to %s as %s...", self.server_url, role)
                self._ws = await websockets.connect(
                    self.server_url,
                    compression=None,
                    max_size=None,
                )
                self._connected = True
                self._role = role
                await self.send({"role": role})
                logger.info("Connected and announced role=%s", role)
                return
            except (ConnectionRefusedError, OSError, websockets.exceptions.WebSocketException) as exc:
                retries += 1
                if max_retries is not None and retries >= max_retries:
                    raise ConnectionError(f"failed to connect to {self.server_url} after {retries} tries") from exc
                logger.warning("Connect failed (%s); retry in %.1fs", exc, self._retry_interval)
                await asyncio.sleep(self._retry_interval)

    async def send(self, data: dict[str, Any]) -> None:
        if not self._ws:
            raise ConnectionError("not connected")
        await self._ws.send(self._packer.pack(data))

    async def receive(self) -> dict[str, Any]:
        if not self._ws:
            raise ConnectionError("not connected")
        raw = await self._ws.recv()
        if isinstance(raw, str):
            raise RuntimeError(f"server error:\n{raw}")
        result = msgpack_numpy.unpackb(raw)
        if not isinstance(result, dict):
            raise TypeError(f"expected dict frame, got {type(result)!r}")
        return result

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass
            self._ws = None
            self._connected = False
            self._role = None

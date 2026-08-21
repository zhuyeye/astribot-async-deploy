"""Continuously receive action chunks as role=receiver."""

from __future__ import annotations

import asyncio
import logging
import queue
from dataclasses import dataclass
from typing import Any

import numpy as np

from common.trace_log import get_tracer
from protocol.base import ProtocolAdapter
from transport.ws_client import RoleWebsocketClient

logger = logging.getLogger(__name__)


@dataclass
class ActionChunkMsg:
    actions_sdk: np.ndarray  # (T, 25) SDK units, gripper [0,100]
    obs_timestamp: float
    raw: dict[str, Any]


class ActionReceiver:
    """Recv broadcast chunks and push onto a thread-safe queue."""

    def __init__(
        self,
        client: RoleWebsocketClient,
        adapter: ProtocolAdapter,
        out_queue: queue.Queue[ActionChunkMsg],
    ) -> None:
        self._client = client
        self._adapter = adapter
        self._queue = out_queue
        self._running = False

    async def run(self) -> None:
        self._running = True
        if not self._client.is_connected():
            await self._client.connect("receiver")

        while self._running:
            try:
                frame = await self._client.receive()
                raw_actions = np.asarray(frame.get("actions", []), dtype=np.float32)
                actions = self._adapter.decode_actions(frame)
                if actions.size == 0:
                    logger.warning("Empty action chunk received; skipping")
                    continue
                ts = float(frame.get("obs_timestamp", 0.0))
                tracer = get_tracer()
                if tracer is not None:
                    tracer.log_recv_actions(
                        raw_actions=raw_actions,
                        decoded_sdk=actions,
                        obs_timestamp=ts,
                    )
                msg = ActionChunkMsg(actions_sdk=actions, obs_timestamp=ts, raw=frame)
                try:
                    self._queue.put_nowait(msg)
                except queue.Full:
                    # Drop oldest to keep latency low.
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._queue.put_nowait(msg)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("ActionReceiver recv failed; reconnecting")
                try:
                    await self._client.close()
                    await self._client.connect("receiver")
                except Exception:  # noqa: BLE001
                    logger.exception("ActionReceiver reconnect failed")
                    await asyncio.sleep(1.0)

    def stop(self) -> None:
        self._running = False

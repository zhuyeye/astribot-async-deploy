"""Continuously send observations as role=sender."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

import numpy as np

from common.trace_log import get_tracer
from protocol.base import InternalObs, ProtocolAdapter
from transport.ws_client import RoleWebsocketClient

logger = logging.getLogger(__name__)

ObsFactory = Callable[[], InternalObs | None]


class ObsSender:
    """Poll sensors, adapt to wire protocol, stream to async policy server."""

    def __init__(
        self,
        client: RoleWebsocketClient,
        adapter: ProtocolAdapter,
        obs_factory: ObsFactory,
        *,
        send_hz: float = 30.0,
        prompt: str = "",
    ) -> None:
        self._client = client
        self._adapter = adapter
        self._obs_factory = obs_factory
        self._send_period = 1.0 / max(send_hz, 1e-3)
        self._prompt = prompt
        self._running = False
        self._last_image_stamp = (0.0, 0.0, 0.0)

    async def run(self) -> None:
        self._running = True
        if not self._client.is_connected():
            await self._client.connect("sender")

        while self._running:
            t0 = time.perf_counter()
            try:
                obs = self._obs_factory()
                if obs is not None:
                    stamps = (
                        float(obs.head_stamp),
                        float(obs.left_stamp),
                        float(obs.right_stamp),
                    )
                    if stamps != self._last_image_stamp and all(s > 0 for s in stamps):
                        if self._prompt and not obs.prompt:
                            obs.prompt = self._prompt
                        wire = self._adapter.encode_obs(obs)
                        tracer = get_tracer()
                        if tracer is not None:
                            tracer.log_send_obs(
                                internal_state_25=obs.state_25_sdk,
                                wire=wire,
                                obs_timestamp=float(obs.obs_timestamp),
                                prompt=obs.prompt,
                            )
                        await self._client.send(wire)
                        self._last_image_stamp = stamps
            except Exception:  # noqa: BLE001
                logger.exception("ObsSender send failed; reconnecting")
                try:
                    await self._client.close()
                    await self._client.connect("sender")
                except Exception:  # noqa: BLE001
                    logger.exception("ObsSender reconnect failed")
                    await asyncio.sleep(1.0)

            elapsed = time.perf_counter() - t0
            await asyncio.sleep(max(0.0, self._send_period - elapsed))

    async def send_reset(self) -> None:
        if self._client.is_connected():
            await self._client.send({"reset": 1})

    def stop(self) -> None:
        self._running = False


def fake_obs_factory(prompt: str = "test") -> ObsFactory:
    """Synthetic observations for connectivity tests."""

    def _make() -> InternalObs:
        now = time.perf_counter()
        return InternalObs(
            head_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
            left_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
            right_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
            state_25_sdk=np.zeros(25, dtype=np.float32),
            obs_timestamp=now,
            head_stamp=now,
            left_stamp=now,
            right_stamp=now,
            prompt=prompt,
        )

    return _make

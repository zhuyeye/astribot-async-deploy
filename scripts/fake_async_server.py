#!/usr/bin/env python3
"""Minimal async dual-channel policy server for local e2e tests.

Usage:
  python scripts/fake_async_server.py --port 8000 --protocol pi05
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from common import msgpack_numpy
import websockets.asyncio.server as ws_server

logger = logging.getLogger(__name__)


class FakeAsyncServer:
    def __init__(self, host: str, port: int, protocol: str, horizon: int = 10) -> None:
        self._host = host
        self._port = port
        self._protocol = protocol
        self._horizon = horizon
        self._packer = msgpack_numpy.Packer()
        self._lock = asyncio.Lock()
        self._latest_obs: dict | None = None
        self._obs_counter = 0
        self._receivers: set = set()

    async def run(self) -> None:
        async with ws_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            logger.info("Fake async server on ws://%s:%d protocol=%s", self._host, self._port, self._protocol)
            broadcast = asyncio.create_task(self._infer_loop())
            try:
                await server.serve_forever()
            finally:
                broadcast.cancel()

    async def _handler(self, websocket: ws_server.ServerConnection) -> None:
        raw = await websocket.recv()
        msg = msgpack_numpy.unpackb(raw)
        role = msg.get("role") if isinstance(msg, dict) else None
        logger.info("Client %s role=%s", websocket.remote_address, role)
        if role == "sender":
            await self._handle_sender(websocket)
        elif role == "receiver":
            self._receivers.add(websocket)
            try:
                while True:
                    await asyncio.sleep(0.5)
            finally:
                self._receivers.discard(websocket)
        else:
            logger.warning("Unknown role: %r", role)

    async def _handle_sender(self, websocket: ws_server.ServerConnection) -> None:
        while True:
            raw = await websocket.recv()
            msg = msgpack_numpy.unpackb(raw)
            if not isinstance(msg, dict):
                continue
            if msg.get("reset") is not None:
                async with self._lock:
                    self._latest_obs = None
                continue
            async with self._lock:
                self._latest_obs = msg
                self._obs_counter += 1

    async def _infer_loop(self) -> None:
        last = 0
        while True:
            await asyncio.sleep(0.05)
            async with self._lock:
                if self._obs_counter == last or self._latest_obs is None:
                    continue
                last = self._obs_counter
                obs = self._latest_obs

            actions = self._fake_actions(obs)
            ts = float(obs.get("obs_timestamp", time.perf_counter()))
            payload = {
                "actions": actions,
                "obs_timestamp": ts,
            }
            packed = self._packer.pack(payload)
            dead = []
            for client in list(self._receivers):
                try:
                    await client.send(packed)
                except Exception:  # noqa: BLE001
                    dead.append(client)
            for c in dead:
                self._receivers.discard(c)
            logger.info("Broadcast actions %s to %d receivers", actions.shape, len(self._receivers))

    def _fake_actions(self, obs: dict) -> np.ndarray:
        if self._protocol == "pi05":
            state = np.asarray(obs.get("observation/state", np.zeros(25)), dtype=np.float32).reshape(-1)
            if state.shape[0] < 25:
                state = np.pad(state, (0, 25 - state.shape[0]))
            base = state[:25].copy()
        else:
            state = np.asarray(obs.get("state", np.zeros(34)), dtype=np.float64).reshape(-1)
            base = np.zeros(34, dtype=np.float64)
            n = min(34, state.shape[0])
            base[:n] = state[:n]
        horizon = self._horizon
        dim = 25 if self._protocol == "pi05" else 34
        out = np.zeros((horizon, dim), dtype=np.float64)
        for i in range(horizon):
            out[i, : min(dim, base.shape[0])] = base[: min(dim, base.shape[0])]
            # Tiny drift so chunks are visible in logs.
            out[i, 7] += 0.01 * i
        return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--protocol", choices=("pi05", "astribot34"), default="pi05")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("-v", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.v else logging.INFO)
    asyncio.run(FakeAsyncServer(args.host, args.port, args.protocol, args.horizon).run())


if __name__ == "__main__":
    main()

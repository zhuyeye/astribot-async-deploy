"""Multi-camera reader with BGR -> RGB conversion."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from protocol.base import PI_CAMERA_KEYS

logger = logging.getLogger(__name__)

SDK_TO_PI_KEY = {sdk_name: pi_key for pi_key, sdk_name in PI_CAMERA_KEYS.items()}
PI_KEYS = list(PI_CAMERA_KEYS.keys())


def pi_key_from_topic(topic_name: str) -> str | None:
    parts = topic_name.strip("/").split("/")
    if len(parts) < 2 or parts[0] != "astribot_camera":
        return None
    return SDK_TO_PI_KEY.get(parts[1])


class CameraReader:
    def __init__(self, astribot: Any, *, activate_timeout_s: float = 15.0) -> None:
        self._astribot = astribot
        self._frames: dict[str, np.ndarray | None] = {key: None for key in PI_KEYS}
        self._stamps: dict[str, float] = {key: 0.0 for key in PI_KEYS}
        self._lock = threading.Lock()
        self._subscribers: list[Any] = []
        self._activate_cameras(activate_timeout_s)

    def _activate_cameras(self, timeout_s: float) -> None:
        self._astribot.activate_camera()
        sdk_names = list(PI_CAMERA_KEYS.values())
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status = self._astribot.get_cameras_info()
            if all(status.get(name, {}).get("activate") for name in sdk_names):
                break
            time.sleep(1.0)
        else:
            status = self._astribot.get_cameras_info()
            inactive = [name for name in sdk_names if not status.get(name, {}).get("activate")]
            raise RuntimeError(f"Failed to activate cameras: {inactive}")

        for pi_key, sdk_name in PI_CAMERA_KEYS.items():
            sub = self._astribot.register_image_callback(
                sdk_name,
                "color",
                self._shared_image_callback,
                need_decode=True,
            )
            self._subscribers.append(sub)
            logger.info("Registered camera callback: %s -> %s", sdk_name, pi_key)

        self._wait_for_first_frames(timeout_s=min(timeout_s, 15.0))

    def _shared_image_callback(
        self,
        topic_name: str,
        msg: Any,
        _width: int,
        _height: int,
        array: np.ndarray,
    ) -> None:
        pi_key = pi_key_from_topic(topic_name)
        if pi_key is None:
            return
        if msg.format.lower() != "jpeg":
            return
        if array.ndim != 3 or array.shape[2] != 3:
            return
        rgb = array[..., ::-1].copy()
        with self._lock:
            self._frames[pi_key] = rgb
            self._stamps[pi_key] = time.perf_counter()

    def _wait_for_first_frames(self, timeout_s: float = 15.0) -> None:
        logger.info("Waiting up to %.0fs for first frames...", timeout_s)
        self.read_rgb(wait_timeout_s=timeout_s)
        logger.info("All cameras have first frames")

    def read_rgb(self, *, wait_timeout_s: float = 10.0) -> dict[str, np.ndarray]:
        deadline = time.monotonic() + wait_timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if all(self._frames[key] is not None for key in PI_KEYS):
                    return {key: self._frames[key].copy() for key in PI_KEYS}  # type: ignore[union-attr]
            time.sleep(0.01)
        missing = [key for key in PI_KEYS if self._frames[key] is None]
        raise TimeoutError(f"Timed out waiting for camera frames: {missing}")

    def read_rgb_with_stamps(self, *, wait_timeout_s: float = 0.0) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        if wait_timeout_s > 0:
            frames = self.read_rgb(wait_timeout_s=wait_timeout_s)
            with self._lock:
                stamps = dict(self._stamps)
            return frames, stamps
        with self._lock:
            if not all(self._frames[key] is not None for key in PI_KEYS):
                raise TimeoutError("cameras not ready")
            frames = {key: self._frames[key].copy() for key in PI_KEYS}  # type: ignore[union-attr]
            stamps = dict(self._stamps)
        return frames, stamps

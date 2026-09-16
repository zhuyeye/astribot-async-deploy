"""High-rate control loop: sample traj -> safety -> SDK."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import queue
import threading
import time
from typing import Any

import numpy as np

from common.trace_log import get_tracer
from control.chunk_traj_bridge import ChunkTrajBridge, ChunkTrajConfig
from protocol.base import freeze_parts
from robot.safety import SafetyConfig, safety_clip
from transport.action_receiver import ActionChunkMsg

logger = logging.getLogger(__name__)


@dataclass
class ExecutorConfig:
    ctrl_hz: float = 250.0
    chunk_hz: float = 30.0
    blend_s: float = 0.12
    max_step_delta_rad: float = 0.15
    freeze_chassis: bool = True
    freeze_torso: bool = False
    freeze_head: bool = False
    dry_run: bool = False
    use_obs_timestamp: bool = True
    playback_lead_s: float = 0.05
    blend_grippers: bool = False
    gripper_binary: bool = True
    gripper_close_enter: float = 70.0
    gripper_open_enter: float = 40.0
    gripper_close_confirm: int = 2
    gripper_open_confirm: int = 5
    gripper_min_close_hold_s: float = 1.2
    gripper_min_open_hold_s: float = 0.35
    gripper_min_close_hold_frames: int = 0
    gripper_deadband: float = 0.0
    auto_episode_reset_open_s: float = 0.6


class Executor:
    """Consume action-chunk queue and stream joint commands."""

    def __init__(
        self,
        joint_io: Any,
        chunk_queue: queue.Queue[ActionChunkMsg],
        config: ExecutorConfig | None = None,
    ) -> None:
        self._joint_io = joint_io
        self._queue = chunk_queue
        self._cfg = config or ExecutorConfig()
        self._bridge = ChunkTrajBridge(
            ChunkTrajConfig(
                chunk_hz=self._cfg.chunk_hz,
                blend_s=self._cfg.blend_s,
                use_obs_timestamp=self._cfg.use_obs_timestamp,
                playback_lead_s=self._cfg.playback_lead_s,
                blend_grippers=self._cfg.blend_grippers,
                gripper_binary=self._cfg.gripper_binary,
                gripper_close_enter=self._cfg.gripper_close_enter,
                gripper_open_enter=self._cfg.gripper_open_enter,
                gripper_close_confirm=self._cfg.gripper_close_confirm,
                gripper_open_confirm=self._cfg.gripper_open_confirm,
                gripper_min_close_hold_s=self._cfg.gripper_min_close_hold_s,
                gripper_min_open_hold_s=self._cfg.gripper_min_open_hold_s,
                gripper_min_close_hold_frames=self._cfg.gripper_min_close_hold_frames,
                gripper_deadband=self._cfg.gripper_deadband,
            )
        )
        self._safety = SafetyConfig(max_step_delta_rad=self._cfg.max_step_delta_rad)
        self._running = False
        self._thread: threading.Thread | None = None
        self._pause_event = threading.Event()
        self._chunks_seen = 0
        self._last_cmd: np.ndarray | None = None
        self._gripper_seeded = False
        self._episode_saw_close = False
        self._episode_open_since: float | None = None
        self._episode_complete = threading.Event()

    @property
    def chunks_seen(self) -> int:
        return self._chunks_seen

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="async-executor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self.resume()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def pause(self) -> None:
        self._pause_event.set()

    def resume(self) -> None:
        self._pause_event.clear()

    def reset(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._bridge.reset()
        self._last_cmd = None
        self._gripper_seeded = False
        self._episode_saw_close = False
        self._episode_open_since = None
        self._episode_complete.clear()

    def pop_episode_complete(self) -> bool:
        if not self._episode_complete.is_set():
            return False
        self._episode_complete.clear()
        return True

    def _update_episode_completion(self, cmd: np.ndarray, now: float) -> None:
        if self._episode_complete.is_set():
            return

        grips = np.asarray(cmd, dtype=np.float32)[[14, 22]]
        any_closed = bool(np.any(grips >= self._cfg.gripper_close_enter))
        all_open = bool(np.all(grips <= self._cfg.gripper_open_enter))

        if any_closed:
            self._episode_saw_close = True
            self._episode_open_since = None
            return

        if not self._episode_saw_close or not all_open:
            self._episode_open_since = None
            return

        if self._episode_open_since is None:
            self._episode_open_since = now
            return

        if now - self._episode_open_since >= self._cfg.auto_episode_reset_open_s:
            self._episode_complete.set()

    def _ensure_gripper_seed(self) -> None:
        if self._gripper_seeded:
            return
        try:
            q = self._joint_io.read_joint_state_25()
            self._bridge.seed_gripper_from_state(q)
            self._gripper_seeded = True
            logger.info(
                "Seeded gripper post from state L=%.1f R=%.1f binary=%s",
                float(q[14]),
                float(q[22]),
                self._cfg.gripper_binary,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to seed gripper post from joint state")

    def _drain_chunks(self) -> None:
        while True:
            try:
                msg = self._queue.get_nowait()
            except queue.Empty:
                break
            self._ensure_gripper_seed()
            self._bridge.ingest(
                msg.actions_sdk,
                obs_timestamp=msg.obs_timestamp,
                now=time.perf_counter(),
            )
            self._chunks_seen += 1
            logger.info(
                "Ingested chunk #%d shape=%s obs_ts=%.4f",
                self._chunks_seen,
                msg.actions_sdk.shape,
                msg.obs_timestamp,
            )

    def _loop(self) -> None:
        period = 1.0 / max(self._cfg.ctrl_hz, 1e-3)
        next_tick = time.monotonic()
        freeze_ref: np.ndarray | None = None

        while self._running:
            if self._pause_event.is_set():
                next_tick = time.monotonic()
                time.sleep(period)
                continue

            self._drain_chunks()

            now = time.perf_counter()
            cmd = self._bridge.sample(now)
            if cmd is not None:
                sampled = cmd.copy()
                if freeze_ref is None:
                    try:
                        freeze_ref = self._joint_io.read_joint_state_25()
                    except Exception:  # noqa: BLE001
                        freeze_ref = cmd.copy()

                prev = self._last_cmd if self._last_cmd is not None else freeze_ref
                cmd = freeze_parts(
                    cmd,
                    freeze_ref,
                    freeze_chassis=self._cfg.freeze_chassis,
                    freeze_torso=self._cfg.freeze_torso,
                    freeze_head=self._cfg.freeze_head,
                )
                after_freeze = cmd.copy()
                safe = safety_clip(cmd, prev, self._safety)
                tracer = get_tracer()
                if tracer is not None:
                    tracer.log_exec(
                        sampled=sampled,
                        after_freeze=after_freeze,
                        after_safety=safe,
                        dry_run=self._cfg.dry_run,
                    )
                if self._cfg.dry_run:
                    if self._chunks_seen > 0 and (
                        self._chunks_seen == 1 or int(time.time() * self._cfg.ctrl_hz) % 50 == 0
                    ):
                        logger.debug(
                            "dry-run cmd[0:3]=%s grip=(%.1f,%.1f)",
                            safe[0:3],
                            safe[14],
                            safe[22],
                        )
                else:
                    self._joint_io.send_joint_position_command(safe, include_grippers=True)
                self._last_cmd = safe.copy()
                self._update_episode_completion(safe, now)

            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

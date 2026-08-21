"""Lightweight RTG: map streaming action chunks onto a time-parameterized trajectory."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import numpy as np

from common.trace_log import get_tracer
from control.gripper_post import GripperPostConfig, GripperPostProcessor
from protocol.base import GRIPPER_INDICES, JOINT_DIM


@dataclass
class ChunkTrajConfig:
    chunk_hz: float = 30.0
    blend_s: float = 0.12
    # If True, timeline starts at obs_timestamp; else at wall clock when chunk arrives.
    use_obs_timestamp: bool = True
    # When obs_timestamp is stale, keep replay slightly in the future so the
    # executor does not jump straight to the chunk tail.
    playback_lead_s: float = 0.05
    # Grippers: skip cross-chunk blend by default; VLA should hold a stable
    # open/close band, then binary sticky discretizes to {0,100}.
    blend_grippers: bool = False
    gripper_binary: bool = True
    # Hysteresis on SDK [0,100]: close >=60 (wire 0.6), open <=40 (wire 0.4).
    gripper_close_enter: float = 60.0
    gripper_open_enter: float = 40.0
    gripper_close_confirm: int = 1
    gripper_open_confirm: int = 2
    # One-shot hold after open→close before open is allowed.
    gripper_min_close_hold_s: float = 0.8
    gripper_min_open_hold_s: float = 0.35
    gripper_min_close_hold_frames: int = 0
    gripper_deadband: float = 0.0


class ChunkTrajBridge:
    """Maintain an active absolute-joint trajectory with cross-chunk blending.

    Pipeline per chunk:
      1) linear blend ``blend_s`` keyframes at the ingest seam (``k_now`` on the
         obs_timestamp timeline when enabled, else key 0) from last sample
         (arms/joints; grippers only if ``blend_grippers``)
      2) discretize grippers (binary sticky) on the (mostly raw) gripper keys
      3) sample: joints linear; grippers ZOH after discretization
    """

    def __init__(self, config: ChunkTrajConfig | None = None) -> None:
        self._cfg = config or ChunkTrajConfig()
        self._lock = threading.Lock()
        self._times: np.ndarray | None = None  # (T,) absolute time
        self._keys: np.ndarray | None = None  # (T, 25) SDK units (post-discrete)
        self._last_sample: np.ndarray | None = None
        self._gripper_post = GripperPostProcessor(
            GripperPostConfig(
                binary=self._cfg.gripper_binary,
                close_enter=self._cfg.gripper_close_enter,
                open_enter=self._cfg.gripper_open_enter,
                close_confirm=self._cfg.gripper_close_confirm,
                open_confirm=self._cfg.gripper_open_confirm,
                min_close_hold_s=self._cfg.gripper_min_close_hold_s,
                min_open_hold_s=self._cfg.gripper_min_open_hold_s,
                min_close_hold_frames=self._cfg.gripper_min_close_hold_frames,
                chunk_hz=self._cfg.chunk_hz,
                deadband=self._cfg.gripper_deadband,
            )
        )
        self._gripper_seeded = False

    def reset(self) -> None:
        with self._lock:
            self._times = None
            self._keys = None
            self._last_sample = None
            self._gripper_post.reset()
            self._gripper_seeded = False

    def seed_gripper_from_state(self, state_25: np.ndarray) -> None:
        with self._lock:
            self._gripper_post.seed_from_state(state_25)
            self._gripper_seeded = True

    def _blend_start_index(self, *, t0: float, now: float, horizon: int) -> int:
        """Key index where playback resumes after ingest (obs timeline seam)."""
        if not self._cfg.use_obs_timestamp:
            return 0
        k = int(np.floor((now - t0) * float(self._cfg.chunk_hz)))
        return int(np.clip(k, 0, max(0, horizon - 1)))

    def ingest(
        self,
        actions_sdk: np.ndarray,
        *,
        obs_timestamp: float,
        now: float | None = None,
    ) -> None:
        actions = np.asarray(actions_sdk, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != JOINT_DIM:
            raise ValueError(f"expected actions (T, {JOINT_DIM}), got {actions.shape}")
        if actions.shape[0] == 0:
            return

        now = time.perf_counter() if now is None else float(now)
        if self._cfg.use_obs_timestamp and np.isfinite(obs_timestamp) and obs_timestamp > 0:
            t0 = float(obs_timestamp)
        else:
            t0 = now

        horizon = actions.shape[0]
        if self._cfg.use_obs_timestamp:
            lead_s = max(0.0, float(self._cfg.playback_lead_s))
            chunk_tail_t = t0 + float(max(0, horizon - 1)) / float(self._cfg.chunk_hz)
            if now >= chunk_tail_t - lead_s:
                t0 = now + lead_s

        times = t0 + np.arange(horizon, dtype=np.float64) / float(self._cfg.chunk_hz)

        with self._lock:
            prev_sample = self._last_sample.copy() if self._last_sample is not None else None
            before = actions.copy()
            keys = actions.copy()
            blend_n = 0
            blend_k0 = 0
            if prev_sample is not None and self._cfg.blend_s > 0:
                blend_n = max(1, int(round(self._cfg.blend_s * self._cfg.chunk_hz)))
                blend_k0 = self._blend_start_index(t0=t0, now=now, horizon=horizon)
                blend_end = min(horizon, blend_k0 + blend_n)
                blend_count = blend_end - blend_k0
                if blend_count > 0:
                    blend_n = blend_count
                    for j, i in enumerate(range(blend_k0, blend_end)):
                        alpha = float(j + 1) / float(blend_count)
                        for dim in range(JOINT_DIM):
                            if dim in GRIPPER_INDICES and not self._cfg.blend_grippers:
                                continue
                            keys[i, dim] = (1.0 - alpha) * float(prev_sample[dim]) + alpha * float(
                                keys[i, dim]
                            )
                else:
                    blend_n = 0

            after_blend = keys.copy()
            # Discretize / smooth grippers AFTER blend (timeline = traj times).
            keys = self._gripper_post.process_keyframes(
                keys, t0=float(t0), hz=float(self._cfg.chunk_hz)
            )

            self._times = times
            self._keys = keys

        tracer = get_tracer()
        if tracer is not None:
            tracer.log_ingest(
                before_keys=before,
                after_keys=keys,
                obs_timestamp=float(obs_timestamp),
                t0=float(t0),
                blend_n=blend_n,
                blend_k0=blend_k0,
                k_now=blend_k0,
                ingest_now=float(now),
                after_blend_keys=after_blend,
            )

    def has_trajectory(self) -> bool:
        with self._lock:
            return self._keys is not None and self._times is not None

    def sample(self, t: float | None = None) -> np.ndarray | None:
        t = time.perf_counter() if t is None else float(t)
        with self._lock:
            if self._keys is None or self._times is None:
                return None
            times = self._times
            keys = self._keys

            if t <= times[0]:
                out = keys[0].copy()
            elif t >= times[-1]:
                out = keys[-1].copy()
            else:
                idx = int(np.searchsorted(times, t, side="right") - 1)
                idx = int(np.clip(idx, 0, len(times) - 2))
                t0, t1 = float(times[idx]), float(times[idx + 1])
                u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
                u = float(np.clip(u, 0.0, 1.0))
                out = keys[idx].copy()
                for dim in range(JOINT_DIM):
                    # After binary post, grippers are {0,100}: ZOH between keys.
                    # Continuous mode: linear interpolate grippers too.
                    if dim in GRIPPER_INDICES and self._cfg.gripper_binary:
                        out[dim] = keys[idx, dim]
                    else:
                        out[dim] = (1.0 - u) * float(keys[idx, dim]) + u * float(keys[idx + 1, dim])

            self._last_sample = out.copy()
            return out

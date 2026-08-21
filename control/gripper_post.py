"""Stateful gripper post-processing: hysteresis binarization (open/close).

Applied AFTER cross-chunk blend on continuous SDK [0,100] keyframes.
Ported from openpi-client/deploy_s1/gripper_post.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from protocol.base import LEFT_GRIPPER_IDX, RIGHT_GRIPPER_IDX

GRIPPER_SIDES = (
    ("left", LEFT_GRIPPER_IDX),
    ("right", RIGHT_GRIPPER_IDX),
)


@dataclass
class GripperPostConfig:
    """Binary gripper hysteresis on SDK [0,100] (== wire [0,1] * 100).

    Default band matches classic Schmitt trigger:
      open  → close when value >= close_enter (60 / 0.6)
      close → open  when value <= open_enter  (40 / 0.4)
      otherwise keep current state

    After open→close, open is blocked for ``min_close_hold_s`` on the keyframe
    timeline (one-shot on that edge; sustained close does NOT refresh hold).
    """

    lo: float = 0.0
    hi: float = 100.0
    binary: bool = True
    open_value: float = 0.0
    close_value: float = 100.0
    # Wire 0.6 / 0.4 → SDK 60 / 40
    close_enter: float = 60.0
    open_enter: float = 40.0
    close_confirm: int = 1
    open_confirm: int = 2
    # One-shot hold after open→close before open is allowed.
    min_close_hold_s: float = 0.8
    # One-shot hold after close→open before close is allowed.
    min_open_hold_s: float = 0.35
    # Optional extra hold counted in processed keyframes (usually leave 0).
    min_close_hold_frames: int = 0
    chunk_hz: float = 30.0
    deadband: float = 0.0
    high_level: float = 80.0
    dip_depth: float = 15.0
    dip_suppress_frames: int = 1
    max_step_up: float = 40.0
    max_step_down: float = 100.0


@dataclass
class _SideState:
    last_raw: float | None = None
    closed: bool | None = None
    close_streak: int = 0
    open_streak: int = 0
    close_hold_remaining: int = 0
    open_hold_remaining: int = 0
    # Timeline time of last close intent (same clock as process_keyframes t0).
    closed_since: float | None = None
    opened_since: float | None = None
    last_out: float | None = None
    dip_count: int = 0


class GripperPostProcessor:
    """Process gripper dims of packed 25-DoF keyframe chunks."""

    def __init__(self, config: GripperPostConfig | None = None) -> None:
        self.config = config or GripperPostConfig()
        self._sides = {
            LEFT_GRIPPER_IDX: _SideState(),
            RIGHT_GRIPPER_IDX: _SideState(),
        }

    def reset(self) -> None:
        self._sides = {
            LEFT_GRIPPER_IDX: _SideState(),
            RIGHT_GRIPPER_IDX: _SideState(),
        }

    def seed_from_state(self, state_25: np.ndarray) -> None:
        state_25 = np.asarray(state_25, dtype=np.float32).reshape(-1)
        cfg = self.config
        now = time.monotonic()
        for idx, st in self._sides.items():
            if idx >= state_25.shape[0]:
                continue
            v = float(np.clip(state_25[idx], cfg.lo, cfg.hi))
            st.last_raw = v
            st.last_out = v
            st.closed = v >= cfg.close_enter
            st.close_streak = 0
            st.open_streak = 0
            st.close_hold_remaining = (
                max(0, int(cfg.min_close_hold_frames)) if st.closed else 0
            )
            st.open_hold_remaining = 0
            # Already closed at seed: do not force another full hold.
            st.closed_since = (
                now - float(cfg.min_close_hold_s) if st.closed else None
            )
            st.opened_since = None
            st.dip_count = 0

    def process_keyframes(
        self,
        keyframe_cmds: np.ndarray,
        *,
        t0: float | None = None,
        hz: float | None = None,
    ) -> np.ndarray:
        out = np.asarray(keyframe_cmds, dtype=np.float32).copy()
        if out.ndim != 2 or out.shape[0] == 0:
            return out
        t_base = time.monotonic() if t0 is None else float(t0)
        rate = float(self.config.chunk_hz if hz is None else hz)
        if rate <= 0.0:
            rate = 30.0
        for _, idx in GRIPPER_SIDES:
            if idx >= out.shape[1]:
                continue
            if self.config.binary:
                out[:, idx] = self._process_side_binary(idx, out[:, idx], t_base, rate)
            else:
                out[:, idx] = self._process_side_continuous(idx, out[:, idx])
        return out

    def _apply_deadband(self, st: _SideState, v: float) -> float:
        cfg = self.config
        if st.last_raw is None:
            st.last_raw = v
            return v
        if abs(v - st.last_raw) < cfg.deadband:
            return float(st.last_raw)
        st.last_raw = v
        return v

    def _arm_close_hold(self, st: _SideState, t_now: float) -> None:
        cfg = self.config
        st.close_hold_remaining = max(0, int(cfg.min_close_hold_frames))
        st.closed_since = float(t_now)
        st.opened_since = None

    def _arm_open_hold(self, st: _SideState, t_now: float) -> None:
        st.open_hold_remaining = 0
        st.opened_since = float(t_now)
        st.closed_since = None

    def _close_hold_active(self, st: _SideState, t_now: float) -> bool:
        cfg = self.config
        if st.close_hold_remaining > 0:
            return True
        hold_s = float(cfg.min_close_hold_s)
        if hold_s <= 0.0 or st.closed_since is None:
            return False
        return (float(t_now) - float(st.closed_since)) < hold_s

    def _open_hold_active(self, st: _SideState, t_now: float) -> bool:
        cfg = self.config
        if st.open_hold_remaining > 0:
            return True
        hold_s = float(cfg.min_open_hold_s)
        if hold_s <= 0.0 or st.opened_since is None:
            return False
        return (float(t_now) - float(st.opened_since)) < hold_s

    def _process_side_binary(
        self,
        idx: int,
        raw: np.ndarray,
        t_base: float,
        rate: float,
    ) -> np.ndarray:
        cfg = self.config
        st = self._sides[idx]
        series = np.clip(np.asarray(raw, dtype=np.float32), cfg.lo, cfg.hi)
        result = np.empty_like(series)
        dt = 1.0 / rate

        if st.closed is None:
            seed = float(series[0]) if st.last_raw is None else float(st.last_raw)
            st.closed = seed >= cfg.close_enter
            st.last_raw = seed
            if st.closed:
                self._arm_close_hold(st, t_base)

        assert st.closed is not None

        for t in range(int(series.shape[0])):
            t_now = t_base + float(t) * dt
            v = self._apply_deadband(st, float(series[t]))
            want_close = v >= cfg.close_enter
            want_open = v <= cfg.open_enter

            if st.closed:
                if want_close:
                    # Sustained close: keep closed, do NOT refresh edge hold.
                    st.open_streak = 0
                    st.close_streak = 0
                    result[t] = cfg.close_value
                elif want_open:
                    if self._close_hold_active(st, t_now):
                        if st.close_hold_remaining > 0:
                            st.close_hold_remaining -= 1
                        st.open_streak = 0
                        result[t] = cfg.close_value
                    else:
                        st.open_streak += 1
                        st.close_streak = 0
                        if st.open_streak >= max(1, int(cfg.open_confirm)):
                            st.closed = False
                            st.open_streak = 0
                            self._arm_open_hold(st, t_now)
                            result[t] = cfg.open_value
                        else:
                            result[t] = cfg.close_value
                else:
                    st.open_streak = 0
                    result[t] = cfg.close_value
            else:
                if want_close:
                    if self._open_hold_active(st, t_now):
                        if st.open_hold_remaining > 0:
                            st.open_hold_remaining -= 1
                        st.close_streak = 0
                        st.open_streak = 0
                    else:
                        st.close_streak += 1
                        st.open_streak = 0
                else:
                    st.close_streak = 0
                if st.close_streak >= max(1, int(cfg.close_confirm)):
                    st.closed = True
                    st.close_streak = 0
                    st.open_streak = 0
                    self._arm_close_hold(st, t_now)
                    result[t] = cfg.close_value
                else:
                    result[t] = cfg.open_value

            st.last_out = float(result[t])

        return result

    def _process_side_continuous(self, idx: int, raw: np.ndarray) -> np.ndarray:
        cfg = self.config
        st = self._sides[idx]
        series = np.clip(np.asarray(raw, dtype=np.float32), cfg.lo, cfg.hi)
        result = np.empty_like(series)

        if st.last_out is None:
            st.last_out = float(series[0])

        for t in range(series.shape[0]):
            v = float(series[t])
            last = float(st.last_out)

            if (
                cfg.dip_suppress_frames > 0
                and last >= cfg.high_level
                and (last - v) >= cfg.dip_depth
            ):
                st.dip_count += 1
                if st.dip_count <= cfg.dip_suppress_frames:
                    v = last
            else:
                st.dip_count = 0

            if abs(v - last) < cfg.deadband:
                v = last

            lo = last - cfg.max_step_down
            hi = last + cfg.max_step_up
            v = float(np.clip(v, lo, hi))
            v = float(np.clip(v, cfg.lo, cfg.hi))

            st.last_out = v
            st.last_raw = v
            result[t] = v

        return result

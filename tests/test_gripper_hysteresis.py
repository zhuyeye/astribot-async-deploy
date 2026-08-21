"""Gripper hysteresis unit tests (wire 0.4/0.6 → SDK 40/60)."""

from __future__ import annotations

import numpy as np

from control.gripper_post import GripperPostConfig, GripperPostProcessor
from protocol.base import LEFT_GRIPPER_IDX


def _run(pp: GripperPostProcessor, wire: list[float], *, t0: float = 0.0, hz: float = 30.0):
    cmds = np.zeros((len(wire), 25), dtype=np.float32)
    cmds[:, LEFT_GRIPPER_IDX] = np.asarray(wire, dtype=np.float32) * 100.0
    return pp.process_keyframes(cmds, t0=t0, hz=hz)[:, LEFT_GRIPPER_IDX]


def test_hysteresis_schmitt_example():
    """Match: 0.2 0.5 0.7 0.55 0.45 0.3 → 开 开 合 合 合 开 (hold disabled)."""
    pp = GripperPostProcessor(
        GripperPostConfig(min_close_hold_s=0.0, min_close_hold_frames=0, open_confirm=1)
    )
    pp.seed_from_state(np.zeros(25, dtype=np.float32))  # open

    out = _run(pp, [0.2, 0.5, 0.7, 0.55, 0.45, 0.3], t0=0.0, hz=30.0)
    expect = [0.0, 0.0, 100.0, 100.0, 100.0, 0.0]
    assert out.tolist() == expect, out.tolist()


def test_min_close_hold_edge_only_no_refresh():
    """Hold arms only on open→close; sustained close must not extend the window."""
    pp = GripperPostProcessor(
        GripperPostConfig(
            min_close_hold_s=0.1,
            min_close_hold_frames=0,
            chunk_hz=20.0,
            open_confirm=1,
        )
    )
    pp.seed_from_state(np.zeros(25, dtype=np.float32))

    # hz=20 → dt=0.05s. Close at t=0; keep close at 0.05/0.10 (no refresh);
    # open at 0.05 would be blocked; open at 0.15 (>0.1) allowed.
    out_early = _run(pp, [0.7, 0.2], t0=0.0, hz=20.0)
    assert out_early.tolist() == [100.0, 100.0], out_early.tolist()

    pp = GripperPostProcessor(
        GripperPostConfig(
            min_close_hold_s=0.1,
            min_close_hold_frames=0,
            chunk_hz=20.0,
            open_confirm=1,
        )
    )
    pp.seed_from_state(np.zeros(25, dtype=np.float32))
    out = _run(pp, [0.7, 0.7, 0.7, 0.2], t0=0.0, hz=20.0)
    # t=0 close, t=0.05/0.10 still close, t=0.15 open after hold
    assert out.tolist() == [100.0, 100.0, 100.0, 0.0], out.tolist()


def test_default_open_confirm_rejects_single_frame_open_dip():
    pp = GripperPostProcessor(GripperPostConfig(min_close_hold_s=0.0, chunk_hz=30.0))
    pp.seed_from_state(np.zeros(25, dtype=np.float32))

    out = _run(pp, [0.8, 1.0, 1.0, 0.22], t0=0.0, hz=30.0)

    assert out.tolist() == [100.0, 100.0, 100.0, 100.0], out.tolist()


def test_default_open_confirm_allows_sustained_open_intent():
    pp = GripperPostProcessor(GripperPostConfig(min_close_hold_s=0.0, chunk_hz=30.0))
    pp.seed_from_state(np.zeros(25, dtype=np.float32))

    out = _run(pp, [0.8, 0.2, 0.2, 0.2], t0=0.0, hz=30.0)

    assert out.tolist() == [100.0, 100.0, 0.0, 0.0], out.tolist()


def test_default_close_hold_blocks_early_open_after_grasp():
    pp = GripperPostProcessor(GripperPostConfig(chunk_hz=10.0))
    pp.seed_from_state(np.zeros(25, dtype=np.float32))

    out = _run(pp, [0.8, 0.0, 0.0, 0.0, 0.0], t0=0.0, hz=10.0)

    assert out.tolist() == [100.0, 100.0, 100.0, 100.0, 100.0], out.tolist()


def test_open_hold_suppresses_release_reclose_bounce():
    pp = GripperPostProcessor(
        GripperPostConfig(
            min_close_hold_s=0.0,
            min_open_hold_s=0.5,
            chunk_hz=10.0,
            open_confirm=1,
            close_confirm=1,
        )
    )
    pp.seed_from_state(np.zeros(25, dtype=np.float32))

    out = _run(pp, [0.8, 0.0, 0.8, 0.8, 0.8, 0.8, 0.8], t0=0.0, hz=10.0)

    assert out.tolist() == [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0], out.tolist()

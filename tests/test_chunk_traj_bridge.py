"""ChunkTrajBridge unit tests."""

from __future__ import annotations

import numpy as np

from control.chunk_traj_bridge import ChunkTrajBridge, ChunkTrajConfig


def test_sample_interpolates_joints_zoh_gripper_after_binary():
    bridge = ChunkTrajBridge(
        ChunkTrajConfig(
            chunk_hz=10.0,
            blend_s=0.0,
            use_obs_timestamp=True,
            playback_lead_s=0.0,
            gripper_binary=True,
            gripper_close_enter=60.0,
            gripper_open_enter=40.0,
            gripper_close_confirm=1,
            gripper_open_confirm=1,
            gripper_min_close_hold_s=0.0,
            gripper_min_close_hold_frames=0,
            gripper_deadband=0.0,
        )
    )
    bridge.seed_gripper_from_state(np.zeros(25, dtype=np.float32))
    actions = np.zeros((3, 25), dtype=np.float32)
    actions[0, 7] = 0.0
    actions[1, 7] = 1.0
    actions[2, 7] = 2.0
    actions[0, 14] = 0.0
    actions[1, 14] = 100.0
    actions[2, 14] = 100.0

    t0 = 100.0
    bridge.ingest(actions, obs_timestamp=t0, now=t0)

    mid = bridge.sample(t0 + 0.05)  # halfway between key 0 and 1
    assert mid is not None
    assert abs(float(mid[7]) - 0.5) < 1e-3
    # binary open at key0 → ZOH until key1
    assert abs(float(mid[14]) - 0.0) < 1e-3

    after = bridge.sample(t0 + 0.15)
    assert after is not None
    assert abs(float(after[14]) - 100.0) < 1e-3


def test_gripper_no_blend_by_default_then_binary():
    bridge = ChunkTrajBridge(
        ChunkTrajConfig(
            chunk_hz=10.0,
            blend_s=0.2,
            use_obs_timestamp=False,
            playback_lead_s=0.0,
            blend_grippers=False,
            gripper_binary=True,
            gripper_close_enter=60.0,
            gripper_open_enter=40.0,
            gripper_close_confirm=1,
            gripper_open_confirm=1,
            gripper_min_close_hold_s=0.0,
            gripper_min_close_hold_frames=0,
            gripper_deadband=0.0,
        )
    )
    bridge.seed_gripper_from_state(np.zeros(25, dtype=np.float32))

    a1 = np.zeros((5, 25), dtype=np.float32)
    a1[:, 7] = 1.0
    a1[:, 14] = 0.0
    bridge.ingest(a1, obs_timestamp=0.0, now=0.0)
    assert abs(float(bridge.sample(0.0)[14]) - 0.0) < 1e-3

    a_close = np.zeros((5, 25), dtype=np.float32)
    a_close[:, 7] = 1.0
    a_close[:, 14] = 80.0  # above close_enter
    bridge.ingest(a_close, obs_timestamp=0.0, now=0.5)
    assert abs(float(bridge.sample(0.5)[14]) - 100.0) < 1e-3

    # New chunk open: gripper jumps via binary (no blend), arm still blends.
    a2 = np.zeros((5, 25), dtype=np.float32)
    a2[:, 7] = 3.0
    a2[:, 14] = 0.0
    bridge.ingest(a2, obs_timestamp=0.0, now=1.0)
    s2 = bridge.sample(1.0)
    assert s2 is not None
    assert float(s2[7]) < 3.0
    assert float(s2[7]) > 1.0
    assert float(s2[14]) in (0.0, 100.0)


def test_seam_blend_at_k_now_with_obs_timestamp():
    """Blend at ingest seam (k_now), not always key0 — mirrors live_ts ingest #39."""
    bridge = ChunkTrajBridge(
        ChunkTrajConfig(
            chunk_hz=30.0,
            blend_s=0.12,
            use_obs_timestamp=True,
            playback_lead_s=0.0,
            gripper_binary=False,
        )
    )
    t0_prev = 100.0
    a1 = np.zeros((10, 25), dtype=np.float32)
    a1[:, 15] = np.linspace(0.21, 0.34, 10, dtype=np.float32)
    bridge.ingest(a1, obs_timestamp=t0_prev, now=t0_prev)

    obs_ts = 100.1397
    ingest_now = obs_ts + 0.140  # ~140ms inference lag after obs capture
    bridge.sample(ingest_now - 0.005)
    prev_arm_r0 = float(bridge._last_sample[15])  # noqa: SLF001

    a2 = np.zeros((10, 25), dtype=np.float32)
    a2[:, 15] = np.array(
        [0.2100, 0.2141, 0.2169, 0.2206, 0.2237, 0.2274, 0.2307, 0.2346, 0.2369, 0.2402],
        dtype=np.float32,
    )
    bridge.ingest(a2, obs_timestamp=obs_ts, now=ingest_now)

    k_now = bridge._blend_start_index(t0=obs_ts, now=ingest_now, horizon=10)  # noqa: SLF001
    assert k_now == 4

    cmd = bridge.sample(ingest_now)
    assert cmd is not None
    raw_at_seam = float(a2[k_now, 15])
    assert abs(float(cmd[15]) - raw_at_seam) > 0.01
    assert abs(float(cmd[15]) - prev_arm_r0) < abs(raw_at_seam - prev_arm_r0)


def test_cross_chunk_blend_arm_only():
    bridge = ChunkTrajBridge(ChunkTrajConfig(chunk_hz=10.0, blend_s=0.2, use_obs_timestamp=False, playback_lead_s=0.0, gripper_binary=False))
    a1 = np.zeros((5, 25), dtype=np.float32)
    a1[:, 7] = 1.0
    bridge.ingest(a1, obs_timestamp=0.0, now=0.0)
    s1 = bridge.sample(0.0)
    assert s1 is not None
    assert abs(float(s1[7]) - 1.0) < 1e-3

    a2 = np.zeros((5, 25), dtype=np.float32)
    a2[:, 7] = 3.0
    bridge.ingest(a2, obs_timestamp=0.0, now=1.0)
    s2 = bridge.sample(1.0)
    assert s2 is not None
    assert float(s2[7]) < 3.0
    assert float(s2[7]) > 1.0


def test_stale_obs_timestamp_clamps_playback_to_future_lead():
    bridge = ChunkTrajBridge(
        ChunkTrajConfig(
            chunk_hz=10.0,
            blend_s=0.0,
            use_obs_timestamp=True,
            playback_lead_s=0.05,
            gripper_binary=False,
        )
    )
    actions = np.zeros((5, 25), dtype=np.float32)
    actions[:, 7] = np.arange(5, dtype=np.float32)

    bridge.ingest(actions, obs_timestamp=100.0, now=101.0)

    # Before this fix, sample(101.0) would see an expired trajectory and return
    # key[-1].  With lead clamp, playback starts at 101.05 and returns key[0].
    s = bridge.sample(101.0)
    assert s is not None
    assert abs(float(s[7]) - 0.0) < 1e-3

    s_mid = bridge.sample(101.10)
    assert s_mid is not None
    assert 0.0 < float(s_mid[7]) < 1.0


def test_fresh_obs_timestamp_keeps_original_timeline():
    bridge = ChunkTrajBridge(
        ChunkTrajConfig(
            chunk_hz=10.0,
            blend_s=0.0,
            use_obs_timestamp=True,
            playback_lead_s=0.05,
            gripper_binary=False,
        )
    )
    actions = np.zeros((5, 25), dtype=np.float32)
    actions[:, 7] = np.arange(5, dtype=np.float32)

    bridge.ingest(actions, obs_timestamp=100.0, now=100.10)

    s = bridge.sample(100.10)
    assert s is not None
    assert abs(float(s[7]) - 1.0) < 1e-3

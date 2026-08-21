"""Protocol adapter unit tests."""

from __future__ import annotations

import numpy as np

from protocol.adapter_astribot34 import Astribot34Adapter, wire_actions_to_25, wire_state_to_25
from protocol.adapter_pi05 import Pi05Adapter
from protocol.base import InternalObs, get_adapter


def _obs() -> InternalObs:
    return InternalObs(
        head_rgb=np.zeros((240, 320, 3), dtype=np.uint8),
        left_rgb=np.zeros((240, 320, 3), dtype=np.uint8),
        right_rgb=np.zeros((240, 320, 3), dtype=np.uint8),
        state_25_sdk=np.linspace(0, 24, 25).astype(np.float32),
        obs_timestamp=1.23,
        head_stamp=1.0,
        left_stamp=1.0,
        right_stamp=1.0,
        prompt="hello",
    )


def test_get_adapter():
    assert get_adapter("pi05").name == "pi05"
    assert get_adapter("astribot34").name == "astribot34"


def test_pi05_roundtrip_actions():
    adapter = Pi05Adapter()
    obs = _obs()
    wire = adapter.encode_obs(obs)
    assert "observation/image" in wire
    assert wire["observation/state"].shape == (25,)
    # grippers scaled to [0,1]
    assert wire["observation/state"][14] <= 1.0

    actions = np.zeros((10, 25), dtype=np.float32)
    actions[:, 14] = 0.5
    actions[:, 22] = 1.0
    out = adapter.decode_actions({"actions": actions, "obs_timestamp": 1.0})
    assert out.shape == (10, 25)
    assert abs(out[0, 14] - 50.0) < 1e-3
    assert abs(out[0, 22] - 100.0) < 1e-3


def test_astribot34_encode_images_chw():
    adapter = Astribot34Adapter()
    wire = adapter.encode_obs(_obs())
    assert wire["images"]["cam_high"].shape[0] == 3
    assert wire["state"].shape == (34,)
    assert wire["obs_timestamp"] == 1.23


def test_astribot34_action_mapping():
    actions = np.random.randn(8, 34).astype(np.float32)
    actions[:, 14] = 0.2
    actions[:, 22] = 0.8
    out = wire_actions_to_25(actions)
    assert out.shape == (8, 25)
    assert abs(out[0, 14] - 20.0) < 1e-3
    assert abs(out[0, 22] - 80.0) < 1e-3


def test_wire_state_gripper_autodetect():
    state = np.zeros(34)
    state[14] = 0.3
    state[22] = 50.0  # already SDK-like
    out = wire_state_to_25(state)
    assert abs(out[14] - 30.0) < 1e-3
    assert abs(out[22] - 50.0) < 1e-3

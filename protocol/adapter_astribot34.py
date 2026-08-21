"""astribot_eval async wire protocol adapter (images.cam_* + state 34).

Dimension mapping (provisional — confirm with server team via SERVER.md):

Wire state/action layout assumed as:
  [0:3]   chassis
  [3:7]   torso
  [7:14]  left arm
  [14]    left gripper (wire [0,1] or [0,100] — auto-detect)
  [15:22] right arm
  [22]    right gripper
  [23:25] head
  [25:34] extra / unused (pad or drop)

If wire dim is 29, treat as first 29 of the above (no 5-DOF tail).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from protocol.base import (
    IMAGE_SIZE,
    JOINT_DIM,
    LEFT_GRIPPER_IDX,
    RIGHT_GRIPPER_IDX,
    InternalObs,
    ProtocolAdapter,
    prep_image,
)


def _to_chw_float01(rgb_hwc_u8: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB -> CHW float32 [0,1]."""
    arr = np.asarray(rgb_hwc_u8)
    if arr.ndim != 3:
        raise ValueError(f"image must be HxWxC, got {arr.shape}")
    if arr.shape[0] == 3:
        chw = arr.astype(np.float32)
    else:
        chw = np.transpose(arr, (2, 0, 1)).astype(np.float32)
    if chw.max() > 1.5:
        chw = chw / 255.0
    return np.clip(chw, 0.0, 1.0)


def _normalize_gripper_wire_to_sdk(value: float) -> float:
    """Accept wire gripper in [0,1] or [0,100]; return SDK [0,100]."""
    v = float(value)
    if v <= 1.0 + 1e-3:
        return float(np.clip(v, 0.0, 1.0) * 100.0)
    return float(np.clip(v, 0.0, 100.0))


def _sdk_gripper_to_wire01(value: float) -> float:
    return float(np.clip(value / 100.0, 0.0, 1.0))


def wire_state_to_25(state: np.ndarray) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float64).reshape(-1)
    out = np.zeros(JOINT_DIM, dtype=np.float32)
    n = min(arr.shape[0], JOINT_DIM)
    out[:n] = arr[:n]
    out[LEFT_GRIPPER_IDX] = _normalize_gripper_wire_to_sdk(out[LEFT_GRIPPER_IDX])
    out[RIGHT_GRIPPER_IDX] = _normalize_gripper_wire_to_sdk(out[RIGHT_GRIPPER_IDX])
    return out


def sdk_25_to_wire_state(state_25: np.ndarray, *, wire_dim: int = 34) -> np.ndarray:
    src = np.asarray(state_25, dtype=np.float64).reshape(-1)
    if src.shape[0] != JOINT_DIM:
        raise ValueError(f"expected state_25 length {JOINT_DIM}, got {src.shape}")
    out = np.zeros(wire_dim, dtype=np.float64)
    n = min(JOINT_DIM, wire_dim)
    out[:n] = src[:n]
    if wire_dim > LEFT_GRIPPER_IDX:
        out[LEFT_GRIPPER_IDX] = _sdk_gripper_to_wire01(src[LEFT_GRIPPER_IDX])
    if wire_dim > RIGHT_GRIPPER_IDX:
        out[RIGHT_GRIPPER_IDX] = _sdk_gripper_to_wire01(src[RIGHT_GRIPPER_IDX])
    return out


def wire_actions_to_25(actions: np.ndarray) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"actions must be rank-2, got {arr.shape}")
    t = arr.shape[0]
    out = np.zeros((t, JOINT_DIM), dtype=np.float32)
    n = min(arr.shape[1], JOINT_DIM)
    out[:, :n] = arr[:, :n]
    for i in range(t):
        out[i, LEFT_GRIPPER_IDX] = _normalize_gripper_wire_to_sdk(out[i, LEFT_GRIPPER_IDX])
        out[i, RIGHT_GRIPPER_IDX] = _normalize_gripper_wire_to_sdk(out[i, RIGHT_GRIPPER_IDX])
    return out


class Astribot34Adapter(ProtocolAdapter):
    """images.cam_* CHW float, state (34,), actions (T, 29..34)."""

    name = "astribot34"

    def __init__(self, *, image_size: int = IMAGE_SIZE, state_dim: int = 34) -> None:
        self._image_size = image_size
        self._state_dim = state_dim

    def encode_obs(self, obs: InternalObs) -> dict[str, Any]:
        head = prep_image(obs.head_rgb, self._image_size)
        left = prep_image(obs.left_rgb, self._image_size)
        right = prep_image(obs.right_rgb, self._image_size)
        payload: dict[str, Any] = {
            "images": {
                "cam_high": _to_chw_float01(head),
                "cam_left_wrist": _to_chw_float01(left),
                "cam_right_wrist": _to_chw_float01(right),
            },
            "state": sdk_25_to_wire_state(obs.state_25_sdk, wire_dim=self._state_dim),
            "obs_timestamp": float(obs.obs_timestamp),
        }
        if obs.prompt:
            payload["prompt"] = obs.prompt
        return payload

    def decode_actions(self, frame: dict[str, Any]) -> np.ndarray:
        if "actions" not in frame:
            raise KeyError(f"missing 'actions'; keys={sorted(frame.keys())}")
        return wire_actions_to_25(frame["actions"])

"""Internal observation / action representation and protocol adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from common import image_tools

JOINT_DIM = 25
LEFT_GRIPPER_IDX = 14
RIGHT_GRIPPER_IDX = 22
GRIPPER_INDICES = (LEFT_GRIPPER_IDX, RIGHT_GRIPPER_IDX)

WHOLE_BODY_INDEX = {
    "astribot_chassis": (0, 3),
    "astribot_torso": (3, 7),
    "astribot_arm_left": (7, 14),
    "astribot_gripper_left": (14, 15),
    "astribot_arm_right": (15, 22),
    "astribot_gripper_right": (22, 23),
    "astribot_head": (23, 25),
}
WHOLE_BODY_NAMES = list(WHOLE_BODY_INDEX.keys())

PI_CAMERA_KEYS = {
    "observation/image": "head_rgbd",
    "observation/wrist_image": "left_wrist_rgbd",
    "observation/wrist_image_right": "right_wrist_rgbd",
}

DEFAULT_PROMPT = "pick up the orange plush toy to the basket"
IMAGE_SIZE = 224

COLLECTION_INIT_25 = np.asarray(
    [
        0.0,
        0.0,
        0.0,
        0.6,
        -1.2,
        0.6,
        0.0,
        0.159,
        -0.022,
        -1.42,
        1.66,
        -0.345,
        0.115,
        0.125,
        0.0,
        -0.159,
        -0.022,
        1.42,
        1.66,
        0.345,
        0.115,
        -0.125,
        0.0,
        0.0,
        0.81,
    ],
    dtype=np.float32,
)

COLLECTION_INIT_MOVE_NAMES = (
    "astribot_torso",
    "astribot_arm_left",
    "astribot_gripper_left",
    "astribot_arm_right",
    "astribot_gripper_right",
    "astribot_head",
)


@dataclass
class InternalObs:
    """Robot-side observation in SDK units (gripper [0,100])."""

    head_rgb: np.ndarray
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    state_25_sdk: np.ndarray
    obs_timestamp: float
    head_stamp: float = 0.0
    left_stamp: float = 0.0
    right_stamp: float = 0.0
    prompt: str = ""


class ProtocolAdapter(ABC):
    name: str

    @abstractmethod
    def encode_obs(self, obs: InternalObs) -> dict[str, Any]:
        ...

    @abstractmethod
    def decode_actions(self, frame: dict[str, Any]) -> np.ndarray:
        """Return (T, 25) absolute targets in SDK units (gripper [0,100])."""
        ...


def gripper_sdk_to_server(state_25: np.ndarray) -> np.ndarray:
    state = np.asarray(state_25, dtype=np.float32).copy()
    state[LEFT_GRIPPER_IDX] = np.clip(state[LEFT_GRIPPER_IDX] / 100.0, 0.0, 1.0)
    state[RIGHT_GRIPPER_IDX] = np.clip(state[RIGHT_GRIPPER_IDX] / 100.0, 0.0, 1.0)
    return state


def gripper_server_to_sdk(cmd_25: np.ndarray) -> np.ndarray:
    cmd = np.asarray(cmd_25, dtype=np.float32).copy()
    cmd[LEFT_GRIPPER_IDX] = np.clip(cmd[LEFT_GRIPPER_IDX], 0.0, 1.0) * 100.0
    cmd[RIGHT_GRIPPER_IDX] = np.clip(cmd[RIGHT_GRIPPER_IDX], 0.0, 1.0) * 100.0
    return cmd


def prep_image(rgb: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb, size, size))


def flatten_joint_groups(groups: list[list[float]]) -> np.ndarray:
    flat: list[float] = []
    for group in groups:
        flat.extend(float(v) for v in group)
    return np.asarray(flat, dtype=np.float32)


def cmd_25_to_sdk_groups(
    cmd: np.ndarray,
    *,
    include_grippers: bool = True,
) -> tuple[list[str], list[list[float]]]:
    names: list[str] = []
    values: list[list[float]] = []
    for name in WHOLE_BODY_NAMES:
        if not include_grippers and name.startswith("astribot_gripper"):
            continue
        start, end = WHOLE_BODY_INDEX[name]
        names.append(name)
        values.append([float(v) for v in cmd[start:end]])
    return names, values


def freeze_parts(
    cmd_sdk: np.ndarray,
    current_sdk: np.ndarray,
    *,
    freeze_chassis: bool = False,
    freeze_torso: bool = False,
    freeze_head: bool = False,
) -> np.ndarray:
    out = np.asarray(cmd_sdk, dtype=np.float32).copy()
    current = np.asarray(current_sdk, dtype=np.float32)
    if freeze_chassis:
        out[0:3] = current[0:3]
    if freeze_torso:
        out[3:7] = current[3:7]
    if freeze_head:
        out[23:25] = current[23:25]
    return out


def get_adapter(name: str, **kwargs: Any) -> ProtocolAdapter:
    key = name.strip().lower()
    if key in ("pi05", "pi05_s1_orange", "openpi"):
        from protocol.adapter_pi05 import Pi05Adapter

        return Pi05Adapter(**kwargs)
    if key in ("astribot34", "astribot_async", "async34"):
        from protocol.adapter_astribot34 import Astribot34Adapter

        return Astribot34Adapter(**kwargs)
    raise ValueError(f"unknown protocol {name!r}; use 'pi05' or 'astribot34'")

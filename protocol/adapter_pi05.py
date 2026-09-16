"""openpi pi05_s1_orange wire protocol adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from protocol.base import (
    IMAGE_SIZE,
    JOINT_DIM,
    InternalObs,
    ProtocolAdapter,
    gripper_sdk_to_server,
    gripper_server_to_sdk,
    prep_image,
)


class Pi05Adapter(ProtocolAdapter):
    """observation/* keys, state/action (T, 25).

    Obs gripper stays SDK [0,100] (server normalizes). Action gripper wire is
    still [0,1] and is scaled to SDK [0,100] on decode.
    """

    name = "pi05"

    def __init__(self, *, image_size: int = IMAGE_SIZE) -> None:
        self._image_size = image_size

    def encode_obs(self, obs: InternalObs) -> dict[str, Any]:
        return {
            "observation/image": prep_image(obs.head_rgb, self._image_size),
            "observation/wrist_image": prep_image(obs.left_rgb, self._image_size),
            "observation/wrist_image_right": prep_image(obs.right_rgb, self._image_size),
            "observation/state": gripper_sdk_to_server(obs.state_25_sdk),
            "obs_timestamp": float(obs.obs_timestamp),
            "prompt": obs.prompt,
        }

    def decode_actions(self, frame: dict[str, Any]) -> np.ndarray:
        if "actions" not in frame:
            raise KeyError(f"missing 'actions'; keys={sorted(frame.keys())}")
        actions = np.asarray(frame["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[1] != JOINT_DIM:
            raise ValueError(f"expected actions (T, {JOINT_DIM}), got {actions.shape}")
        out = np.stack([gripper_server_to_sdk(actions[i]) for i in range(actions.shape[0])], axis=0)
        return out

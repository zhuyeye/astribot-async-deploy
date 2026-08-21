"""Gripper unit helpers only — no binary/effector post-processing."""

from __future__ import annotations

import numpy as np

from protocol.base import LEFT_GRIPPER_IDX, RIGHT_GRIPPER_IDX, gripper_sdk_to_server, gripper_server_to_sdk

__all__ = [
    "LEFT_GRIPPER_IDX",
    "RIGHT_GRIPPER_IDX",
    "gripper_sdk_to_server",
    "gripper_server_to_sdk",
    "step_hold_grippers",
]


def step_hold_grippers(dense: np.ndarray, keyframes: np.ndarray, hold_idx: np.ndarray) -> np.ndarray:
    """Apply ZOH on gripper dims given keyframe hold indices."""
    out = dense.copy()
    out[:, LEFT_GRIPPER_IDX] = keyframes[hold_idx, LEFT_GRIPPER_IDX]
    out[:, RIGHT_GRIPPER_IDX] = keyframes[hold_idx, RIGHT_GRIPPER_IDX]
    return out

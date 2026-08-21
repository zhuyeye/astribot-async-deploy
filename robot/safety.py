"""Safety clipping for 25-dim joint commands (SDK units: gripper 0-100)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from protocol.base import GRIPPER_INDICES, WHOLE_BODY_INDEX, WHOLE_BODY_NAMES

SDK_LIMITS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "astribot_torso": (
        (-0.04, -2.30, -0.40, -1.20),
        (1.30, 0.06, 2.30, 1.20),
    ),
    "astribot_arm_left": (
        (-3.10, -1.53, -3.10, -0.06, -2.56, -0.76, -1.53),
        (3.10, 0.46, 3.10, 2.61, 2.56, 0.76, 1.53),
    ),
    "astribot_arm_right": (
        (-3.10, -1.53, -3.10, -0.06, -2.56, -0.76, -1.53),
        (3.10, 0.46, 3.10, 2.61, 2.56, 0.76, 1.53),
    ),
    "astribot_head": ((-1.57, -1.22), (1.57, 1.22)),
}


@dataclass
class SafetyConfig:
    max_step_delta_rad: float = 0.15
    max_abs_rad: float = 3.2
    gripper_min: float = 0.0
    gripper_max: float = 100.0
    clip_to_limits: bool = True


def safety_clip(cmd: np.ndarray, current: np.ndarray, config: SafetyConfig) -> np.ndarray:
    out = np.asarray(cmd, dtype=np.float32).copy()
    current = np.asarray(current, dtype=np.float32)

    delta = out - current
    mask = np.ones(out.shape[0], dtype=bool)
    mask[list(GRIPPER_INDICES)] = False
    delta[mask] = np.clip(delta[mask], -config.max_step_delta_rad, config.max_step_delta_rad)
    out = current.copy()
    out[mask] = current[mask] + delta[mask]
    for idx in GRIPPER_INDICES:
        out[idx] = cmd[idx]
        out[idx] = float(np.clip(out[idx], config.gripper_min, config.gripper_max))

    if config.clip_to_limits:
        for name in WHOLE_BODY_NAMES:
            if name.startswith("astribot_gripper"):
                continue
            limits = SDK_LIMITS.get(name)
            if limits is None:
                continue
            lower, upper = limits
            start, end = WHOLE_BODY_INDEX[name]
            for i, idx in enumerate(range(start, end)):
                if i < len(lower):
                    out[idx] = np.clip(out[idx], lower[i], upper[i])
                out[idx] = np.clip(out[idx], -config.max_abs_rad, config.max_abs_rad)

    return out

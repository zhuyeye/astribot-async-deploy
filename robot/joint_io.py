"""Astribot SDK joint state read / command write for 25-DOF."""

from __future__ import annotations

from typing import Any

import numpy as np

from protocol.base import (
    COLLECTION_INIT_25,
    COLLECTION_INIT_MOVE_NAMES,
    WHOLE_BODY_INDEX,
    WHOLE_BODY_NAMES,
    cmd_25_to_sdk_groups,
    flatten_joint_groups,
)


class JointIO:
    def __init__(
        self,
        astribot: Any,
        *,
        control_way: str = "filter",
        gripper_control_way: str | None = "direct",
    ) -> None:
        self._astribot = astribot
        self._control_way = control_way
        self._gripper_control_way = gripper_control_way

    def read_joint_state_25(self) -> np.ndarray:
        groups = self._astribot.get_current_joints_position(WHOLE_BODY_NAMES)
        return flatten_joint_groups(groups)

    def send_joint_position_command(
        self,
        cmd: np.ndarray,
        *,
        include_grippers: bool = True,
    ) -> None:
        if not include_grippers or not self._gripper_control_way:
            names, values = cmd_25_to_sdk_groups(cmd, include_grippers=include_grippers)
            self._astribot.set_joints_position(
                names,
                values,
                control_way=self._control_way,
                use_wbc=False,
                add_default_torso=False,
            )
            return

        if self._gripper_control_way == self._control_way:
            names, values = cmd_25_to_sdk_groups(cmd, include_grippers=True)
            self._astribot.set_joints_position(
                names,
                values,
                control_way=self._control_way,
                use_wbc=False,
                add_default_torso=False,
            )
            return

        names, values = cmd_25_to_sdk_groups(cmd, include_grippers=False)
        self._astribot.set_joints_position(
            names,
            values,
            control_way=self._control_way,
            use_wbc=False,
            add_default_torso=False,
        )

        gripper_names: list[str] = []
        gripper_values: list[list[float]] = []
        for name in ("astribot_gripper_left", "astribot_gripper_right"):
            start, end = WHOLE_BODY_INDEX[name]
            gripper_names.append(name)
            gripper_values.append([float(v) for v in np.asarray(cmd, dtype=np.float32)[start:end]])
        self._astribot.set_joints_position(
            gripper_names,
            gripper_values,
            control_way=self._gripper_control_way,
            use_wbc=False,
            add_default_torso=False,
        )

    def stop(self) -> None:
        self._astribot.stop_robot()

    def move_to_home(self) -> None:
        self._astribot.move_to_home()

    def move_to_collection_init(self, *, duration_s: float = 3.0) -> None:
        cmd = np.asarray(COLLECTION_INIT_25, dtype=np.float32)
        names: list[str] = []
        values: list[list[float]] = []
        for name in COLLECTION_INIT_MOVE_NAMES:
            start, end = WHOLE_BODY_INDEX[name]
            names.append(name)
            values.append([float(v) for v in cmd[start:end]])
        self._astribot.move_joints_position(
            names,
            values,
            duration=duration_s,
            use_wbc=False,
            add_default_torso=False,
        )

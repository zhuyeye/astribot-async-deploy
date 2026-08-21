"""JointIO command grouping tests."""

from __future__ import annotations

import numpy as np

from robot.joint_io import JointIO


class _FakeAstribot:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def set_joints_position(
        self,
        names: list[str],
        values: list[list[float]],
        *,
        control_way: str,
        use_wbc: bool,
        add_default_torso: bool,
    ) -> None:
        self.calls.append(
            {
                "names": names,
                "values": values,
                "control_way": control_way,
                "use_wbc": use_wbc,
                "add_default_torso": add_default_torso,
            }
        )


def test_joint_io_splits_gripper_control_way() -> None:
    fake = _FakeAstribot()
    joint_io = JointIO(fake, control_way="filter", gripper_control_way="direct")

    cmd = np.arange(25, dtype=np.float32)
    joint_io.send_joint_position_command(cmd, include_grippers=True)

    assert len(fake.calls) == 2
    body, grippers = fake.calls
    assert body["control_way"] == "filter"
    assert body["names"] == [
        "astribot_chassis",
        "astribot_torso",
        "astribot_arm_left",
        "astribot_arm_right",
        "astribot_head",
    ]
    assert grippers["control_way"] == "direct"
    assert grippers["names"] == ["astribot_gripper_left", "astribot_gripper_right"]
    assert grippers["values"] == [[14.0], [22.0]]


def test_joint_io_keeps_single_call_when_control_way_matches() -> None:
    fake = _FakeAstribot()
    joint_io = JointIO(fake, control_way="filter", gripper_control_way="filter")

    joint_io.send_joint_position_command(np.zeros(25, dtype=np.float32), include_grippers=True)

    assert len(fake.calls) == 1
    assert fake.calls[0]["control_way"] == "filter"

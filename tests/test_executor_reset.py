"""Executor reset tests."""

from __future__ import annotations

import queue

import numpy as np

from control.executor import Executor, ExecutorConfig
from transport.action_receiver import ActionChunkMsg


class _FakeJointIO:
    def __init__(self) -> None:
        self._q = np.zeros(25, dtype=np.float32)

    def read_joint_state_25(self) -> np.ndarray:
        return self._q.copy()


def test_executor_reset_clears_queue_and_bridge_state() -> None:
    q: queue.Queue[ActionChunkMsg] = queue.Queue()
    q.put(ActionChunkMsg(actions_sdk=np.ones((2, 25), dtype=np.float32), obs_timestamp=1.0, raw={}))
    executor = Executor(_FakeJointIO(), q, ExecutorConfig())

    executor._bridge.ingest(np.ones((2, 25), dtype=np.float32), obs_timestamp=1.0, now=1.0)  # noqa: SLF001
    assert executor._bridge.has_trajectory()  # noqa: SLF001

    executor.reset()

    assert q.empty()
    assert not executor._bridge.has_trajectory()  # noqa: SLF001


def test_episode_complete_requires_stable_open_after_close() -> None:
    executor = Executor(
        _FakeJointIO(),
        queue.Queue(),
        ExecutorConfig(auto_episode_reset_open_s=0.5),
    )
    cmd = np.zeros(25, dtype=np.float32)

    executor._update_episode_completion(cmd, 0.0)  # noqa: SLF001
    assert not executor.pop_episode_complete()

    cmd[22] = 100.0
    executor._update_episode_completion(cmd, 0.1)  # noqa: SLF001
    assert not executor.pop_episode_complete()

    cmd[22] = 0.0
    executor._update_episode_completion(cmd, 0.2)  # noqa: SLF001
    executor._update_episode_completion(cmd, 0.6)  # noqa: SLF001
    assert not executor.pop_episode_complete()

    cmd[22] = 100.0
    executor._update_episode_completion(cmd, 0.7)  # noqa: SLF001
    cmd[22] = 0.0
    executor._update_episode_completion(cmd, 0.8)  # noqa: SLF001
    executor._update_episode_completion(cmd, 1.31)  # noqa: SLF001

    assert executor.pop_episode_complete()
    assert not executor.pop_episode_complete()

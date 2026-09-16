"""Async dual-channel deploy runner."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import logging
import os
import queue
import signal
import sys
import time
from typing import Any

import numpy as np

from control.executor import Executor, ExecutorConfig
from common.trace_log import TraceConfig, TraceLogger, set_tracer
from protocol.base import DEFAULT_PROMPT, InternalObs, get_adapter
from transport.action_receiver import ActionChunkMsg, ActionReceiver
from transport.obs_sender import ObsSender, fake_obs_factory
from transport.ws_client import RoleWebsocketClient

logger = logging.getLogger(__name__)

# ROS/SDK often replace SIGINT; without this, Ctrl-C appears to do nothing.
_STOP_SIGNAL_COUNT = 0


def _on_stop_signal(stop: asyncio.Event | None = None) -> None:
    global _STOP_SIGNAL_COUNT
    _STOP_SIGNAL_COUNT += 1
    if _STOP_SIGNAL_COUNT >= 2:
        logger.warning("Forced exit on second Ctrl-C / SIGTERM")
        os._exit(130)
    logger.warning("Stop requested (Ctrl-C). Press again to force exit.")
    if stop is not None:
        stop.set()


def _restore_sync_sigint() -> None:
    """Raise KeyboardInterrupt in Python even while blocked in time.sleep / SDK."""

    def _handler(_signum: int, _frame: object) -> None:
        _on_stop_signal(None)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _install_stop_signals(loop: asyncio.AbstractEventLoop, stop: asyncio.Event) -> None:
    def _on_stop() -> None:
        _on_stop_signal(stop)

    def _sync_handler(_signum: int, _frame: object) -> None:
        try:
            loop.call_soon_threadsafe(_on_stop)
        except RuntimeError:
            os._exit(130)

    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError, RuntimeError, ValueError):
            loop.remove_signal_handler(sig)
        try:
            loop.add_signal_handler(sig, _on_stop)
        except (NotImplementedError, RuntimeError, ValueError):
            signal.signal(sig, _sync_handler)


@dataclass
class DeployConfig:
    host: str = "192.168.81.88"
    port: int = 8000
    protocol: str = "pi05"
    prompt: str = DEFAULT_PROMPT
    send_hz: float = 30.0
    chunk_hz: float = 30.0
    ctrl_hz: float = 250.0
    blend_s: float = 0.12
    max_step_delta_rad: float = 0.15
    # SDK control mode for body joints. 'filter' is smoother but can feel laggy;
    # 'direct' is more responsive but may increase jitter.
    control_way: str = "filter"
    # Optional: override SDK filter parameters (only meaningful when control_way="filter").
    # SDK comment: lower is higher smooth, so to be more responsive you typically increase this.
    filter_scale: float | None = None
    gripper_filter_scale: float | None = None
    freeze_chassis: bool = True
    freeze_torso: bool = False
    freeze_head: bool = False
    dry_run: bool = False
    fake_sensors: bool = False
    move_init: bool = True
    init_duration_s: float = 3.0
    use_obs_timestamp: bool = True
    playback_lead_s: float = 0.05
    image_size: int = 224
    max_runtime_s: float = 0.0
    max_chunks: int = 0  # stop after N received action chunks (0=forever)
    trace: bool = False
    trace_dir: str = "action_logs/async_trace"
    trace_every_send: int = 10
    trace_every_recv: int = 1
    trace_every_exec: int = 50
    trace_save_arrays_every: int = 10
    trace_save_images: bool = False
    trace_full_vectors: bool = True
    blend_grippers: bool = False
    gripper_binary: bool = True
    gripper_close_enter: float = 70.0
    gripper_open_enter: float = 40.0
    gripper_close_confirm: int = 2
    gripper_open_confirm: int = 5
    gripper_min_close_hold_s: float = 1.2
    gripper_min_open_hold_s: float = 0.35
    gripper_min_close_hold_frames: int = 0
    gripper_deadband: float = 0.0
    stdin_reset: bool = True
    auto_episode_reset: bool = True
    auto_episode_reset_open_s: float = 0.6
    auto_episode_reset_delay_s: float = 0.5


class FakeJointIO:
    """No-op joint I/O for connectivity tests."""

    def __init__(self) -> None:
        self._q = np.zeros(25, dtype=np.float32)
        self.commands: list[np.ndarray] = []

    def read_joint_state_25(self) -> np.ndarray:
        return self._q.copy()

    def send_joint_position_command(self, cmd: np.ndarray, *, include_grippers: bool = True) -> None:
        self._q = np.asarray(cmd, dtype=np.float32).copy()
        self.commands.append(self._q.copy())

    def stop(self) -> None:
        pass

    def move_to_collection_init(self, *, duration_s: float = 3.0) -> None:
        pass


class AsyncDeployRunner:
    def __init__(self, config: DeployConfig) -> None:
        self._cfg = config
        self._adapter = get_adapter(config.protocol, image_size=config.image_size)
        self._chunk_queue: queue.Queue[ActionChunkMsg] = queue.Queue(maxsize=8)

    def run(self) -> None:
        asyncio.run(self._run_async())

    async def _reset_episode(
        self,
        *,
        executor: Executor,
        sender: ObsSender,
        joint_io: Any,
    ) -> None:
        logger.info("Episode reset requested")
        executor.pause()
        await asyncio.sleep(2.0 / max(self._cfg.ctrl_hz, 1e-3))
        executor.reset()
        with suppress(Exception):
            await sender.send_reset()
        if self._cfg.move_init:
            await asyncio.to_thread(
                joint_io.move_to_collection_init,
                duration_s=self._cfg.init_duration_s,
            )
        executor.reset()
        executor.resume()
        logger.info("Episode reset complete")

    async def _stdin_reset_loop(
        self,
        *,
        executor: Executor,
        sender: ObsSender,
        joint_io: Any,
    ) -> None:
        logger.info("Press 'r' + Enter to reset episode")
        loop = asyncio.get_running_loop()
        lines: asyncio.Queue[str] = asyncio.Queue()

        def _on_stdin() -> None:
            line = sys.stdin.readline()
            lines.put_nowait(line)

        loop.add_reader(sys.stdin.fileno(), _on_stdin)
        try:
            while True:
                line = await lines.get()
                if line == "":
                    await asyncio.sleep(0.5)
                    continue
                cmd = line.strip().lower()
                if cmd in ("r", "reset"):
                    await self._reset_episode(executor=executor, sender=sender, joint_io=joint_io)
        finally:
            with suppress(Exception):
                loop.remove_reader(sys.stdin.fileno())

    async def _auto_reset_loop(
        self,
        *,
        executor: Executor,
        sender: ObsSender,
        joint_io: Any,
    ) -> None:
        while True:
            await asyncio.sleep(0.05)
            if not executor.pop_episode_complete():
                continue
            logger.info(
                "Episode complete detected; resetting in %.2fs",
                self._cfg.auto_episode_reset_delay_s,
            )
            await asyncio.sleep(max(0.0, self._cfg.auto_episode_reset_delay_s))
            await self._reset_episode(executor=executor, sender=sender, joint_io=joint_io)

    async def _run_async(self) -> None:
        ros_mw = None
        joint_io: Any
        camera = None
        tracer: TraceLogger | None = None
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        _restore_sync_sigint()
        if self._cfg.trace:
            save_every = self._cfg.trace_save_arrays_every
            # Short verification runs: dump arrays every chunk/send by default.
            if self._cfg.max_chunks > 0 and save_every == 10:
                save_every = 1
            tracer = TraceLogger(
                TraceConfig(
                    enabled=True,
                    out_dir=self._cfg.trace_dir,
                    every_send=self._cfg.trace_every_send,
                    every_recv=self._cfg.trace_every_recv,
                    every_exec=self._cfg.trace_every_exec,
                    save_arrays_every=save_every,
                    full_vectors=self._cfg.trace_full_vectors,
                    save_images=self._cfg.trace_save_images,
                )
            )
            set_tracer(tracer)

        if self._cfg.fake_sensors:
            joint_io = FakeJointIO()
            obs_factory = fake_obs_factory(self._cfg.prompt)
            logger.info("Fake-sensor mode")
        else:
            from robot.camera_reader import CameraReader
            from robot.joint_io import JointIO
            from robot.sdk_env import import_astribot_sdk

            ros_mw, Astribot = import_astribot_sdk()
            astribot = Astribot(freq=self._cfg.ctrl_hz, high_control_rights=True)
            _restore_sync_sigint()
            if self._cfg.control_way == "filter" and (self._cfg.filter_scale is not None or self._cfg.gripper_filter_scale is not None):
                # SDK: lower filter_scale = more smoothing/less responsive.
                # Provide a fallback when only one side is set.
                filter_scale = self._cfg.filter_scale if self._cfg.filter_scale is not None else self._cfg.gripper_filter_scale
                gripper_filter_scale = (
                    self._cfg.gripper_filter_scale
                    if self._cfg.gripper_filter_scale is not None
                    else (self._cfg.filter_scale if self._cfg.filter_scale is not None else filter_scale)
                )
                astribot.set_filter_parameters(float(filter_scale), float(gripper_filter_scale))
            joint_io = JointIO(
                astribot,
                control_way=self._cfg.control_way,
                gripper_control_way="direct",
            )
            camera = CameraReader(astribot)
            _restore_sync_sigint()
            if self._cfg.move_init:
                logger.info("Moving to collection init (%.1fs)...", self._cfg.init_duration_s)
                joint_io.move_to_collection_init(duration_s=self._cfg.init_duration_s)

            def obs_factory() -> InternalObs | None:
                try:
                    frames, stamps = camera.read_rgb_with_stamps(wait_timeout_s=0.0)
                except TimeoutError:
                    return None
                q = joint_io.read_joint_state_25()
                now = time.perf_counter()
                return InternalObs(
                    head_rgb=frames["observation/image"],
                    left_rgb=frames["observation/wrist_image"],
                    right_rgb=frames["observation/wrist_image_right"],
                    state_25_sdk=q,
                    obs_timestamp=now,
                    head_stamp=stamps["observation/image"],
                    left_stamp=stamps["observation/wrist_image"],
                    right_stamp=stamps["observation/wrist_image_right"],
                    prompt=self._cfg.prompt,
                )

        sender_client = RoleWebsocketClient(self._cfg.host, self._cfg.port)
        receiver_client = RoleWebsocketClient(self._cfg.host, self._cfg.port)
        sender = ObsSender(
            sender_client,
            self._adapter,
            obs_factory,
            send_hz=self._cfg.send_hz,
            prompt=self._cfg.prompt,
        )
        receiver = ActionReceiver(receiver_client, self._adapter, self._chunk_queue)
        executor = Executor(
            joint_io,
            self._chunk_queue,
            ExecutorConfig(
                ctrl_hz=self._cfg.ctrl_hz,
                chunk_hz=self._cfg.chunk_hz,
                blend_s=self._cfg.blend_s,
                max_step_delta_rad=self._cfg.max_step_delta_rad,
                freeze_chassis=self._cfg.freeze_chassis,
                freeze_torso=self._cfg.freeze_torso,
                freeze_head=self._cfg.freeze_head,
                dry_run=self._cfg.dry_run,
                use_obs_timestamp=self._cfg.use_obs_timestamp,
                playback_lead_s=self._cfg.playback_lead_s,
                blend_grippers=self._cfg.blend_grippers,
                gripper_binary=self._cfg.gripper_binary,
                gripper_close_enter=self._cfg.gripper_close_enter,
                gripper_open_enter=self._cfg.gripper_open_enter,
                gripper_close_confirm=self._cfg.gripper_close_confirm,
                gripper_open_confirm=self._cfg.gripper_open_confirm,
                gripper_min_close_hold_s=self._cfg.gripper_min_close_hold_s,
                gripper_min_open_hold_s=self._cfg.gripper_min_open_hold_s,
                gripper_min_close_hold_frames=self._cfg.gripper_min_close_hold_frames,
                gripper_deadband=self._cfg.gripper_deadband,
                auto_episode_reset_open_s=self._cfg.auto_episode_reset_open_s,
            ),
        )

        logger.info(
            "Starting async deploy: host=%s:%d protocol=%s dry_run=%s fake=%s",
            self._cfg.host,
            self._cfg.port,
            self._cfg.protocol,
            self._cfg.dry_run,
            self._cfg.fake_sensors,
        )
        executor.start()
        _install_stop_signals(loop, stop)
        send_task = asyncio.create_task(sender.run(), name="obs-sender")
        recv_task = asyncio.create_task(receiver.run(), name="action-receiver")
        reset_task = None
        if self._cfg.stdin_reset and sys.stdin.isatty():
            reset_task = asyncio.create_task(
                self._stdin_reset_loop(executor=executor, sender=sender, joint_io=joint_io),
                name="stdin-reset",
            )
        auto_reset_task = None
        if self._cfg.auto_episode_reset:
            auto_reset_task = asyncio.create_task(
                self._auto_reset_loop(executor=executor, sender=sender, joint_io=joint_io),
                name="auto-reset",
            )

        try:
            if self._cfg.max_chunks > 0:
                logger.info("Will stop after %d action chunks", self._cfg.max_chunks)
                while executor.chunks_seen < self._cfg.max_chunks and not stop.is_set():
                    await asyncio.sleep(0.05)
                if not stop.is_set():
                    await asyncio.sleep(0.2)
                    logger.info(
                        "Reached max_chunks=%d (seen=%d)",
                        self._cfg.max_chunks,
                        executor.chunks_seen,
                    )
            elif self._cfg.max_runtime_s > 0:
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=self._cfg.max_runtime_s)
            else:
                await stop.wait()
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            logger.warning("Interrupted")
            stop.set()
        finally:
            sender.stop()
            receiver.stop()
            if reset_task is not None:
                reset_task.cancel()
            if auto_reset_task is not None:
                auto_reset_task.cancel()
            send_task.cancel()
            recv_task.cancel()
            tasks = [send_task, recv_task]
            if reset_task is not None:
                tasks.append(reset_task)
            if auto_reset_task is not None:
                tasks.append(auto_reset_task)

            async def _cleanup() -> None:
                await asyncio.gather(*tasks, return_exceptions=True)
                with suppress(Exception):
                    await sender.send_reset()
                await sender_client.close()
                await receiver_client.close()
                executor.stop()
                with suppress(Exception):
                    joint_io.stop()
                if ros_mw is not None:
                    with suppress(Exception):
                        ros_mw.shutdown()
                if tracer is not None:
                    tracer.close()
                    set_tracer(None)

            try:
                await asyncio.wait_for(_cleanup(), timeout=5.0)
            except (asyncio.TimeoutError, KeyboardInterrupt):
                logger.error("Shutdown hung; forcing exit")
                os._exit(130)
            logger.info("Stopped; chunks_seen=%d", executor.chunks_seen)

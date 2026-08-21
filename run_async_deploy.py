#!/usr/bin/env python3
"""CLI entry for async dual-channel Astribot S1 deploy."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from protocol.base import DEFAULT_PROMPT
from runner import AsyncDeployRunner, DeployConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Async dual-channel S1 policy deploy")
    p.add_argument("--host", default="192.168.81.88", help="GPU async policy server IP")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--protocol",
        choices=("pi05", "astribot34"),
        default="pi05",
        help="Wire protocol adapter (default pi05 until server checklist locks astribot34)",
    )
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument(
        "--send-hz",
        "--obs-send-hz",
        dest="send_hz",
        type=float,
        default=30.0,
        help="Observation send rate",
    )
    p.add_argument("--chunk-hz", type=float, default=30.0, help="Action keyframe rate")
    p.add_argument("--ctrl-hz", type=float, default=250.0, help="SDK command rate")
    p.add_argument("--blend-s", type=float, default=0.12, help="Cross-chunk blend seconds")
    p.add_argument("--max-step-delta", type=float, default=0.15)
    p.add_argument("--freeze-chassis", action="store_true", default=True)
    p.add_argument("--no-freeze-chassis", action="store_false", dest="freeze_chassis")
    p.add_argument("--freeze-torso", action="store_true")
    p.add_argument("--freeze-head", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Do not send commands to SDK")
    p.add_argument("--fake-sensors", action="store_true", help="Synthetic obs; no SDK")
    p.add_argument("--move-init", action="store_true", default=True)
    p.add_argument("--no-move-init", action="store_false", dest="move_init")
    p.add_argument("--init-duration", type=float, default=3.0)
    p.add_argument(
        "--ignore-obs-timestamp",
        action="store_true",
        help="Timeline chunks from arrival time instead of obs_timestamp",
    )
    p.add_argument(
        "--playback-lead-s",
        type=float,
        default=0.05,
        help="Minimum lead time before a chunk becomes playable on the robot side",
    )
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--max-runtime", type=float, default=0.0, help="Exit after N seconds (0=forever)")
    p.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Exit after N action chunks received (0=forever). Use e.g. 5 for a short verify run.",
    )
    p.add_argument(
        "--trace",
        action="store_true",
        help="Log send/recv and pre/post processing to JSONL (+ console samples)",
    )
    p.add_argument("--trace-dir", default="action_logs/async_trace")
    p.add_argument("--trace-every-send", type=int, default=10, help="Console every N sends (JSONL always)")
    p.add_argument("--trace-every-recv", type=int, default=1, help="Console every N action chunks")
    p.add_argument("--trace-every-exec", type=int, default=50, help="Console every N control ticks")
    p.add_argument(
        "--trace-save-arrays-every",
        type=int,
        default=10,
        help="Save NPZ every N events (0=never; auto=1 when --max-chunks>0)",
    )
    p.add_argument(
        "--trace-save-images",
        action="store_true",
        help="Also dump RGB images into send_*.npz (large)",
    )
    p.add_argument(
        "--blend-grippers",
        action="store_true",
        help="Also cross-chunk blend gripper dims (default: off; binary only)",
    )
    p.set_defaults(blend_grippers=False)
    p.add_argument(
        "--gripper-binary",
        action="store_true",
        default=True,
        help="Binarize grippers to {0,100} AFTER blend (default: on)",
    )
    p.add_argument("--no-gripper-binary", action="store_false", dest="gripper_binary")
    p.add_argument("--gripper-close-enter", type=float, default=60.0, help="SDK units; wire 0.6 → close")
    p.add_argument("--gripper-open-enter", type=float, default=40.0, help="SDK units; wire 0.4 → open")
    p.add_argument("--gripper-close-confirm", type=int, default=1)
    p.add_argument("--gripper-open-confirm", type=int, default=2)
    p.add_argument(
        "--gripper-min-close-hold-s",
        type=float,
        default=0.8,
        help="One-shot hold after open→close before open allowed (default 0.8s; not refreshed by sustained close)",
    )
    p.add_argument(
        "--gripper-min-open-hold-s",
        type=float,
        default=0.35,
        help="One-shot hold after close→open before close allowed (default 0.35s; suppresses release bounce)",
    )
    p.add_argument("--gripper-min-close-hold-frames", type=int, default=0)
    p.add_argument("--gripper-deadband", type=float, default=0.0)
    p.add_argument(
        "--stdin-reset",
        action="store_true",
        default=True,
        help="Allow 'r' + Enter in the terminal to reset the episode (default when stdin is a TTY)",
    )
    p.add_argument("--no-stdin-reset", action="store_false", dest="stdin_reset")
    p.add_argument(
        "--auto-episode-reset",
        action="store_true",
        default=True,
        help="Automatically reset after a completed grasp/place episode (default)",
    )
    p.add_argument("--no-auto-episode-reset", action="store_false", dest="auto_episode_reset")
    p.add_argument(
        "--auto-episode-reset-open-s",
        type=float,
        default=0.6,
        help="Require grippers to stay open this long after a close before auto reset",
    )
    p.add_argument(
        "--auto-episode-reset-delay-s",
        type=float,
        default=0.5,
        help="Delay after completion detection before moving back to init",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = DeployConfig(
        host=args.host,
        port=args.port,
        protocol=args.protocol,
        prompt=args.prompt,
        send_hz=args.send_hz,
        chunk_hz=args.chunk_hz,
        ctrl_hz=args.ctrl_hz,
        blend_s=args.blend_s,
        max_step_delta_rad=args.max_step_delta,
        freeze_chassis=args.freeze_chassis,
        freeze_torso=args.freeze_torso,
        freeze_head=args.freeze_head,
        dry_run=args.dry_run,
        fake_sensors=args.fake_sensors,
        move_init=args.move_init,
        init_duration_s=args.init_duration,
        use_obs_timestamp=not args.ignore_obs_timestamp,
        playback_lead_s=args.playback_lead_s,
        image_size=args.image_size,
        max_runtime_s=args.max_runtime,
        max_chunks=args.max_chunks,
        trace=args.trace,
        trace_dir=args.trace_dir,
        trace_every_send=args.trace_every_send,
        trace_every_recv=args.trace_every_recv,
        trace_every_exec=args.trace_every_exec,
        trace_save_arrays_every=args.trace_save_arrays_every,
        trace_save_images=args.trace_save_images,
        blend_grippers=args.blend_grippers,
        gripper_binary=args.gripper_binary,
        gripper_close_enter=args.gripper_close_enter,
        gripper_open_enter=args.gripper_open_enter,
        gripper_close_confirm=args.gripper_close_confirm,
        gripper_open_confirm=args.gripper_open_confirm,
        gripper_min_close_hold_s=args.gripper_min_close_hold_s,
        gripper_min_open_hold_s=args.gripper_min_open_hold_s,
        gripper_min_close_hold_frames=args.gripper_min_close_hold_frames,
        gripper_deadband=args.gripper_deadband,
        stdin_reset=args.stdin_reset,
        auto_episode_reset=args.auto_episode_reset,
        auto_episode_reset_open_s=args.auto_episode_reset_open_s,
        auto_episode_reset_delay_s=args.auto_episode_reset_delay_s,
    )
    AsyncDeployRunner(cfg).run()


if __name__ == "__main__":
    main()

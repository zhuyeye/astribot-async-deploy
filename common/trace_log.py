"""Structured send/recv/processing traces for debugging async deploy."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from protocol.base import LEFT_GRIPPER_IDX, RIGHT_GRIPPER_IDX

logger = logging.getLogger(__name__)


def _arr_summary(x: np.ndarray | None, *, name: str = "arr") -> dict[str, Any]:
    if x is None:
        return {name: None}
    a = np.asarray(x)
    out: dict[str, Any] = {
        f"{name}_shape": list(a.shape),
        f"{name}_dtype": str(a.dtype),
    }
    if a.size == 0:
        return out
    flat = a.astype(np.float64).ravel()
    out[f"{name}_min"] = float(np.min(flat))
    out[f"{name}_max"] = float(np.max(flat))
    out[f"{name}_mean"] = float(np.mean(flat))
    return out


def _as_list(x: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(x, dtype=np.float64).ravel()]


def _compact_for_console(value: Any) -> Any:
    """Keep console trace readable while preserving full JSONL payloads."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if k.endswith("_full") or k.endswith("_rows"):
                continue
            out[k] = _compact_for_console(v)
        return out
    if isinstance(value, list):
        if len(value) > 8:
            head = value[:3]
            tail = value[-2:]
            return {
                "len": len(value),
                "head": [_compact_for_console(v) for v in head],
                "tail": [_compact_for_console(v) for v in tail],
            }
        return [_compact_for_console(v) for v in value]
    return value


def _cmd25_summary(cmd: np.ndarray, *, prefix: str = "cmd", full: bool = False) -> dict[str, Any]:
    c = np.asarray(cmd, dtype=np.float64).reshape(-1)
    d: dict[str, Any] = {
        f"{prefix}_dim": int(c.shape[0]),
        f"{prefix}_chassis": [float(v) for v in c[0:3]] if c.shape[0] >= 3 else [],
        f"{prefix}_torso": [float(v) for v in c[3:7]] if c.shape[0] >= 7 else [],
        f"{prefix}_arm_l": [float(v) for v in c[7:14]] if c.shape[0] >= 14 else [],
        f"{prefix}_arm_r": [float(v) for v in c[15:22]] if c.shape[0] >= 22 else [],
        f"{prefix}_grip_L": float(c[LEFT_GRIPPER_IDX]) if c.shape[0] > LEFT_GRIPPER_IDX else None,
        f"{prefix}_grip_R": float(c[RIGHT_GRIPPER_IDX]) if c.shape[0] > RIGHT_GRIPPER_IDX else None,
        f"{prefix}_head": [float(v) for v in c[23:25]] if c.shape[0] >= 25 else [],
    }
    if full:
        d[f"{prefix}_full"] = _as_list(c)
    return d


def _chunk_summary(actions: np.ndarray, *, prefix: str = "actions", full: bool = False) -> dict[str, Any]:
    a = np.asarray(actions, dtype=np.float64)
    d = _arr_summary(a, name=prefix)
    if a.ndim == 2 and a.shape[0] > 0 and a.shape[1] >= 25:
        d[f"{prefix}_first"] = _cmd25_summary(a[0], prefix=f"{prefix}_t0", full=full)
        d[f"{prefix}_last"] = _cmd25_summary(a[-1], prefix=f"{prefix}_tN", full=full)
        d[f"{prefix}_grip_L_seq"] = [float(v) for v in a[:, LEFT_GRIPPER_IDX]]
        d[f"{prefix}_grip_R_seq"] = [float(v) for v in a[:, RIGHT_GRIPPER_IDX]]
        if full:
            d[f"{prefix}_rows"] = [_as_list(row) for row in a]
    return d


@dataclass
class TraceConfig:
    enabled: bool = True
    out_dir: str | None = "action_logs/async_trace"
    # Console sampling (JSONL always records every event when enabled).
    every_send: int = 10
    every_recv: int = 1
    every_exec: int = 50
    # Persist full arrays periodically (0 = never).
    save_arrays_every: int = 10
    # Include full 25-d / (T,25) vectors in JSONL (for short verification runs).
    full_vectors: bool = True
    # Also dump RGB images into send_*.npz (heavy).
    save_images: bool = False


class TraceLogger:
    """Thread-safe JSONL + optional NPZ dumps."""

    def __init__(self, config: TraceConfig | None = None) -> None:
        self._cfg = config or TraceConfig()
        self._lock = threading.Lock()
        self._send_i = 0
        self._recv_i = 0
        self._exec_i = 0
        self._ingest_i = 0
        self._dir: Path | None = None
        self._jsonl = None
        if self._cfg.enabled and self._cfg.out_dir:
            self._dir = Path(self._cfg.out_dir)
            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._dir / f"trace_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
            self._jsonl = path.open("a", encoding="utf-8")
            logger.info("Trace logging to %s", path)

    def close(self) -> None:
        with self._lock:
            if self._jsonl is not None:
                self._jsonl.close()
                self._jsonl = None

    def _write(self, event: str, payload: dict[str, Any], *, console: bool = True, level: int = logging.INFO) -> None:
        if not self._cfg.enabled:
            return
        row = {"t": time.time(), "event": event, **payload}
        line = json.dumps(row, ensure_ascii=False, default=str)
        with self._lock:
            if self._jsonl is not None:
                self._jsonl.write(line + "\n")
                self._jsonl.flush()
        if console:
            console_payload = _compact_for_console(payload)
            logger.log(
                level,
                "TRACE %s %s",
                event,
                json.dumps(console_payload, ensure_ascii=False, default=str),
            )

    def _maybe_save_npz(self, name: str, arrays: dict[str, np.ndarray]) -> str | None:
        if self._dir is None:
            return None
        path = self._dir / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        return str(path)

    def _should_save_arrays(self, idx: int) -> bool:
        every = self._cfg.save_arrays_every
        return every > 0 and idx % max(1, every) == 0

    def log_send_obs(
        self,
        *,
        internal_state_25: np.ndarray,
        wire: dict[str, Any],
        obs_timestamp: float,
        prompt: str,
    ) -> None:
        """Always JSONL; console every ``every_send`` frames."""
        self._send_i += 1
        full = self._cfg.full_vectors
        wire_state = wire.get("observation/state", wire.get("state"))
        payload: dict[str, Any] = {
            "idx": self._send_i,
            "obs_timestamp": float(obs_timestamp),
            "prompt": prompt,
            "internal": _cmd25_summary(internal_state_25, prefix="state_sdk", full=full),
        }
        ws = None
        if wire_state is not None:
            ws = np.asarray(wire_state)
            payload["wire"] = _cmd25_summary(ws, prefix="state_wire", full=full) if ws.ndim == 1 else _arr_summary(ws, name="state_wire")
            if ws.ndim == 1 and ws.shape[0] >= 25:
                payload["wire_grip_L"] = float(ws[LEFT_GRIPPER_IDX])
                payload["wire_grip_R"] = float(ws[RIGHT_GRIPPER_IDX])
                # Obs gripper is passed through in SDK units (no /100).
                payload["grip_scale_check_L"] = {
                    "sdk": float(np.asarray(internal_state_25).ravel()[LEFT_GRIPPER_IDX]),
                    "wire": float(ws[LEFT_GRIPPER_IDX]),
                }
        for k in (
            "observation/image",
            "observation/wrist_image",
            "observation/wrist_image_right",
        ):
            if k in wire and hasattr(wire[k], "shape"):
                payload[f"wire_{k}"] = list(np.asarray(wire[k]).shape)
        if "images" in wire and isinstance(wire["images"], dict):
            payload["wire_images"] = {
                ik: list(np.asarray(iv).shape) for ik, iv in wire["images"].items() if hasattr(iv, "shape")
            }
        if self._should_save_arrays(self._send_i):
            arrays: dict[str, np.ndarray] = {
                "state_sdk": np.asarray(internal_state_25, dtype=np.float32),
                "obs_timestamp": np.asarray([obs_timestamp], dtype=np.float64),
            }
            if ws is not None:
                arrays["state_wire"] = np.asarray(ws, dtype=np.float32)
            if self._cfg.save_images:
                for k in (
                    "observation/image",
                    "observation/wrist_image",
                    "observation/wrist_image_right",
                ):
                    if k in wire and hasattr(wire[k], "shape"):
                        arrays[k.replace("/", "_")] = np.asarray(wire[k])
            path = self._maybe_save_npz(f"send_{self._send_i:06d}", arrays)
            if path:
                payload["npz"] = path
        console = self._send_i % max(1, self._cfg.every_send) == 0
        self._write("send_obs", payload, console=console, level=logging.DEBUG)

    def log_recv_actions(
        self,
        *,
        raw_actions: np.ndarray,
        decoded_sdk: np.ndarray,
        obs_timestamp: float,
    ) -> None:
        self._recv_i += 1
        full = self._cfg.full_vectors
        payload: dict[str, Any] = {
            "idx": self._recv_i,
            "obs_timestamp": float(obs_timestamp),
            "raw": _chunk_summary(raw_actions, prefix="raw", full=full),
            "decoded_sdk": _chunk_summary(decoded_sdk, prefix="sdk", full=full),
        }
        # Gripper scale check: wire[0,1] * 100 ≈ sdk.
        raw_a = np.asarray(raw_actions)
        sdk_a = np.asarray(decoded_sdk)
        if raw_a.ndim == 2 and raw_a.shape[1] > RIGHT_GRIPPER_IDX:
            payload["grip_scale_check"] = {
                "raw_L0": float(raw_a[0, LEFT_GRIPPER_IDX]),
                "sdk_L0": float(sdk_a[0, LEFT_GRIPPER_IDX]),
                "raw_L0_x100": float(raw_a[0, LEFT_GRIPPER_IDX]) * 100.0,
                "raw_R0": float(raw_a[0, RIGHT_GRIPPER_IDX]),
                "sdk_R0": float(sdk_a[0, RIGHT_GRIPPER_IDX]),
            }
        if self._should_save_arrays(self._recv_i):
            path = self._maybe_save_npz(
                f"recv_{self._recv_i:06d}",
                {
                    "raw_actions": np.asarray(raw_actions, dtype=np.float32),
                    "decoded_sdk": np.asarray(decoded_sdk, dtype=np.float32),
                    "obs_timestamp": np.asarray([obs_timestamp], dtype=np.float64),
                },
            )
            if path:
                payload["npz"] = path
        console = self._recv_i % max(1, self._cfg.every_recv) == 0
        self._write("recv_actions", payload, console=console)

    def log_ingest(
        self,
        *,
        before_keys: np.ndarray,
        after_keys: np.ndarray,
        obs_timestamp: float,
        t0: float,
        blend_n: int,
        blend_k0: int = 0,
        k_now: int | None = None,
        ingest_now: float | None = None,
        after_blend_keys: np.ndarray | None = None,
    ) -> None:
        self._ingest_i += 1
        full = self._cfg.full_vectors
        payload = {
            "idx": self._ingest_i,
            "obs_timestamp": float(obs_timestamp),
            "traj_t0": float(t0),
            "blend_n": int(blend_n),
            "blend_k0": int(blend_k0),
            "before": _chunk_summary(before_keys, prefix="before", full=full),
            "after": _chunk_summary(after_keys, prefix="after", full=full),
        }
        if ingest_now is not None:
            payload["ingest_now"] = float(ingest_now)
        if k_now is not None:
            payload["k_now"] = int(k_now)
        if after_blend_keys is not None:
            payload["after_blend"] = _chunk_summary(after_blend_keys, prefix="blend", full=full)
        if before_keys.shape[0] and after_keys.shape[0] and before_keys.shape[1] >= 14:
            payload["t0_delta_arm"] = float(np.max(np.abs(after_keys[0, 7:14] - before_keys[0, 7:14])))
        if after_blend_keys is not None and after_blend_keys.shape[0]:
            payload["grip_L_before"] = [float(v) for v in before_keys[:, LEFT_GRIPPER_IDX]]
            payload["grip_L_blend"] = [float(v) for v in after_blend_keys[:, LEFT_GRIPPER_IDX]]
            # after_keys keep continuous grippers; discrete happens on sample.
            payload["grip_L_traj"] = [float(v) for v in after_keys[:, LEFT_GRIPPER_IDX]]
            payload["grip_L_discrete"] = payload["grip_L_traj"]  # back-compat alias
            payload["grip_R_before"] = [float(v) for v in before_keys[:, RIGHT_GRIPPER_IDX]]
            payload["grip_R_traj"] = [float(v) for v in after_keys[:, RIGHT_GRIPPER_IDX]]
        if self._should_save_arrays(self._ingest_i):
            arrays = {
                "before_keys": np.asarray(before_keys, dtype=np.float32),
                "after_keys": np.asarray(after_keys, dtype=np.float32),
                "obs_timestamp": np.asarray([obs_timestamp], dtype=np.float64),
                "traj_t0": np.asarray([t0], dtype=np.float64),
                "blend_n": np.asarray([blend_n], dtype=np.int32),
            }
            if after_blend_keys is not None:
                arrays["after_blend_keys"] = np.asarray(after_blend_keys, dtype=np.float32)
            path = self._maybe_save_npz(f"ingest_{self._ingest_i:06d}", arrays)
            if path:
                payload["npz"] = path
        console = self._ingest_i % max(1, self._cfg.every_recv) == 0
        self._write("ingest_blend", payload, console=console)

    def log_exec(
        self,
        *,
        sampled: np.ndarray,
        after_freeze: np.ndarray,
        after_safety: np.ndarray,
        dry_run: bool,
    ) -> None:
        self._exec_i += 1
        full = self._cfg.full_vectors
        payload = {
            "idx": self._exec_i,
            "dry_run": dry_run,
            "sampled": _cmd25_summary(sampled, prefix="sampled", full=full),
            "after_freeze": _cmd25_summary(after_freeze, prefix="freeze", full=full),
            "after_safety": _cmd25_summary(after_safety, prefix="safe", full=full),
        }
        if self._should_save_arrays(self._exec_i) and self._exec_i % max(1, self._cfg.every_exec) == 0:
            path = self._maybe_save_npz(
                f"exec_{self._exec_i:06d}",
                {
                    "sampled": np.asarray(sampled, dtype=np.float32),
                    "after_freeze": np.asarray(after_freeze, dtype=np.float32),
                    "after_safety": np.asarray(after_safety, dtype=np.float32),
                },
            )
            if path:
                payload["npz"] = path
        console = self._exec_i % max(1, self._cfg.every_exec) == 0
        self._write("exec_cmd", payload, console=console, level=logging.DEBUG)


_GLOBAL: TraceLogger | None = None


def get_tracer() -> TraceLogger | None:
    return _GLOBAL


def set_tracer(tracer: TraceLogger | None) -> None:
    global _GLOBAL
    _GLOBAL = tracer

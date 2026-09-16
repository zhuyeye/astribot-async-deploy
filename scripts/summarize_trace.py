#!/usr/bin/env python3
"""Print a short review of async deploy traces (obs / recv / processed)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _latest_jsonl(trace_dir: Path) -> Path:
    files = sorted(trace_dir.glob("trace_*.jsonl"))
    if not files:
        raise SystemExit(f"no trace_*.jsonl in {trace_dir}")
    return files[-1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trace-dir", default="action_logs/async_trace")
    p.add_argument("--jsonl", default="", help="Explicit jsonl path (default: latest)")
    p.add_argument("--limit", type=int, default=5, help="How many recv rounds to print")
    args = p.parse_args()

    path = Path(args.jsonl) if args.jsonl else _latest_jsonl(Path(args.trace_dir))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"file={path}")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["event"]] = counts.get(r["event"], 0) + 1
    print("counts:", counts)

    sends = [r for r in rows if r["event"] == "send_obs"]
    recvs = [r for r in rows if r["event"] == "recv_actions"]
    ingests = [r for r in rows if r["event"] == "ingest_blend"]
    execs = [r for r in rows if r["event"] == "exec_cmd"]

    print("\n=== uploaded obs (sample) ===")
    for r in sends[: min(3, len(sends))]:
        internal = r.get("internal", {})
        print(
            f"  send#{r['idx']} ts={r['obs_timestamp']:.4f} "
            f"sdk_grip=({internal.get('state_sdk_grip_L')},{internal.get('state_sdk_grip_R')}) "
            f"wire_grip=({r.get('wire_grip_L')},{r.get('wire_grip_R')}) "
            f"npz={r.get('npz')}"
        )
        chk = r.get("grip_scale_check_L")
        if chk:
            print(f"    grip_scale_check_L: {chk}")

    print(f"\n=== received + decoded actions (first {args.limit}) ===")
    for r in recvs[: args.limit]:
        raw = r.get("raw", {})
        sdk = r.get("decoded_sdk", {})
        print(
            f"  recv#{r['idx']} obs_ts={r['obs_timestamp']:.4f} "
            f"shape={raw.get('raw_shape')} "
            f"raw_grip_L={raw.get('raw_grip_L_seq')} "
            f"sdk_grip_L={sdk.get('sdk_grip_L_seq')}"
        )
        print(f"    grip_scale_check={r.get('grip_scale_check')} npz={r.get('npz')}")

    print(f"\n=== after blend (first {args.limit}) ===")
    for r in ingests[: args.limit]:
        print(
            f"  ingest#{r['idx']} blend_n={r.get('blend_n')} "
            f"t0_delta_arm={r.get('t0_delta_arm')} npz={r.get('npz')}"
        )

    print("\n=== exec sample (first few) ===")
    for r in execs[: min(5, len(execs))]:
        safe = r.get("after_safety", {})
        print(
            f"  exec#{r['idx']} dry_run={r.get('dry_run')} "
            f"safe_grip=({safe.get('safe_grip_L')},{safe.get('safe_grip_R')}) "
            f"safe_arm_l0={safe.get('safe_arm_l', [None])[0] if safe.get('safe_arm_l') else None}"
        )

    print(
        "\nCheck: wire_grip ≈ sdk_grip on send (no /100); "
        "sdk_grip ≈ raw_grip*100 on recv; "
        "ingest after differs from before only on early non-gripper frames when blend_n>0."
    )


if __name__ == "__main__":
    main()

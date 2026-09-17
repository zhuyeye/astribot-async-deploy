# Async policy server requirements

This document is for the GPU / policy-server owner. The robot client in
`astribot-async-deploy` speaks an **async dual-channel** WebSocket protocol
(aligned with Kai0 `websocket_policy_asynchronous_astribot`).

A **synchronous** openpi `infer(obs)→action` server is **not** enough.

## Required server behavior

```text
Client A  -- role:sender  -->  Server worker  -->  shared latest obs
Client B  -- role:receiver -->  Server worker  <--  broadcast actions
                                    ^
                                    |
                              main process infer loop
```

1. On connect, first client frame is `{"role": "sender"}` or `{"role": "receiver"}`.
2. **Sender**: streams observation dicts (msgpack+numpy). Latest obs overwrites a
   single buffer. Optional `{"reset": 1}` clears / resets policy.
3. **Infer loop**: when `input_counter` advances, read **current** buffer (latest
   obs; intermediate frames may be skipped), run `policy.infer`, publish actions.
4. **Receiver**: keep connection open; server **broadcasts** each new result:
   ```python
   {
     "actions": np.ndarray,      # (horizon, action_dim)
     "obs_timestamp": float,     # timestamp from the obs that produced this chunk
   }
   ```
5. WebSocket settings: `compression=None`, large/`None` `max_size`.

Reference implementation:
`openpi-client/websocket_reference_code/origin_code/Kai0_s1/server/websocket_policy_asynchronous_astribot.py`

Launch example (Kai0 tree):

```bash
# adjust to your checkout
uv run scripts/serve_policy_astribot_async.py ...
```

## Protocol checklist (please reply with checks)

Client can speak either profile via `--protocol`. **Confirm one** before real-robot runs.

### Profile A — `pi05` (default on robot)

- [ ] Image keys: `observation/image`, `observation/wrist_image`, `observation/wrist_image_right`
- [ ] Images: HWC `uint8`, resized/padded to 224×224 RGB
- [ ] State key: `observation/state`, shape `(25,)`, gripper dims in **`[0,1]`**
- [ ] Action key: `actions`, shape `(T, 25)`, absolute joint targets, gripper **`[0,1]`**
- [ ] `obs_timestamp`: float present on obs and echoed on action broadcast
- [ ] `prompt`: string (required for language-conditioned policies)
- [ ] Port / bind address: ___________
- [ ] Api-Key required? yes/no ___________
- [ ] Server-managed RTC (`prev_action_chunk` / `inference_delay`)? yes/no ___________

### Profile B — `astribot34`

- [ ] Image keys under `images`: `cam_high`, `cam_left_wrist`, `cam_right_wrist`
- [ ] Images: CHW `float32` in `[0,1]`
- [ ] State: shape `(34,)`, dtype float64 — **please paste dim semantics**
- [ ] Actions: shape `(T, 29..34)` — confirm exact `action_dim` and gripper indices/units
- [ ] Mapping of dims 25..33 (extra) for robot execution: keep / drop / freeze?
- [ ] `obs_timestamp` + optional `prompt`
- [ ] RTC enabled? yes/no ___________

### Provisional 25↔34 mapping used by robot client (confirm or correct)

| SDK 25 index | Meaning | Assumed wire index |
|-------------|---------|--------------------|
| 0:3 | chassis | 0:3 |
| 3:7 | torso | 3:7 |
| 7:14 | left arm | 7:14 |
| 14 | left gripper | 14 |
| 15:22 | right arm | 15:22 |
| 22 | right gripper | 22 |
| 23:25 | head | 23:25 |
| — | unused / pad | 25:34 |

Gripper wire units: client auto-detects `[0,1]` vs `[0,100]` on decode.

## Bring-up order

1. Start async server on GPU; `curl` /health if available.
2. On Orin: `python scripts/fake_async_server.py` locally first (sanity of client only), then point client at GPU.
3. Robot: `--fake-sensors --dry-run` against GPU server → expect chunk broadcasts.
4. Robot: real cameras `--dry-run`.
5. Robot: execute with small motion / freeze chassis.

## What the robot does after each chunk

1. Decode `actions` → internal `(T,25)` SDK units (gripper `[0,100]`).
2. `ChunkTrajBridge`: time-align with `obs_timestamp`, **blend non-gripper** across chunks; keep **continuous** gripper keys (no ingest lookahead binarize).
3. Control loop at `--ctrl-hz` samples trajectory; **gripper sticky binary SM runs on sample** (per keyframe advance only) → `set_joints_position` (grippers included).
4. Do not binarize the full playable suffix at ingest — that closes early when H≫ frames actually played (RTC).

## Contact fields for server reply

- Chosen profile: `pi05` / `astribot34`
- Exact action shape: `(T, _) =`
- Control frequency assumed for RTC: ____ Hz
- Example `serve` command you will run:

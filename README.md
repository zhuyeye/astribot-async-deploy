# Async dual-channel S1 deploy

Robot-side (Orin) client: continuous observation stream + continuous action-chunk receive,
then cross-chunk trajectory smoothing and SDK joint execution.

Sibling of `openpi-client`. Communication follows `websocket_reference_code` async
sender/receiver roles. Execution fills the missing RTG / SDK path.

## Layout

| Path | Role |
|------|------|
| `transport/` | Async WebSocket + `role=sender\|receiver` |
| `protocol/` | `pi05` and `astribot34` wire adapters |
| `control/` | `ChunkTrajBridge` (RTG) + high-rate executor |
| `robot/` | Astribot SDK camera / joints |
| `SERVER.md` | Requirements for the GPU async policy server |
| `scripts/fake_async_server.py` | Local dual-channel mock server |

## Install

```bash
cd /home/astribot/workspace/astribot-async-deploy
pip install -r requirements.txt
```

## 1. Local connectivity (no robot)

Terminal A:

```bash
python scripts/fake_async_server.py --port 8000 --protocol pi05
```

Terminal B:

```bash
PYTHONPATH=. python run_async_deploy.py \
  --host 127.0.0.1 --port 8000 \
  --protocol pi05 --fake-sensors --dry-run --max-runtime 5 -v
```

Expect logs with `Ingested chunk` / `chunks_seen>0`.

## 2. Real robot dry-run

```bash
./scripts/start_async.sh \
  --host <GPU_IP> --port 8000 \
  --protocol pi05 \
  --prompt "pick up the orange plush toy to the basket" \
  --dry-run -v
```

## 3. Real robot execute

```bash
./scripts/start_async.sh \
  --host <GPU_IP> --port 8000 \
  --protocol pi05 \
  --prompt "pick up the orange plush toy to the basket" \
  --ctrl-hz 250 --blend-s 0.12 --freeze-chassis
```

## Protocol switch

- `--protocol pi05` (default): openpi `observation/*`, action `(T,25)`, gripper wire `[0,1]`
- `--protocol astribot34`: `images.cam_*`, state `(34,)`, action `(T,29..34)` — confirm with server via `SERVER.md` checklist first

## Gripper policy (v1)

No binary / effector post-process. Gripper travels with arms via
`set_joints_position(..., control_way=direct)` after `[0,1]→[0,100]` conversion.
Cross-chunk blend applies to non-gripper joints only; grippers use step-hold.

## Tests

```bash
PYTHONPATH=. pytest -q
```

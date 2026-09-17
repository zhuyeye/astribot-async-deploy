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
  --send-hz 10 --chunk-hz 30 --ctrl-hz 250 --blend-s 0.12 --freeze-chassis
```

### Proven Costa handover (Orin → `192.168.0.100`)

RTC + causal gripper post (defaults below). Server horizon H=20 is fine with this client.

```bash
./scripts/start_async.sh \
  --host 192.168.0.100 --port 8003 \
  --protocol pi05 \
  --send-hz 10 --chunk-hz 30 --ctrl-hz 250 \
  --blend-s 0.12 --freeze-chassis \
  --gripper-binary \
  --gripper-close-enter 70 --gripper-open-enter 40 \
  --gripper-close-confirm 2 --gripper-open-confirm 5 \
  --gripper-min-close-hold-s 1.2 --gripper-min-open-hold-s 0.35
```

## Protocol switch

- `--protocol pi05` (default): openpi `observation/*`, action `(T,25)`, gripper wire `[0,1]`
- `--protocol astribot34`: `images.cam_*`, state `(34,)`, action `(T,29..34)` — confirm with server via `SERVER.md` checklist first

## Gripper policy (locked)

Binary sticky hysteresis on SDK `[0,100]` after `[0,1]→[0,100]` decode.

| Knob | Default | Notes |
|------|---------|--------|
| `--gripper-binary` | on | `{0,100}` open/close |
| `--gripper-close-enter` / `--open-enter` | 70 / 40 | Schmitt band |
| `--gripper-close-confirm` / `--open-confirm` | 2 / 5 | keyframe counts |
| `--gripper-min-close-hold-s` / `--min-open-hold-s` | 1.2 / 0.35 | one-shot edge holds |
| `--blend-grippers` | off | arms blend; grippers do not |

**Causal (important):** ingest only stores continuous gripper targets (arm blend
unchanged). The sticky SM advances on **sample**, only when the active keyframe
index moves — current + history, never the unplayed chunk tail. Ingest-time
`process_keyframes` on `[k_now:]` caused H=20 early grasp under RTC
(`send-hz≈10` plays ~3 frames/chunk while close sat at frames 17–19).

Arms still cross-chunk blend (`--blend-s`); grippers ZOH discrete after the SM.

## Tests

```bash
PYTHONPATH=. pytest -q
```

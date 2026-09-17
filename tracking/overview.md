---
id: astribot-async-deploy
status: active
updated: 2026-09-17
phase: rtc-live-validation
related:
  - openpi-client
---

# Overview

## 项目用途

Orin 侧（机器人端）异步双通道部署客户端：持续发送 observation，持续接收
action chunk，做跨 chunk 轨迹平滑后经 Astribot SDK 执行关节指令。

与 `openpi-client` 配套；通信对齐 `websocket_reference_code` 的 async
sender/receiver 角色。同步 `infer(obs)→action` 服务端不够用，需见
`SERVER.md`。

## 当前阶段

阶段：`rtc-live-validation`（真机 / RTC 联调与验证）。

完成标准：对选定协议（默认 `pi05`）能稳定跑通
fake → dry-run → 真机执行；chunk 时序、blend、夹爪策略可复现且有测试覆盖；
服务端 checklist（`SERVER.md`）与实际 GPU 服务对齐。

## 目标

- [x] 异步双通道 transport（sender / receiver）
- [x] `pi05` / `astribot34` 协议适配
- [x] `ChunkTrajBridge` + 高频率 executor
- [x] fake server + 本地连通性验证路径
- [ ] 与 GPU async policy server 的正式联调 checklist 闭环
- [ ] 真机 RTC / live 行为稳定可复现（含 trace 分析）
- [x] 夹爪策略与协议单位约定文档化并锁定（因果 SM + Costa 推荐参数）

## 工作主线

- 实现 / 开发: `transport/`、`protocol/`、`control/`、`robot/`、`runner.py`
- 数据 / 输入: 相机 + 关节状态；action chunk + `obs_timestamp`
- 验证 / 运维: `scripts/fake_async_server.py`、`pytest`、`action_logs/`、`summarize_trace.py`
- 文档 / 笔记: `README.md`、`SERVER.md`、本目录 `tracking/`

## 里程碑

- 已完成: 客户端骨架、双协议适配、RTG blend、fake e2e、多项单测、trace 日志
- 进行中: RTC / live 真机验证（见 `action_logs/rtc_*`、`live_*`）
- 下一步: 服务端协议 checklist 确认；固化稳定 launch 参数与回归用例

## 入口与参考

- 对外说明: [README.md](../README.md)
- 服务端要求: [SERVER.md](../SERVER.md)
- CLI: `run_async_deploy.py`、`scripts/start_async.sh`
- 本地 mock: `scripts/fake_async_server.py`
- 深度笔记 / handoff: 本目录 `current.md` + `daily/`

# Decisions

## 2026-09-17 - 夹爪状态机改到 sample 因果推进（禁止 ingest 展望）

- Context: Costa handover、H=20、RTC、`send-hz≈10` 时每块只播 ~3 帧；ingest 对 `[k_now:]` 跑 GripperPost 会吃到 frame 17–19 的关，下一块 EXEC 已是 100，手臂仍在大位移 → 提前夹。
- Decision: ingest 只存连续夹爪；`GripperPost` 仅在 `sample` 且 **keyframe 下标前进** 时推进一步。默认滞回参数锁定为 close/open enter 70/40、confirm 2/5、hold 1.2s/0.35s；`blend_grippers=False`。真机推荐 `192.168.0.100:8003`、`--send-hz 10 --chunk-hz 30 --blend-s 0.12` + RTC。
- Why: 语义应为「当前+历史」；smoke 回放同一 trace 首次闭合从 ingest#54 推迟到 #63（~0.9s），与体感一致。H=10 几乎无此病，根因是展望而非模型从不关。
- Follow-up: 勿回退到 suffix 批处理；臂速不均优先当策略节奏（只播 chunk 头），不是夹爪回归。

## 2026-08-21 - 使用根目录 `tracking/` 做进展跟踪

- Context: 需要可提交、可跨会话恢复的工作进展记录；仓库里已有 `action_logs/` 存放运行时 trace。
- Decision: 在仓库根目录建立 `tracking/`，不放在 `docs/` 或 `notes/`；日志正文默认中文。
- Why: 与 repo-tracking skill 约定一致；与 `action_logs/` 的运行时日志职责分离，避免混淆。
- Follow-up: 之后实质工作结束时更新 `current.md` 与当天 daily；阶段/目标变化时再改 `overview.md`。

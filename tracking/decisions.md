# Decisions

## 2026-08-21 - 使用根目录 `tracking/` 做进展跟踪

- Context: 需要可提交、可跨会话恢复的工作进展记录；仓库里已有 `action_logs/` 存放运行时 trace。
- Decision: 在仓库根目录建立 `tracking/`，不放在 `docs/` 或 `notes/`；日志正文默认中文。
- Why: 与 repo-tracking skill 约定一致；与 `action_logs/` 的运行时日志职责分离，避免混淆。
- Follow-up: 之后实质工作结束时更新 `current.md` 与当天 daily；阶段/目标变化时再改 `overview.md`。

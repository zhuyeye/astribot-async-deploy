---
status: active
updated: 2026-09-17
phase: rtc-live-validation
related:
  - openpi-client
---

# Current

- 当前焦点: Costa handover 真机 RTC；夹爪已改为 sample 因果 SM，并通过 `orin100_send10_p8003_rtc` 复测。
- 为什么现在做: H=20 提前夹已定位为 ingest 展望；臂速快慢不均经 trace 判定主要为策略节奏。
- 下一步: 保持推荐 launch 参数做回归；按需调 `blend-s` / send-hz 抹平跨块手感（非夹爪）。
- 阻塞: None
- 最近日志: [2026-09-17](daily/2026-09-17.md)
- 锁定参数: 见 `README.md`「Proven Costa handover」与 `.cursor/rules/gripper-rtc.mdc`

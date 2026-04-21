# Phase 13: Experiment Section Draft

Date: 2026-04-21

## Goal

承接 Phase 12 的正式结果盘点，把第 4 周成果转成论文实验章节草稿，而不是继续停留在日志和表格层面。

## Output

新增论文材料：

```text
docs/毕设报告/实验章节草稿.md
```

该草稿包括：

- 实验设置：TEACh、`|h|` 与 `--n-samples` 的区别、主指标定义。
- 主结果表：复用 Phase 12 中已跑满 100 QA 的 h=50/h=100 可追溯结果。
- 小规模稳定性诊断：记录 h=15 n=15 Active-H-EMV diagnostic 的 `S_c/S_p/T/valid`。
- 消融观察：baseline vs graph、zero-shot vs few-shot、结构化 guardrail。
- 案例分析：图邻居扩展、无记录对象提前停止、时间事件结构化回答、修正/遗忘模块。
- 威胁与限制：API 不稳定、summary cache 缺口、缺少 h=100 graph 正式结果、修正/遗忘缺少大规模量化。

## Source Records

- `docs/record/phase12_week4_result_inventory.md`
- `docs/record/phase10_h15_auto_eval_and_error_analysis.md`
- `docs/record/phase11_when_event_lookup_fix.md`
- `docs/record/phase9_h15_task_tools_diagnostics.md`
- `docs/record/phase3_correction_api.md`
- `docs/record/phase2_forgetting_smoke.md`

## Key Writing Decisions

- 主表只使用已跑满 100 QA 且有 auto-eval 的正式结果。
- h=15 n=15 只作为 diagnostic / guardrail 证据，不写成正式规模化结果。
- 不夸大 Active-H-EMV full system：当前大规模主表支撑最强的是 Graph-H-EMV；Active guardrail、修正、遗忘以小样本和机制案例支撑。
- 明确写出缺少 `|h|=100` Graph-H-EMV 正式结果，避免答辩时被追问过度声明。

## Next

1. 将 `docs/毕设报告/实验章节草稿.md` 合并进正式论文正文结构。
2. 继续写方法章节中的公式和系统图说明。
3. 如果还有预算，只做 h=15 targeted temporal ambiguity 小修；不默认启动新的 h>=25 大实验。

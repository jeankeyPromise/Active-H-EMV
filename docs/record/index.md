# Active-H-EMV 阶段记录索引

本文档用于防止项目上下文漂移。后续每次关键代码修改、实验运行、指标整理或论文材料更新后，都应新增或更新一条 `docs/record/phase*.md` 记录。

## 已有阶段记录

| 阶段 | 文件 | 核心内容 | 当前结论 |
| --- | --- | --- | --- |
| Phase 0 | `phase0_preflight.md` | 环境、数据、项目恢复预检 | 基础环境与数据路径已恢复 |
| Phase 1 | `phase1_graph_smoke.md` | 图构建与图增强检索 smoke | 图结构和检索链路可运行 |
| Phase 2 | `phase2_forgetting_smoke.md` | 遗忘/巩固模块 smoke | 效用遗忘逻辑已具备基础验证 |
| Phase 3 | `phase3_correction_api.md` | 修正模块与 API 验证 | 修正链路可运行，但依赖稳定 Judge/API |
| Phase 4 | `phase4_metrics_comparison.md` | h=50 graph/baseline pilot 对比 | 图增强在 `S_c` 上有提升，token 与 API 稳定性仍需控制 |
| Phase 5 | `phase5_guardrails_checkpoint_smoke.md` | no-record、checkpoint、resume、空回复控制 | 失败恢复和 guardrail 已改善 |
| Phase 6 | `phase6_temporal_neighbor_h15_n50.md` | h=15 temporal neighbor smoke | QA token 未爆炸，发现在线摘要隐藏成本 |
| Phase 7 | `phase7_completion_plan_implementation.md` | 毕设完成计划首批落地 | 新增 date/event lookup、VQA guard、记录规范；h=15 n=2 smoke T=1.79K |
| Phase 8 | `phase8_graph_trace_and_cache_workflow.md` | 图检索案例日志与缓存预构建流程 | 新增 graph trace JSONL 与显式 history cache 预构建入口 |
| Phase 9 | `phase9_h15_task_tools_diagnostics.md` | h=15 task tools 与稳定性诊断 | 新增 task_list/task_lookup/object_lookup；h=15 n=15 最终 15/15 有效，平均 prompt 2.51K |
| Phase 10 | `phase10_h15_auto_eval_and_error_analysis.md` | h=15 n=15 自动语义评估与错误分析 | `S_c=46.7%`, `S_p=20.0%`, valid=100%, `T=2.70K`；下一步修 when/event 与 temporal ambiguity |
| Phase 11 | `phase11_when_event_lookup_fix.md` | `when did you X` 事件日期修复 | targeted sample 从 wrong 变 correct，`T=1.72K`，过滤 plant 误匹配 |
| Phase 12 | `phase12_week4_result_inventory.md` | 第 4 周正式结果盘点与论文主表草案 | 复算 h=50/h=100 正式主指标；新增 `scripts/evaluate_result.sh` 固化 smoke 后 correctness evaluation |
| Phase 13 | `phase13_experiment_section_draft.md` | 论文实验章节草稿 | 将主表、h=15 诊断、三类案例和限制整理成可进入正文的实验章节 |
| Phase 14 | `phase14_h50_cached_probe.md` | h=50 缓存前缀可行性预检 | 新增 history cache audit/require guard；h=50 n=10 cached probe valid=100%, `S_c=80%`, `T=2.25K`；暂不继续 n=60 |
| Phase 15 | `phase15_h50_n20_cached_probe.md` | h=50 n=20 缓存前缀诊断 | n=20 valid=100%, `S_c=70%`, `S_p=25%`, `T=2.26K`；新增 answer 语法回退和 temporal 位置/整体任务约束 |
| Phase 16 | `phase16_h50_n40_cached_probe.md` | h=50 n=40 缓存前缀诊断 | n=40 valid=100%, `S_c=55%`, `S_p=25%`, `T=2.26K`；修复 pillow/sofa task_lookup runaway，下一步收紧 temporal/event 约束 |

## 固定实验口径

- `|h|` 表示每段长历史包含的基础情景个数，不等于 `--n-samples`。
- 标准 TEACh 表格每个 `|h|` 应跑满 10 段长历史 × 10 个问答 = 100 QA。
- `--n-samples` 只用于 pilot/smoke 截断，不能把 `--n-samples 50` 写成标准 `|h|=50` 结果。
- 论文主指标优先使用 `S_c`、`S_p`、`T`、valid answer rate。
- BLEU/ROUGE/METEOR 只作为辅助表面指标，不作为本文核心有效性结论。

## 实验安全规则

- 未经再次确认，不启动新的 `h=25`、`h=50`、`h=100` 大规模实验。
- 调参默认使用 `data/teach/test_set_15.pkl`，先跑 `n-samples=2/5/15`。
- 任何实验都应输出到项目内，例如 `experiments/results/teach/smoke/` 或 `experiments/results/teach/metrics/`。
- 运行时应优先使用 `--resume`、`--retry-errors`、`--max-prompt-tokens-per-sample`、`--max-average-prompt-tokens-per-sample`、`--max-seconds-per-sample`。
- 如果发现在线摘要构建、空回复重试、无效 VQA 或重复搜索导致 token 异常，应暂停并记录，不继续硬跑。
- h=15 `n-samples=15` 或类似 smoke 跑完后，使用 `scripts/evaluate_result.sh` 同步运行 correctness evaluation 和 `calc_metrics --primary-only`。

## 每次记录建议包含

- 修改目标。
- 涉及文件。
- 运行命令。
- 结果文件。
- 指标摘要。
- 异常与原因。
- 下一步建议。

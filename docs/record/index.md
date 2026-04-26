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
| Phase 10 | `phase10_h15_auto_eval_and_error_analysis.md` | h=15 n=15 自动语义评估与错误分析 | `S_c=46.7%`, `S_p=66.7%`, valid=100%, `T=2.70K`；下一步修 when/event 与 temporal ambiguity |
| Phase 11 | `phase11_when_event_lookup_fix.md` | `when did you X` 事件日期修复 | targeted sample 从 wrong 变 correct，`T=1.72K`，过滤 plant 误匹配 |
| Phase 12 | `phase12_week4_result_inventory.md` | 第 4 周正式结果盘点与论文主表草案 | 复算 h=50/h=100 正式主指标；新增 `scripts/evaluate_result.sh` 固化 smoke 后 correctness evaluation |
| Phase 13 | `phase13_experiment_section_draft.md` | 论文实验章节草稿 | 将主表、h=15 诊断、三类案例和限制整理成可进入正文的实验章节 |
| Phase 14 | `phase14_h50_cached_probe.md` | h=50 缓存前缀可行性预检 | 新增 history cache audit/require guard；h=50 n=10 cached probe valid=100%, `S_c=80%`, `T=2.25K`；暂不继续 n=60 |
| Phase 15 | `phase15_h50_n20_cached_probe.md` | h=50 n=20 缓存前缀诊断 | n=20 valid=100%, `S_c=70%`, `S_p=95%`, `T=2.26K`；新增 answer 语法回退和 temporal 位置/整体任务约束 |
| Phase 16 | `phase16_h50_n40_cached_probe.md` | h=50 n=40 缓存前缀诊断 | n=40 valid=100%, `S_c=55%`, `S_p=80%`, `T=2.26K`；修复 pillow/sofa task_lookup runaway，下一步收紧 temporal/event 约束 |
| Phase 17 | `phase17_n40_precision_fixes.md` | n=40 暴露问题的精度修复 | 收紧 event/date 与 temporal target 约束；修复 tomato-bowl、armchair 多报；low-action 仍需 raw-action 辅助索引 |
| Phase 18 | `phase18_h50_n40_patched_probe.md` | h=50 n=40 patched probe | n=40 valid=100%, `S_c=62.5%`, `S_p=80.0%`, `T=2.37K`；precision patch 净提升，瓶颈转向 low-action/raw-action 召回 |
| Phase 19 | `phase19_raw_action_lookup_probe.md` | raw-action lookup 与 h=50 n=40 action probe | low-action targeted n=4 valid=100%, `S_c=25%`, `S_p=100%`, `T=2.16K`；n=40 action_v1 valid=100%, `S_c=55%`, `S_p=75%`, `T=2.31K`，未超过 Phase 18，已补 pillow/sofa v1.1 误路由 |
| Phase 20 | `phase20_h50_action_v12_probe.md` | h=50 动作 v1.2 提示词探测 | 定向 n=5 valid=100%, `S_c=60%`, `T=2.08K`；n=40 valid=100%, `S_c=60%`, `S_p=80%`, `T=2.29K`，收复了 action_v1 的部分退化，但仍未超过 Phase 18 |
| Phase 21 | `phase21_h50_action_v13_n20.md` | h=50 动作 v1.3 稳定性探测 | n=20 valid=100%, `S_c=60%`, `S_p=85%`, `T=2.35K`；`clean all` 约束更稳，但未超过 Phase 15/18，目前更适合作为诊断分支 |
| Phase 22 | `phase22_h50_action_v13_n60_resume.md` | h=50 动作 v1.3 n=60 续跑完成 | 从 `46/60` 安全续跑到 `60/60`；valid=100%, `S_c=53.3%`, `S_p=75.0%`, `T=2.45K`；低层动作结构化直答护栏有效消除了 token runaway |
| Phase 23 | `phase23_h50_fullqa_cache_completion.md` | h=50 全量 QA 缓存补齐与正式评测 | 先修复短对象 `cd` 路径，再将 `50ep` history cache 从 `6/10` 补齐到 `10/10`，最终 full-QA `100/100` 完成；valid=100%, `S_c=48.0%`, `S_p=72.0%`, `T=2.06K` |
| Phase 24 | `phase24_h100_fullqa_cache_completion.md` | h=100 全量 QA 缓存补齐与正式评测 | 先将 `100ep` history cache 从 `0/10` 补齐到 `10/10`，再完成 full-QA `100/100`；valid=98.0%, `S_c=49.0%`, `S_p=75.0%`, `T=2.29K`，相对原文 `h=100` 基线仍显著占优 |
| Phase 25 | `phase25_armarx_forgetting_guarded_probe.md` | ARMARX 遗忘模块受控探测 | 遗忘模块已能压缩 ARMARX 记忆树；`ubpf_ultra` 将文件比例压到 `0.797`，但 one-pass prompt 仍高达 `220,648` tokens，说明遗忘应与层级检索配合评估，而不能只依赖 full-history flatten |
| Phase 26 | `phase26_armarx_forgetting_prompt_stats.md` | ARMARX 遗忘模块格式化历史长度统计 | 在不新增 LLM 调用的前提下，统计 one-pass 格式化历史长度；`random_medium` 压到基线的 `98.73%`，`medium/aggressive` 为 `95.44%`，`ultra` 为 `94.61%`，再次说明遗忘有效但无法单独解决 one-pass flatten 的长上下文瓶颈 |
| Phase 27 | `phase27_armarx_l2_detail_probe.md` | ARMARX L2 事件层细节题受控探测 | L2 事件层视图下 `base/random/medium/medium_graph/ultra` 的格式化输入完全一致；只跑 baseline 即可。细节题 `n=6` 上 valid=`83.3%`、`S_c=16.7%`、`T=45.66K`，说明仅保留到事件层不足以支撑大多数细节敏感 QA |
| Phase 28 | `phase28_armarx_evidence_audit.md` | ARMARX 遗忘模块关键证据保真审计 | 在零新增 LLM token 下审计 question-local evidence；`medium_graph` 比 `medium/ultra` 更能保留 relation-rich 过程事件。117 个高细节密度事件上，`mean relations` 分别为 `130.22` vs `112.63/111.82`，说明图中心性版本更保守、更利于细节保真 |
| Phase 29 | `phase29_armarx_local_detail_probe.md` | ARMARX 局部低层细节视图单题探测 | 以 `a7a-merged-objects-5` 为单题 probe，只给局部最新 scene graph；`base` 已能回答出包含 `OrangeJuice` 的关系性线索，而 `medium/ultra` 因 relations 被清空而直接拒答，说明局部低层视图能把遗忘差异真实传递到 QA |
| Phase 30 | `phase30_armarx_local_detail_slice.md` | ARMARX 局部低层细节切片实验 | 5 个超小局部 probe 继续验证机制；在前 4 个 relation-sensitive probe 上，`base=4/4`、`random_medium=3/4`、`medium=0/4`、`medium_graph=4/4`、`ultra=0/4`，说明 `medium_graph` 对细节保真明显优于 `medium/ultra`，且随机遗忘表现不稳定 |

## 固定实验口径

- `|h|` 表示每段长历史包含的基础情景个数，不等于 `--n-samples`。
- 标准 TEACh 表格每个 `|h|` 应跑满 10 段长历史 × 10 个问答 = 100 QA。
- `--n-samples` 只用于 pilot/smoke 截断，不能把 `--n-samples 50` 写成标准 `|h|=50` 结果。
- 论文主指标优先使用 `S_c`、`S_p`、`T`、valid answer rate。
- 其中 `S_p` 表示“至少部分正确率”，即 `correct + partial`，不是“仅部分正确”。
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

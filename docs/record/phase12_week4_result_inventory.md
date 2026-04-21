# Phase 12: Week 4 Result Inventory and Thesis Table Draft

Date: 2026-04-21

## Goal

进入规划中的第 4 周：不启动新的 h=25/50/100 大实验，优先复用已有正式结果和 h=15 诊断结果，整理论文主表草案、消融证据和案例素材。

同时把 h=15 `--n-samples 15` 这类 smoke 后续的 correctness evaluation 固化为标准流程。

## Current Progress

- 第 1 周“冻结工程主线与实验口径”已完成：阶段记录入口、主指标口径、checkpoint/resume/token budget 已建立。
- 第 2 周“低风险工程问题”已基本完成：`date_lookup` / `event_date_lookup`、task/object lookup、VQA guard、graph trace、cache workflow 都已落地并 smoke。
- 第 3 周“h=15 小规模稳定性实验”已完成到可交付 checkpoint：
  - h=15 n=15：valid=100%，`S_c=46.7%`，`S_p=20.0%`，`T=2.70K`。
  - `when did you X` targeted fix：wrong -> correct，`T=1.72K`。
  - h=15 cache 仍不完整，所以不默认继续 h=15 n=50。
- 当前进入第 4 周：正式结果整理、论文主表、最小消融和案例材料。

## Evaluation Helper

新增：

```text
scripts/evaluate_result.sh
```

用途：结果 JSON 生成后，一步运行语义 correctness evaluation 和论文主指标统计。

```bash
PYTHON_BIN="conda run --no-capture-output -n active-h-emv python" \
  scripts/evaluate_result.sh experiments/results/teach/smoke/task_tools_h15_n15.json
```

默认使用：

```text
llm_emv/config/llm_eval/gemini_2.5_pro.yaml
```

并写出：

```text
<result_stem>_llm_eval.log
<result_stem>_metrics.log
```

已同步更新 `docs/Experiment Design/Command.md`。后续 h=15 n=15 或类似 smoke 跑完后，应立即运行这个 helper，避免只记录 token/valid rate 而遗漏 correctness。

## Recomputed Formal Metrics

本阶段只复用已有 auto-eval 文件，运行 `calc_metrics --primary-only` 复算主指标，没有重新调用 LLM judge。

```bash
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval.metrics.calc_metrics --primary-only \
  experiments/results/teach/h_emv_50.json

conda run --no-capture-output -n active-h-emv python -m llm_emv.eval.metrics.calc_metrics --primary-only \
  experiments/results/teach/h_emv_graph_aug_50_zs.json

conda run --no-capture-output -n active-h-emv python -m llm_emv.eval.metrics.calc_metrics --primary-only \
  experiments/results/teach/h_emv_graph_aug_50_fs.json

conda run --no-capture-output -n active-h-emv python -m llm_emv.eval.metrics.calc_metrics --primary-only \
  experiments/results/teach/h_emv_gemini_2.5_pro_100_fs.json

conda run --no-capture-output -n active-h-emv python -m llm_emv.eval.metrics.calc_metrics --primary-only \
  experiments/results/teach/test_set/gemini_2.5_pro/zs/h_emv_gemini_2.5_pro_100.json
```

| Scope | Method / Config | Result file | Auto-eval file | QA | Valid | `S_c` | `S_p` | Wrong / no-answer | `T` prompt K |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `|h|=50` formal | H-EMV baseline fs | `experiments/results/teach/h_emv_50.json` | `h_emv_50.gemini_2.5_pro-f60fb2.auto_eval.json` | 100 | 96.0% | 41.0 | 27.0 | 32.0 | 27.63 |
| `|h|=50` formal | Graph-H-EMV zs | `experiments/results/teach/h_emv_graph_aug_50_zs.json` | `h_emv_graph_aug_50_zs.gemini_2.5_pro-20bf9e.auto_eval.json` | 100 | 100.0% | 43.0 | 23.0 | 34.0 | 5.19 |
| `|h|=50` formal | Graph-H-EMV fs | `experiments/results/teach/h_emv_graph_aug_50_fs.json` | `h_emv_graph_aug_50_fs.gemini_2.5_pro-20bf9e.auto_eval.json` | 100 | 98.0% | 44.0 | 23.0 | 33.0 | 16.15 |
| `|h|=100` formal | H-EMV baseline zs | `experiments/results/teach/test_set/gemini_2.5_pro/zs/h_emv_gemini_2.5_pro_100.json` | `h_emv_gemini_2.5_pro_100.gemini_2.5_pro-3bb6e1.auto_eval.json` | 100 | 90.0% | 42.0 | 27.0 | 31.0 | 7.08 |
| `|h|=100` formal | H-EMV baseline fs | `experiments/results/teach/h_emv_gemini_2.5_pro_100_fs.json` | `h_emv_gemini_2.5_pro_100_fs.gemini_2.5_pro-f60fb2.auto_eval.json` | 100 | 95.0% | 38.0 | 21.0 | 41.0 | 29.85 |

## Thesis Main Table Draft

建议论文主表先采用“可追溯、跑满 100 QA”的结果：

| Method | `|h|` | Shot | Valid | `S_c` | `S_p` | `T` prompt K | Suggested use |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| H-EMV baseline | 50 | fs | 96.0% | 41.0 | 27.0 | 27.63 | baseline 对照 |
| Graph-H-EMV | 50 | zs | 100.0% | 43.0 | 23.0 | 5.19 | 主推：低成本且正确率提升 |
| Graph-H-EMV | 50 | fs | 98.0% | 44.0 | 23.0 | 16.15 | 消融：few-shot 对图增强略增正确率但成本变高 |
| H-EMV baseline | 100 | zs | 90.0% | 42.0 | 27.0 | 7.08 | 长历史 baseline 趋势 |
| H-EMV baseline | 100 | fs | 95.0% | 38.0 | 21.0 | 29.85 | few-shot 长历史反而更贵且不优 |

可写结论：

- 图增强在 `|h|=50` 上相对 baseline 提升完全正确率：`41 -> 43/44`。
- zero-shot graph 的 token 成本显著低于 h=50 few-shot baseline：`5.19K vs 27.63K`。
- few-shot 不稳定地提升或降低 correctness，且显著增加 token，论文中应作为对照而不是默认主配置。
- `|h|=100` 当前只有 baseline 可追溯结果，没有同规模 graph 正式结果；不能声称 graph 在 h=100 上已验证。

## Diagnostic / Pilot Evidence

这些结果不放入论文主表，但可作为工程稳定性和消融案例：

| Purpose | File | QA | Result |
| --- | --- | ---: | --- |
| h=15 Active-H-EMV diagnostic | `experiments/results/teach/smoke/task_tools_h15_n15.json` | 15 | valid=100%，`S_c=46.7%`，`S_p=20.0%`，`T=2.70K` |
| h=15 `when did you X` fix | `experiments/results/teach/smoke/event_when_lookup_h15_tasks_to_exact_times_n1.json` | 1 | wrong -> correct，`T=1.72K` |
| h=15 no-record object guard | `experiments/results/teach/smoke/object_lookup_h15_object_n1.json` | 1 | `armchair` 无记录时直接回答 “No, I have no record of that.” |
| h=15 graph trace | `experiments/results/teach/traces/graph_trace_h15_direct_search.jsonl` | offline trace | 48 条 trace 中 9 条出现 `expanded_indices` |

## Case Study Candidates

1. 图邻居扩展帮助召回：
   - `graph_trace_h15_direct_search.jsonl` line 2：`sliced tomato` 从 seed item 0 通过 `similar_action` 扩展到 item 2。
   - line 27：`cook potato` 从 March 14 seed 通过 `co_object` 扩展到 March 13 的完整餐食任务。
   - line 30：`dirty mug` 通过 `similar_action` 扩展到另一日的清洗/整理记录。

2. 无记录/低相似度提前停止：
   - `object_lookup_h15_object_n1.log`：`object_lookup('armchair')` 没有 task-sized summary 命中，直接推荐 negative answer，避免继续 synonym search / VQA。

3. 时间事件结构化工具：
   - `event_when_lookup_h15_tasks_to_exact_times_n1.log`：`event_date_lookup('When did you water the plant?')` 返回 2024/03/13、2024/03/15、2024/03/16，并给出 exact-date recommended answer。

4. 修正/遗忘模块：
   - 修正模块已有 `docs/record/phase3_correction_api.md` 的真实 API 链路验证。
   - 遗忘模块已有 `docs/record/phase2_forgetting_smoke.md` 的 isolated smoke。
   - 第 4 周论文案例中只做闭环示例，不强行补昂贵大规模 correction experiment。

## Risks and Boundaries

- h=15 cache 在前 30 个 QA 左右出现缺失并开始在线 summarization；不要自动继续 h=15 n=50。
- Phase 4 中 `--n-samples 50` 的 merged pair 是 50-QA pilot，不作为标准主表结果；可用于趋势和案例。
- `|h|=100` 缺少 Graph-H-EMV 正式 100 QA 结果。若论文时间和 token 预算不允许补跑，应明确写成未来工作/限制。
- Active-H-EMV 当前完整版本的大规模主表结果尚未跑满 100 QA。论文主表可以用 Graph-H-EMV 正式结果支撑核心图增强贡献，用 h=15 diagnostic 支撑 guardrail/active 工程改进。

## Next

1. 开始写论文实验章节草稿：先用本文件的主表草案，不再临时翻旧日志。
2. 为三类案例各整理一段可直接进入论文的 “query -> retrieval evidence -> answer/result” 描述。
3. 如要补最小消融，优先做 h=15 targeted before/after temporal ambiguity 修复；不默认启动 h=25/50/100 新实验。
4. 保持每个新实验后运行 `scripts/evaluate_result.sh`，并把 metrics log 路径写入 `docs/record/`。

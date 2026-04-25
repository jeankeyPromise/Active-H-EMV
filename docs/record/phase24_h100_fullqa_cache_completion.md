# Phase 24: h=100 全量 QA 缓存补全与 100-QA 运行

## 目标

在已完成的 `h=50` full-QA 结果基础上，把第 4 周实验主线推进到 `h=100`，同时保持相同的运行纪律：

- 先解释并稳定 cache 工作流；
- 在 QA 前补齐所有缺失的 `100ep` multi-history cache；
- 使用仅 cache 的执行方式完成完整 `100 QA`；
- 运行完成后立即进行正确性评估，并记录论文表格所需的最终指标。

本阶段是当前 Active-H-EMV 分支上的首个正式 `h=100` full-QA 结果。

## Cache 工作流

补充文档：

- `docs/论文写作准备/cache原理说明.md`

该文档解释了：

1. 为什么需要 multi-history cache；
2. 为什么缺失 cache 会悄悄触发在线 summarization；
3. 为什么 `--audit-history-cache` 与 `--require-history-cache` 是必要护栏；
4. 为什么逐 history、带边界的 cache 填充比一次性整体 precompute 更安全。

## 缓存补全

初始审计：

- selected histories: `10`
- cached histories: `0`
- missing histories: `10`

安全补全策略：

- 数据集：`data/teach/test_set_100.pkl`
- 配置：`teach/simplified/full_graph_aug_zs_fast`
- 通过 `--skip-first-n-episodes <ep>` 与 `--n-samples 10` 做逐 history 的有界 precompute
- summarizer 模型：`gemini-2.5-pro`
- summarizer prompt 轻量化：`few_shot_k=0`

代表性命令：

```bash
set -a
source .env
set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_100.pkl \
  --output experiments/results/teach/smoke/h100_cache_fill_ep07_placeholder.json \
  --precompute-history-cache \
  --skip-first-n-episodes 7 \
  --n-samples 10 \
  --llm-summarizer-cfg "{'llm': {'model_name': 'gemini-2.5-pro', 'request_timeout': 120, 'max_retries': 5}, 'example_db_name': 'teach', 'few_shot_k': 0}"
```

补全后的最终审计：

- selected histories: `10`
- cached histories: `10`
- missing histories: `0`

这说明 `h=100` 的 QA 运行无需回退到在线递归 summarization。

## Full-QA 运行

正式输出：

- `experiments/results/teach/smoke/h100_current_zs_fast_n100_fullqa.json`
- `experiments/results/teach/smoke/h100_current_zs_fast_n100_fullqa.jsonl`

初始带护栏运行命令：

```bash
set -a
source .env
set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_100.pkl \
  --output experiments/results/teach/smoke/h100_current_zs_fast_n100_fullqa.json \
  --n-samples 100 \
  --require-history-cache \
  --max-prompt-tokens-per-sample 12000 \
  --max-average-prompt-tokens-per-sample 5000 \
  --max-seconds-per-sample 240
```

在长时间运行过程中，实验曾从 checkpoint 继续恢复。最终完成的结果文件记录的恢复后运行配置为：

- `resume=True`
- `max_prompt_tokens_per_sample=25000`
- `max_average_prompt_tokens_per_sample=7000`
- `max_seconds_per_sample=240`

这意味着最终的 `100/100` 完成是在严格 cache-only 执行下达到的，但在恢复阶段稍微放宽了 prompt 预算上限，使 `h=100` 下的 just-before/after 和长距离日期检索问题能够完成，而不是过早被截断。

## 评估

评估命令：

```bash
set -a
source .env
set +a
PYTHON_BIN="conda run --no-capture-output -n active-h-emv python" \
  scripts/evaluate_result.sh experiments/results/teach/smoke/h100_current_zs_fast_n100_fullqa.json
```

生成的评估产物：

- `experiments/results/teach/smoke/h100_current_zs_fast_n100_fullqa.gemini_2.5_pro-5634d4.auto_eval.json`
- `experiments/results/teach/smoke/h100_current_zs_fast_n100_fullqa_llm_eval.log`
- `experiments/results/teach/smoke/h100_current_zs_fast_n100_fullqa_metrics.log`

## 稳定性结果

最终完成情况：

- 最终 JSON 结果数：`100`
- 最终 checkpoint 行数：`100`

运行时观察：

- QA 过程中未发生在线 `group and summarize`
- 没有 `###ERROR###`
- 没有空回复级联
- 早期部分运行中观察到一次样本级 timeout，但补完后的续跑仍然达到了 `100/100`
- 低层动作结构化直答仍然在部分样本中节省了 token（`0 token` 路径仍然有效）

## 指标

此处采用的指标约定：

- `S_c`：完全正确率
- `S_p`：至少部分正确率（`correct + partial`）

来自 `scripts/evaluate_result.sh` 的主要指标：

| Run | Total | Valid | `S_c` | `S_p` | Wrong/no-answer | `T` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original H-EMV `h=100` full multimodal (`ICL=1`, full) | 100 | 100% | 34.0% | 62.0% | 38.0% | 10.4K |
| Phase 23 `h=50` full QA | 100 | 100% | 48.0% | 72.0% | 28.0% | 2.06K |
| Phase 24 `h=100` full QA | 100 | 98.0% | 49.0% | 75.0% | 25.0% | 2.29K |

Phase 24 细分结果：

- `correct = 49/100`
- `partially_correct = 26/100`
- `at_least_partially_correct = 75/100`
- `wrong = 25/100`
- 合法答案率 = `98.0%`
- 错误/空答案率 = `2.0%`
- 每个 QA 的 prompt tokens = `2.29K`
- 每个 QA 的 completion tokens = `0.36K`

## 解读

本阶段是一个明确利好论文的结果。

与原论文 `h=100` full multimodal H-EMV 那一行相比：

1. `S_c` 从 `34.0%` 提升到 `49.0%`（`+15` 个点）。
2. prompt-token 成本从 `10.4K` 降到 `2.29K`。
3. 即便存在 `2%` 的 invalid/error rate，整体语义效用仍明显强于原始基线。

与 Phase 23 相比：

- 在更长 history 范围下，`h=100` 基本保持了相同级别的语义正确性（`49.0%` vs `48.0%`）；
- `S_p` 也保持强势（`75.0%` vs `72.0%`），说明即使在更长 history 下，大多数问题仍至少部分正确；
- token 成本只小幅上升（`2.29K` vs `2.06K`）；
- `h=100` 新增的主要成本是运行时延，尤其体现在 temporal-neighbor 和长距离日期问题上。

因此，当前系统在 `h=100` 下的行为最适合描述为：

- 已足够支撑论文中相对于原始基线的主张；
- 在显式 cache 控制下具备可运行性；
- 已接近实际瓶颈，再想继续提升，可能需要针对性精度优化，而不是单纯继续拉长 history。

## 结论

对于论文主表：

- 保留 **Phase 23** 作为稳定的 `h=50` full-QA 结果点；
- 增加 **Phase 24** 作为 `h=100` full-QA 结果点；
- 使用 Phase 24 支撑“即使在测试的最长 history 长度下，改进后的系统仍优于原始 H-EMV 基线”的论点。

## 下一步

默认情况下，最合理的下一步并不是再跑一个更大的实验。

而是：

1. 将 Phase 23 和 Phase 24 合并进最终论文结果表与讨论中；
2. 选取 2-3 个代表性的 `h=100` 成功案例，以及 1-2 个失败案例做定性分析；
3. 只有在代码改动能够精确命中某个残余问题时，才继续推进，例如短对象 yes/no 精度或 temporal-neighbor 歧义。

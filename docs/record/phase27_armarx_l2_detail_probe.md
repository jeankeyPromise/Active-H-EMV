# Phase 27: ARMARX L2 事件层细节题受控探测

日期：2026-04-25

## 目标

根据 [one-pass与树结构遗忘的关系.md](../Key%20Concept/one-pass与树结构遗忘的关系.md) 中的后续实验指导，本阶段不再把重点放在“one-pass 是否变短”上，而是转向一个更贴近遗忘模块本体的问题：

> 当我们保留到 `HigherLevelSummary -> GoalBasedSummary -> EventBasedSummary` 这一层时，系统还能支撑多少细节敏感问题？

同时，本阶段继续遵守实验安全规则：

- 先做零新增 LLM token 的长度统计；
- 只有在 prompt 成本可控时才启动正式 QA；
- 如果不同遗忘设置在该视图下输入完全一致，则不重复消耗 token 做无信息增益的五组全跑。

## 方法

### 1. 构造细节敏感 QA 子集

新增文件：

- `data/armarx_lt_mem/qa_forgetting_detail_probe_v1.json`

共 6 题，覆盖：

- `temporal_event`
- `object_detail`
- `event_statistics`
- `problem_analysis`

具体题目类型有意偏向“需要较具体历史细节”的问题，而不是高层 overview。

### 2. 使用 L2 事件层视图做零成本长度统计

配置：

- `armarx_lt_mem/zs_1pass_l2_gemini`
- `armarx_lt_mem/zs_1pass_l2_gemini_forget_random`
- `armarx_lt_mem/zs_1pass_l2_gemini_forget_medium`
- `armarx_lt_mem/zs_1pass_l2_gemini_forget_medium_graph`
- `armarx_lt_mem/zs_1pass_l2_gemini_forget_ultra`

统计命令：

```bash
conda run --no-capture-output -n active-h-emv python scripts/armarx_forgetting_prompt_stats.py \
  --history-dir data/armarx_lt_mem \
  --qa-file data/armarx_lt_mem/qa_forgetting_detail_probe_v1.json \
  --settings \
    base=armarx_lt_mem/zs_1pass_l2_gemini \
    random_medium=armarx_lt_mem/zs_1pass_l2_gemini_forget_random \
    medium=armarx_lt_mem/zs_1pass_l2_gemini_forget_medium \
    medium_graph=armarx_lt_mem/zs_1pass_l2_gemini_forget_medium_graph \
    ultra=armarx_lt_mem/zs_1pass_l2_gemini_forget_ultra \
  --prepared-history-dirs \
    random_medium=experiments/results/armarx_lt_mem/forgetting/random_medium \
    medium=experiments/results/armarx_lt_mem/forgetting/ubpf_medium \
    medium_graph=experiments/results/armarx_lt_mem/forgetting/ubpf_medium_graph \
    ultra=experiments/results/armarx_lt_mem/forgetting/ubpf_ultra \
  --output experiments/results/armarx_lt_mem/forgetting_probe/prompt_length_stats_l2_detail_probe_v1.json
```

结果文件：

- `experiments/results/armarx_lt_mem/forgetting_probe/prompt_length_stats_l2_detail_probe_v1.json`

### 3. 仅运行一组正式 QA baseline

因为长度统计显示五组输入完全一致，所以只运行 `base` 一组，避免无效 token 消耗。

运行命令：

```bash
set -a; source .env; set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg armarx_lt_mem/zs_1pass_l2_gemini \
  --output experiments/results/armarx_lt_mem/forgetting_probe/zs_1pass_l2_gemini_detail_base_n6_env.json \
  --dataset simple \
  --history-dir data/armarx_lt_mem \
  --qa-file data/armarx_lt_mem/qa_forgetting_detail_probe_v1.json \
  --n-samples 6 \
  --max-prompt-tokens-per-sample 60000 \
  --max-average-prompt-tokens-per-sample 60000 \
  --max-seconds-per-sample 180
```

说明：

- 第一次直接运行时，由于当前 shell 未加载 `.env`，出现 `OPENAI_API_KEY` 缺失报错；
- 随后按仓库既有方式 `source .env` 后重跑成功；
- 正式结果以 `_env.json` 版本为准。

评测命令：

```bash
set -a; source .env; set +a
PYTHON_BIN="conda run --no-capture-output -n active-h-emv python" \
  scripts/evaluate_result.sh \
  experiments/results/armarx_lt_mem/forgetting_probe/zs_1pass_l2_gemini_detail_base_n6_env.json
```

## 结果

### 1. L2 事件层视图在五种遗忘设置下完全一致

`prompt_length_stats_l2_detail_probe_v1.json` 显示：

| Setting | Mean prompt chars | 相对 base |
| --- | ---: | ---: |
| `base` | `152,475` | `1.0000` |
| `random_medium` | `152,475` | `1.0000` |
| `medium` | `152,475` | `1.0000` |
| `medium_graph` | `152,475` | `1.0000` |
| `ultra` | `152,475` | `1.0000` |

进一步对格式化后的历史字符串做哈希校验，五组完全一致：

| Setting | History chars | MD5 |
| --- | ---: | --- |
| `base` | `152,064` | `7250c8cfd9da8b4d6e20eff00e2319f8` |
| `random_medium` | `152,064` | `7250c8cfd9da8b4d6e20eff00e2319f8` |
| `medium` | `152,064` | `7250c8cfd9da8b4d6e20eff00e2319f8` |
| `medium_graph` | `152,064` | `7250c8cfd9da8b4d6e20eff00e2319f8` |
| `ultra` | `152,064` | `7250c8cfd9da8b4d6e20eff00e2319f8` |

这说明：

- 当前遗忘设置虽然会压缩树结构；
- 但在“保留到 L2+事件摘要”这个视图下，送给 LLM 的文本并没有变化；
- 因而继续把 `base/random/medium/medium_graph/ultra` 五组都跑一遍正式 QA，不会得到有意义的新信息。

### 2. 事件层细节题 baseline 结果较弱

正式结果文件：

- `experiments/results/armarx_lt_mem/forgetting_probe/zs_1pass_l2_gemini_detail_base_n6_env.json`
- `experiments/results/armarx_lt_mem/forgetting_probe/zs_1pass_l2_gemini_detail_base_n6_env.gemini_2.5_pro-c0fc2d.auto_eval.json`

主指标：

- Total QA: `6`
- Valid answer rate: `83.3%`
- Error/empty answer rate: `16.7%`
- `S_c`: `16.7%` (`1/6`)
- `S_p`: `0.0%`
- T prompt tokens per QA: `45.66K`
- Completion tokens per QA: `0.25K`

总 token：

- Prompt tokens: `273,957`
- Completion tokens: `1,513`

单题 prompt token 大约在 `54.8K` 左右，未超过本阶段设置的 `60K` 单题阈值，说明护栏有效。

### 3. 各题表现

从 auto-eval 分类看：

- `correct_summarized`: `1`
- `partially_correct_`: `1`
- `wrong`: `1`
- `no_answer`: `3`

对应样例：

1. `a7a-merged-problems-3`  
   正确。模型抓到了 “先尝试 grasp `Moog`，随后改抓 milk” 这一关键因果线。

2. `a7a-merged-req-at-time-1`  
   仅部分正确。模型定位到了 6 月 26 日晚 8 点后的一段行为，但没有精确恢复“先 receive 再 hand over”的目标答案。

3. `a7a-merged-action-detail-2`  
   错误。模型把多个 `WhatCanYouSee*` 阶段混在一起，给出了更宽泛的对象集合，而不是 “unknown object 并询问名称”。

4. `a7a-merged-objects-5`  
   无答案，网络连接错误。

5. `a7a-merged-events-4`、`a7a-merged-events-1`  
   无有效答案，输出被截断在推理起始处，说明该视图对这类题仍然过重且不稳。

## 结论

本阶段得到两个很重要的结论。

### 结论 1：保留到事件层时，当前遗忘设置不会改变这一视图下的输入

也就是说，当前 ARMARX 遗忘实验在公开 `pkl` 数据上，更像是在做：

- “树结构压缩”
- “底层细节裁剪”

而不是在改变 `HigherLevelSummary / GoalBasedSummary / EventBasedSummary` 这一层的文本内容。

因此：

- 如果评测视图只看到这一层；
- 那么不同遗忘强度自然不会表现出差异。

### 结论 2：仅保留到事件层，不足以支撑大多数细节敏感问题

这与遗忘模块的理论预期是一致的：

- 高层总结可以支撑 overview 类问题；
- 但 object 邻接、精确时序、特定失败原因等题型，往往还依赖更低层细节；
- 因而“只保留总结/事件摘要”会带来明确的问答能力边界。

## 为什么本阶段没有继续做五组全跑

这是有意的实验监督决策，而不是遗漏。

原因是：

1. 零成本长度统计已证明五组输入完全相同；
2. 哈希校验进一步证明格式化历史字节级一致；
3. 在这种前提下继续做五组正式 QA，只会重复消耗约 `5 x 6 x 45K` 量级的 prompt token；
4. 但不会带来可解释的新结论。

因此，本阶段将 token 预算集中在“确认事件层视图的能力边界”上，而不是做无增益重复。

## 下一步

更有信息量的下一步，不是继续重复 L2 事件层五组 QA，而是选一条真正能暴露遗忘差异的路径：

1. 设计一个“更低层细节可见”的受控视图，只暴露与问题局部相关的底层细节，而不是整棵树全平铺；
2. 或者做离线结构评测，直接统计被遗忘节点覆盖到的 scene / relation / raw payload 信息损失；
3. 再在极小规模 detail slice 上比较 `random` 与 `UBPF`，验证“忘什么”是否优于“随便删”。

在当前证据下，可以先把论文中的表述写稳：

> 在 ARMAR-7 公布的长期记忆树数据上，遗忘模块能够压缩树结构；  
> 但当系统仅保留到高层总结/事件描述时，细节敏感 QA 能力会明显受限。  
> 因而遗忘模块的价值应被表述为“树级压缩与信息保真权衡”，而不是“任何视图下都能无损回答细节问题”。

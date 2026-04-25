# Phase 26: ARMARX 遗忘模块格式化历史长度统计

日期：2026-04-25

## 目标

在不触发新的 LLM 调用、避免额外 token 消耗的前提下，继续推进 ARMARX 遗忘模块实验，回答一个更具体的问题：

> 遗忘后的记忆树，在 one-pass QA 真正送入模型之前，是否已经显著缩短了“格式化后的历史文本”？

这个问题很重要，因为 Phase 25 已经表明：

- 压缩后的树结构是存在的；
- 但 one-pass QA 仍然会面临极高的 prompt 成本；
- 因此需要进一步区分：问题究竟出在“遗忘不够有效”，还是“one-pass flatten 本身就是主要瓶颈”。

## 方法

### 1. 新增脚本

- `scripts/armarx_forgetting_prompt_stats.py`

该脚本不调用 LLM，只做以下事情：

1. 读取 `data/armarx_lt_mem/qa.json` 中的 30 个问题；
2. 按 one-pass 配置复用 `ZeroShotOnePassSemiFlatQA` 的历史格式化逻辑；
3. 统计每个设置下：
   - `history_chars`
   - `history_words`
   - `prompt_chars`
   - `prompt_words`
4. 按问题类型进行分组汇总。

### 2. 为避免额外重算与网络风险，直接复用已生成的遗忘后历史

本次统计没有重新在线计算 forgetting，也没有重新加载搜索嵌入模型，而是直接复用 Phase 25 已生成的遗忘后 `pkl`：

- `experiments/results/armarx_lt_mem/forgetting/random_medium/`
- `experiments/results/armarx_lt_mem/forgetting/ubpf_medium/`
- `experiments/results/armarx_lt_mem/forgetting/ubpf_medium_graph/`
- `experiments/results/armarx_lt_mem/forgetting/ubpf_aggressive/`
- `experiments/results/armarx_lt_mem/forgetting/ubpf_ultra/`

这样可以避免：

- 重新加载 `all-mpnet-base-v2` 带来的 HuggingFace 网络握手问题；
- 在统计实验中引入新的外部不稳定性；
- 重复消耗不必要的时间和资源。

### 3. 设置

对比组如下：

- `base`
- `random_medium`
- `medium`
- `medium_graph`
- `aggressive`
- `ultra`

说明：

- `medium_graph` 没有单独的 one-pass gemini 配置文件，因此在本次长度统计中复用了 `medium` 的 one-pass prompt 配置，但读取的是 `ubpf_medium_graph` 对应的遗忘后历史；
- 由于 one-pass 的历史文本本身与问题类型无关，因此不同问题类型的差异主要来自问题文本本身，而不是 history 结构。

## 运行命令

```bash
conda run --no-capture-output -n active-h-emv python scripts/armarx_forgetting_prompt_stats.py \
  --history-dir data/armarx_lt_mem \
  --qa-file data/armarx_lt_mem/qa.json \
  --settings \
    base=armarx_lt_mem/zs_1pass_flat_gemini \
    random_medium=armarx_lt_mem/zs_1pass_flat_gemini_forget_random \
    medium=armarx_lt_mem/zs_1pass_flat_gemini_forget_medium \
    medium_graph=armarx_lt_mem/zs_1pass_flat_gemini_forget_medium \
    aggressive=armarx_lt_mem/zs_1pass_flat_gemini_forget_aggressive \
    ultra=armarx_lt_mem/zs_1pass_flat_gemini_forget_ultra \
  --prepared-history-dirs \
    random_medium=experiments/results/armarx_lt_mem/forgetting/random_medium \
    medium=experiments/results/armarx_lt_mem/forgetting/ubpf_medium \
    medium_graph=experiments/results/armarx_lt_mem/forgetting/ubpf_medium_graph \
    aggressive=experiments/results/armarx_lt_mem/forgetting/ubpf_aggressive \
    ultra=experiments/results/armarx_lt_mem/forgetting/ubpf_ultra \
  --output experiments/results/armarx_lt_mem/forgetting_guarded/prompt_length_stats_full.json
```

结果文件：

- `experiments/results/armarx_lt_mem/forgetting_guarded/prompt_length_stats_full.json`

## 总体结果

以 30 个问题的平均 `prompt_chars` 为主指标：

| Setting | Mean prompt chars | 相对 base | 相对 base 变化 |
| --- | ---: | ---: | ---: |
| `base` | `596,767.7` | `1.0000` | `0` |
| `random_medium` | `589,209.7` | `0.9873` | `-7,558` |
| `medium_graph` | `585,863.7` | `0.9817` | `-10,904` |
| `medium` | `569,561.7` | `0.9544` | `-27,206` |
| `aggressive` | `569,561.7` | `0.9544` | `-27,206` |
| `ultra` | `564,573.7` | `0.9461` | `-32,194` |

对应的平均 `history_words`：

| Setting | Mean history words | 相对 base |
| --- | ---: | ---: |
| `base` | `43,728` | `1.0000` |
| `random_medium` | `42,840` | `0.9797` |
| `medium_graph` | `42,489` | `0.9717` |
| `medium` | `40,526` | `0.9268` |
| `aggressive` | `40,526` | `0.9268` |
| `ultra` | `39,906` | `0.9126` |

## 关键发现

### 1. 遗忘确实能缩短 one-pass 输入，但幅度有限

最强设置 `ultra` 将平均格式化 prompt 长度从：

- `596,767.7 chars`

降到：

- `564,573.7 chars`

平均减少：

- `32,194 chars`
- 约 `5.39%`

这说明遗忘对 one-pass 输入长度并不是完全无效，但下降幅度也远称不上“根本性缓解”。

### 2. 随机遗忘最弱，效用引导遗忘更有效

`random_medium` 仅将 prompt 长度压到基线的：

- `98.73%`

而 `medium` / `aggressive` 达到：

- `95.44%`

这说明：

- “忘什么”依然比“是否忘”更关键；
- UBPF 至少在压缩输入长度这件事上，明显优于随机遗忘。

### 3. 图中心性版本更保守

`medium_graph` 仅压到：

- `98.17%`

比 `medium` 的：

- `95.44%`

保守很多。

这与 Phase 25 中的树级压缩统计一致，说明：

- `use_graph_centrality=true` 会显著保留更多结构；
- 它可能更有利于后续层级检索稳定性；
- 但对 one-pass flatten 的长度缓解作用有限。

### 4. `medium` 与 `aggressive` 在当前 one-pass 长度统计中表现相同

这次结果中：

- `medium`
- `aggressive`

两者的平均 `prompt_chars` 完全一致。

这说明至少在当前这份 merged history 和 one-pass L0 格式化方式下，二者虽然在遗忘层级上可能不同，但最终落到展平文本长度时，表现没有拉开。

这进一步支持一个判断：

> 当前真正主导成本的，不是遗忘参数微调本身，而是 one-pass flatten 这种评测方式。

## 与 Phase 25 的关系

Phase 25 的 one-pass token probe 结论是：

- `base` 第一题约 `228,174 prompt tokens`
- `ultra` 第一题约 `220,648 prompt tokens`
- token 仅下降约 `3.3%`

本阶段的格式化长度统计得到：

- `ultra` 平均 `prompt_chars` 约为 base 的 `94.61%`

二者方向一致：

1. 遗忘确实会让 one-pass 输入变短；
2. 但幅度不大；
3. 因而 one-pass flatten 依然是主瓶颈。

换句话说，这一阶段为 Phase 25 的 token probe 提供了一个**完全不依赖新 LLM 调用的支撑证据层**。

## 结论

本阶段得到的最重要结论是：

1. 遗忘模块不仅能压缩树结构，也能实际缩短 one-pass 输入历史；
2. 但即便在 `ultra` 下，缩短幅度也只有约 `5%`；
3. 随机遗忘明显弱于效用引导遗忘；
4. 图中心性版本更保守，更像是“为层级检索保真”而不是“为 one-pass 减长”服务；
5. 因此，遗忘模块的主价值应优先在**层级检索场景**中验证，而不是指望它单独解决 full-history flatten 的 token 问题。

## 下一步

最合理的下一步是：

1. 基于 `qa.json` 做一个按问题类型分层的小型正式 QA 子集；
2. 优先保留：
   - summary / overview
   - temporal event
   - object detail
   - problem analysis
3. 在该子集上做极小规模、强护栏的遗忘对比；
4. 同时继续避免直接启动 merged ARMARX 的 interactive/full 全量运行。

# Phase 25: ARMARX 遗忘模块受控探测

日期：2026-04-24

## 目标

在显式 token 风险监督下，推进 `data/armarx_lt_mem` 上的遗忘模块实验。

本阶段目标不是立刻完成一张正式总表，而是先回答三个运行层面的问题：

1. 我们能否生成遗忘后的 ARMARX history，并在本地量化压缩效果？
2. 当前 interactive/full QA 路径在合并后的 ARMARX history 上是否依然安全？
3. 如果不安全，是否能通过受控的一次性 one-pass probe，在不失控消耗 token 的前提下得到信息保留信号？

## 新增产物

配置：

- `llm_emv/config/armarx_lt_mem/full_gemini_forget_medium.yaml`
- `llm_emv/config/armarx_lt_mem/full_gemini_forget_medium_graph.yaml`
- `llm_emv/config/armarx_lt_mem/full_gemini_forget_random_medium.yaml`
- `llm_emv/config/armarx_lt_mem/full_gemini_forget_aggressive.yaml`
- `llm_emv/config/armarx_lt_mem/full_gemini_forget_ultra.yaml`
- `llm_emv/config/armarx_lt_mem/full_gemini_guarded.yaml`
- `llm_emv/config/armarx_lt_mem/full_gemini_guarded_forget_medium.yaml`
- `llm_emv/config/armarx_lt_mem/zs_1pass_flat_gemini.yaml`
- `llm_emv/config/armarx_lt_mem/zs_1pass_flat_gemini_forget_medium.yaml`
- `llm_emv/config/armarx_lt_mem/zs_1pass_flat_gemini_forget_random.yaml`
- `llm_emv/config/armarx_lt_mem/zs_1pass_flat_gemini_forget_aggressive.yaml`
- `llm_emv/config/armarx_lt_mem/zs_1pass_flat_gemini_forget_ultra.yaml`

脚本：

- `scripts/armarx_forgetting_prepare.py`

结果：

- `experiments/results/armarx_lt_mem/forgetting/*/compression_stats.json`
- `experiments/results/armarx_lt_mem/forgetting_guarded/*.json`
- `experiments/results/armarx_lt_mem/forgetting_guarded/*.jsonl`

## 压缩结果

在 `2024-a7a-merged-summary.pkl` 上的本地预处理成功完成。

关键设置如下：

| Setting | File ratio | Scene ratio | Relation ratio | Forgetting levels |
| --- | ---: | ---: | ---: | --- |
| `random_medium` | `0.952` | `0.940` | `0.933` | `L0=2823, L1=390, L2=389` |
| `ubpf_medium` | `0.813` | `0.790` | `0.728` | `L0=1244, L1=1139, L2=1219` |
| `ubpf_medium_graph` | `0.959` | `0.963` | `0.900` | `L0=1251, L1=2225, L2=126` |
| `ubpf_ultra` | `0.797` | `0.762` | `0.705` | `L0=145, L1=1898, L2=1559` |

观察：

1. UBPF 确实可以明显压缩已发布的 ARMARX memory tree。
2. `use_graph_centrality=true` 会让该数据集上的遗忘变得明显更保守。
3. 在当前 immunity/boundary 规则下，随机遗忘的压缩力度要弱得多。
4. `ubpf_ultra` 是第一个让 Level 2 summary retention 占比超过 40% 的设置。

## 受控 QA 发现

### 1. 当前 interactive/full 路径在合并 ARMARX history 上并不安全

尝试路径：

- `armarx_lt_mem/full_gemini`
- `armarx_lt_mem/full_gemini_guarded`

这两次运行都在完成前被手动停止，因为第一个样本就进入了非常嘈杂的重复 FAISS-search 行为。虽然尚未确认已经出现 token runaway，但风险已经足够高，继续运行会违反本次“受控实验”的要求。

结论：

> 当前 interactive/full agent 路径并不适合作为 ARMARX 合并 history 上遗忘模块的第一条评估路径。

### 2. one-pass gemini probe 可控，并且真实暴露了 token 规模

因此改为使用如下 one-pass probe：

- `armarx_lt_mem/zs_1pass_flat_gemini`

第一个样本（`a7a-merged-explain-1`）产生了：

- prompt tokens: `228,174`
- completion tokens: `508`

该运行被如下护栏自动停止：

- `--max-prompt-tokens-per-sample 20000`

这正是我们希望护栏发挥的行为。

解读：

> 对未压缩的合并 ARMARX history 进行 one-pass QA，在运行层面是不可接受的。

### 3. 中等/激进遗忘在 one-pass 问答时并未降低 prompt tokens

使用：

- `zs_1pass_flat_gemini_forget_medium`
- `zs_1pass_flat_gemini_forget_aggressive`

第一个样本依然产生了：

- prompt tokens: `228,174`

原因：

- 在该问题的实际查询时间点下，安全下限拉低了有效阈值，导致 Level 2 实际上没有发生；
- 没有 Level 2 summary retention，L0 one-pass history 的格式化结果就几乎不变。

### 4. ultra 遗忘产生了真实的 Level 2 保留，并带来了小幅 prompt 下降

使用：

- `zs_1pass_flat_gemini_forget_ultra`

第一个样本产生了：

- prompt tokens: `220,648`
- completion tokens: `642`

相较基线，减少了：

- `7,526` 个 prompt tokens
- 约 `3.3%`

解读：

> 即便是非常激进的树级遗忘，在该样本上对展平后的 one-pass prompt 也只能带来有限下降。这说明 one-pass flatten 本身才是主瓶颈，而遗忘机制更适合与层级检索结合，而不是和整段历史全量倾倒绑定在一起。

## 主要结论

1. 遗忘模块已经能在 ARMARX memory tree 上正常工作，并产生可量化的压缩效果。
2. 压缩强度对 `min_retain_ratio` 非常敏感。
3. 对于合并 ARMARX history，one-pass prompt 成本极端高，因此反而很适合作为 stress-test 基线。
4. 当前证据支持一种很强的论文论述方式：

   - 遗忘可以压缩记忆树；
   - 但仅靠树级压缩，无法解决 one-pass flatten 下的长历史 token 问题；
   - 因此，遗忘应与层级检索配合评估，而不是只和 full-history dumping 一起比较。

## 建议的下一步

现在还不应直接发起完整 ARMARX 正式 QA 运行。

更合理的下一步是：

1. 增加一个按问题类型分组的小型自定义 QA 子集；
2. 在第一轮正式遗忘对比中优先覆盖 summary-style 和 event-style 问题；
3. 增加一个紧凑指标，用于统计 LLM 调用前的 formatted-history 长度；
4. 只有在加入更强循环诊断或更严格搜索上限后，才重新尝试 interactive/full ARMARX 路径。

# Phase 35: TEACh `|h|=50` Level 2 参数探测

## 目标

检查当前 forgetting 参数为什么一直打不到 `Level 2`，
并构造一个真正进入 `Level 2` 的配置，
观察结构压缩和 QA 表现会发生什么变化。

## 新配置

- `llm_emv/config/teach/simplified/full_graph_aug_zs_fast_forget_l2probe.yaml`

相对于当前主 setting，核心调整是：

- `theta_1: 0.55`
- `theta_2: 0.30`
- `half_life: 1800`
- `min_retain_ratio: 0.10`

其中最关键的是 `min_retain_ratio`：
此前主 setting / ultra 都会触发安全下限，
把 `effective_theta_1` 和 `effective_theta_2` 明显下拉到约 `0.23 / 0.09`，
导致几乎不可能真正进入 `Level 2`。

## 结果文件

- 存储统计：
  - `experiments/results/teach/forgetting/h50_forget_l2probe_n20_storage.json`
- QA 预跑：
  - `experiments/results/teach/forgetting/h50_forget_l2probe_n20_probe.json`
  - `experiments/results/teach/forgetting/h50_forget_l2probe_n20_probe.gemini_2.5_pro-573e41.auto_eval.json`

## 存储统计（n=20）

在前 20 个 QA 对应的 2 段长 history 上：

- 文件大小比：`1.0427`
- scene 比：`0.9294`
- relation 比：`0.5034`
- 事件比：`1.0000`

遗忘层级分布：

- before: `L0 = 10545`
- after: `L0 = 1230`, `L1 = 2924`, `L2 = 6391`

这说明新参数已经显著进入 `Level 2`：

- `L2` 占全部事件约 `60.6%`
- relation 总量约减半
- scene 总量也出现了实际下降

## 一个意外但重要的发现

虽然结构压缩已经很明显，但 pickle 文件大小**反而增加**了：

- `file_size_ratio = 1.0427`

进一步查看统计：

- `with_summary_override = 6391`
- `with_cached_summary = 6391`

这说明当前 `Level 2` 实现里，
大量事件同时持有 `_summary_override` 与 `_cached_nl_summary`，
带来了额外序列化开销。

所以当前 `Level 2` 配置回答了两个问题：

1. 参数确实可以调到让 history 真正进入 `Level 2`；
2. 但当前实现还不适合直接拿 pickle 大小做“更强遗忘更省存储”的结论，
   因为表示层本身引入了额外负担。

## QA 预跑（n=20）

`l2_probe` 在同一批 `n=20` 上的 auto-eval 结果为：

- Valid: `100.0%`
- `S_c = 55.0%`
- `S_p = 20.0%`
- Wrong: `25.0%`
- `T = 2.13K`

表面上看，与当前 `Base / Forget+Graph / Ultra` 的 `n=20` 结果相比：

- Base: `S_c = 60.0%`, `S_p = 20.0%`, `T = 2.13K`
- L2 probe: `S_c = 55.0%`, `S_p = 20.0%`, `T = 2.13K`

但继续做 raw answer 对齐后，得到一个更关键的事实：

- `l2_probe` 与 `Base` 在这 `20/20` 个样本上的**答案文本逐条完全一致**
- prompt token 总量也几乎一致：
  - Base: `42693`
  - L2 probe: `42688`

因此，这里的 `55%` 对 `60%` 不应解读为真实 QA 行为下降，
更像是 LLM judge 的批次波动。

更准确的说法是：

1. 在当前这 20 题上，即使大量进入 `Level 2`，最终回答仍未变化；
2. 结构压缩已经发生，但尚未在这一小批题上转化为稳定可见的 QA 退化。

## 关键结论

### 1. 参数确实需要调整，才能观察到真正的 Level 2 行为

当前主 setting 和 ultra 的问题不是“遗忘无效”，
而是安全下限把阈值压得太低，
导致遗忘长期停留在 `Level 1`。

### 2. `min_retain_ratio` 是当前最关键的旋钮

把它从 `0.30 / 0.20` 降到 `0.10` 后，
`effective_theta_1` 与 `effective_theta_2` 不再被强制下拉，
于是 `Level 2` 大量出现。

### 3. 当前 Level 2 已经能压结构，但还没有真正压 pickle

这是当前实现层面的主要瓶颈：

- 结构指标是正向的
- 序列化大小却是反向的

所以如果后面要在论文里强调“更强遗忘更省存储”，
最好先修一下 `Level 2` 表示中的摘要缓存冗余。

## 当前建议

在现阶段，不建议直接把这个 `l2_probe` 扩到 full `100/100` 正式实验。

更合适的路径是：

1. 保留当前 Phase 34 作为正式主结果：
   - `Forget+Graph`
   - full `100/100`
   - 答案与基线完全一致
   - 存储压缩约 `5.26%`
2. 将本阶段作为机制验证：
   - 证明参数调节可以真正触发 `Level 2`
   - 证明 scene / relation 可以被显著压缩
   - 同时暴露出当前 `Level 2` 序列化实现仍需优化

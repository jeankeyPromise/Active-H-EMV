# Phase 34: TEACh `|h|=50` Forget+Graph `100/100` 正式对齐

## 目标

把遗忘模块主 setting `Forget+Graph` 扩展到 full `100/100` QA，
直接与现有正式基线
`experiments/results/teach/smoke/h50_current_zs_fast_n100_fullqa.json`
对齐，并补齐同一批 10 段 history 的存储统计。

## 使用配置

- 基线：`teach/simplified/full_graph_aug_zs_fast`
- 遗忘主 setting：`teach/simplified/full_graph_aug_zs_fast_forget`

## 结果文件

- Forget+Graph full QA:
  - `experiments/results/teach/forgetting/h50_forget_graph_n100_fullqa.json`
  - `experiments/results/teach/forgetting/h50_forget_graph_n100_fullqa.gemini_2.5_pro-573e41.auto_eval.json`
- Forget+Graph 存储统计：
  - `experiments/results/teach/forgetting/h50_forget_graph_n100_storage.json`
- 对齐基线：
  - `experiments/results/teach/smoke/h50_current_zs_fast_n100_fullqa.json`

## QA 对齐结果

### 1. raw answer 逐条核对

最关键的结果是：

- `Forget+Graph` 与现有正式基线在 `100/100` QA 上的**答案文本逐条完全一致**
- sample_id 也完全一一对应

因此，从最终问答输出角度看，
当前 `Forget+Graph` 没有改变 `|h|=50` 正式实验的回答行为。

### 2. auto-eval 指标

`Forget+Graph` 当前这次 auto-eval 的结果为：

- Valid: `99.0%`
- `S_c = 45.0%`
- `S_p = 23.0%`
- Wrong: `32.0%`
- `T = 1.95K`

但这里要特别说明：

- 现有正式基线的旧 auto-eval 文件来自更早一轮 judge
- 本轮重跑基线 auto-eval 后，judge 分类仍然会有轻微波动
- 由于 **raw answer 100/100 完全一致**，因此这部分 `45/23` 对 `48/24`
  不应解读为真实 QA 行为变化，更像是 LLM judge 的非确定性波动

因此，本阶段更可靠的结论是：

> `Forget+Graph` 与正式基线在 full `100/100` 上保持了相同的最终回答。

### 3. token 成本

- 正式基线 `T ≈ 2.06K`
- Forget+Graph `T ≈ 1.95K`

也就是说，在答案不变的前提下，当前 Forget+Graph 还有小幅 prompt 降低。

## 存储统计

同一批 `100` 个 QA 对应 `10` 段长 history 的聚合统计如下：

- 选中 history：`10`
- 选中 QA：`100`
- 文件大小比：`0.9474`
- scene 比：`1.0000`
- relation 比：`1.0000`
- 事件比：`1.0000`

遗忘层级分布：

- before: `L0 = 60836`
- after: `L0 = 18672`, `L1 = 42164`

这说明：

1. 当前 full `100/100` 正式口径下，遗忘已经稳定带来约 `5.26%` 的 pickle 压缩；
2. 压缩仍然全部停留在 `Level 1`；
3. scene / relation 结构尚未被压到，因此结构化证据没有明显减少。

## 关键观察

### 1. 主实验已经闭环

这轮已经能支撑一句很稳的实验结论：

> 在 TEACh `|h|=50` 的真实分层检索问答管线上，
> 当前效用遗忘主 setting 在保持最终回答不变的同时，
> 为 10 段长 history 带来了约 `5.26%` 的存储压缩。

### 2. 当前不建议直接扩展 Ultra 到 full `100/100`

原因有两个：

1. Phase 33 的 `n=20` 里，`Ultra` 与 `Base / Forget+Graph` 输出完全一致；
2. `Ultra` 当前也仍然没有进入 `Level 2`，只是把 `Level 1` 比例再推高一些，
   相对 `Forget+Graph` 的额外压缩收益有限。

因此，继续把 `Ultra` 扩到 full `100/100` 的信息增量不高，
不如把精力放到“如何真正进入 `Level 2` 并观察结构压缩-准确率权衡”。

## 当前结论

1. `Forget+Graph` 已经在 TEACh `|h|=50` full `100/100` 上完成正式对齐；
2. 与现有正式基线相比，最终答案 `100/100` 完全一致；
3. 同时带来约 `5.26%` 的存储压缩；
4. 但当前压缩仍然只到 `Level 1`，尚未进入更强的结构压缩区间。

## 下一步

下一步不再优先跑 `Ultra full 100/100`，而是转向：

1. 调整 forgetting 参数，使部分 history 真正进入 `Level 2`；
2. 观察 scene / relation 缩减是否出现；
3. 再评估这种更强遗忘是否带来可见的 QA 退化。

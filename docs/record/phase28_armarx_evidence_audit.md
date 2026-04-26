# Phase 28: ARMARX 遗忘模块关键证据保真审计

日期：2026-04-25

## 目标

Phase 27 已经表明：

- `L2 + 事件层` 视图下，不同遗忘设置的格式化输入完全一致；
- 因而继续做 `base/random/medium/medium_graph/ultra` 五组正式 QA 不会带来额外信息；
- 但这并不等于不同遗忘设置在树结构内部没有差异。

因此，本阶段改做一个**零新增 LLM token** 的离线审计实验，直接回答：

> 对于细节敏感问题依赖的关键过程事件，不同遗忘设置到底保留了多少底层证据？

这里的“底层证据”主要指：

- `scene` 数量
- `relation` 数量
- `objects` 数量
- 是否被压成 `Level 2` 的摘要保留
- 是否触发 `_summary_override`

## 方法

### 1. 新增脚本

- `scripts/armarx_forgetting_evidence_audit.py`

该脚本做两类统计：

1. **关键 probe 事件审计**
   - 选取与细节题最相关的几个代表性事件；
   - 对比它们在 `base/random/medium/medium_graph/ultra` 中的保留状态。

2. **全局 relation-rich / process-rich 事件聚合统计**
   - 以 base history 为参照；
   - 选出 `scene >= 15` 或 `relation >= 60` 的事件；
   - 统计这些“高细节密度事件”在各遗忘设置中分别落到 `Level 0 / 1 / 2` 的数量；
   - 并汇总每个设置下的平均 `scene` / `relation` 保留量。

### 2. 关键 probe 事件

本阶段选择了 4 个代表性 probe：

1. `unknown_object_scan`
   - 对应 QA：`a7a-merged-action-detail-2`
   - 关注 `WhatCanYouSee` 与未知物体识别相关过程

2. `moog_failure`
   - 对应 QA：`a7a-merged-problems-3`
   - 关注 “Moog” 导致失败的第一次 grasp

3. `soy_milk_predefined_grasp`
   - 对应 QA：`a7a-merged-objects-5`
   - 关注成功 grasp milk 前的 `soy-milk` 过程事件
   - 这是一个明显的 relation-rich event

4. `dishwasher_success`
   - 对应 QA：`a7a-merged-events-1`
   - 关注成功 load dishwasher 的事件

### 3. 聚合阈值

本阶段将以下事件定义为“高细节密度事件”：

- `scene >= 15`
  或
- `relation >= 60`

在 base history 中共选出：

- `117` 个事件

### 4. 运行命令

```bash
conda run --no-capture-output -n active-h-emv python scripts/armarx_forgetting_evidence_audit.py \
  --base-history data/armarx_lt_mem/2024-a7a-merged-summary.pkl \
  --setting \
    base=data/armarx_lt_mem/2024-a7a-merged-summary.pkl \
    random_medium=experiments/results/armarx_lt_mem/forgetting/random_medium/2024-a7a-merged-summary.pkl \
    medium=experiments/results/armarx_lt_mem/forgetting/ubpf_medium/2024-a7a-merged-summary.pkl \
    medium_graph=experiments/results/armarx_lt_mem/forgetting/ubpf_medium_graph/2024-a7a-merged-summary.pkl \
    ultra=experiments/results/armarx_lt_mem/forgetting/ubpf_ultra/2024-a7a-merged-summary.pkl \
  --output experiments/results/armarx_lt_mem/forgetting_probe/evidence_audit_v1.json
```

结果文件：

- `experiments/results/armarx_lt_mem/forgetting_probe/evidence_audit_v1.json`

## 关键结果

### 1. 细节题最依赖的 relation-rich 过程事件，在不同设置下保留差异明显

最有代表性的 probe 是：

- `soy_milk_predefined_grasp`
- 对应 `a7a-merged-objects-5`

该事件在 base 中为：

- `47 scenes`
- `282 relations`
- `752 objects`

各设置如下：

| Setting | Forget level | Scenes | Relations | Summary override |
| --- | ---: | ---: | ---: | --- |
| `base` | `0` | `47` | `282` | `False` |
| `random_medium` | `1` | `47` | `282` | `False` |
| `medium` | `2` | `1` | `0` | `True` |
| `medium_graph` | `1` | `47` | `282` | `False` |
| `ultra` | `2` | `1` | `0` | `True` |

这说明：

- `medium` 和 `ultra` 会把这个关键细节过程直接压成摘要保留；
- `random_medium` 和 `medium_graph` 则仍保留完整的过程结构；
- 图中心性版本确实更偏向保留结构上“值得保住”的 relation-rich 节点。

### 2. 失败事件与带对话片段整体仍被保住

例如 `moog_failure`：

- 各设置均为 `forgetting_level = 0`
- `3 scenes`
- `18 relations`

这与遗忘模块的设计是一致的：

- 失败事件受到豁免保护；
- 带对话的片段也更不容易被遗忘。

因此，本阶段也从侧面验证了当前遗忘策略至少遵守了“重要失败经历不被误删”这一安全目标。

### 3. 全局上，`medium_graph` 比 `medium/ultra` 更能保住高细节密度事件

对 117 个高细节密度事件的聚合统计如下：

| Setting | L0 | L1 | L2 | Mean scenes | Mean relations |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base` | `117` | `0` | `0` | `47.31` | `145.44` |
| `random_medium` | `102` | `6` | `9` | `45.01` | `137.20` |
| `medium` | `71` | `23` | `23` | `39.74` | `112.63` |
| `medium_graph` | `71` | `36` | `10` | `43.89` | `130.22` |
| `ultra` | `66` | `27` | `24` | `39.58` | `111.82` |

可以看到：

1. `medium` 与 `ultra` 的 `Level 2` 数量都明显更高；
2. `medium_graph` 的 `Level 2` 只有 `10`，明显少于 `medium` 的 `23`；
3. `medium_graph` 的平均 `relations` 保留量 `130.22`，也显著高于 `medium` 的 `112.63`；
4. `random_medium` 在这个阈值下相对更保守，但缺乏明确的“该保哪里”的方法约束。

## 解释

这阶段最重要的意义在于，它把“遗忘差异”从 prompt 层面重新拉回到了树结构本体。

Phase 26/27 的情况是：

- 不同设置在某些 one-pass 视图下 prompt 完全相同；
- 所以 LLM QA 很难分出差别。

但 Phase 28 说明：

- 差异其实真实存在于事件内部的 `scene / relation / object` 保留层；
- 尤其在 relation-rich 过程事件上，`medium_graph` 与 `medium/ultra` 的行为已经明显分化；
- 这正是遗忘模块应该被验证的地方。

换句话说：

> “问模型时看起来差不多”，并不代表“树里面保存的细节一样多”。

## 结论

本阶段得到三条可以直接写进论文的结论：

1. 遗忘模块的差异主要体现在**事件内部细节保留**，而不一定会立刻体现在某个固定 one-pass prompt 视图上；
2. 对高细节密度事件，`medium_graph` 明显比 `medium/ultra` 更能保住 `scene / relation` 结构；
3. 失败事件与带对话片段整体被稳定保留，说明当前遗忘策略具备基本安全性。

## 下一步

现在最有价值的下一步已经不是继续全量 one-pass，而是做一轮**局部低层细节视图**的小规模验证：

1. 针对 `a7a-merged-objects-5` 这类题，只给模型与该问题局部相关的过程事件；
2. 对比 `random_medium / medium / medium_graph / ultra`；
3. 由于输入局部化，token 成本会远低于 full-history flatten；
4. 而且这时不同遗忘设置在输入层就会真正产生差异。

这会比继续重复 L2 事件层五组正式 QA 更有信息量，也更节省预算。

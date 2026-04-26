# Phase 29: ARMARX 局部低层细节视图单题探测

日期：2026-04-25

## 目标

Phase 28 已经说明：

- 不同遗忘设置在树内部确实保留了不同量级的细节；
- 尤其在 relation-rich 过程事件上，`medium_graph` 与 `medium/ultra` 已经明显分化；
- 但这种差异在 full-history 或 L2 one-pass 视图里不容易直接传递到 QA。

因此，本阶段做一个**极小规模、低 token 成本**的验证：

> 如果只给模型与问题局部相关的低层细节片段，不同遗忘设置会不会在回答上表现出差异？

## 方法

### 1. 新增脚本

- `scripts/armarx_local_detail_probe.py`

该脚本只做一件事：

1. 针对一个局部 probe；
2. 在每个遗忘设置中定位对应事件；
3. 只抽取该事件的**最新 scene graph** 作为局部上下文；
4. 用一个极短 prompt 直接问同一问题。

### 2. 选择的 probe

本阶段选择：

- `objects_5_local_latest_scene`
- 原问题：`Which object was next to the soy milk last time you grasped it today in the afternoon?`
- Ground truth：`Orange juice bottle`

对应的局部事件是：

- `Grasping::KnownObject::PredefinedGrasp(object=soy-milk) <Succeeded>`

这是一个很好的 probe，因为：

- 在 `base` / `medium_graph` 中，该事件保留了 relation 行；
- 在 `medium` / `ultra` 中，该事件已被 `Level 2` 压缩，relations 被清空；
- 因而局部输入层本身就会产生差异。

### 3. 运行命令

```bash
set -a; source .env; set +a
conda run --no-capture-output -n active-h-emv python scripts/armarx_local_detail_probe.py \
  --setting \
    base=data/armarx_lt_mem/2024-a7a-merged-summary.pkl \
    medium=experiments/results/armarx_lt_mem/forgetting/ubpf_medium/2024-a7a-merged-summary.pkl \
    medium_graph=experiments/results/armarx_lt_mem/forgetting/ubpf_medium_graph/2024-a7a-merged-summary.pkl \
    ultra=experiments/results/armarx_lt_mem/forgetting/ubpf_ultra/2024-a7a-merged-summary.pkl \
  --output experiments/results/armarx_lt_mem/forgetting_probe/local_detail_probe_objects5_v1.json
```

结果文件：

- `experiments/results/armarx_lt_mem/forgetting_probe/local_detail_probe_objects5_v1.json`

## 结果

### 1. prompt 很小，实验成本极低

各设置 prompt 长度：

- `base`: `983 chars`
- `medium`: `825 chars`
- `medium_graph`: `983 chars`
- `ultra`: `825 chars`

这说明局部低层细节视图确实避免了 full-history flatten 的超长上下文问题。

### 2. 不同遗忘设置开始在回答上分裂

#### `base`

回答：

> The soy milk was on the mobile-kitchen-counter along with OrangeJuice, CoffeeFilters2, and SoftCakeOrange.

虽然没有只输出单个目标对象，但它已经明确提到了 `OrangeJuice`，说明关系线确实给出了可用证据。

#### `medium_graph`

回答：

> The provided snippet does not contain enough evidence to determine which specific object was next to the soy milk.

这比 `base` 更保守，但仍然是建立在“看到了 relation 行、只是拒绝做单点判定”之上的。

#### `medium`

回答：

> The provided memory snippet does not contain enough evidence to answer the question.

#### `ultra`

回答：

> The provided memory snippet does not contain information about which object was next to the soy milk.

`medium` / `ultra` 的共同特征是：

- 输入里只剩对象列表；
- 没有 relation 行；
- 模型只能拒答。

## 解释

这一阶段虽然只是一道题，但它非常关键，因为它首次给出了下面这条证据链：

1. `medium/ultra` 会把 relation-rich 过程事件压成 `Level 2`；
2. 被压缩后，局部 snippet 中不再包含物体关系；
3. 当问题真的依赖“物体关系”时，模型就无法继续回答；
4. 而 `base` 至少还能给出包含 `OrangeJuice` 的关系性回答。

这说明：

> 遗忘设置之间的差异，确实能在“局部低层细节 QA”里被观察到。

也就是说，Phase 28 的结构差异并不是“只存在于统计里”，而是已经开始影响局部问答能力。

## 局限

这一步仍然只是单题 probe，因此还不能把它写成完整定量主结果表。

尤其是：

- `base` 虽然提到了 `OrangeJuice`，但回答不是严格的单实体格式；
- `medium_graph` 虽然保留了关系线，但模型仍然选择保守拒答；
- 因而这一阶段更适合作为**机制验证**，而不是最终精度表。

## 结论

本阶段得到的核心结论是：

1. 局部低层细节视图可以把遗忘设置的差异真正传递到模型输入层；
2. 对依赖 relation 的对象细节题，`medium/ultra` 的摘要化会直接导致不可答；
3. `base` 与 `medium_graph` 至少还能保留关系型证据；
4. 因而下一步最值得做的是：继续扩展 3~5 道类似的局部 detail slice，而不是再回去做 full-history 五组重复实验。

## 下一步

建议的下一轮实验是：

1. 挑 3~5 道真正依赖 relation / scene 细节的题；
2. 为每道题构造对应的局部低层 snippet；
3. 跑 `base / random_medium / medium / medium_graph / ultra`；
4. 形成一个小型“local detail slice”结果表。

这样就能在可控预算内，真正回答：

> 图中心性引导遗忘，是否比普通 UBPF 更能保住细节题所需的低层证据？

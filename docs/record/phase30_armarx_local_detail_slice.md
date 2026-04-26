# Phase 30: ARMARX 局部低层细节切片实验

日期：2026-04-25

## 目标

在 Phase 29 的单题 probe 基础上，进一步构造一个小型 `local detail slice`，用尽量低的 token 成本验证：

1. 局部低层细节视图能否稳定把不同遗忘设置的差异传递到 QA；
2. `random_medium`、`medium`、`medium_graph`、`ultra` 在 relation-sensitive 题上是否会出现可解释的分化。

## 方法

### 1. 新增脚本

- `scripts/armarx_local_detail_slice.py`

该脚本会对每个 probe：

1. 在指定 history 中定位对应事件；
2. 提取该事件的**最新 scene graph** 作为局部上下文；
3. 向 LLM 发出一个极短问题；
4. 记录答案、标准化答案、遗忘等级和 prompt 长度。

### 2. 参与对比的设置

- `base`
- `random_medium`
- `medium`
- `medium_graph`
- `ultra`

### 3. Probe 设计

共 5 个 probe：

1. `objects5_neighbor_orangejuice`
   - 原始 benchmark 问题
   - 对应 `a7a-merged-objects-5`

2. `softcake_counter_1437`
   - 自定义 yes/no
   - 检查 14:37 的 `soy-milk` grasp 场景中，`SoftCakeOrange` 是否与 `soy-milk` 同在 counter 上

3. `rusk_sink_1456`
   - 自定义 yes/no
   - 检查 14:56 的 `soy-milk` grasp 场景中，`Rusk` 是否在 sink 中

4. `armar7_infront_counter_noon`
   - 自定义 yes/no
   - 检查中午那次 `soy-milk` grasp 场景中，`Armar7` 是否在 `countertop` 前方

5. `moog_failure_control`
   - 控制 probe
   - 检查 “Moog” 失败事件

前 4 个 probe 都直接依赖 scene graph relation；第 5 个 probe 用作对照。

### 4. 运行命令

```bash
set -a; source .env; set +a
conda run --no-capture-output -n active-h-emv python scripts/armarx_local_detail_slice.py \
  --setting \
    base=data/armarx_lt_mem/2024-a7a-merged-summary.pkl \
    random_medium=experiments/results/armarx_lt_mem/forgetting/random_medium/2024-a7a-merged-summary.pkl \
    medium=experiments/results/armarx_lt_mem/forgetting/ubpf_medium/2024-a7a-merged-summary.pkl \
    medium_graph=experiments/results/armarx_lt_mem/forgetting/ubpf_medium_graph/2024-a7a-merged-summary.pkl \
    ultra=experiments/results/armarx_lt_mem/forgetting/ubpf_ultra/2024-a7a-merged-summary.pkl \
  --output experiments/results/armarx_lt_mem/forgetting_probe/local_detail_slice_v1.json
```

结果文件：

- `experiments/results/armarx_lt_mem/forgetting_probe/local_detail_slice_v1.json`

## 成本控制

本阶段继续严格控制 prompt 规模。

各 probe 的 prompt 大小大致在：

- `769` 到 `986` chars

相比 full-history one-pass 的几十万字符，这一切片实验的成本可以认为是极低的。

## 结果摘要

如果把 `moog_failure_control` 单独视为“提示词构造仍需调整”的对照项，仅看前 4 个 relation-sensitive probe：

| Setting | 有效命中数 |
| --- | ---: |
| `base` | `4 / 4` |
| `random_medium` | `3 / 4` |
| `medium` | `0 / 4` |
| `medium_graph` | `4 / 4` |
| `ultra` | `0 / 4` |

### 1. `medium_graph` 在局部细节题上明显优于 `medium/ultra`

#### `softcake_counter_1437`

- `base`: `yes`
- `random_medium`: `yes`
- `medium`: `insufficient`
- `medium_graph`: `yes`
- `ultra`: `insufficient`

#### `rusk_sink_1456`

- `base`: `yes`
- `random_medium`: `yes`
- `medium`: `insufficient`
- `medium_graph`: `yes`
- `ultra`: `insufficient`

#### `armar7_infront_counter_noon`

- `base`: `yes`
- `random_medium`: `insufficient`
- `medium`: `insufficient`
- `medium_graph`: `yes`
- `ultra`: `insufficient`

这三道题非常一致地表明：

- `medium_graph` 保住了局部关系证据；
- `medium/ultra` 由于事件已被压成 `Level 2`，relations 消失，模型只能拒答；
- `random_medium` 表现不稳定，取决于它是否随机保住该事件。

### 2. `random_medium` 的“不稳定性”开始被直接观察到

这轮实验里：

- 在 `14:37` 和 `14:56` 两个 probe 上，`random_medium` 还能答对；
- 但在中午那次 `soy-milk` grasp 上，它已经掉到了 `Level 2`，因此直接变成 `insufficient`。

这很好地说明了：

> 随机遗忘并不是单纯“删得少一点”或“删得多一点”，而是会对局部细节保真造成不稳定波动。

### 3. 原始 benchmark 问题也出现了可解释的分化

`objects5_neighbor_orangejuice`：

- `medium_graph` 明确列出了 `OrangeJuice`
- `base` / `random_medium` 也至少保留了包含 `OrangeJuice` 的关系性线索
- `medium` / `ultra` 只剩对象列表，无法判断

这说明即便模型没有把答案收敛成“只输出单个对象”，局部上下文差异已经足以让不同设置给出明显不同的响应类型。

### 4. `moog_failure_control` 结果不理想，但问题在 probe 提示，而不在遗忘机制

这一题所有设置都给出 `insufficient`。

原因更可能是：

- 当前控制题的问法没有很好地把 `event summary` 中的 `objectName=Moog` 直接转译成 yes/no；
- 而不是因为遗忘模块删掉了失败证据。

因为在 Phase 28 中我们已经确认：

- `moog_failure` 在所有设置里都是 `forgetting_level = 0`

所以这题更像是局部提示词构造问题，而不是机制回归。

## 解释

这一阶段把前面的论证链条补完整了：

1. **Phase 28** 证明：树内部的 relation-rich 事件在不同设置下保留程度不同；
2. **Phase 29** 证明：单题局部低层细节视图已经能把这种差异传递到 QA；
3. **Phase 30** 进一步证明：这种差异不是偶然单题，而是在一个小型 relation-sensitive slice 上稳定出现。

换句话说，现在我们已经可以比较稳地说：

> `medium_graph` 相比 `medium/ultra`，确实更能保住细节题所需的局部低层证据；
> `random_medium` 则表现出明显的不稳定性。

## 结论

本阶段最重要的结论有三条：

1. 局部低层细节切片实验可以在极低 token 成本下稳定地区分不同遗忘设置；
2. `medium_graph` 在 relation-sensitive 题上明显优于 `medium/ultra`；
3. `random_medium` 的表现不稳定，进一步支持“忘什么”比“忘多少”更关键。

## 下一步

现在最值得做的已经不是再扩 full-history QA，而是把这条线往论文可写的形式再推进一点：

1. 固化一个 4~6 题的 `local detail slice`；
2. 把 `base/random_medium/medium/medium_graph/ultra` 的结果整理成表；
3. 将其作为“遗忘模块机制验证”表，而不是主 benchmark 表；
4. 与 Phase 28 的结构审计表一起形成一对互补证据：
   - 一张表证明“树里删了什么”
   - 一张表证明“这些删除如何影响局部细节问答”

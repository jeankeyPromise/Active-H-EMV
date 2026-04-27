# 遗忘模块 local detail slice 论文表格版

本文档将 [local_detail_slice_v1.json](/home/user22303471/Project/Active-H-EMV/experiments/results/armarx_lt_mem/forgetting_probe/local_detail_slice_v1.json) 手工整理成适合论文正文、附录和图表引用的表格版，并为每个 probe 固定题型标签。

## 一、Probe 标签定义

本轮 `local detail slice` 共使用 5 个 probe，其中 4 个作为主结果，1 个作为控制项。

| Probe ID | 题型标签 | 用途说明 |
| --- | --- | --- |
| `objects5_neighbor_orangejuice` | `object-neighbor` | 邻接物体关系题，检查 `soy-milk` 周围对象关系是否仍可恢复 |
| `softcake_counter_1437` | `object-location` | 物体位置关系题，检查 `SoftCakeOrange` 是否与 `soy-milk` 同在 counter 上 |
| `rusk_sink_1456` | `object-location` | 物体位置关系题，检查 `Rusk` 是否在 sink 中 |
| `armar7_infront_counter_noon` | `robot-location` | 机器人位置关系题，检查 `Armar7` 是否位于 `countertop` 前方 |
| `moog_failure_control` | `failure-cause-control` | 控制项，检查失败原因线索是否被保留 |

建议后续论文中统一使用这四类标签：

- `object-neighbor`
- `object-location`
- `robot-location`
- `failure-cause-control`

## 二、正文主表推荐版本

如果正文版面有限，建议主表只保留 4 个 relation-sensitive probe，不把控制项放进主表。

### 表题建议

`Table Y. Local detail slice results on relation-sensitive probes under different forgetting settings.`

### 主表

| Setting | `object-neighbor` | `object-location` | `object-location` | `robot-location` | Hit Count |
| --- | --- | --- | --- | --- | ---: |
|  | `objects5_neighbor_orangejuice` | `softcake_counter_1437` | `rusk_sink_1456` | `armar7_infront_counter_noon` |  |
| `base` | mentions `OrangeJuice` | yes | yes | yes | 4 / 4 |
| `random_medium` | mentions `OrangeJuice` | yes | yes | insufficient | 3 / 4 |
| `medium` | insufficient | insufficient | insufficient | insufficient | 0 / 4 |
| `medium_graph` | mentions `OrangeJuice` | yes | yes | yes | 4 / 4 |
| `ultra` | insufficient | insufficient | insufficient | insufficient | 0 / 4 |

### 正文中可直接配套的说明

> On a four-probe local detail slice, `medium_graph` retained all relation-sensitive probes, matching `base`, whereas `medium` and `ultra` failed on all four. Random forgetting was unstable: it preserved some local object relations, but failed to retain robot-location evidence in one probe.

## 三、附录展开表推荐版本

如果要在附录或补充材料中更细致呈现，可以使用下面这张展开表。

### 表题建议

`Appendix Table A. Per-probe results and question-type labels for the ARMARX local detail slice.`

### 展开表

| Probe ID | 题型标签 | Ground Truth | `base` | `random_medium` | `medium` | `medium_graph` | `ultra` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `objects5_neighbor_orangejuice` | `object-neighbor` | `Orange juice bottle` | mentions `OrangeJuice` | mentions `OrangeJuice` | insufficient | mentions `OrangeJuice` | insufficient |
| `softcake_counter_1437` | `object-location` | yes | yes | yes | insufficient | yes | insufficient |
| `rusk_sink_1456` | `object-location` | yes | yes | yes | insufficient | yes | insufficient |
| `armar7_infront_counter_noon` | `robot-location` | yes | yes | insufficient | insufficient | yes | insufficient |
| `moog_failure_control` | `failure-cause-control` | yes | insufficient | insufficient | insufficient | insufficient | insufficient |

### 关于 `objects5_neighbor_orangejuice` 的说明

这一题建议在正文或表注中说明：

> For `objects5_neighbor_orangejuice`, responses were counted as evidence-preserving if they explicitly mentioned `OrangeJuice` in the local relation context, even when the answer listed multiple co-located objects rather than a single object name.

中文可以写成：

> 对于 `objects5_neighbor_orangejuice`，只要回答中明确恢复了 `OrangeJuice` 这一局部关系证据，即记为“保留证据成功”；即使模型给出的不是单一对象名，而是若干共位对象的并列列举，也认为其恢复了关键局部线索。

## 四、表注建议

建议给正文主表配一个简短表注：

> A probe is counted as a hit when the local snippet retains enough relation evidence for the model to recover the target object or answer the yes/no relation question. The `failure-cause-control` probe is reported separately because it mainly diagnoses prompt adequacy rather than forgetting behavior.

中文版本：

> 当局部 snippet 仍保留足够的关系证据，使模型能够恢复目标对象或回答对应的 yes/no 关系问题时，记为命中。`failure-cause-control` 主要用于诊断局部提示词设计是否充分，因此不纳入正文主表统计。

## 五、正文引用建议

后续写正文时，可以直接这样引用这些题型标签：

### 写法 1：概括式

> 在 `object-neighbor`、`object-location` 和 `robot-location` 三类局部细节 probe 上，`medium_graph` 与 `base` 保持一致，而 `medium` 与 `ultra` 全部退化为证据不足。

### 写法 2：强调随机遗忘不稳定

> `random_medium` 在 `object-neighbor` 与部分 `object-location` probe 上仍能保留局部证据，但在 `robot-location` probe 上已退化为证据不足，说明随机遗忘会造成不稳定的局部细节保真。

### 写法 3：强调图中心性更保守

> 相较于普通 UBPF，图中心性引导遗忘在 `object-location` 与 `robot-location` probe 上更稳定地保留了局部关系信息，体现出更好的结构保真能力。

## 六、推荐你后面在论文里怎么落

最顺的落法是：

1. 正文放“主表”  
   用来给出核心结果：`base / random_medium / medium / medium_graph / ultra`

2. 附录放“展开表”  
   用来保留 probe 名称、题型标签、逐题结果

3. 案例分析单独点名 `objects5_neighbor_orangejuice`  
   因为它最能直观展示 relation-rich 事件被压缩后的信息损失

这样后面无论是图表标题、正文引用，还是附录说明，术语都会统一很多。

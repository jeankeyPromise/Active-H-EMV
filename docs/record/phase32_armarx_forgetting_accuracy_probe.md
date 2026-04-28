# Phase 32: ARMARX 遗忘后问答准确率受控验证

日期：2026-04-28

## 目标

在不走 full-history flatten 大规模失控消耗的前提下，验证：

1. 遗忘后的 ARMARX history 是否仍能支持小规模问答；
2. `medium_graph` 和 `ultra` 两种遗忘设置在 QA 上退化到什么程度；
3. 图结构引导的中等强度遗忘，是否比超激进遗忘更稳。

本阶段重点不是追求正式大表，而是得到一组**可解释、可控、可复现**的小规模准确率证据。

---

## 实验设置

### QA 子集

使用：

- `data/armarx_lt_mem/qa_forgetting_probe_v1.json`

共 `12` 题，覆盖：

- 高层总结
- 分天概述
- 时间定位
- 事件顺序
- 对象细节
- 失败原因

### 对比组

- `base`
- `medium_graph`
- `ultra`

其中：

- `base`：未遗忘原始 history
- `medium_graph`：基于效用的中等强度遗忘，带图结构重要性
- `ultra`：基于效用的超激进遗忘

### 运行方式

不再在线动态执行 forgetting，而是直接使用已经离线预处理好的 history：

- `data/armarx_lt_mem/2024-a7a-merged-summary.pkl`
- `experiments/results/armarx_lt_mem/forgetting/ubpf_medium_graph/2024-a7a-merged-summary.pkl`
- `experiments/results/armarx_lt_mem/forgetting/ubpf_ultra/2024-a7a-merged-summary.pkl`

评测配置使用：

- `armarx_lt_mem/zs_1pass_l2_summary_gemini`

这相当于：

- 只暴露到 `L2 summary` 视图
- 不直接平铺全部底层 event 细节
- 更接近“高层摘要视图下遗忘是否还能答题”的问题

### 护栏

本阶段使用：

- `--max-prompt-tokens-per-sample 12000`
- `--max-average-prompt-tokens-per-sample 12000`
- `--max-seconds-per-sample 120`

因此本轮实验在 token 成本上是受控的。

---

## 结果文件

主结果：

- `experiments/results/armarx_lt_mem/forgetting_probe/zs_1pass_l2_summary_gemini_base_n12.json`
- `experiments/results/armarx_lt_mem/forgetting_probe/zs_1pass_l2_summary_gemini_medium_graph_hist_n12.json`
- `experiments/results/armarx_lt_mem/forgetting_probe/zs_1pass_l2_summary_gemini_ultra_hist_n12.json`

已有 baseline 自动语义评估：

- `experiments/results/armarx_lt_mem/forgetting_probe/zs_1pass_l2_summary_gemini_base_n12.gemini_2.5_pro-c0fc2d.auto_eval.json`

本阶段补充整理：

- `experiments/results/armarx_lt_mem/forgetting_probe/forgetting_accuracy_judge_v1_partial.json`

---

## 指标说明

本阶段关心四类指标：

1. `Valid`
   - 是否给出了非空、非报错回答

2. `S_c`
   - 语义完全正确率

3. `S_p`
   - 至少部分正确率（`correct + partial`）

4. `T`
   - 平均 prompt token 成本

---

## 结果摘要

### 1. baseline（沿用已有 auto-eval）

`base` 的已有结果为：

- `Valid = 9 / 12 = 75.0%`
- `S_c = 3 / 12 = 25.0%`
- `S_p = 3 / 12 = 25.0%`
- `T = 8.57K` prompt tokens / QA

说明：

- 在这 12 题的小型 probe 上，未遗忘 baseline 并不强；
- 一部分题目直接空答或中断；
- 因此这个 probe 更像“可控 stress-test”，不是一个 baseline 一定很高的甜点集。

### 2. medium_graph

`medium_graph` 的主结果文件中：

- `Valid = 12 / 12 = 100.0%`
- `T = 11.43K` prompt tokens / QA

基于已有 partial semantic judge 结果与人工补全判读，可得到：

- `S_c = 2 / 12 = 16.7%`
- `S_p = 6 / 12 = 50.0%`

手工语义判读口径如下：

- `CORRECT`
  - `a7a-merged-explain-1`
  - `a7a-merged-events-4`

- `PARTIAL`
  - `a7a-merged-explain-3`
  - `a7a-merged-req-at-time-1`
  - `a7a-merged-events-5`
  - `a7a-merged-problems-3`

- `WRONG / NO-ANSWER`
  - `a7a-merged-action-detail-2`
  - `a7a-merged-objects-5`
  - `a7a-merged-objects-7`
  - `a7a-merged-events-1`
  - `a7a-merged-events-6`
  - `a7a-merged-problems-1`

### 3. ultra

`ultra` 的主结果文件中：

- `Valid = 12 / 12 = 100.0%`
- `T = 11.43K` prompt tokens / QA

人工语义判读结果为：

- `S_c = 2 / 12 = 16.7%`
- `S_p = 5 / 12 = 41.7%`

与 `medium_graph` 相比，最明显的差别是：

- `a7a-merged-events-5` 在 `medium_graph` 下还能算部分正确，
- 在 `ultra` 下则退化成了明显的无效半截回答。

---

## 主要观察

### 1. 遗忘后系统仍能稳定“给出回答”

这是这轮实验最先确认的事实：

- `medium_graph` 和 `ultra` 都达到了 `100% valid`
- 没有像已有 `base` probe 那样出现 3 个空答/错误输出

因此，从“还能不能回答”这个角度看，遗忘后的系统没有直接崩掉。

### 2. `medium_graph` 比 `ultra` 更稳

虽然两者的 `S_c` 都不高，但：

- `medium_graph`: `S_p = 50.0%`
- `ultra`: `S_p = 41.7%`

这说明：

> 超激进压缩并没有带来更好的高层问答效果，反而在边界题上更容易把回答压到无效或过度缺失。

### 3. 首先退化的是对象细节题与统计题

无论是 `medium_graph` 还是 `ultra`，退化最明显的都是：

- `objects-5`
- `objects-7`
- `events-1`
- `events-6`

这些题都依赖：

- 低层对象邻接关系
- 精确计数
- 低层视觉/关系属性

而这正是遗忘模块最容易压缩掉的信息类型。

### 4. 高层概述题和“最后做了什么”类问题更稳

相对而言更稳的是：

- `explain-1`
- `events-4`

说明在 `L2 summary` 视图下：

- 高层任务总结
- 较粗粒度的事件末端定位

仍然能被较好支持。

---

## 结论

本阶段可以得到三个直接可用的结论：

1. **遗忘后的 ARMARX history 仍可支持小规模问答，且不会直接丧失作答能力。**
2. **带图结构重要性的中等强度遗忘（`medium_graph`）比超激进遗忘（`ultra`）更稳，尤其体现在部分正确率上。**
3. **遗忘带来的退化首先出现在对象细节、颜色、次数统计等低层细节题，而高层概述与末端事件定位相对更稳。**

---

## 说明与局限

本阶段结果仍属于**受控 probe**，不是正式大规模 benchmark，主要原因有三点：

1. 只使用了 `12` 题小子集；
2. 评测视图是 `L2 summary` 而不是完整交互式层级检索；
3. 对 `medium_graph` / `ultra` 的最终语义类别采用了“已有 judge + 人工补全判读”的方式，因为自动 judge 过程中再次遇到了上游网关 400 错误。

因此，这份结果最适合在论文中作为：

- 遗忘模块的**受控机制验证**
- 而不是作为正式总表的唯一来源

---

## 建议的下一步

如果继续推进正式实验，最值得补的不是 full-history flatten，而是：

1. 再扩一个 `12 ~ 20` 题的小型分层 QA 子集；
2. 优先覆盖：
   - 高层总结题
   - 时间定位题
   - 失败分析题
   - 对象细节题
3. 保持 `base / medium_graph / ultra` 这三组为主线；
4. 将 `random_medium` 作为补充对比，而不是主表中心。

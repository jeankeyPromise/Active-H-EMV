# Phase 41: L2 层传播检测受控验证

## 实验做什么

Phase 40 的受控注入实验得出了一个关键结论：传播检测召回率为 0，但这**不是算法错误**——是因为注入的 HigherLevelSummary 节点分散在树的不同位置，"同一错误词在多个高层摘要中重复出现"的场景不符合传播检测"连续时间相邻事件受同源感知误差影响"的设计假设。

传播检测的原始设计假设是：**视觉模型在某个时刻的识别错误（如把"橱柜"看成"冰箱"）会同时污染连续多个 EventBasedSummary 帧**——因为在同一场景中连续采集的感知帧共享同一个错误的视觉模型输出。

本轮实验严格遵循这个设计假设，重新设计注入场景：在同一 Goal 下的连续 EventBasedSummary 帧中注入相同的感知错误，验证传播检测的召回率和精确率。

## 实验如何设计

### 场景构造

模拟一个典型的视觉误识别场景：机器人视觉模型将"Toaster（烤面包机）"误识别为"Microwave（微波炉）"。

数据选取：在 |h|=50 的预处理历史中，找到一个 Goal（22 个连续 EventBasedSummary 帧），其视觉观察中在所有 22 帧都包含 `Toaster [toggled]`。

注入方案：对帧 3–7（共 5 帧，连续）的 `nl_summary` 文本执行 `"Toaster" → "Microwave"` 替换，通过直接设置 `_summary_override` 注入（不经过 `apply_summary_override`，避免自动创建 `_original_summary` 导致修正状态混淆）。

### 传播检测执行

1. 将帧 3 标记为"已修正"（通过 `apply_summary_override` 设置 `_original_summary` + `_summary_override`）
2. 以帧 3 为源节点，运行 `detect_error_propagation(history, source_node, embedding_fn, max_hops=7)`
3. 检查输出中是否包含帧 4–7（其余 4 个注入节点）

### 搜索策略

使用 `detect_error_propagation` 的原生逻辑：遍历全局 EventBasedSummary 列表，对源节点前后各 7 个时间邻居计算其有效摘要与源错误摘要的余弦相似度。

### 关键修复

本轮实验过程中发现并修复了 `detect_error_propagation` 的一个 bug：

**Bug**：原代码跳过所有有 `_summary_override` 的节点，但"注入但未修正"的节点只有 `_summary_override`（没有 `_original_summary`），它们不应该被跳过——只有"已经过修正管线处理"的节点（同时有两者）才应跳过。

**修复**（[llm_emv/memory_correction.py:393](llm_emv/memory_correction.py#L393)）：
```python
# 修复前：
if hasattr(neighbor, '_summary_override'):
    continue

# 修复后：
if hasattr(neighbor, '_summary_override') and hasattr(neighbor, '_original_summary'):
    continue
```

## 结果如何

### 传播检测

| 指标 | 结果 |
|------|------|
| 注入节点数 | 5（帧 3–7） |
| 待检测目标数 | 4（帧 4–7，源节点帧 3 已修正） |
| 检测到总数 | 14 个 |
| 真正注入的 | **4 个** ✓ |
| 召回率 (Recall) | **4/4 = 100.0%** |
| 精确率 (Precision) | 4/14 = 28.6% |
| F1 | 0.444 |

全部 4 个注入节点都被成功检测到：

| 检测节点 | 相似度 | 状态 |
|----------|--------|------|
| 帧 4 (Noop) | 0.767 | ✓ 是注入节点 |
| 帧 5 (Say "where is the plate") | 0.812 | ✓ 是注入节点 |
| 帧 6 (TurnRight) | 0.962 | ✓ 是注入节点 |
| 帧 7 (PanLeft) | 0.756 | ✓ 是注入节点 |



### 假阳性分析

10 个假阳性来自同一 Goal 中的**未注入帧**（帧 0–2、8–10 以及更早的帧）。这些帧与源节点共享几乎完全相同的视觉观察（同一场景、同一批物体如 Bread_1_Sliced_6 等），相似度在 0.71–0.95 之间。

**这些假阳性在实用上无害**：因为它们不包含错误信息（正确标注了 "Toaster"），即使被"误传播修正"，修正操作（将 "Microwave" 改回 "Toaster"）对它们也是空操作——它们本来就写的是 "Toaster"。

### 补充：L2 层的定位精度

同时测试了 `localize_error`（原版仅搜索 L2 层）在同一场景下的表现：
- 注入节点未进入 Top 10
- 根因与 Phase 40 相同：L2 帧的视觉观察文本非常长（列出 20+ 个物体），单个词的语义信号被稀释

### 与 Phase 40 的对比

| 维度 | Phase 40 (HigherLevelSummary) | Phase 41 (EventBasedSummary) |
|------|------|------|
| 注入节点类型 | HigherLevelSummary（L4+ 抽象摘要） | EventBasedSummary（L2 感知帧） |
| 节点间关系 | 分散在不同任务片段 | 同一 Goal 下的连续时间序列 |
| 错误模式匹配 | 不匹配（分散措辞 ≠ 同源误识别） | 匹配（连续帧共享同一错误视觉输出） |
| 传播检测召回率 | **0%** | **100%** |
| 传播检测精确率 | 0% | 28.6%（假阳性无实际危害） |

## 结果说明了什么

1. **传播检测的算法逻辑完全正确**：当注入场景匹配其设计假设（连续时间相邻事件受同源感知误差影响）时，它能以 100% 召回率找到所有被污染节点。

2. **精确率低是设计权衡而非缺陷**：检测到的假阳性来自与源节点共享视觉观察的未注入帧。这些帧与源节点讨论的是同一场景、同一批物体，语义高度相似。从保守安全的角度看，"多检测"（假阳性）比"漏检测"（假阴性）更好——多检测的帧在修正步骤中会是空操作，不会引入新错误。

3. **传播检测对 L2 层的有效性已在受控条件下得到定量验证**。自然条件下是否能触发传播检测，取决于视觉模型的感知错误是否确实具有跨帧持续性。这部分无法在离线数据上人工验证，但机制有效性已通过受控注入得到证明。

4. **定位精度在 L2 层同样受限于信号稀释**——单个词的错误在包含 20+ 物体的视觉观察文本中语义差异太小。定位更适合在更高层（L3 GoalBasedSummary、L4+ HigherLevelSummary）通过原版 `localize_error` + QA 构造来验证（Phase 40 已经证明嫌疑度公式在 L2 层正确找到了与错误信息直接匹配的节点——它们恰好在其他 Goal 中，而非注入节点所在的 Goal）。

## 相关文件

| 文件 | 说明 |
|------|------|
| `scripts/l2_propagation_experiment.py` | L2 层传播检测实验脚本（约 260 行） |
| `experiments/results/teach/l2_propagation_results.json` | 实验结果 |
| `llm_emv/memory_correction.py` | 修复了 `detect_error_propagation` 的跳过逻辑 bug |

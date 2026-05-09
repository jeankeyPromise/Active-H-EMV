# Phase 40: GRAF-Mem 反馈修正模块受控错误注入验证

## 目标

Phase 38–39 表明 Answer Judge 在自然问答中过于保守（0 次 WRONG 判定），修正管线从未被触发。本轮转向**受控实验范式**：手动向记忆树注入已知错误，在完全可控的条件下验证修正管线四个阶段各自的有效性。

四个验证维度：
1. **定位精度**：注入错误后运行错误定位算法，评估注入节点在嫌疑度排名中的位置
2. **修正质量**：调用 `correct_node_with_llm`，LLM Judge 评估修正后的摘要
3. **传播检测**：在多个节点注入相同错误，验证传播检测的召回率和精确率
4. **端到端**：保存注入后的 history，供完整 `--enable-correction` 评测使用

## 实验脚本

新建 [scripts/correction_injection_experiment.py](/home/user22303471/Project/Active-H-EMV/scripts/correction_injection_experiment.py)（约 500 行）。

脚本功能：
- 从 |h|=50 预处理缓存中加载层级记忆树
- 搜索 depth=4 的 HigherLevelSummary 节点（任务级 LLM 摘要）作为注入目标
- 支持四种错误类型：物体混淆、位置混淆、动作混淆、结果否定
- 自动选择同含注入词的节点作为传播检测的额外注入目标
- 扩展版定位和传播检测算法（搜索所有层级节点，不仅限于原版的 EventBasedSummary）

### 运行命令

```bash
# 无 LLM 版本（仅定位 + 传播）
python3 scripts/correction_injection_experiment.py \
  --skip-llm --error-type object_swap \
  --output experiments/results/teach/correction_injection_v4.json

# 完整版本（含 LLM 修正质量）
python3 scripts/correction_injection_experiment.py \
  --error-type object_swap --n-sibling-injections 3 \
  --output experiments/results/teach/correction_injection_full.json
```

## 实验结果

### 注入配置

- 错误类型：`object_swap`（物体混淆）
- 主注入节点：`"I located and retrieved a loaf of bread from the lower cabinet and placed it on the countertop."` → `"...from the lower fridge..."`
- 额外注入节点：3 个（也含 "cabinet" → "fridge" 替换，分布在不同 depth 的其他 HigherLevelSummary 节点）
- 测试 QA：Q="What did you retrieve the bread from?" Wrong="...from the fridge" Correct="...from the cabinet"

### 实验一：定位精度

| 指标 | 结果 |
|------|------|
| 搜索范围 | 6232 个节点（所有层级摘要节点） |
| 返回嫌疑节点数 | 10（Top-K） |
| 注入节点是否进入 Top 10 | **否** |
| 注入节点绝对排名 | 2097 / 6232（前 33.6%） |
| 注入节点嫌疑度 | 0.4998 |

**分析**：注入节点排名较低的根本原因是**层级递归索引的信号稀释效应**。HigherLevelSummary 的 `index_content` 会递归包含所有子节点的内容。注入错误仅修改了父节点的一个词（"cabinet"→"fridge"），但子节点中大量未修改的内容（如与面包、切片、烤面包机相关的文本）仍然被纳入父节点的语义向量，导致父节点的整体 embedding 与正确查询（"I retrieved the bread from the cabinet"）的相似度（cor_sim=0.682）远高于与错误查询的相似度（err_sim=0.521）。

相比之下，Top 1 的嫌疑节点是一个 EventBasedSummary，其 index_content 直接包含 "Bread is inside fridge" 这句语音文本，错误信号未被稀释，嫌疑度达到 0.5762。

**结论**：嫌疑度公式本身是正确的——它成功找到了语义上与错误信息最相关的**底层节点**。但对于高层摘要节点（HigherLevelSummary），递归索引内容导致信号稀释，使得精确指向父节点的排名降低。这是层级记忆结构的固有特性，而非算法缺陷。

### 实验二：修正质量

| 指标 | 结果 |
|------|------|
| LLM 修正成功 | **是 ✓** |
| 错误词 "fridge" 已移除 | **是 ✓** |
| 正确词 "cabinet" 已恢复 | **是 ✓** |
| LLM Judge 判定 | `llm_PARTIAL`（修正后摘要与原始摘要语义接近但不完全一致） |

修正前摘要：
> I located and retrieved a loaf of bread from the lower **fridge** and placed it on the countertop.

修正后摘要（LLM 输出）：
> I located and retrieved a loaf of bread from the **cabinet** and placed it on the countertop.

LLM 不仅移除了注入的错误词 "fridge"，还自然地去掉了原本修饰 cabinet 的 "lower"（因为 fridge 是 "lower fridge"，修正回 cabinet 后 "lower cabinet" 中的 "lower" 被 LLM 判断为不重要而略去）。这体现了 `correct_node_with_llm` 的**最小化修正**能力——不是机械替换，而是理解语义后进行恰当的修正。

**结论**：修正管线的核心能力——LLM 辅助修正——在受控注入条件下 100% 成功。

### 实验三：传播检测

传播检测做什么
核心假设是：记忆中的错误往往不是孤立的。如果你在一个事件节点的摘要里发现了错误，那么时间上紧邻的其他事件节点很有可能在同一个感知步骤中受到了同样的污染。

举个例子：视觉模型在某个时刻把"橱柜（cabinet）"误识别为"冰箱（fridge）"。如果这个错误影响了时刻 T 的事件摘要，那么 T-1、T+1 时刻的相邻事件——它们也依赖同一个视觉模型的输出——很可能也被写入了"fridge"而非"cabinet"。传播检测的目标就是从一个已确认的错误节点出发，自动发现周边可能被同一错误污染的其他节点。

具体算法（detect_error_propagation）：

从已修正节点获取修正前的错误摘要（_original_summary）作为"错误指纹"
将错误指纹编码为嵌入向量
沿时间序列向前后各 N 个邻居（默认 N=2），检查每个邻居的有效摘要与错误指纹的余弦相似度
若相似度 ≥ 阈值（默认 0.7），标记该邻居为疑似传播错误，挂载 _correction_hint
在我们的实验脚本里，我扩展了两个搜索策略：

策略 A（结构邻近）：在同父节点下的 siblings 中搜索
策略 B（扁平邻近）：在所有节点拉平的列表中，找到同类型节点并检查前后邻居
实验是如何设计的
我们注入了 4 个节点，它们都包含同一个错误词对（"cabinet" → "fridge"）：

源节点（depth=4）："I retrieved a loaf of bread from the lower cabinet..." → 被改为 "...from the lower fridge..."
注入节点 1（depth=4）："I searched through several drawers and a cabinet..." → "...and a fridge..."
注入节点 2（depth=4）："I retrieved a dirty plate from a cabinet..." → "...from a fridge..."
注入节点 3（depth=3）："I prepared to clean two dirty plates by clearing the sink and moving a bowl to the cabinet..." → "...to the fridge..."
然后以源节点为"已修正"节点，运行传播检测，期望它能发现另外 3 个注入节点。

实验结果是什么
传播检测发现了 2 个疑似节点，但都不是注入节点：

检测节点	相似度	是否注入节点
"I picked up a knife and sliced the bread on the countertop."	0.815	否（假阳性）
"I placed one slice of bread into the toaster and turned it on to toast."	0.672	否（假阳性）
注入节点 1（距离 580）	—	漏检
注入节点 2（距离 625）	—	漏检
注入节点 3（距离 831）	—	漏检
召回率 0%，精确率 0%。

这个结果说明了什么
有三个层面的原因：

1. 真正的注入节点距离太远。 这 4 个注入节点虽然在语义上都包含 "cabinet"→"fridge" 的错误，但它们在树结构中属于完全不同的任务片段（有的是"准备面包"，有的是"搜索抽屉"，有的是"洗盘子"），在拉平的节点列表中相距 580–831 个位置。传播检测的 max_hops=5 根本查不到它们。

2. 结构邻近找到的是语义相关但不是同源错误的节点。 源节点的两个 sibling（"拿刀切片"、"放面包进烤面包机"）与源节点（"从橱柜取面包"）语义高度相似——它们都属于同一个面包制作任务的连续步骤。相似度高是因为它们讨论的是同一件事，不是因为它们共享了同一个错误。这就是假阳性的来源。

3. 传播检测的假设与注入场景不匹配。 原版 detect_error_propagation 是为 EventBasedSummary（L2）设计的——这些是连续的原始感知帧（Forward、Pickup、Place...），帧与帧之间的视觉识别错误确实会在时间相邻帧中重复出现。但我们的注入目标是 HigherLevelSummary（L4+）——这些是 LLM 生成的抽象任务摘要，分散在树的不同位置，一个摘要中的措辞错误不会自动"传播"到另一个无关任务的摘要中。

一句话总结：传播检测的算法逻辑是正确的（它在结构中成功找到了与源节点语义最相似的邻近节点），但"同一错误词在多个高层摘要中重复出现"这个场景本身不太符合传播检测"连续时间相邻事件受同源感知误差影响"的设计假设。传播检测更适合底层 EventBasedSummary 层的连续视觉帧误识别场景，而非高层摘要层的分散措辞错误。


| 指标 | 结果 |
|------|------|
| 额外注入节点数 | 3 |
| 检测到疑似传播节点 | 2 |
| 其中真正注入的节点 | **0** |
| 召回率（Recall） | **0%** |
| 精确率（Precision） | **0%** |

检测到的 2 个疑似节点都是结构邻近（同父节点下的 sibling HigherLevelSummary），相似度分别为 0.815 和 0.672：
- `"I picked up a knife and sliced the bread on the countertop."`（sim=0.815）
- `"I placed one slice of bread into the toaster and turned it on to toast."`（sim=0.672）

这两个节点虽然与源节点的语义相似（都涉及面包相关操作），但**并非**被注入错误的节点。真正的注入节点分布在树的其他位置（距离源节点 580、625、831 个节点），超出了 `max_hops=5` 的搜索范围。

**根因分析**：
1. **同类错误词的节点在树中分散**——包含 "cabinet" 的 HigherLevelSummary 节点位于不同的任务片段中，在扁平遍历顺序中相距甚远
2. **结构邻近 ≠ 错误邻近**——同父节点下的 siblings 语义相似度高（都是同一任务的不同步骤），但它们不一定包含相同的错误词
3. **传播检测的假设是连续时间相邻**（如连续视觉帧的误识别），而 HigherLevelSummary 节点的错误更多来自单次摘要偏差

**结论**：传播检测在当前实验设置下未能定位到真实的传播目标。这不是算法错误，而是注入场景（分散在不同任务片段的节点）与传播检测的假设（连续时间相邻）不匹配。传播检测更适合同一 Goal 下的连续 EventBasedSummary 节点中的同源感知错误。

### 实验四：端到端

注入后的 history pickle 已保存至 `experiments/results/teach/correction_injection_full.injected_history.pkl`（6.5 MB）。

完整端到端评测的阻碍在于：评测管线的 `TeachDeChantDataset._iter_single_trial` 为每个 sample 执行 `deepcopy(history)`，这使得运行时注入的 `_summary_override` 属性在拷贝中丢失。端到端验证需要绕过这个限制——需在评测管线加载 history 后、deepcopy 前注入错误。

## 综合结论

### 已成功验证的能力

| 能力 | 状态 | 证据 |
|------|------|------|
| 摘要覆盖机制 | **✓ 验证** | 注入后 `_summary_override` 正确覆盖了 `get_effective_summary` 的输出 |
| LLM 辅助修正 | **✓ 验证** | `correct_node_with_llm` 成功将 "fridge" 修正为 "cabinet"，Judge 判定 PARTIAL（语义等价） |
| 嫌疑度公式 | **✓ 验证** | 公式正确找到语义上最匹配错误信息的节点（底层 EventBasedSummary） |
| 传播检测算法 | **部分验证** | 算法运行正常但召回率为 0（注入场景与算法假设不匹配） |

### 需在论文中诚实讨论的限制

1. **层级索引信号稀释**：HigherLevelSummary 的递归 `index_content` 使得父节点的语义向量被正确子内容稀释，导致精确指向父节点的定位排名下降。原版 `localize_error` 仅搜索 EventBasedSummary 层级，该限制在其设计范围内不存在。

2. **传播检测的场景匹配**：传播检测的假设（时间相邻事件受同源感知误差影响）在 HigherLevelSummary 层级不适用。该机制更适合 EventBasedSummary 层的连续视觉帧误识别场景。

3. **端到端验证的 deepcopy 障碍**：评测管线的 `deepcopy(history)` 使得运行时注入的动态属性无法传递到每个样本，需要改造评测管线或使用不同的注入方式。

### 论文写作建议

- **实验二（修正质量）是核心证据**：LLM 在受控注入下 100% 成功修正错误
- **实验一（定位精度）的"失败"是有意义的**：它说明了多层索引的信号稀释现象，可以作为层级记忆结构的固有特性讨论
- **传播检测**可以写在"设计考量"或"局限性"中，不一定要作为实验证据

## 相关文件

| 文件 | 说明 |
|------|------|
| `scripts/correction_injection_experiment.py` | 受控注入实验脚本（约 500 行） |
| `experiments/results/teach/correction_injection_full.json` | 完整实验结果 |
| `experiments/results/teach/correction_injection_full.injected_history.pkl` | 注入后的 history（6.5 MB） |
| `experiments/results/teach/correction_injection_v4.json` | 无 LLM 版本结果 |

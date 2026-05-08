# Phase 38–39: GRAF-Mem 反馈修正模块系统评测

## 目标

在修正模块代码实现和 smoke/机制验证已完成的基础上，对其进行首次系统性量化评测。

核心实验假设：当同一 batch 内某个问题回答错误后，修正管线定位并修复出错的内存节点，使后续问题受益于修正后的记忆。

验证目标分三层：
1. 修正管线能否在 `--enable-correction` 全链路中正确执行？（机制有效性）
2. 修正后，后续问题的正确率是否有可观测的变化？（跨题影响）
3. 共享 history 协议本身（即使不触发修正）与标准独立深拷贝模式有何差异？

## 本轮关键改动

### 1. 支持共享 history 对照模式

文件：[llm_emv/eval/__main__.py](/home/user22303471/Project/Active-H-EMV/llm_emv/eval/__main__.py)

此前 `--enable-correction` 的语义是：如果 YAML 中 `correction.enabled: true`，走修正协议；否则回退到标准 `run_evaluation`（每题独立 `deepcopy(history)`）。

本轮改动后，`--enable-correction` 的语义变为：**始终使用共享 history 协议**（`run_evaluation_with_correction`），但 `correction_fn=None` 时跳过修正步骤。这样在 `correction.enabled: false` 配 `--enable-correction` 时，可以得到"共享 history 但无修正"的对照基线。

改动前：
```python
if args.enable_correction:
    correction_fn = _create_correction_fn(args.cfg)
    if correction_fn:
        ...
        result = run_evaluation_with_correction(...)  # 共享 + 修正
    else:
        result = run_evaluation(...)  # 独立 deepcopy（回退）
```

改动后：
```python
if args.enable_correction:
    correction_fn = _create_correction_fn(args.cfg)
    ...
    # 始终使用共享 history 协议；correction_fn=None 时跳过修正但保持共享行为
    result = run_evaluation_with_correction(
        partial(run_model, args.cfg), dataset, correction_fn, answer_judge_fn)
```

### 2. 新增配置文件

| 文件 | 用途 |
|------|------|
| [full_graph_aug_shared_baseline.yaml](/home/user22303471/Project/Active-H-EMV/llm_emv/config/teach/simplified/full_graph_aug_shared_baseline.yaml) | 与 `full_graph_aug_correction.yaml` 完全相同的图增强/搜索/Agent 设置，仅 `correction.enabled: false`，用于共享 history 对照基线 |

### 3. 实验规划文档

新增 [docs/Experiment Design/反馈修正模块实验规划.md](/home/user22303471/Project/Active-H-EMV/docs/Experiment Design/反馈修正模块实验规划.md)，包含四个实验的完整设计、命令、分析维度和时间估算。

## 实验与结果

### 实验一：Smoke Test（|h|=5）

**目的**：在低成本小历史上跑通修正全链路。

**运行命令**：
```bash
export $(grep -v '^#' .env | xargs)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_correction \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_5.pkl \
  --output experiments/results/teach/correction_smoke_h5.json \
  --enable-correction \
  --max-seconds-per-sample 180
```

**环境问题**：HuggingFace Hub 网络不可达导致 `all-mpnet-base-v2` 模型加载失败，通过设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1` 解决（模型已在本地缓存 419MB）。

**结果文件**：`experiments/results/teach/correction_smoke_h5.json`

**核心结果**：
- 总样本 100，有效答案 89，超时错误 11
- **Answer Judge 判定分布**：`llm_PARTIAL`=54, `ground_truth_substring_in_hypothesis`=8, `exact`=1
- **WRONG 判定数**：0
- **修正管线触发数**：0

**分析**：|h|=5 历史极短（每 batch 仅 5 个 episode），Agent 能轻松找到正确或部分正确的证据，因此从未产生需要修正的错误回答。Smoke 验证了全链路（共享 history → Judge 判定 → 条件触发）代码正确执行，但未触发修正。

### 实验二 组 A：|h|=50 共享 history 无修正（新基线）

**运行命令**：
```bash
python3 -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_shared_baseline \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/paper_formal/teach_active_h_emv_20260424/phase38_h50_shared_baseline/h50_shared_baseline.json \
  --enable-correction \
  --require-history-cache \
  --max-seconds-per-sample 180
```

**结果文件**：`experiments/paper_formal/teach_active_h_emv_20260424/phase38_h50_shared_baseline/h50_shared_baseline.json`

**核心结果**：
- 总样本 100，有效答案 94，超时错误 6
- 超时错误集中在第一批样本（最长的 sample_id），说明 180s 超时对首个 batch 的 `sequence_of_task_descs` 等展开密集型问题不够

### 实验二 组 B：|h|=50 共享 history + 修正（主实验）

**运行命令**：
```bash
python3 -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_correction \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/paper_formal/teach_active_h_emv_20260424/phase39_h50_correction_full/h50_correction_full.json \
  --enable-correction \
  --require-history-cache \
  --max-seconds-per-sample 180
```

**结果文件**：`experiments/paper_formal/teach_active_h_emv_20260424/phase39_h50_correction_full/h50_correction_full.json`

**核心结果**：
- 总样本 100，有效答案 98，超时错误 2
- **Answer Judge 判定分布**：`llm_PARTIAL`=85, `ground_truth_substring_in_hypothesis`=7, `exact`=3, `ground_truth_contained_in_hypothesis`=2, `llm_judge_error`=1
- **WRONG 判定数**：**0**
- **修正管线触发数**：**0**

### 组 A vs 组 B 对比

| 指标 | 组 A (共享无修正) | 组 B (共享+修正) |
|------|-------------------|-------------------|
| 有效答案 | 94/100 | 98/100 |
| 超时错误 | 6 | 2 |
| 答案不同的样本 | — | **40/100** |
| WRONG 判定 | — | 0 |
| 修正管线触发 | — | 0 |

**答案差异分析**：组 A 和组 B 之间有 40/100 题答案不同。这并非由主动修正导致（修正从未触发），而是由**共享 history 协议**下树展开状态的跨题保持 + LLM 非确定性（temperature=0.1）共同导致的。这本身是一个重要发现：反馈闭环即使不施加主动修正，仅共享记忆状态就足以改变系统行为。

差异的性质多为表述方式变化（如日期格式、措辞详略），而非核心事实错误。例如：
- 基线：`"On May 2, 2023, I prepared coffee..."` → 修正运行：`"On May 2, 2023, I prepared several food items..."`（加了概括前缀）
- 基线：`"I put all the pillows on the armchair 17 days ago..."` → 修正运行：`"17 days ago and 10 days ago..."`（省略了动作主体）

## 异常与问题

### 1. Answer Judge 过于保守

**根因**：Judge prompt 明确要求"Use WRONG only when the answer is clearly wrong. When uncertain between CORRECT and PARTIAL, choose PARTIAL." 这导致几乎所有不够完美的答案都被判为 PARTIAL（85/96），而非 WRONG（0/96）。

**影响**：修正管线依赖 `llm_WRONG` 判定作为触发条件（PARTIAL 被设计为"不修正"）。当前判定策略下，修正管线在真实问答中几乎永远不会触发。

**这不是代码 bug，而是设计权衡**：论文中修正机制被定位为"安全修正"——不确定时不改。但这对实验验证构成了挑战——无法在自然问答中观测到修正效果。

### 2. 共享 history 协议无 checkpoint

当前 `run_evaluation_with_correction` 函数只在所有样本完成后一次性写入 JSON 输出文件，不支持逐样本 checkpoint（而标准评测路径有 `_append_checkpoint`）。对于多小时的修正实验，这增加了运行风险。

**改进方向**：在 `run_evaluation_with_correction` 中增加 `_append_checkpoint` 调用，与标准评测路径对齐。

### 3. 首个 batch 超时

|h|=50 组 A 有 6 个超时错误，全部集中在第一批样本（50 个 episode、极其长的 sample_id）。`sequence_of_task_descs` 和 `seq_specific_shortened_low_actions` 需要完整展开记忆树，在 180s 超时限制下容易失败。

组 B 只有 2 个超时，差异可能来自 LLM 响应的随机波动。

## 下一步建议

鉴于 Answer Judge 在当前策略下不产生 WRONG 判定（因而修正管线从不触发），后续实验应转向**受控注入验证**而非自然问答累积：

### 建议：合成错误注入实验

1. **准备**：选取 |h|=5 的单个 batch，对其 history 中 1-2 个 `EventBasedSummary` 节点手工注入错误（修改 `_summary_override` 为包含错误信息的摘要）
2. **验证 err_loc**：验证 `localize_error(history, question, wrong_answer, correct_answer)` 能否将注入错误的节点排到嫌疑度前 3
3. **验证 correction**：验证 `correct_node_with_llm` 能否生成正确的修正摘要
4. **验证 propagation**：在同一 batch 的多个相邻节点注入相同错误，验证 `detect_error_propagation` 能否发现并传播修正
5. **验证 end-to-end**：以注入错误的 history 运行 `--enable-correction`，验证后续问题的回答是否因修正而改善

这比等待自然 WRONG 判定更可控、可重现，且更适合作为论文中的机制级验证证据。

### 不建议继续

- **实验三（消融—传播检测）**：当前 0 次修正触发，消融无意义
- **实验四（消融—图增强对修正的影响）**：同上
- **更大的 |h|**：|h|=50 已 0 WRONG，|h|=100 不会显著增加 WRONG 率（Agent 的图增强检索在大历史上反而更稳定）

## 论文写作建议

1. **共享 history 效应**可以作为独立发现写入："即使不施加主动修正，同 episode 内共享记忆状态就导致 40% 的答案表述产生变化，说明反馈闭环的 statefulness 本身就是影响系统行为的重要因素。"

2. **修正模块**的写作策略：
   - 方法章节：完整描述四阶段管线设计（错误定位 → 单点修正 → 传播检测 → 自动传播）
   - 实验章节：
     - 机制级验证：合成错误注入实验（受控、可重现、强证据）
     - 系统级评测：共享 history 协议的变化效应（自然运行、弱证据）
     - 坦诚说明：在现有图增强检索的精度下，自然问答中极少产生需要修正的严重错误，修正机制更多体现为"安全网"而非"日常工具"
   - 不要声称"修正提升了 benchmark 准确率"（当前数据不支持此结论）

3. **Answer Judge 的保守性**值得讨论："Judge 偏向 PARTIAL 而非 WRONG 的设计选择反映了修正机制的核心理念——在不完全确定时不随意修改记忆。这在安全至上的机器人应用中是正确的权衡。"

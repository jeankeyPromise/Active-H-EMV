# Phase 36: TEACh 遗忘模块的 pickle 压缩-准确率权衡验证

## 目标

围绕遗忘模块的核心目的，给出一条更直接的证据链：

1. 遗忘是否真的压缩了 pickle 文件大小？
2. 在压缩后，问答准确率是否保持稳定，至少不会明显下降？

## 本轮关键改动

### 1. 修正 `Level 2` 的表示方式

在此前的 `Level 2` 探测中，虽然结构已经被压缩，但 pickle 反而变大。
原因是：

- 同一份摘要文本被同时保存到 `_cached_nl_summary` 与 `_summary_override`
- `Level 2` 事件虽然只保留摘要语义，但仍携带对象列表、额外文本字段等冗余内容

本轮做了两处实现修正：

- [llm_emv/memory_consolidation.py](/home/user22303471/Project/Active-H-EMV/llm_emv/memory_consolidation.py)
  - `Level 2` 只保留 `_summary_override`
  - 清理 `_cached_nl_summary`
  - 清空 `objects` / `relations`
  - 清理 `audio_description` / `action_parameter_summary` / `asr_recognition`
- [em/em_tree.py](/home/user22303471/Project/Active-H-EMV/em/em_tree.py)
  - `EventBasedSummary.nl_summary` 优先返回 `_summary_override`
  - `EventBasedSummary.index_content` 在有 `_summary_override` 时仅使用摘要文本

这使 `Level 2` 真正变成“摘要保留”，而不是“摘要 + 尾部细节一起背着走”。

## 使用结果

### A. 稳态主结果：Forget+Graph full `100/100`

结果文件：

- QA:
  - `experiments/results/teach/forgetting/h50_forget_graph_n100_fullqa.json`
- 存储：
  - `experiments/results/teach/forgetting/h50_forget_graph_n100_storage.json`

核心结果：

- 与正式基线 `h50_current_zs_fast_n100_fullqa` 的答案文本 **100/100 逐条一致**
- 文件大小比：`0.9474`

也就是：

- **pickle 压缩约 `5.26%`**
- **最终回答不变**

这是当前最稳的正式主结果。

### B. 强压缩结果：修正后的 Level 2 probe（`n=20`）

结果文件：

- 存储：
  - `experiments/results/teach/forgetting/h50_forget_l2probe_n20_storage_v3.json`
- QA:
  - `experiments/results/teach/forgetting/h50_forget_l2probe_n20_probe_v2.json`
  - `experiments/results/teach/forgetting/h50_forget_l2probe_n20_probe_v2.gemini_2.5_pro-573e41.auto_eval.json`

配置：

- `teach/simplified/full_graph_aug_zs_fast_forget_l2probe`

核心结果：

- 文件大小比：`0.6569`
- scene 比：`0.9294`
- relation 比：`0.5034`
- `L2` 事件数：`6391 / 10545`

也就是：

- **pickle 压缩约 `34.31%`**
- **relation 约减半**
- **scene 数也出现实际下降**

同时，在同一批 `n=20` QA 上：

- 与 `Base` 的答案文本 **20/20 逐条一致**
- auto-eval:
  - Valid: `100.0%`
  - `S_c = 60.0%`
  - `S_p = 15.0%`
  - `T = 2.13K`

这里最关键的不是单次 judge 的 `S_p` 波动，
而是 **raw answer 20/20 完全一致**。

## 结果解读

### 1. 遗忘已经被证明能有效压缩 pickle

当前至少有两档明确结果：

| Setting | QA 范围 | pickle 比 | 压缩幅度 | 答案变化 |
| --- | --- | ---: | ---: | --- |
| Forget+Graph | full `100/100` | `0.9474` | `5.26%` | `100/100` 一致 |
| L2 probe | `n=20` | `0.6569` | `34.31%` | `20/20` 一致 |

这已经能直接支撑“遗忘能压缩 pickle 文件大小”这个论点。

### 2. 当前压缩并没有明显伤害 QA

更重要的是，在上述两档实验里：

- 稳态主 setting：full `100/100` 最终答案完全不变
- 更强压缩 setting：`n=20` 最终答案仍完全不变

因此，目前更准确的说法是：

> 遗忘已经能够在压缩 pickle 的同时，保持问答输出基本稳定。

### 3. 图结构增强仍然重要

稳态 full `100/100` 主结果使用的是 `Forget+Graph`，
它说明图结构增强版本适合作为正式主 setting。

而更激进的 `Level 2` 探测表明：

- 当我们希望更大幅度压缩结构时，
- 仍然可以先通过图增强检索保持问答稳定性。

## 当前最适合论文使用的表述

可以把遗忘模块总结为两层结果：

1. **正式主结果**  
   在 TEACh `|h|=50` full `100/100` 上，
   `Forget+Graph` 在保持最终答案不变的前提下，
   带来约 `5.26%` 的 pickle 压缩。

2. **强压缩机制验证**  
   在修正 `Level 2` 表示后，
   同一框架下可将 pickle 压缩到基线的 `65.69%`，
   同时在 `n=20` QA 上保持相同答案输出，
   说明遗忘具备进一步压缩长期记忆存储的潜力。

## 当前结论

本阶段已经能够回答遗忘模块最核心的问题：

1. **遗忘可以有效压缩 pickle 文件大小；**
2. **在当前验证范围内，压缩后 QA 没有出现明显下降；**
3. **遗忘模块的核心目标已经得到实验性支持。**

## 下一步

如果后面还要继续强化这条线，最自然的下一步不是再堆更多 setting，
而是二选一：

1. 将修正后的 `Level 2` 配置扩到更大的 QA 子集（如 `n=40` / `n=60`），继续验证“强压缩但低退化”；
2. 直接把本阶段结果整理成论文表格与正文分析，作为遗忘模块的主证据链。

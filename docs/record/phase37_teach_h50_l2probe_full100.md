# Phase 37: TEACh `|h|=50` 修正后 Level 2 配置的 full 100/100 验证

## 目标

在已经修正 `Level 2` 表示冗余的前提下，将强压缩配置扩展到 TEACh `|h|=50` 的 full `100/100` QA，验证两件事：

1. 是否能在更大 QA 范围内继续显著压缩 pickle 文件大小；
2. 是否能在强压缩下仍保持最终问答行为基本不变。

## 使用配置

- 配置文件：
  - [full_graph_aug_zs_fast_forget_l2probe.yaml](/home/user22303471/Project/Active-H-EMV/llm_emv/config/teach/simplified/full_graph_aug_zs_fast_forget_l2probe.yaml)
- 对齐基线：
  - [h50_current_zs_fast_n100_fullqa.json](/home/user22303471/Project/Active-H-EMV/experiments/results/teach/smoke/h50_current_zs_fast_n100_fullqa.json)
- 对齐温和遗忘 setting：
  - [h50_forget_graph_n100_fullqa.json](/home/user22303471/Project/Active-H-EMV/experiments/results/teach/forgetting/h50_forget_graph_n100_fullqa.json)

## 结果文件

- QA 结果：
  - [h50_forget_l2probe_n100_fullqa.json](/home/user22303471/Project/Active-H-EMV/experiments/results/teach/forgetting/h50_forget_l2probe_n100_fullqa.json)
- correctness evaluation：
  - [h50_forget_l2probe_n100_fullqa.gemini_2.5_pro-573e41.auto_eval.json](/home/user22303471/Project/Active-H-EMV/experiments/results/teach/forgetting/h50_forget_l2probe_n100_fullqa.gemini_2.5_pro-573e41.auto_eval.json)
- 存储统计：
  - [h50_forget_l2probe_n100_storage.json](/home/user22303471/Project/Active-H-EMV/experiments/results/teach/forgetting/h50_forget_l2probe_n100_storage.json)

## 核心结果

### 1. 强压缩在 full `100/100` 上成立

聚合后的 10 段 long history 统计如下：

- pickle 比：`0.6830`
- scene 比：`0.9288`
- relation 比：`0.4917`
- event 比：`1.0000`

对应总量变化：

- base:
  - events: `60836`
  - scenes: `68214`
  - relations: `577052`
  - file size: `88782961` bytes
- after:
  - events: `60836`
  - scenes: `63355`
  - relations: `283721`
  - file size: `60639023` bytes

也就是：

- **pickle 压缩约 `31.70%`**
- **scene 数减少约 `7.12%`**
- **relation 数减少约 `50.83%`**

### 2. 大量事件真正进入 Level 2

遗忘等级分布：

- `L0 = 7071`
- `L1 = 16623`
- `L2 = 37142`

也就是说，当前强压缩配置下：

- 超过 `61%` 的事件进入 `Level 2`
- 强压缩不再只是停留在 `Level 1`

### 3. full `100/100` 的最终答案保持不变

将 [h50_forget_l2probe_n100_fullqa.json](/home/user22303471/Project/Active-H-EMV/experiments/results/teach/forgetting/h50_forget_l2probe_n100_fullqa.json) 与以下两组结果逐条比较：

- [h50_current_zs_fast_n100_fullqa.json](/home/user22303471/Project/Active-H-EMV/experiments/results/teach/smoke/h50_current_zs_fast_n100_fullqa.json)
- [h50_forget_graph_n100_fullqa.json](/home/user22303471/Project/Active-H-EMV/experiments/results/teach/forgetting/h50_forget_graph_n100_fullqa.json)

结果均为：

- **`100/100` 最终答案文本逐条一致**

因此，这轮实验中最硬的准确率证据不是 judge 分类波动，
而是 **强压缩配置并没有改变最终输出答案**。

## 结果解读

### 1. 遗忘模块的核心目标已经被直接支持

这轮结果可以直接支撑：

> 遗忘模块能够显著压缩长期记忆树的 pickle 存储，同时在正式 `100/100` QA 范围内保持最终问答输出稳定。

### 2. 现在已经形成两档清晰结果

| Setting | QA 范围 | Pickle ratio | Scene ratio | Relation ratio | 最终答案变化 |
| --- | --- | ---: | ---: | ---: | --- |
| Forget+Graph | full `100/100` | `0.9474` | `1.0000` | `1.0000` | `100/100` 一致 |
| Level 2 probe | full `100/100` | `0.6830` | `0.9288` | `0.4917` | `100/100` 一致 |

这意味着：

- 温和遗忘已经在正式主 setting 上证明“稳定压缩”；
- 强压缩配置已经进一步证明“更大幅度压缩”也是可能的；
- 两档设置都没有在 full `100/100` 上改写最终答案。

### 3. full-history flatten 不再是主证据

这轮 full `100/100` 验证是在真实分层检索问答管线上完成的。
因此，遗忘模块的主证据链现在应明确落在：

- 分层检索
- 图增强
- 遗忘前后正式 QA 对齐
- 存储统计

而不是 one-pass flatten。

## 当前结论

修正后的 `Level 2` 配置已经完成 full `100/100` 验证，并给出一条比此前更强的主证据链：

1. **pickle 可压到基线的 `68.30%`；**
2. **relation 可压到基线的 `49.17%`；**
3. **正式 `100/100` QA 的最终答案仍与基线逐条一致。**

因此，遗忘模块已经能够被表述为：

> 在 TEACh `|h|=50` 的真实分层检索设置下，基于效用的遗忘不仅能够温和压缩存储，还能在更激进的 `Level 2` 压缩下显著降低 pickle 大小，而不会在当前验证范围内引入明显的问答退化。

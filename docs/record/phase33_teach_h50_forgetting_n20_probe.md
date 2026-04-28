# Phase 33: TEACh `|h|=50` 遗忘模块 `n=20` 主线预跑

## 目标

把遗忘模块放回真实的 TEACh 分层检索问答管线中验证，而不是继续使用 one-pass flatten。

本轮重点回答两个问题：

1. 在 `|h|=50`、正式 graph-augmented zero-shot 口径下，遗忘后问答准确率是否明显下降？
2. 在同一批 history 上，遗忘是否已经带来可测的存储压缩？

## 本轮新增

- 新配置：
  - `llm_emv/config/teach/simplified/full_graph_aug_zs_fast_forget.yaml`
  - `llm_emv/config/teach/simplified/full_graph_aug_zs_fast_forget_ultra.yaml`
- 新脚本：
  - `scripts/teach_forgetting_storage_stats.py`
- 运行时优化：
  - `llm_emv/setup.py` 增加 forgetting cache，按
    `history structure + q_time + forgetting cfg`
    复用遗忘后的 history，避免同一段长 history 的 10 个问题重复做记忆巩固。
  - `llm_emv/eval/__main__.py` 跟交互入口一致，默认关闭 LangChain verbose prompt dump。

## 评测设置

- 数据：`data/teach/test_set_50.pkl`
- 历史缓存：`--require-history-cache`
- QA 样本数：`n=20`
- 对比组：
  - `Base`: `teach/simplified/full_graph_aug_zs_fast`
  - `Forget+Graph`: `teach/simplified/full_graph_aug_zs_fast_forget`
  - `Ultra`: `teach/simplified/full_graph_aug_zs_fast_forget_ultra`
- Token / 时长护栏：
  - `--max-prompt-tokens-per-sample 12000`
  - `--max-average-prompt-tokens-per-sample 5000`
  - `--max-seconds-per-sample 240`

## 结果文件

- Base:
  - `experiments/results/teach/forgetting/h50_base_zs_fast_n20_probe.json`
  - `experiments/results/teach/forgetting/h50_base_zs_fast_n20_probe.gemini_2.5_pro-573e41.auto_eval.json`
- Forget+Graph:
  - `experiments/results/teach/forgetting/h50_forget_graph_n20_probe.json`
  - `experiments/results/teach/forgetting/h50_forget_graph_n20_probe.gemini_2.5_pro-573e41.auto_eval.json`
- Ultra:
  - `experiments/results/teach/forgetting/h50_forget_ultra_n20_probe.json`
  - `experiments/results/teach/forgetting/h50_forget_ultra_n20_probe.gemini_2.5_pro-573e41.auto_eval.json`
- 存储统计：
  - `experiments/results/teach/forgetting/h50_forget_graph_n20_storage.json`
  - `experiments/results/teach/forgetting/h50_forget_ultra_n20_storage.json`

## QA 指标

三组在当前 `n=20` slice 上得到**完全相同**的指标：

| Setting | Valid | `S_c` | `S_p` | Wrong | `T` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 100.0% | 60.0% | 20.0% | 20.0% | 2.13K |
| Forget+Graph | 100.0% | 60.0% | 20.0% | 20.0% | 2.13K |
| Ultra | 100.0% | 60.0% | 20.0% | 20.0% | 2.13K |

进一步核对：

- `Base` vs `Forget+Graph` 的 `results` JSON MD5 完全一致
- `Base` vs `Ultra` 的 token 统计也完全一致

这说明在当前这 20 题对应的两段 `|h|=50` history 上，遗忘没有改变最终问答输出。

## 存储统计

### Forget+Graph

- 选中 history：2
- 选中 QA：20
- 文件大小比：`0.9455`
- scene 比：`1.0000`
- relation 比：`1.0000`
- 遗忘层级：
  - before: `L0 = 10545`
  - after: `L0 = 3229`, `L1 = 7316`

### Ultra

- 选中 history：2
- 选中 QA：20
- 文件大小比：`0.9373`
- scene 比：`1.0000`
- relation 比：`1.0000`
- 遗忘层级：
  - before: `L0 = 10545`
  - after: `L0 = 2333`, `L1 = 8212`

## 关键观察

### 1. 当前 `n=20` slice 上，遗忘后的 QA 准确率没有下降

这是一个正结果。至少在这批题上：

- 主 setting 没有破坏问答能力
- 更激进的 `Ultra` 也没有立即把输出打坏

### 2. 当前压缩主要来自 `Level 1`，还没有进入 `Level 2`

无论是 `Forget+Graph` 还是 `Ultra`，当前两段 history 都只出现了：

- `Level 0`
- `Level 1`

而没有出现 `Level 2`。

这也解释了为什么：

- `scene_ratio = 1.0`
- `relation_ratio = 1.0`

因为当前压缩还没有删到 scene / relation 这一层，主要是通过 event 的细节字段压缩和摘要字段重写，带来了大约 `5.5%` 到 `6.3%` 的 pickle 大小下降。

### 3. 这轮结果更适合表述为“在不损伤当前 QA slice 的前提下，已获得温和存储压缩”

而不是宣称：

- 遗忘已经显著减少结构化底层证据
- 或者 Ultra 已经明显损伤 QA

目前证据还不支持这两种更强说法。

## 运行层面的经验

- 若不做 forgetting cache，同一 history 内 10 个问题会重复执行记忆巩固，开销过高。
- 加入 cache 后，日志能稳定看到：
  - `[Forgetting] 命中缓存`
  - `[GraphAug] 命中图缓存`
- 这使得 TEACh `|h|=50` 上的遗忘 QA 预跑变得可控。

## 当前结论

在 TEACh `|h|=50` 的前 20 个正式 QA 上：

1. `Forget+Graph` 与 `Base` 的问答结果逐项一致；
2. `Ultra` 也未在这一 slice 上造成额外问答退化；
3. `Forget+Graph` 已带来约 `5.5%` 的 pickle 大小压缩；
4. `Ultra` 已带来约 `6.3%` 的 pickle 大小压缩；
5. 当前压缩主要停留在 `Level 1`，尚未压到 scene / relation 层。

## 下一步

最自然的下一步是：

1. 将 `Forget+Graph` 扩展到 full `100/100` QA，直接与现有 `h50_current_zs_fast_n100_fullqa` 正式基线对齐；
2. 若 full `100/100` 仍然稳定，再决定是否把 `Ultra` 也扩展到 full `100/100`；
3. 同时检查是否需要调整 forgetting 参数，使部分 history 真正进入 `Level 2`，从而观察更明显的结构压缩-准确率权衡。

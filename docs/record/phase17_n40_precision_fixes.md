# Phase 17: n=40 Precision Fixes

## 目标

针对 Phase 16 的 h=50 `n=40` cached probe 暴露出的错误类型做低风险修复：优先处理 event/date 多报、temporal target 宽匹配和低层动作 task lookup 候选不准，不启动新的正式大跑。

## 修改文件

- `llm_emv/emv_api.py`

## 关键修改

1. Event/date lookup 约束收紧
   - 增加 object/action 约束词组：`bowl`、`pot`、`pillow`、`remote`、`tissue`、`watch`、`box` 等。
   - 增加 location 约束：`plate`、`bowl`、`drawer`。
   - 将 `armchair` 与普通 `chair` 拆开：查询 `armchair` 时不再接受普通 `chair` 命中。
   - 对 `tomato/potato/lettuce + plate/bowl` 增加对象-容器关系检查，避免同一宽摘要中 “potato in bowl” 与 “tomato on plate” 被拼成 “tomato in bowl”。

2. 数量约束从硬过滤改为排序/推荐约束
   - `1 slice` 不再硬过滤，避免合法摘要省略 `one/1` 时漏召回。
   - `2/3/5 slices` 加入轻量 score adjustment。
   - 当查询含明确数量且至少一个候选明确匹配数量时，`event_date_lookup()` 的 Recommended answer 优先只推荐数量匹配候选。

3. Temporal neighbor target 约束
   - temporal target 排名阶段复用 object/action/location relation 约束，减少把相似 cooking/cleaning 段误当目标任务。

4. Low-action task lookup 推荐
   - 对 `toggle/open/pick up/place/put` 这类低层动作查询，将 Recommended answer 从 top-3 扩展到 top-5。
   - 给 `drawer` 增加精确 location pattern，降低 `open drawer` 被 `cabinet/safe` 候选挤占的概率。

## 验证命令

语法检查：

```bash
conda run --no-capture-output -n active-h-emv python -m py_compile llm_emv/emv_api.py
```

直接工具验证使用 `TeachDeChantDataset + EMVerbalizationAPI`，只调用本地 embedding 与已缓存 history，不调用 QA LLM。

## Targeted 验证结果

- `serve 1 slice(s) of tomato in a bowl` / Dec sample
  - GT：`14 days ago and 6 days ago`
  - 修复后 Recommended answer：`14 days ago and 6 days ago`

- `serve 1 slice(s) of tomato in a bowl` / Apr sample
  - GT：`24, 22, 10, 9, 8 days ago`
  - 修复前多报：`21 days ago`
  - 修复后 Recommended answer：`24 days ago and 22 days ago and 10 days ago and 9 days ago and 8 days ago`

- `put all remote control on one armchair`
  - GT：`12 days ago`
  - 修复前多报：`16, 12, 6 days ago`
  - 修复后 Recommended answer：`12 days ago`

- `open the drawer`
  - 修复前 top candidates 被 cabinet/safe/watch/newspaper 等候选主导。
  - 修复后 top candidates 覆盖 tissue-box 与 sandwich 相关任务；`remote control on one dresser` 仍未进入 top-5，是后续残留召回问题。

## 残留问题

- `cook 3 slice(s) of potato and serve in a bowl`
  - GT：`Oct 01, 2024`
  - 修复后仍推荐 `2024/09/27` 与 `2024/10/01`。
  - 原因：`2024/09/27` 的摘要语义上也包含 “cooked potato slices into bowl”，属于和 GT 极近的历史项；不宜用日期或样本特化规则硬删。

- Low-action task lookup 仍是弱点
  - `open drawer` 已改善，但对 GT 中的 remote-control task 召回不足。
  - `toggle faucet`、`pick up butterknife` 仍建议后续单独做 raw-action/leaf-level 辅助索引，而不是继续堆 summary 关键词规则。

## 当前结论

本阶段修复解决了 n=40 中最明确的三类多报：`tomato bowl` 宽摘要拼接、`armchair` 被普通 chair 吞并、低层动作推荐过短。修复仍保持低风险：不改历史构建、不触发在线 summarization、不引入新外部依赖。

## 下一步建议

- 不急着直接扩大到 n=60；先在 h=50 cached 前缀上跑一个新的 `n=40` patched probe，并同步 `scripts/evaluate_result.sh`，比较 Phase 16 的 `S_c=55%` 是否回升。
- 若 low-action 仍拖后腿，下一步应增加 leaf/raw-action 索引，而不是继续放宽 task summary 召回。

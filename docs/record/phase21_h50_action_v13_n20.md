# Phase 21: h=50 动作 v1.3 n=20 稳定性探测

## 目标

通过一组刻意保持小而低风险的改动，继续推进第 4 周 h=50 实验循环。本阶段的目标不是追求新的峰值分数，而是：

- 减少仍会导致长 completion 或格式错误 `answer("...")` 的冗长低层动作推荐答案；
- 让 `clean all X` 风格问题在 `task_lookup` / `event_date_lookup` / 时间邻近检索中的匹配更保守；
- 验证更新后的系统在缓存好的 h=50 前缀上仍能稳定运行，并保持与前几轮探测相近的竞争力。

本阶段紧接 Phase 20，核心配置保持不变：

- `teach/simplified/full_graph_aug_zs_fast`
- `--require-history-cache`
- 仅使用 h=50 缓存前缀
- 运行后立即进行正确性评估

## 代码改动

编辑文件：

- `llm_emv/emv_api.py`

改动如下：

1. 缩短低层动作的推荐答案：
   - 从 top-5 任务短语缩减为 top-3；
   - 短语长度上限从 `max_len=110` 降为 `max_len=85`；
   - 增加强提示，要求直接基于 `Recommended answer` 作答。

2. 新增共享辅助函数 `_clean_all_candidate_matches(...)`：
   - 用于识别 cups/mugs/drinkware/pots/pans/plates 这类 `all + object-group` 查询；
   - 要求证据中提到同一对象组；
   - 要求证据中包含 `clean*`、`wash*`、`rins*`、`dirty`、`sink` 或 `faucet` 等清洁相关词。

3. 将该辅助函数应用到：
   - `_event_candidate_satisfies_constraints(...)`
   - `_target_candidate_satisfies_constraints(...)`
   - `_task_lookup_candidate_satisfies_constraints(...)`

4. 用通用辅助逻辑替换掉此前脆弱的、硬编码的 cup/mug 特判。

## 命令

语法检查：

```bash
python -m py_compile llm_emv/emv_api.py
```

运行：

```bash
set -a
source .env
set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_current_zs_fast_n20_action_v13.json \
  --n-samples 20 \
  --require-history-cache \
  --max-prompt-tokens-per-sample 12000 \
  --max-average-prompt-tokens-per-sample 5000 \
  --max-seconds-per-sample 240
```

评估：

```bash
PYTHON_BIN="conda run --no-capture-output -n active-h-emv python" \
  scripts/evaluate_result.sh experiments/results/teach/smoke/h50_current_zs_fast_n20_action_v13.json
```

## 结果文件

- `experiments/results/teach/smoke/h50_current_zs_fast_n20_action_v13.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n20_action_v13.jsonl`
- `experiments/results/teach/smoke/h50_current_zs_fast_n20_action_v13.gemini_2.5_pro-5b7e74.auto_eval.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n20_action_v13_llm_eval.log`
- `experiments/results/teach/smoke/h50_current_zs_fast_n20_action_v13_metrics.log`

## 稳定性

- Cache 审计：抽查 history `2/2` 已缓存，missing `0`。
- 最终 QA：`20/20`。
- 合法答案率：`100.0%`。
- 错误/空答案率：`0.0%`。
- 未发生在线 `group and summarize`。
- 未出现 prompt 失控增长。
- 主运行过程中无需 API 重试。

仍然观察到的残余问题：

- `clean all the cups` / `clean all the mugs` 这类事件日期查询仍然偏宽，可能列出多个关联较弱的日期。
- 一些较长的推荐答案仍会诱导模型输出不完整的 `answer("...` 调用，但结构化回退已经处理了这些情况，避免了硬失败。

## 指标

来自 `scripts/evaluate_result.sh` 的主要指标：

| Run | Total | Valid | `S_c` | `S_p` | Wrong/no-answer | `T` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 15 cached v1 | 20 | 100% | 70.0% | 25.0% | 5.0% | 2.26K |
| Phase 21 action v1.3 | 20 | 100% | 60.0% | 25.0% | 15.0% | 2.35K |

Phase 21 细分结果：

- `correct = 12/20`
- `partially_correct = 5/20`
- `wrong = 3/20`

## 解读

Phase 21 并不是新的 n=20 最优结果，因此不应取代 Phase 15 作为更强的 h=50、n=20 缓存前缀参考。但它依然有价值：

- 它证明了通用化的 `clean all X` 约束不会破坏 h=50 缓存前缀的稳定性；
- 合法答案率依旧保持在 `100%`；
- prompt 使用仍受控（`T=2.35K`）；
- 它为后续进一步做精度修复提供了一个更稳的基线，同时没有重新引入空回复或缓存触发的隐藏成本。

换句话说：这个版本适合作为稳定性/诊断分支，但还不适合直接扩规模。

## 结论

对于论文主线，当前建议仍然是：

- 保留 **Phase 18** 作为最佳 h=50、n=40 pilot；
- 保留 **Phase 15** 作为最强的 h=50、n=20 缓存前缀参考；
- 将 **Phase 21** 视作稳定性/诊断检查点，而不是主结果。

## 下一步

不要立刻把 Phase 21 扩展到 n=40。下一步低风险改进应只针对两个残余问题：

1. 收紧 `event_date_lookup(...)` 对 `clean all cups/mugs/drinkware` 的处理，只保留高置信日期；
2. 进一步减少格式错误的 `answer("...")`，方式是缩短或简化低层动作问题的结构化推荐答案。

在做完下一次定向精度修复后，再按以下顺序重跑：

1. 少量定向 `clean all` / 低层动作样本；
2. 然后进行一次 h=50 的 `n=20` 探测；
3. 之后才考虑再次尝试 `n=40`。

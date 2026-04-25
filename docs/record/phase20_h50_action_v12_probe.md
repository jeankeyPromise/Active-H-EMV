# Phase 20: h=50 动作 v1.2 探测实验

## 目标

在 Phase 19 之后继续推进第 4 周实验循环。Phase 19 表明，原始动作检索对低层动作问题有帮助，但 `action_v1` 让 h=50、n=40 的 pilot 结果出现了回退。本阶段测试更小幅度的 v1.2 prompt 调整：

- 保留 v1.1 中 raw-action / pillow-sofa 的修复；
- 让最终回答优先使用 `answer("...")`，而不是 `answer(reasoning="...", answer="...")`；
- 检查这样是否能在保留低层动作增益的同时，减少格式错误的 `answer(...)` 截断问题。

## 代码改动

- `llm_emv/config/teach/simplified/system_zero_shot.prompt.txt`
  - 将默认最终回答指令改为 `answer("...")`。
  - 仅在确实需要简短理由时，才把 `answer(reasoning="...", answer="...")` 作为可选形式。
  - 将无记录时的回复改为 `answer("I have no record of that.")`。
- `llm_emv/config/teach/simplified/usage.prompt.py`
  - 将最终回答的 usage 示例更新为 `answer("...")`。

## 定向探测

精确索引的定向问题如下：

- `place the mug on the coffeemachine`
- `toggle on the faucet`
- `pick up the butterknife`
- `put all pillow on any sofa`
- `open the drawer`

结果文件：

- `experiments/results/teach/smoke/h50_action_lookup_v11_targeted_n5.json`
- `experiments/results/teach/smoke/h50_action_lookup_v12_targeted_n5.json`
- `experiments/results/teach/smoke/h50_action_lookup_v12_targeted_n5_metrics.log`

指标：

| Probe | Total | Valid | `S_c` | `S_p` | Wrong/no-answer | `T` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v11 targeted | 5 | 100% | 40.0% | 60.0% | 0.0% | 2.10K |
| v12 targeted | 5 | 100% | 60.0% | 40.0% | 0.0% | 2.08K |

观察：

- `put all pillow on any sofa` 仍然保持修复状态，重新给出了正确的两个枕头答案。
- `pick up the butterknife` 和 `open the drawer` 现在会输出合法的 `answer("...")` 调用。
- 定向集合中的语法回退从 3 例下降到 2 例。
- 仍有两个较长的低层动作总结触发了 completion 截断，并走了结构化回退。

## h=50 n=40 探测

命令：

```bash
set -a
source .env
set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v12.json \
  --n-samples 40 \
  --require-history-cache \
  --max-prompt-tokens-per-sample 12000 \
  --max-average-prompt-tokens-per-sample 5000 \
  --max-seconds-per-sample 240
```

该运行中遇到了两次瞬时 API 连接错误。二者先被记录为 checkpoint error，随后通过如下命令清除：

```bash
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v12.json \
  --n-samples 40 \
  --resume \
  --retry-errors \
  --require-history-cache \
  --max-prompt-tokens-per-sample 12000 \
  --max-average-prompt-tokens-per-sample 5000 \
  --max-seconds-per-sample 240
```

正确性评估：

```bash
PYTHON_BIN="conda run --no-capture-output -n active-h-emv python" \
  scripts/evaluate_result.sh experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v12.json
```

结果文件：

- `experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v12.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v12.jsonl`
- `experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v12.log`
- `experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v12.gemini_2.5_pro-082378.auto_eval.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n40_action_v12_metrics.log`

## 稳定性

- Cache 审计：抽查的 4/4 个 history 已缓存完成，missing=0。
- 最终 QA：40/40。
- 合法答案率：100.0%。
- 错误/空答案率：经重试后为 0.0%。
- 平均 prompt tokens：2290.4（`T=2.29K`）。
- 最大 prompt tokens：3847。
- 平均 completion tokens：476.6。
- 最大 completion tokens：764。
- 接近上限的 completion（`>=760`）：5 个样本。
- 结构化回退：4 例。
- API 连接错误：影响 2 个样本，均通过 `--resume --retry-errors` 修复。
- 未发生在线 `group and summarize`。
- 未触发真实 VQA answer-path 调用。

## 指标

| Run | Total | Valid | `S_c` | `S_p` | Wrong/no-answer | `T` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 18 patched v1 | 40 | 100% | 62.5% | 17.5% | 20.0% | 2.37K |
| Phase 19 action v1 | 40 | 100% | 55.0% | 20.0% | 25.0% | 2.31K |
| Phase 20 action v1.2 | 40 | 100% | 60.0% | 20.0% | 20.0% | 2.29K |

Phase 20 收复了 Phase 19 中 action_v1 带来的大部分回退，并且 prompt 成本略有下降，但在 `S_c` 上仍未超过 Phase 18。

## 结论

v1.2 的 prompt 调整作为稳定性清理是有价值的，应该保留：定向低层动作正确率提升了，h=50、n=40 的 pilot 也回到了与 Phase 18 相同的 wrong/no-answer 水平，同时 prompt 成本更低。不过，当前 h=50、n=40 的最佳 pilot 仍然是 Phase 18（`S_c=62.5%`）。

在继续扩展到 n=40 以上之前，下一步改进不应再是大范围 prompt 改写，而应聚焦于精度问题：

- 降低 `clean all X` 一类时间匹配的过宽召回；
- 将低层动作答案规范为任务名风格，而不是冗长总结；
- 处理不完整的 `answer("...")` 生成，或在结构化工具已经给出高置信推荐答案时，避免额外 LLM answer 步骤；
- 在再次进行 n=40/n=60 探测前，先只重跑定向样本。

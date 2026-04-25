# Phase 22: h=50 动作 v1.3 n=60 续跑完成

## 目标

补完此前中断的 h=50 缓存前缀 `n=60` 实验。上一次运行在 `46/60` 处停止，原因是一个低层动作样本（`pick up the remotecontrol`）陷入了反复的自由形式工具调用，并触发了单样本 prompt-token 护栏。

本阶段的目标是：

- 保留 Phase 21 分支，不做整轮重跑；
- 增加一个最小化护栏，让低层动作问题在已有结构化答案时直接停止，而不是继续漂移到临时搜索；
- 从 checkpoint 恢复中断的 `n=60` 运行；
- 在完成后立即评估结果。

## 代码改动

编辑文件：

- `llm_emv/simplified_agent/simple_coding_emv.py`

改动：

1. 新增 `_is_low_action_task_query(...)`，用于识别那些表面上是 task-description，实质上属于低层动作的问题，例如：
   - `pick up ...`
   - `place ...`
   - `put ...`
   - `toggle/open/...`

2. 在自动 `task_lookup(...)` hint 运行后，如果查询属于低层动作，且已经有结构化 `Recommended answer`，agent 现在会直接返回该答案：
   - 不再进入自由形式 LLM 循环；
   - 不再额外调用 `search(...)` / `history.search(...)`；
   - 不再在同一个低层动作问题上反复游走调用工具。

这个改动刻意保持保守：它不改变检索来源，只改变在低层动作结构化答案已存在时的停止规则。

## 命令

语法检查：

```bash
python -m py_compile llm_emv/simplified_agent/simple_coding_emv.py llm_emv/emv_api.py
```

续跑：

```bash
set -a
source .env
set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_current_zs_fast_n60_action_v13.json \
  --n-samples 60 \
  --require-history-cache \
  --resume \
  --max-prompt-tokens-per-sample 12000 \
  --max-average-prompt-tokens-per-sample 5000 \
  --max-seconds-per-sample 240
```

评估：

```bash
PYTHON_BIN="conda run --no-capture-output -n active-h-emv python" \
  scripts/evaluate_result.sh experiments/results/teach/smoke/h50_current_zs_fast_n60_action_v13.json
```

## 结果文件

- `experiments/results/teach/smoke/h50_current_zs_fast_n60_action_v13.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n60_action_v13.jsonl`
- `experiments/results/teach/smoke/h50_current_zs_fast_n60_action_v13.gemini_2.5_pro-5b7e74.auto_eval.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n60_action_v13_llm_eval.log`
- `experiments/results/teach/smoke/h50_current_zs_fast_n60_action_v13_metrics.log`

## 续跑结果

- 续跑前状态：已完成 `46/60`，因 token 护栏中止。
- 修复后续跑：成功完成到 `60/60`。
- 最终 JSON 结果数：`60`。
- 最终 checkpoint 行数：`60`。
- 错误样本数：`0`。

整个过程中，history-cache 安全性保持完好：

- selected histories: `6`
- cached histories: `6`
- missing histories: `0`

未发生在线 `group and summarize`。

## 稳定性说明

本阶段最关键的行为收益是：此前有问题的低层动作分支不再继续烧掉 prompt 预算。

- `task_lookup('put all remote control on any sofa')` 现在会直接返回结构化答案；
- `task_lookup('place the egg on the countertop')` 也会直接返回；
- 这两类样本在结构化 hint 之后额外消耗的 prompt/completion tokens 都为 `0`。

续跑过程未出现以下问题：

- 空回复；
- API/runtime 硬错误；
- 隐式 summarization；
- 单样本 token 预算中止。

日志中仍可见的残余问题：

- `object_lookup('cd')` 仍然会在词面上与 `credit card` 证据发生碰撞，虽然模型最终回答仍较保守。这是精度问题，不是稳定性问题。

## 指标

来自 `scripts/evaluate_result.sh` 的主要指标：

| Run | Total | Valid | `S_c` | `S_p` | Wrong/no-answer | `T` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 21 action v1.3 n=20 | 20 | 100% | 60.0% | 25.0% | 15.0% | 2.35K |
| Phase 22 action v1.3 n=60 | 60 | 100% | 53.3% | 21.7% | 25.0% | 2.45K |

Phase 22 细分结果：

- `correct = 32/60`
- `partially_correct = 13/60`
- `wrong = 15/60`

其他评估摘要：

- 合法答案率 = `100.0%`
- 错误/空答案率 = `0.0%`
- 每个 QA 的 completion tokens = `0.49K`

## 解读

Phase 22 是一次成功的稳定性补完，不是新的 headline 最优分数。

它证明了：

- 缓存好的 h=50 前缀可以安全地从小规模 pilot 扩展到 `n=60`；
- checkpoint/resume 工作流在实践中是可用的；
- 低层动作 direct-answer 护栏消除了此前中断运行的那一类具体失败模式；
- 即便从 `n=20` 扩到 `n=60`，整体 token 使用仍然健康（`T=2.45K`）。

它尚未证明的是：

- 这个分支在质量上超过了 Phase 18 的更强缓存 pilot；
- 当前对象/实体精度问题已经完全清理干净。

## 结论

对于论文主线：

- 保留 **Phase 18** 作为更强的 h=50 `n=40` 质量 pilot；
- 保留 **Phase 22** 作为 action-v1.3 分支在更大缓存前缀规模下具备运行稳定性的当前最强证据；
- 在讨论 checkpoint 恢复、护栏和低层动作稳定化时引用 Phase 22。

## 下一步

下一步应继续保持小步、聚焦精度：

1. 收紧 `object_lookup(...)`，避免 `cd` 这类短字符串过宽匹配到 `credit card`；
2. 可选地重跑一个很小的定向 object/no-answer smoke；
3. 然后再决定这个分支是否值得继续推向完整 100-QA 的缓存安全边界，或者是否仍应以 Phase 18 作为更干净的主结果分支。

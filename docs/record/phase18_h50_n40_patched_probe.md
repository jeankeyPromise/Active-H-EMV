# Phase 18: h=50 n=40 Patched Probe

## 目标

在 Phase 17 的 precision fixes 后，重新运行 h=50 缓存安全前缀 `n=40` probe，并同步 correctness auto-eval，检查修复是否带来净收益。

## 运行命令

Cache audit:

```bash
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_current_zs_fast_n40_patched_v1.audit.json \
  --n-samples 40 \
  --audit-history-cache
```

Main run:

```bash
set -a
source .env
set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_current_zs_fast_n40_patched_v1.json \
  --n-samples 40 \
  --resume \
  --require-history-cache \
  --max-prompt-tokens-per-sample 12000 \
  --max-average-prompt-tokens-per-sample 5000 \
  --max-seconds-per-sample 240 \
  2>&1 | tee -a experiments/results/teach/smoke/h50_current_zs_fast_n40_patched_v1.log
```

Correctness evaluation:

```bash
set -a
source .env
set +a
PYTHON_BIN="conda run --no-capture-output -n active-h-emv python" \
  scripts/evaluate_result.sh experiments/results/teach/smoke/h50_current_zs_fast_n40_patched_v1.json \
  2>&1 | tee experiments/results/teach/smoke/h50_current_zs_fast_n40_patched_v1_eval.log
```

## 结果文件

- `experiments/results/teach/smoke/h50_current_zs_fast_n40_patched_v1.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n40_patched_v1.jsonl`
- `experiments/results/teach/smoke/h50_current_zs_fast_n40_patched_v1.log`
- `experiments/results/teach/smoke/h50_current_zs_fast_n40_patched_v1.gemini_2.5_pro-dd6f9e.auto_eval.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n40_patched_v1_eval.log`
- `experiments/results/teach/smoke/h50_current_zs_fast_n40_patched_v1_llm_eval.log`
- `experiments/results/teach/smoke/h50_current_zs_fast_n40_patched_v1_metrics.log`

## 稳定性结果

- Cache audit：4/4 selected histories cached，missing=0。
- Final QA：40/40。
- Valid answer rate：100.0%。
- Error/empty answer rate：0.0%。
- 平均 prompt token：2374.1，即 `T=2.37K`。
- 最大 prompt token：5320。
- 平均 completion token：285.3。
- 最大 completion token：764。
- completion=764 的结构化回退：2 个样本。
- 未发现在线 `group and summarize`。
- 未发现真实 VQA 调用进入答案流程。

## Auto-Eval 指标

与 Phase 16 的 h=50 n=40 cached probe 对比：

| 指标 | Phase 16 cached v1 | Phase 18 patched v1 |
| --- | ---: | ---: |
| Total QA | 40 | 40 |
| Valid | 100.0% | 100.0% |
| `S_c` semantic correct | 55.0% | 62.5% |
| `S_p` partially correct | 25.0% | 17.5% |
| Wrong/no-answer | 20.0% | 20.0% |
| `T` prompt tokens / QA | 2.26K | 2.37K |

细分类别：

- `correct*`：25
- `partially_correct*`：7
- `wrong`：4
- `no_answer`：4

其中两个 `no_answer` 是 object yes/no 的否定回答：`painting` 和 `desk` 的 GT 都是 `no`，系统回答 `No, I have no record of that.`，人工看应计为正确；当前 evaluator 仍把这类否定答案归为 no_answer。

## 已改善样本

- `serve 1 slice(s) of tomato in a bowl` / Dec sample
  - Phase 18 推荐并回答：`14 days ago and 6 days ago`，对齐 GT。

- `serve 1 slice(s) of tomato in a bowl` / Apr sample
  - Phase 16 多报 `21 days ago`。
  - Phase 18 回答：`24, 22, 10, 9, 8 days ago`，对齐 GT。

- `put all remote control on one armchair`
  - Phase 16 多报 `16/12/6 days ago`。
  - Phase 18 回答：`12 days ago`，对齐 GT。

- `clean all the pots` just-after
  - Phase 16 错到 potato preparation。
  - Phase 18 命中 `put all remote control on one sofa`，方向和任务类型明显改善。

## 主要残留错误

1. Low-action task lookup 仍然弱
   - `toggle on the faucet` 只召回一个 plate-cleaning 任务，漏掉 GT 中大量 `clean all mugs/pots/drinkwares/plates` 等任务。
   - `pick up the butterknife` 只答 toast，漏 plate/tomato/potato 相关任务。
   - `open the drawer` 已覆盖 tissue-box 和 sandwich，但仍漏 remote-control task。

2. `clean all X` temporal target 仍易误定位
   - `clean all the mugs` before：GT 是 `make a salad`，系统答成 `open cabinet`。
   - `clean all the drinkwares` before：GT 是 `put all pillow on any sofa`，系统答成 coffee/toast。
   - `clean all the pots` before：GT 是 `put all remote control on one dresser`，系统答成 `open cabinet`。

3. 数量精确匹配仍不够硬
   - `cook 3 slice(s) of potato and serve in a bowl` 仍多报 `2024/09/27`。
   - `serve 1 slice(s) of tomato on a plate` 仍多报多个 salad/sandwich assembly 日期。

4. Completion 截断/回退仍存在
   - 2 个样本 completion=764 并触发结构化回退。
   - 回退保证 valid=100%，但 date_lookup 的 Recommended answer 对整日查询仍偏长。

## 当前结论

Phase 17 的 precision fixes 带来正向收益：`S_c` 从 55.0% 提升到 62.5%，且 valid 维持 100%。主要收益来自 event/date 位置关系约束和 armchair/chair 拆分。代价是平均 prompt token 从 2.26K 增至 2.37K，仍远低于 5K guard。

当前瓶颈已经更集中：summary 级结构化工具难以可靠覆盖 low-action 与 `clean all X` 的邻接任务。下一步如果继续提升 correctness，应优先做 raw-action/leaf-level 辅助索引，而不是继续扩大 summary 关键词规则。

# Phase 11: `when did you X` Event Lookup Fix

Date: 2026-04-21

## Goal

修复 Phase 10 auto-eval 暴露出的低风险错误：`When did you water the plant?` 虽然能走 `event_date_lookup()`，但回答成了 “N days ago”，且候选中混入了 faucet/water 但不涉及 plant 的记录。

## Implemented Changes

文件：`llm_emv/emv_api.py`

1. `event_date_lookup()` 增加 answer mode。
   - `when did you X` 返回具体日期。
   - `how many days ago did you X` 继续返回相对天数。

2. `water the plant` 事件匹配增加简单对象约束。
   - 当 event query 包含 `plant` 时，候选摘要必须包含 `plant` 或 `houseplant`。
   - 这会过滤 faucet-only / sink-cleaning 误匹配。

## Verification

语法检查：

```bash
conda run --no-capture-output -n active-h-emv python -m py_compile \
  llm_emv/emv_api.py \
  llm_emv/simplified_agent/simple_coding_emv.py
```

Targeted smoke:

```bash
set -a; source .env; set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_15.pkl \
  --output experiments/results/teach/smoke/event_when_lookup_h15_tasks_to_exact_times_n1.json \
  --use-only-question-types tasks_to_exact_times \
  --n-samples 1 \
  --max-prompt-tokens-per-sample 8000 \
  --max-average-prompt-tokens-per-sample 6000 \
  --max-seconds-per-sample 180
```

Auto-eval:

```bash
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval.metrics.llm_eval \
  llm_emv/config/llm_eval/gemini_2.5_pro.yaml \
  experiments/results/teach/smoke/event_when_lookup_h15_tasks_to_exact_times_n1.json

conda run --no-capture-output -n active-h-emv python -m llm_emv.eval.metrics.calc_metrics \
  --primary-only \
  experiments/results/teach/smoke/event_when_lookup_h15_tasks_to_exact_times_n1.json
```

## Result

Files:

- `experiments/results/teach/smoke/event_when_lookup_h15_tasks_to_exact_times_n1.json`
- `experiments/results/teach/smoke/event_when_lookup_h15_tasks_to_exact_times_n1.jsonl`
- `experiments/results/teach/smoke/event_when_lookup_h15_tasks_to_exact_times_n1.log`
- `experiments/results/teach/smoke/event_when_lookup_h15_tasks_to_exact_times_n1.gemini_2.5_pro-19eb87.auto_eval.json`
- `experiments/results/teach/smoke/event_when_lookup_h15_tasks_to_exact_times_n1_llm_eval.log`
- `experiments/results/teach/smoke/event_when_lookup_h15_tasks_to_exact_times_n1_metrics.log`

Before fix in h=15 n=15:

- Question: `When did you water the plant?`
- Answer: `I watered the plant 7 days ago, 6 days ago, 4 days ago, 2 days ago, and 1 day ago.`
- Auto-eval: `wrong`
- Problem: faucet-only false positives plus relative-time answer mode.

After fix:

- Candidates reduced to real plant events:
  - 2024/03/13
  - 2024/03/15
  - 2024/03/16
- Answer: `I watered the plant on March 13th, March 15th, and March 16th.`
- Prompt tokens: 1721.
- Completion tokens: 63.
- Auto-eval: `correct`.
- Primary metrics on targeted sample: `S_c=100%`, `S_p=0%`, valid=100%, `T=1.72K`.

## Conclusion

The low-risk event lookup fix directly converts the identified Phase 10 date/event error from wrong to correct, while reducing token use. The remaining correctness risk is now mainly temporal before/after target ambiguity, not date/event lookup formatting.

## Next

Move to Week 4 result consolidation:

1. Inventory existing h=50/h=100 formal results and auto-eval files.
2. Draft the thesis main result table using traceable file paths.
3. Prepare case studies for graph expansion, no-record guardrails, and correction/forgetting modules.

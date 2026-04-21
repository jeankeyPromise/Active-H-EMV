# Phase 10: h=15 n=15 Auto-Eval and Error Analysis

Date: 2026-04-21

## Goal

按后续实验规范，h=15 `n-samples=15` 跑完后同步进行 correctness evaluation，不只记录有效回答率和 token。评估使用项目已有的 LLM category judge，输出论文主指标 `S_c`、`S_p`、valid answer rate 和 `T`。

## Inputs

- QA result: `experiments/results/teach/smoke/task_tools_h15_n15.json`
- Checkpoint: `experiments/results/teach/smoke/task_tools_h15_n15.jsonl`
- Eval config: `llm_emv/config/llm_eval/gemini_2.5_pro.yaml`

## Commands

```bash
set -a; source .env; set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval.metrics.llm_eval \
  llm_emv/config/llm_eval/gemini_2.5_pro.yaml \
  experiments/results/teach/smoke/task_tools_h15_n15.json \
  2>&1 | tee experiments/results/teach/smoke/task_tools_h15_n15_llm_eval.log

conda run --no-capture-output -n active-h-emv python -m llm_emv.eval.metrics.calc_metrics \
  --primary-only \
  experiments/results/teach/smoke/task_tools_h15_n15.json \
  2>&1 | tee experiments/results/teach/smoke/task_tools_h15_n15_metrics.log
```

Generated auto-eval file:

```text
experiments/results/teach/smoke/task_tools_h15_n15.gemini_2.5_pro-6d4da8.auto_eval.json
```

## Result

Primary metrics:

| Metric | Value |
| --- | ---: |
| Total QA | 15 |
| Valid answer rate | 100.0% (15/15) |
| `S_c` semantic correct | 46.7% (7/15) |
| `S_p` partially correct | 20.0% (3/15) |
| Wrong / no-answer | 33.3% (5/15) |
| `T` prompt tokens per QA | 2.70K |
| Completion tokens per QA | 0.18K |

Notes:

- The `T=2.70K` value comes from `task_tools_h15_n15.json` cumulative `openai_costs`, so it includes the one failed sample attempt and its later retry.
- The final deduplicated JSONL sample average from Phase 9 is lower: 2.51K prompt tokens per QA.
- For论文表格, use the `calc_metrics` primary metric value unless explicitly reporting “final successful attempts only”.

Fine categories:

| Category | Count |
| --- | ---: |
| `correct` | 2 |
| `correct_summarized` | 1 |
| `correct_tmi` | 4 |
| `partially_correct_tmi` | 3 |
| `wrong` | 4 |
| `no_answer` | 1 |

Broad categories:

- Correct: 7.
- Partially correct: 3.
- Wrong/no-answer: 5.

## Error Analysis

1. `when did you X` answer mode was too coarse.
   - `When did you water the plant?` was made valid by routing to `event_date_lookup()`, but the answer used “days ago” wording.
   - The judge marked it wrong because this question expects concrete dates/times, not only relative day counts.
   - Next fix: make `event_date_lookup()` return dates for `when did you X`, while keeping days-ago wording for `how many days ago did you X`.

2. Event matching for `water the plant` is too permissive.
   - Some candidates involving faucet/water but not a plant were included.
   - Next fix: require plant/houseplant evidence when the event query contains `plant`.

3. Temporal before/after matching still has ambiguous target selection.
   - Two potato before/after questions were judged wrong.
   - Current `temporal_neighbor()` often finds plausible adjacent substeps, but not always the episode-level task intended by the QA ground truth.
   - Next fix should be conservative: improve candidate ranking or add a task-level exact target preference, then retest only targeted before/after samples.

4. Object no-record answers may be judged as `no_answer`.
   - `Was there a pillow ?` answered `No, I have no record of that.`
   - The broad metric treats `no_answer` as wrong, even though this may be acceptable behavior for no-record object questions.
   - For论文, mention that no-record guardrails improve safety/validity but may be penalized by generic semantic judges.

## Current Conclusion

h=15 n=15 is now stable enough as a Week 3 engineering checkpoint:

- It meets the valid-rate target.
- It stays well under the token budget.
- It has no VQA misfires, no connection errors, and no hidden online summary generation.
- Correctness is competitive with existing h=50 graph pilot trends but still exposes two fixable semantic issues: `when did you X` formatting/filtering and temporal neighbor target ambiguity.

## Next

1. Fix `event_date_lookup()` answer mode for `when did you X`.
2. Tighten plant-event matching to remove faucet-only false positives.
3. Rerun only the affected h=15 n=15 failed/wrong temporal/date samples, not the whole experiment.
4. After that, start Week 4 work: consolidate existing h=50/h=100 formal results and prepare the paper’s main table plus case studies.

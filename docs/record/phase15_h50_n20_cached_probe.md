# Phase 15: h=50 n=20 Cached Prefix Probe

Date: 2026-04-22

## Goal

继续第 4 周实验验证，在不触发在线 history summarization 的前提下，把 `|h|=50` cached-prefix probe 从 n=10 扩到 n=20，并同步运行 correctness evaluation。

本阶段不进入 Week 5 写作主线，也不使用旧的 interrupted n=60 checkpoint 作为 aggregate 结果。

## Cache Audit

Command:

```bash
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_current_zs_fast_n20_cached_v1.audit.json \
  --n-samples 20 \
  --audit-history-cache
```

Result:

- Selected histories: 2.
- Cached histories: 2.
- Missing histories: 0.
- The n=20 prefix uses the first two h=50 multi-history caches only, so it is safe from hidden online summarization.

## Code Changes

Files:

- `llm_emv/simplified_agent/simple_coding_emv.py`
- `llm_emv/emv_api.py`

Changes:

1. Structured fallback on malformed final `answer(...)`.
   - If a structured tool already produced a `Recommended answer`, and the LLM later emits a truncated `answer(...)` that raises `SyntaxError`, return the structured recommendation instead of retrying until `###ERROR###`.
   - This fixed repeated failures where the model generated an unterminated final answer string after `temporal_neighbor()` or `task_lookup()`.

2. `temporal_neighbor()` target-location constraints.
   - When the target phrase includes explicit locations such as `chairs`, `bed`, `side table`, `countertop`, etc., temporal target ranking now requires the candidate summary to mention the same location family.
   - This avoids matching `put all newspaper on any chairs` to a side-table newspaper task.

3. `temporal_neighbor()` all-task preference.
   - For target phrases containing `all`, rank whole-task summaries such as `collected three newspapers ... placed them all ...` above intermediate substeps such as `first newspaper`.
   - This makes before/after questions prefer the next task after the completed all-object task, not the next sub-action inside the same task.

## Main Run

Command:

```bash
set -a; source .env; set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_current_zs_fast_n20_cached_v1.json \
  --n-samples 20 \
  --require-history-cache \
  --max-prompt-tokens-per-sample 12000 \
  --max-average-prompt-tokens-per-sample 5000 \
  --max-seconds-per-sample 240
```

The first attempt was stopped after a repeated final-answer `SyntaxError`. After adding structured fallback, the run was resumed with:

```bash
set -a; source .env; set +a
conda run --no-capture-output -n active-h-emv python -m llm_emv.eval \
  --cfg teach/simplified/full_graph_aug_zs_fast \
  --dataset teach-dechant \
  --teach-base dataset/TEACh \
  --qa-file data/teach/test_set_50.pkl \
  --output experiments/results/teach/smoke/h50_current_zs_fast_n20_cached_v1.json \
  --n-samples 20 \
  --resume \
  --retry-errors \
  --require-history-cache \
  --max-prompt-tokens-per-sample 12000 \
  --max-average-prompt-tokens-per-sample 5000 \
  --max-seconds-per-sample 240
```

## Result Files

- `experiments/results/teach/smoke/h50_current_zs_fast_n20_cached_v1.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n20_cached_v1.jsonl`
- `experiments/results/teach/smoke/h50_current_zs_fast_n20_cached_v1.log`
- `experiments/results/teach/smoke/h50_current_zs_fast_n20_cached_v1.gemini_2.5_pro-96033d.auto_eval.json`
- `experiments/results/teach/smoke/h50_current_zs_fast_n20_cached_v1_llm_eval.log`
- `experiments/results/teach/smoke/h50_current_zs_fast_n20_cached_v1_metrics.log`

## Stability Metrics

Final JSON contains 20/20 results. The JSONL checkpoint has 21 lines because one stale error row was retried and superseded by the later row for the same sample.

| Metric | Value |
| --- | ---: |
| QA | 20 |
| Valid answer rate | 100.0% |
| Error / empty answer rate | 0.0% |
| Average prompt tokens | 2.26K |
| Max prompt tokens | 3.71K |
| Average completion tokens | 0.40K |
| Max completion tokens | 764 |
| Total prompt tokens | 45,197 |
| Total completion tokens | 7,982 |

Runtime observations:

- No missing h=50 history cache.
- No online `group and summarize` history construction.
- No VQA calls for textual questions.
- No prompt budget stop.
- Structured fallback triggered 5 times after final `answer(...)` truncation; all 5 returned non-empty answers.

## Correctness Evaluation

Command:

```bash
PYTHON_BIN="conda run --no-capture-output -n active-h-emv python" \
  scripts/evaluate_result.sh experiments/results/teach/smoke/h50_current_zs_fast_n20_cached_v1.json
```

Auto-eval result:

| Metric | Value |
| --- | ---: |
| `S_c` semantic correct | 70.0% (14/20) |
| `S_p` partially correct | 25.0% (5/20) |
| Wrong / no-answer | 5.0% (1/20) |

Non-full-correct items:

| Index | Category | Question | Diagnosis |
| ---: | --- | --- | --- |
| 2 | `partially_correct_tmi` | just after putting book on bed | Correct task `boil potato`, but answer includes extra details. |
| 5 | `partially_correct_tmi` | place mug on coffeemachine | Captures coffee preparation but misses some listed GT task variants. |
| 12 | `wrong` | just after putting all newspapers on chairs | Temporal target was mis-ranked to side-table / substep newspaper task before the post-run fix. |
| 15 | `partially_correct_missing` | toggle on faucet | Recall is relevant but incomplete for many faucet-related tasks. |
| 17 | `partially_correct_tmi` | cook 3 potato slices and serve on plate | Includes extra date beyond GT. |
| 18 | `partially_correct_tmi` | serve 1 tomato slice in bowl days ago | Includes extra days beyond GT. |

## Targeted Post-Run Fix

After the n=20 aggregate, the only wrong sample was diagnosed offline without additional LLM calls.

Targeted query:

```python
temporal_neighbor('put all newspaper on any chairs', direction='after')
```

Before the fix:

- Recommended answer: picked up the second newspaper from the dresser and placed it on the side table / armchair.
- Problem: selected a substep or wrong-location newspaper task instead of the next task after the completed chair task.

After the fix:

- Top target: `collected three newspapers ... placed them all on the armchair`.
- Recommended answer: `I prepared a lettuce sandwich by slicing the bread and lettuce, toasting the bread slices, and then assembling the sandwich on a plate...`
- This aligns with GT: `make a sandwich`.

This fix is not reflected in the already-written n=20 aggregate metrics above. It should be validated in the next fresh n=20/n=40 run.

## Conclusion

h=50 cached-prefix n=20 is now technically stable:

- Cache guard works for the selected prefix.
- Prompt token usage remains far below the 12K per-sample and 5K average stop lines.
- Valid answer rate is 100%.
- The remaining correctness bottlenecks are precision issues in temporal/date/task recall, not runaway token or hidden summarization.

Recommended next step:

1. Run a fresh h=50 n=20 or n=40 cached-prefix probe with the post-run temporal fix.
2. Evaluate immediately with `scripts/evaluate_result.sh`.
3. Only consider h=50 n=60 after the fresh probe has no `###ERROR###`, no online summarization, and no new repeated-search/token spikes.
